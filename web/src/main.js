// UI wiring: theme, matte tools, model picker with cached downloads, generation, real-time mode.

import '@fontsource-variable/jost';
import './style.css';
import { MatteEditor } from './matte.js';
import { loadVersions, versionFiles, cachedFlags, cacheBytes, clearCache } from './models.js';
import { countEmbeddings, clearEmbeddings } from './embcache.js';
import { formatBytes, formatEta, rate, mattePixelsToTensor, decodeToPixels, SIZE } from './util.js';
// pipeline.js (with ONNX Runtime) is a separate chunk: ORT's WASM worker re-imports its own
// module URL, which must not be this file (it touches the DOM).
const { Pipeline } = await import('./pipeline.js');

const REALTIME_DEBOUNCE_MS = 120;   // wait this long after the last edit before a real-time run
const PROGRESS_REDRAW_MS = 100;     // throttle for progress-bar text updates
const SEED_RANGE = 2 ** 31;

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const modelsBase = new URL(params.get('models') ?? '../models/web/', location.href).href;

const state = { versions: [], selected: null, pipeline: null, busy: false, pending: false, timer: null, last: null };

/** Command. Shows a message in the status line (red when it is an error). */
function status(text, isError = false) {
  $('status').textContent = text;
  $('status').style.color = isError ? 'var(--accent)' : '';
}

/** Command. Reports an error in the UI and rethrows it, so it also reaches the console. */
function fail(e) {
  status(`Error: ${e.message}`, true);
  throw e;
}

// ---------- theme ----------

/** Command. Switches light ↔ dark and stores the choice. */
function toggleTheme() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
  editor.drawOverlay();
}

// ---------- matte editor ----------

const editor = new MatteEditor($('matte'), $('overlay'), () => scheduleRealtime());

/** Command. Draws the starter matte: a white disc, so Generate works right away. */
function starterMatte() {
  editor.objects.push({ type: 'ellipse', x: 262, y: 262, w: 500, h: 500 });
  editor.changed();
}

for (const b of document.querySelectorAll('[data-tool]')) {
  b.addEventListener('click', () => {
    document.querySelectorAll('[data-tool]').forEach((x) => x.classList.toggle('on', x === b));
    editor.setTool(b.dataset.tool);
  });
}
$('brush-size').addEventListener('input', (e) => { editor.brushSize = Number(e.target.value); $('brush-size-v').textContent = e.target.value; });
$('text-value').addEventListener('input', (e) => editor.setText({ value: e.target.value }));
$('text-font').addEventListener('change', (e) => editor.setText({ font: e.target.value }));
$('text-weight').addEventListener('change', (e) => editor.setText({ weight: Number(e.target.value) }));
$('text-size').addEventListener('input', (e) => editor.setText({ size: Number(e.target.value) }));
$('invert').addEventListener('click', () => editor.toggleInvert());
$('clear').addEventListener('click', () => editor.clear());
$('delete').addEventListener('click', () => editor.deleteSelected());
$('upload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) editor.upload(await createImageBitmap(file));
  e.target.value = '';
});
addEventListener('keydown', (e) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && !e.target.closest('input, textarea, select')) editor.deleteSelected();
});

// ---------- output ----------

for (const t of document.querySelectorAll('.tab')) {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
    $('out-gray').hidden = t.dataset.view !== 'gray';
    $('out-rgba').hidden = t.dataset.view !== 'rgba';
  });
}

/** Command. Downloads a canvas as PNG. */
function download(canvas, name) {
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }, 'image/png');
}
$('dl-gray').addEventListener('click', () => download($('out-gray'), `${state.last}_on_gray.png`));
$('dl-rgba').addEventListener('click', () => download($('out-rgba'), `${state.last}_rgba.png`));

/** Command. Fills the timings table. */
function showTimings(t) {
  const rows = ['tokenize', 'text_encoder', 'vae_encoder', 'transformer', 'vae_decoder', 'total'].filter((k) => k in t);
  $('timings').innerHTML = rows.map((k) => {
    const v = typeof t[k] === 'number' ? `${t[k].toFixed(1)} ms` : t[k];
    return `<tr class="${k === 'total' ? 'total' : ''}"><td>${k.replace('_', ' ')}</td><td>${v}</td></tr>`;
  }).join('');
}

// ---------- generation ----------

/** Command. Runs one generation from the current editor state; queues one more if busy. */
async function generate() {
  if (!state.pipeline) return;
  if (state.busy) { state.pending = true; return; }
  state.busy = true;
  status('Generating…');
  const matte = editor.pixels();
  const { image, timings } = await state.pipeline.generate({
    prompt: $('prompt').value, matte: mattePixelsToTensor(matte, SIZE * SIZE), matteRev: editor.rev, seed: Number($('seed').value),
  }).catch(fail).finally(() => { state.busy = false; });
  const { onGray, cutout } = decodeToPixels(image, matte, SIZE * SIZE);
  $('out-gray').getContext('2d').putImageData(new ImageData(onGray, SIZE, SIZE), 0, 0);
  $('out-rgba').getContext('2d').putImageData(new ImageData(cutout, SIZE, SIZE), 0, 0);
  $('out-empty').hidden = true;
  $('dl-gray').disabled = $('dl-rgba').disabled = false;
  state.last = `${state.pipeline.version.id}_seed${$('seed').value}`;
  showTimings(timings);
  status(`Done in ${timings.total.toFixed(0)} ms (${state.pipeline.ep}).`);
  refreshCacheInfo();
  if (state.pending) { state.pending = false; generate(); }
}

/** Command. In real-time mode, schedules a debounced generation after an edit. */
function scheduleRealtime() {
  if (!$('realtime').checked || !state.pipeline) return;
  clearTimeout(state.timer);
  state.timer = setTimeout(generate, REALTIME_DEBOUNCE_MS);
}

$('generate').addEventListener('click', generate);
$('prompt').addEventListener('input', scheduleRealtime);
$('seed').addEventListener('input', scheduleRealtime);
$('realtime').addEventListener('change', scheduleRealtime);
$('new-seed').addEventListener('click', () => { $('seed').value = Math.floor(Math.random() * SEED_RANGE); scheduleRealtime(); });
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate(); });

// ---------- models ----------

/**
 * Query. 'webgpu' when an adapter is available (or ?ep=…), else 'wasm', plus a label naming the
 * adapter and whether it has shader-f16 (fp16 graphs need it; software adapters lack it).
 *
 * @returns {Promise<{ep: string, label: string}>}
 */
async function pickEp() {
  const adapter = navigator.gpu && (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));
  const ep = params.get('ep') ?? (adapter ? 'webgpu' : 'wasm');
  if (ep !== 'webgpu') return { ep, label: adapter ? 'WASM' : 'WASM (no WebGPU)' };
  const info = adapter.info ?? {};
  const f16 = adapter.features.has('shader-f16') ? 'f16' : 'no f16';
  return { ep, label: `WebGPU · ${info.vendor || ''} ${info.architecture || ''} · ${f16}`.replace(/\s+/g, ' ') };
}

/** Command. Updates cache size and prompt-cache count. */
async function refreshCacheInfo() {
  $('cache-size').textContent = `cache ${formatBytes(await cacheBytes())}`;
  $('emb-count').textContent = `${await countEmbeddings()} cached prompts`;
}

/** Command. Renders the version cards with their cached status. */
async function renderVersions() {
  const box = $('versions');
  box.innerHTML = '';
  for (const v of state.versions) {
    const files = versionFiles(v);
    const flags = await cachedFlags(files);
    const total = files.reduce((a, f) => a + f.bytes, 0);
    const nCached = flags.filter(Boolean).length;
    const card = document.createElement('div');
    card.className = `version${state.selected === v ? ' selected' : ''}`;
    card.innerHTML = `
      <span class="version-name"></span>
      <span class="label"></span>
      <span class="meta desc"></span>
      <span class="meta">${formatBytes(total)} · ${nCached === files.length ? 'cached' : `${nCached}/${files.length} files cached`}</span>
      <div class="row"><button>${state.pipeline?.version === v ? 'Loaded' : 'Load'}</button></div>`;
    card.querySelector('.version-name').textContent = v.id;
    card.querySelector('.label').textContent = v.manifest.label ?? '';
    card.querySelector('.desc').textContent = v.manifest.description ?? '';
    const btn = card.querySelector('button');
    btn.disabled = state.pipeline?.version === v || v.manifest.missing.length > 0;
    if (v.manifest.missing.length) btn.textContent = `Not ready: no ${v.manifest.missing.join(', ')}`;
    btn.addEventListener('click', () => loadVersion(v).catch(fail));
    box.append(card);
  }
}

/** Command. Creates one progress row per file and returns its updater. */
function progressRows(files) {
  $('files').innerHTML = '';
  const t0 = performance.now();
  const rows = new Map();
  for (const f of files) {
    const row = document.createElement('div');
    row.className = 'file';
    row.innerHTML = '<span class="name"></span><div class="bar"><div></div></div><span class="info meta">waiting</span>';
    row.querySelector('.name').textContent = f.name;
    $('files').append(row);
    rows.set(f.url, { row, drawn: 0 });
  }
  return (f, done, total, fromCache) => {
    const r = rows.get(f.url);
    const now = performance.now();
    if (done < total && now - r.drawn < PROGRESS_REDRAW_MS) return;
    r.drawn = now;
    r.row.querySelector('.bar > div').style.width = `${(100 * done) / total}%`;
    r.row.classList.toggle('cached', fromCache);
    if (fromCache) { r.row.querySelector('.info').textContent = `${formatBytes(total)} · from cache`; return; }
    const seconds = (now - t0) / 1000;
    const { mbps, eta } = rate(done, total, seconds);
    r.row.querySelector('.info').textContent = done === total
      ? `${formatBytes(total)} · downloaded in ${seconds.toFixed(1)} s · ${mbps.toFixed(1)} MB/s`
      : `${formatBytes(done)} / ${formatBytes(total)} · ${mbps.toFixed(1)} MB/s · ETA ${formatEta(eta)}`;
  };
}

/** Command. Downloads (or reads from cache) and initializes a version. */
async function loadVersion(v) {
  state.selected = v;
  state.pipeline = null;
  $('generate').disabled = true;
  await renderVersions();
  const { ep, label } = await pickEp();
  $('ep').textContent = label;
  status(`Loading ${v.id}…`);
  const t0 = performance.now();
  state.pipeline = await Pipeline.load(v, ep, progressRows(versionFiles(v)));
  const url = new URL(location.href);
  url.searchParams.set('version', v.id);
  history.replaceState(null, '', url);
  $('generate').disabled = false;
  status(`${v.id} ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (${ep}). Press Generate.`);
  await renderVersions();
  await refreshCacheInfo();
  scheduleRealtime();
}

$('clear-cache').addEventListener('click', async () => {
  await clearCache();
  await renderVersions();
  await refreshCacheInfo();
});
$('clear-emb').addEventListener('click', async () => {
  await clearEmbeddings();
  if (state.pipeline) state.pipeline.memory.clear();
  await refreshCacheInfo();
});
$('theme').addEventListener('click', toggleTheme);

/** Command. Startup: matte, versions; auto-loads ?version= or a fully cached version. */
async function init() {
  if (!window.isSecureContext) status('This page is not a secure context (use https or localhost): WebGPU and the model cache are unavailable.', true);
  starterMatte();
  state.versions = await loadVersions(modelsBase);
  const wanted = params.get('version');
  const pre = state.versions.find((v) => v.id === wanted);
  if (wanted && !pre) status(`Unknown version "${wanted}"; pick one below.`, true);
  state.selected = pre ?? state.versions[0];
  await renderVersions();
  await refreshCacheInfo();
  const fully = (await cachedFlags(versionFiles(state.selected))).every(Boolean);
  if (state.selected.manifest.missing.length) return status(`${state.selected.id} is not ready: no ${state.selected.manifest.missing.join(', ')}.`, true);
  if (pre || fully) await loadVersion(state.selected);
}

init().catch(fail);
