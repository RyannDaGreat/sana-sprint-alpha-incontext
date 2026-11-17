// UI wiring: theme, matte tools, model picker with cached downloads, generation, real-time mode.

import '@fontsource-variable/jost';
import './style.css';
import { MatteEditor } from './matte.js';
import { fillFontSelect, ensureFont } from './fonts.js';
import { MaskGallery } from './gallery.js';
import { loadVersions, versionFiles, cachedFlags, cacheBytes, clearCache } from './models.js';
import { countEmbeddings, clearEmbeddings } from './embcache.js';
import { formatBytes, formatEta, rate, mattePixelsToTensor, decodeToPixels, fitRect, cropRect, SIZE } from './util.js';
// pipeline.js (with ONNX Runtime) is a separate chunk: ORT's WASM worker re-imports its own
// module URL, which must not be this file (it touches the DOM).
const { Pipeline, sprintTimesteps } = await import('./pipeline.js');
const { Segmenter, SEG_FILE } = await import('./segment.js');

const REALTIME_DEBOUNCE_MS = 120;   // wait this long after the last edit before a real-time run
const PROGRESS_REDRAW_MS = 100;     // throttle for progress-bar text updates
const SEED_RANGE = 2 ** 31;

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// On GitHub Pages the models come from the repo's `models` branch (sharded; see src/sana_alpha/publish_models.py).
const PAGES_MODELS = 'https://raw.githubusercontent.com/RyannDaGreat/sana-sprint-alpha-incontext/models/web/';
const defaultModels = location.hostname.endsWith('github.io') ? PAGES_MODELS : '../models/web/';
const modelsBase = new URL(params.get('models') ?? defaultModels, location.href).href;

const state = { versions: [], selected: null, pipeline: null, busy: false, pending: false, timer: null, last: null, segmenter: null, hasPhoto: false };

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
  editor.resetHistory();
}

/** Command. Enables Undo / Redo to match the editor history. */
function syncHistoryUi() {
  $('undo').disabled = !editor.canUndo();
  $('redo').disabled = !editor.canRedo();
  $('th-apply').disabled = $('th-cancel').disabled = !editor.preview;
}
editor.onHistory = syncHistoryUi;

/** Command. One undo step for a finished text-field edit, when it changed a text object. */
function commitText() { if (editor.selected?.type === 'text') editor.commit(); }

/** Command. Loads the chosen font (and weight) before applying it, so the canvas measures the real glyphs. */
async function applyFont() {
  const font = $('text-font').value, weight = Number($('text-weight').value);
  await ensureFont(font, weight);
  editor.setText({ font, weight });
  commitText();
}

/** Command. Shows the threshold preview for the slider values. */
function previewThreshold() {
  $('th-level-v').textContent = $('th-level').value;
  $('th-soft-v').textContent = $('th-soft').value;
  editor.previewThreshold(Number($('th-level').value), Number($('th-soft').value));
  syncHistoryUi();
}

/** Command. Loads a gallery mask (any image, SVG included) into the matte as an upload (one undo step). */
async function loadMask(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  editor.upload(img, img.naturalWidth || SIZE, img.naturalHeight || SIZE, true);
  showPhoto(null);
}

const gallery = new MaskGallery($('gallery'), (url) => loadMask(url).catch(fail));
$('masks').addEventListener('click', () => gallery.open().catch(fail));

for (const b of document.querySelectorAll('[data-tool]')) {
  b.addEventListener('click', () => {
    document.querySelectorAll('[data-tool]').forEach((x) => x.classList.toggle('on', x === b));
    editor.setTool(b.dataset.tool);
  });
}
$('brush-size').addEventListener('input', (e) => { editor.brushSize = Number(e.target.value); $('brush-size-v').textContent = e.target.value; });
$('text-value').addEventListener('input', (e) => editor.setText({ value: e.target.value }));
fillFontSelect($('text-font'));
$('text-font').value = editor.text.font;
$('text-font').addEventListener('change', () => applyFont().catch(fail));
$('text-weight').addEventListener('change', () => applyFont().catch(fail));
$('text-size').addEventListener('input', (e) => editor.setText({ size: Number(e.target.value) }));
for (const id of ['text-value', 'text-size']) $(id).addEventListener('change', commitText);
$('th-level').addEventListener('input', previewThreshold);
$('th-soft').addEventListener('input', previewThreshold);
$('th-apply').addEventListener('click', () => { editor.applyThreshold(); syncHistoryUi(); });
$('th-cancel').addEventListener('click', () => { editor.cancelThreshold(); syncHistoryUi(); });
$('undo').addEventListener('click', () => editor.undo());
$('redo').addEventListener('click', () => editor.redo());
$('invert').addEventListener('click', () => editor.toggleInvert());
$('clear').addEventListener('click', () => editor.clear());
$('delete').addEventListener('click', () => editor.deleteSelected());
$('upload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) { editor.upload(await createImageBitmap(file)); showPhoto(null); }
  e.target.value = '';
});
addEventListener('keydown', (e) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && !e.target.closest('input, textarea, select')) editor.deleteSelected();
  // Undo/redo shortcuts; text fields keep their own native undo.
  if (!(e.ctrlKey || e.metaKey) || e.target.closest('input[type="text"], input[type="number"], textarea')) return;
  const key = e.key.toLowerCase();
  if (key === 'z' && !e.shiftKey) { e.preventDefault(); editor.undo(); }
  if ((key === 'z' && e.shiftKey) || key === 'y') { e.preventDefault(); editor.redo(); }
});

// ---------- photo → matte ----------

/** Command. Draws the photo letterboxed like the matte (or clears it for null) and applies "Show photo". */
function showPhoto(bitmap) {
  const c = $('photo').getContext('2d');
  c.clearRect(0, 0, SIZE, SIZE);
  if (bitmap) { const r = fitRect(bitmap.width, bitmap.height); c.drawImage(bitmap, r.x, r.y, r.w, r.h); }
  state.hasPhoto = !!bitmap;
  $('matte-square').classList.toggle('photo-on', state.hasPhoto && $('show-photo').checked);
}

/** Command. Segments a photo in the browser (model loaded once, cached in OPFS) and makes its soft mask the matte. */
async function photoToMatte(file) {
  const seg = $('seg-status');
  const bitmap = await createImageBitmap(file);
  if (!state.segmenter) {
    const { ep, label, f16 } = await pickEp();
    const segEp = ep === 'webgpu' && !f16 ? { ep: 'wasm', label: 'WASM (WebGPU has no f16)' } : { ep, label };
    seg.textContent = `Loading BEN2 (${segEp.label})…`;
    state.segmenter = await Segmenter.load(segEp.ep, progressRows([SEG_FILE], $('seg-files')));
    state.segmenter.label = `BEN2 fp16 · ${segEp.label}`;
  }
  seg.textContent = `Segmenting (${state.segmenter.label})…`;
  const { gray, w, h, ms } = await state.segmenter.run(bitmap);
  editor.placeGray(gray, w, h, true);
  showPhoto(bitmap);
  seg.textContent = `Matte in ${ms.toFixed(0)} ms · ${state.segmenter.label}`;
  refreshCacheInfo();
}

$('photo-upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) photoToMatte(file).catch((err) => { $('seg-status').textContent = `Error: ${err.message}`; fail(err); });
});
$('show-photo').addEventListener('change', () => $('matte-square').classList.toggle('photo-on', state.hasPhoto && $('show-photo').checked));

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

/** Query. The sampler settings in the UI: steps, guidance, the TrigFlow angles they give, and the output size. */
function samplerSettings() {
  const steps = Number($('steps').value);
  const sMid = steps === 2 ? Number($('s-mid').value) : null;
  const [width, height] = $('size').value.split('x').map(Number);
  return { steps, guidance: Number($('guidance').value), timesteps: sprintTimesteps(steps, Number($('s-max').value), sMid), width, height,
    adapterScale: Number($('lambda').value) };
}

/**
 * Query. The editor matte's centered width : height crop, resampled to width × height.
 *
 * @returns {Uint8ClampedArray} (height, width, 4) uint8 RGBA
 */
function matteAtSize(width, height) {
  const r = cropRect(width, height);
  const c = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
  c.drawImage($('matte'), r.x, r.y, r.w, r.h, 0, 0, width, height);
  return c.getImageData(0, 0, width, height).data;
}

/**
 * Pure function. The full equation of a generation: weights, precision and sampler.
 *
 * @param {{weights?: string, precision: string, adapter: object|null}} manifest - weights contain "λ" when manifest.adapter
 * @param {{steps: number, guidance: number, timesteps: number[], width: number, height: number, adapterScale: number}} sampler
 * @param {number} seed
 * @returns {string}
 *
 * @example equationText({weights: 'W_sprint + (W_control − W_adapter) + gates · control_v2 step 1500', precision: 'transformer fp16 · Gemma fp16'},
 *   {steps: 2, guidance: 4.5, timesteps: [1.5708, 1.3, 0], width: 1024, height: 1024}, 0)
 * // 'W_sprint + (W_control − W_adapter) + gates · control_v2 step 1500 · transformer fp16 · Gemma fp16 · Sprint sampler, 2 steps at s = 1.5708, 1.3 · guidance 4.5 (embedded) · 1024 × 1024 · seed 0'
 */
function equationText(manifest, { steps, guidance, timesteps, width, height, adapterScale }, seed) {
  const s = timesteps.slice(0, -1).map((v) => +v.toFixed(4)).join(', ');
  const weights = manifest.weights ?? 'weight equation missing from manifest.json';
  return `${manifest.adapter ? weights.replace('λ', adapterScale) : weights} · ${manifest.precision} · ` +
    `Sprint sampler, ${steps} step${steps > 1 ? 's' : ''} at s = ${s} · guidance ${guidance} (embedded) · ${width} × ${height} · seed ${seed}`;
}

/**
 * Command. Syncs the sampler controls: s_mid only applies to 2 steps; the guidance value label; sizes other than
 * 1024² need a version with dynamic-size graphs; the dashed frame shows the matte crop of a non-square size.
 */
function syncSamplerUi() {
  $('s-mid').disabled = $('steps').value !== '2';
  $('guidance-v').textContent = $('guidance').value;
  $('lambda-v').textContent = $('lambda').value;
  $('lambda').disabled = !state.pipeline?.hasLambda;
  const dynamic = state.pipeline?.version.manifest.dynamic_size ?? true;
  for (const o of $('size').options) o.disabled = !dynamic && o.value !== '1024x1024';
  if ($('size').selectedOptions[0].disabled) $('size').value = '1024x1024';
  const [w, h] = $('size').value.split('x').map(Number), r = cropRect(w, h), pct = (v) => `${(100 * v) / SIZE}%`;
  Object.assign($('crop-frame').style, { left: pct(r.x), top: pct(r.y), width: pct(r.w), height: pct(r.h) });
  $('crop-frame').hidden = w === h;
}

/** Command. Fills the timings table. */
function showTimings(t) {
  const order = ['tokenize', 'text_encoder', 'vae_encoder', ...Object.keys(t).filter((k) => k.startsWith('transformer')), 'vae_decoder', 'total'];
  const rows = order.filter((k) => k in t);
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
  const sampler = samplerSettings(), seed = Number($('seed').value), { width, height } = sampler;
  const matte = matteAtSize(width, height);
  const { image, timings } = await state.pipeline.generate({
    prompt: $('prompt').value, matte: mattePixelsToTensor(matte, width * height), matteRev: editor.rev, seed, ...sampler,
  }).catch(fail).finally(() => { state.busy = false; });
  $('equation').textContent = equationText(state.pipeline.version.manifest, sampler, seed);
  const { onGray, cutout } = decodeToPixels(image, matte, width * height);
  for (const [id, pixels] of [['out-gray', onGray], ['out-rgba', cutout]]) {
    Object.assign($(id), { width, height });
    $(id).getContext('2d').putImageData(new ImageData(pixels, width, height), 0, 0);
  }
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
for (const id of ['steps', 'guidance', 's-max', 's-mid', 'size', 'lambda']) $(id).addEventListener('input', () => { syncSamplerUi(); scheduleRealtime(); });
$('realtime').addEventListener('change', scheduleRealtime);
$('new-seed').addEventListener('click', () => { $('seed').value = Math.floor(Math.random() * SEED_RANGE); scheduleRealtime(); });
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate(); });

// ---------- models ----------

/**
 * Query. 'webgpu' when an adapter is available (or ?ep=…), else 'wasm', plus a label naming the
 * adapter and whether it has shader-f16 (fp16 graphs need it; software adapters lack it).
 *
 * @returns {Promise<{ep: string, label: string, f16: boolean}>}
 */
async function pickEp() {
  const adapter = navigator.gpu && (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));
  const ep = params.get('ep') ?? (adapter ? 'webgpu' : 'wasm');
  if (ep !== 'webgpu') return { ep, label: adapter ? 'WASM' : 'WASM (no WebGPU)', f16: false };
  const info = adapter.info ?? {};
  const f16 = adapter.features.has('shader-f16');
  return { ep, f16, label: `WebGPU · ${info.vendor || ''} ${info.architecture || ''} · ${f16 ? 'f16' : 'no f16'}`.replace(/\s+/g, ' ') };
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

/** Command. Creates one progress row per file in `box` and returns its updater. */
function progressRows(files, box = $('files')) {
  box.innerHTML = '';
  const t0 = performance.now();
  const rows = new Map();
  for (const f of files) {
    const row = document.createElement('div');
    row.className = 'file';
    row.innerHTML = '<span class="name"></span><div class="bar"><div></div></div><span class="info meta">waiting</span>';
    row.querySelector('.name').textContent = f.name;
    box.append(row);
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
  syncSamplerUi();
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
  syncSamplerUi();
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
