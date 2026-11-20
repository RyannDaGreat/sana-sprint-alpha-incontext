// UI wiring: theme, matte tools, model picker with cached downloads, generation, real-time mode.

import '@fontsource-variable/inter'; // the UI font
import '@fontsource-variable/jost';   // a built-in choice of the text tool (fonts.js)
import './style.css';
import { MatteEditor } from './matte.js';
import { initPanes } from './panes.js';
import { attachPanZoom } from './panzoom.js';
import { fillFontSelect, ensureFont } from './fonts.js';
import { MaskGallery } from './gallery.js';
import { buildTrimaskUi, DEFAULT_GUI } from './trimask_ui.js';
import { levelFractions, rgbaToMask } from './trimask_ops.js';
import { highwayWeight } from './sneakpeek.js';
import { loadVersions, versionFiles, cachedFlags, cacheState, cacheBytes, clearCache } from './models.js';
import { countEmbeddings, clearEmbeddings } from './embcache.js';
import { saveUiState, loadUiState, clearUiState, readInputs, writeInputs } from './persist.js';
import { addItem, listItems, deleteItem, clearItems, itemBytes } from './history.js';
import { TurnScheduler } from './scheduler.js';
import { rgbaFromComposites, splitterPixels } from './splitter.js';
import { formatBytes, formatEta, rate, mattePixelsToTensor, decodeToPixels, fitRect, cropRect, SIZE, progressLine } from './util.js';
// pipeline.js (with ONNX Runtime) is a separate chunk: ORT's WASM worker re-imports its own
// module URL, which must not be this file (it touches the DOM).
const { Pipeline, sprintTimesteps, fibo2Sigmas, fibo2Mu, canLoadHere } = await import('./pipeline.js');
const { Segmenter, SEG_FILE } = await import('./segment.js');

const REALTIME_DEBOUNCE_MS = 120;   // wait this long after the last edit before a real-time run
const STEP_CHOICES = { sana: [1, 2, 3, 4], fibo2: [1, 2, 4, 8, 12, 20, 30, 50] };  // Sprint: few-step; FIBO2: a flow model (training previews: 20)
const PROGRESS_REDRAW_MS = 100;     // throttle for progress-bar text updates
const PROGRESS_TICK_MS = 1000;      // rows being read or turned into a session count their seconds (m00881)
const SEED_RANGE = 2 ** 31;

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// On GitHub Pages the models come from the one Hugging Face repo (sharded; see src/sana_alpha/publish_models.py).
const PAGES_MODELS = 'https://huggingface.co/OneOverZero/sana-sprint-alpha-incontext/resolve/main/web/';  // the one Hugging Face repo (manifest §12 "Hosting", m00727)
const defaultModels = location.hostname.endsWith('github.io') ? PAGES_MODELS : '../models/web/';
const modelsBase = new URL(params.get('models') ?? defaultModels, location.href).href;

const state = { versions: [], selected: null, pipeline: null, busy: false, pending: false, timer: null, last: null, segmenter: null, hasPhoto: false,
  progress: null, progressFrame: 0 };  // progress: the newest progress event; progressFrame: its pending animation frame

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

let levelsFrame = 0; // pending readout frame (declared first: the editor's constructor already reports a change)
// persistence state (manifest §20), declared before the editor for the same reason
const PERSIST_IDS = ['prompt', 'seed', 'steps', 'guidance', 's-max', 's-mid', 'size', 'lambda', 'gamma', 'mu', 'lambda-v', 'gamma-v', 'mu-v', 'realtime', 'sneak-peek', 'every-layer', 'peek-decoder'];
const SAVE_DEBOUNCE_MS = 400;  // after the last edit
const persist = { ready: false, timer: 0, warned: false };  // ready: only after the saved state was restored (else the starter disc overwrites it)

const editor = new MatteEditor($('matte'), $('overlay'), () => { scheduleRealtime(); scheduleLevels(); scheduleSave(); });
/** Command. Next frame, shows how much of the matte is white / gray / black (readout under the title). */
function scheduleLevels() {
  cancelAnimationFrame(levelsFrame);
  levelsFrame = requestAnimationFrame(() => {
    const f = levelFractions(rgbaToMask(editor.pixels())), pct = (v) => `${(100 * v).toFixed(1)}%`;
    $('levels').textContent = `now: white ${pct(f.white)} · gray ${pct(f.gray)} · black ${pct(f.black)}`;
  });
}

// ---------- persistence (manifest §20, m00848): matte, prompt, seed and settings survive a reload until "Reset all" ----------


/** Command. Debounced save of the editor and the inputs to IndexedDB; a storage failure is shown once, never hidden. */
function scheduleSave() {
  if (!persist.ready) return;
  clearTimeout(persist.timer);
  persist.timer = setTimeout(async () => {
    try {
      await saveUiState(await uiSnapshot());
    } catch (e) {
      if (!persist.warned) status(`Could not save your matte and prompt in this browser (${e.message}); they will not survive a reload.`, true);
      persist.warned = true;
    }
  }, SAVE_DEBOUNCE_MS);
}

/** Query. The editor and inputs as one storable object (persistence and the gallery's "Load settings"). */
async function uiSnapshot() {
  return { raster: await editor.raster.convertToBlob({ type: 'image/png' }), objects: structuredClone(editor.objects),
           invert: editor.invert, inputs: readInputs($, PERSIST_IDS), mode: padMode() ? 'pad' : 'single',
           pad: pad.slots.map(({ prompt, seed, off }) => ({ prompt, seed, off })) };
}

/** Command. Restores the saved matte and inputs, if any; returns whether something was restored. */
async function restoreUiState() {
  let saved;
  try { saved = await loadUiState(); } catch (e) { status(`Could not read your saved matte and prompt (${e.message}).`, true); }
  persist.ready = true;
  if (!saved) return false;
  await applyUiState(saved);
  return true;
}

/** Command. Puts a stored snapshot (uiSnapshot) back into the editor and the inputs. */
async function applyUiState(saved) {
  editor.rctx.clearRect(0, 0, editor.raster.width, editor.raster.height);
  editor.rctx.drawImage(await createImageBitmap(saved.raster), 0, 0);
  editor.rasterRev++;
  editor.objects = saved.objects; editor.invert = saved.invert; editor.selected = null;
  editor.changed();
  editor.resetHistory();
  writeInputs($, saved.inputs);
  if (saved.pad?.length) buildPad(saved.pad);
  if (saved.mode === 'pad') setMode('pad');
}

$('reset-all').addEventListener('click', async () => {
  persist.ready = false;  // no save may land after the delete
  clearTimeout(persist.timer);
  await clearUiState().catch((e) => status(`Could not clear the saved state (${e.message}).`, true));
  location.reload();
});
for (const id of PERSIST_IDS) for (const ev of ['input', 'change']) $(id)?.addEventListener(ev, scheduleSave);

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
  $('th-apply').disabled = $('th-cancel').disabled = !editor.preview || editor.preview.flat;
  $('preview-badge').hidden = !editor.preview;
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

buildTrimaskUi($('trimask-ui'), editor, (params.get('gui') ?? DEFAULT_GUI).toUpperCase());

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
$('th-apply').addEventListener('click', () => { editor.applyPreview(); syncHistoryUi(); });
$('th-cancel').addEventListener('click', () => { editor.cancelPreview(); syncHistoryUi(); });
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
  const inField = e.target.closest('input, textarea, select');
  if ((e.key === 'Delete' || e.key === 'Backspace') && !inField) editor.deleteSelected();
  if (e.key === 'Enter' && editor.path && !inField) { e.preventDefault(); editor.finishPath(); }
  if (e.key === 'Escape' && editor.path) editor.cancelPath();
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

for (const t of document.querySelectorAll('#single-head .tab')) {  // the output views only (the mode and pad tabs have their own)
  t.addEventListener('click', () => {
    document.querySelectorAll('#single-head .tab').forEach((x) => x.classList.toggle('on', x === t));
    for (const view of ['gray', 'rgba', 'alpha', 'uncertainty', 'timeline']) $(`out-${view}`).hidden = t.dataset.view !== view;
    $('timeline-controls').hidden = t.dataset.view !== 'timeline';
    $('tl-plot').hidden = t.dataset.view !== 'timeline' || !tape.errors;
    if (t.dataset.view === 'timeline') showFrame(Number($('tl-slider').value));
  });
}

// ---------- gallery (m00876, m00878: KISS): every finished generation, newest first, in its own IndexedDB store ----------

const GALLERY_THUMB_PX = 192;  // grid thumbnails only; the full view shows the stored lossless original at 1:1
const gal = { items: [], urls: [], open: -1, output: 0 };

/**
 * Command. Saves the finished generation (asynchronous, after the run; never on the compute path): each output
 * canvas as a lossless PNG, a thumbnail, the full UI state and a one-line summary.
 *
 * @param {string[]} ids - output canvases, e.g. ['out-gray', 'out-rgba']
 * @param {string} info - summary (version, prompt, seed, sampler)
 */
async function saveToGallery(ids, info) {
  try {
    const outputs = await Promise.all(ids.map(async (id) => ({ name: id.replace('out-', ''), blob: await new Promise((r) => $(id).toBlob(r, 'image/png')) })));
    const src = $(ids[0]), k = GALLERY_THUMB_PX / Math.max(src.width, src.height);
    const t = new OffscreenCanvas(Math.round(src.width * k), Math.round(src.height * k));
    t.getContext('2d').drawImage(src, 0, 0, t.width, t.height);
    await addItem({ ts: Date.now(), version: state.pipeline.version.id, info, state: await uiSnapshot(), outputs, thumb: await t.convertToBlob({ type: 'image/png' }) });
    if (!$('out-gallery').hidden) renderGallery();  // the gallery page is open
  } catch (e) { status(`Could not save this generation to the gallery (${e.message}).`, true); }
}

/** Command. Lists the stored generations as a newest-first thumbnail grid, with the space they use. */
async function renderGallery() {
  gal.urls.forEach((u) => URL.revokeObjectURL(u));
  gal.items = await listItems().catch((e) => { status(`Could not read the gallery (${e.message}).`, true); return []; });
  gal.urls = gal.items.map((it) => URL.createObjectURL(it.thumb));
  $('gal-grid').innerHTML = '';
  gal.items.forEach((it, i) => {
    const b = document.createElement('button');
    b.className = 'gal-cell';
    b.title = it.info;
    b.innerHTML = `<img src="${gal.urls[i]}" alt=""><span class="meta">${new Date(it.ts).toLocaleString()}</span>`;
    b.addEventListener('click', () => openItem(i));
    $('gal-grid').append(b);
  });
  $('gal-empty').hidden = gal.items.length > 0;
  $('gal-used').textContent = `${gal.items.length} generation${gal.items.length === 1 ? '' : 's'} · ${formatBytes(gal.items.reduce((n, it) => n + itemBytes(it), 0))}`;
}

/** Command. Opens item i in the full view: its first output, lossless at 1:1 (the view scrolls). */
function openItem(i) {
  gal.open = i;
  gal.output = 0;
  const it = gal.items[i];
  $('gal-info').textContent = `${i + 1} / ${gal.items.length} · ${new Date(it.ts).toLocaleString()} · ${it.info}`;
  const other = state.pipeline && it.version !== state.pipeline.version.id;  // m00887: settings only, never a model switch
  $('gal-model-note').hidden = !other;
  $('gal-model-note').textContent = other ? `made with ${it.version}` : '';
  $('gal-outputs').innerHTML = '';
  it.outputs.forEach((o, k) => {
    const b = document.createElement('button');
    b.className = `tab${k === 0 ? ' on' : ''}`;
    b.textContent = o.name;
    b.addEventListener('click', () => showOutput(k));
    $('gal-outputs').append(b);
  });
  showOutput(0);
  $('gal-view').hidden = false;
}

/** Command. Shows output k of the open item. */
function showOutput(k) {
  gal.output = k;
  [...$('gal-outputs').children].forEach((b, j) => b.classList.toggle('on', j === k));
  if ($('gal-img').src) URL.revokeObjectURL($('gal-img').src);
  $('gal-img').src = URL.createObjectURL(gal.items[gal.open].outputs[k].blob);
}

$('gal-close').addEventListener('click', () => { $('gal-view').hidden = true; });
$('gal-load').addEventListener('click', async () => {
  await applyUiState(gal.items[gal.open].state);
  syncSamplerUi();
  scheduleSave();
  $('gal-view').hidden = true;
  setMode(padMode() ? 'pad' : 'single');  // back to the editor with the loaded settings
  status(`Loaded the settings of ${new Date(gal.items[gal.open].ts).toLocaleString()} (made with ${gal.items[gal.open].version}; the loaded model is unchanged).`);
});
$('gal-download').addEventListener('click', () => {
  const it = gal.items[gal.open], o = it.outputs[gal.output], a = document.createElement('a');
  a.href = URL.createObjectURL(o.blob);
  a.download = `${it.version}_${it.ts}_${o.name}.png`;
  a.click();
  URL.revokeObjectURL(a.href);
});
$('gal-delete').addEventListener('click', async () => {
  await deleteItem(gal.items[gal.open].id);
  $('gal-view').hidden = true;
  renderGallery();
});
$('gal-clear').addEventListener('click', async () => {
  if ($('gal-clear').dataset.armed !== '1') {  // two presses (no confirm() in some hosts): the first arms, the second clears
    $('gal-clear').dataset.armed = '1';
    $('gal-clear').textContent = 'Press again to delete all';
    setTimeout(() => { $('gal-clear').dataset.armed = ''; $('gal-clear').textContent = 'Clear gallery'; }, 4000);
    return;
  }
  await clearItems();
  $('gal-clear').dataset.armed = '';
  $('gal-clear').textContent = 'Clear gallery';
  renderGallery();
});
document.addEventListener('keydown', (e) => {
  if ($('gal-view').hidden) return;
  if (e.key === 'Escape') $('gal-view').hidden = true;
  else if (e.key === 'ArrowRight' && gal.open < gal.items.length - 1) openItem(gal.open + 1);
  else if (e.key === 'ArrowLeft' && gal.open > 0) openItem(gal.open - 1);
});

// ---------- sneak peek timeline (m00875): every preview of the last generation, scrubbable after (and during) the run ----------

const TIMELINE_CAP_BYTES = 300 * 2 ** 20;  // keeps the tab's memory bounded; the oldest previews go first beyond it
const TIMELINE_FPS = 12;
const ERROR_PX = 64;  // frames and the final image are compared at this size (a mean error needs no more pixels)
const tape = { frames: [], bytes: 0, dropped: 0, run: 0, next: 0, pending: 0, timer: 0, errors: null };  // not 'timeline': generate() destructures the pipeline's timeline

/** Command. Empties the timeline (a new generation starts); bumps the run so late frames of the old run are ignored. */
function timelineReset() {
  for (const f of tape.frames) f.bitmap.close();
  Object.assign(tape, { frames: [], bytes: 0, dropped: 0, run: tape.run + 1, next: 0, pending: 0, errors: null });
  timelinePlay(false);
  updateTimeline();
}

/**
 * Command. Adds one shown preview, as displayed at its own resolution (lossless: an ImageBitmap of the same pixels).
 * Asynchronous and off the compute path: createImageBitmap resolves later; frames keep their arrival order.
 *
 * @param {ImageData|HTMLCanvasElement} source
 * @param {string} label
 * @param {{w: number, h: number}} size - source pixels (bitmap size)
 * @returns {Promise<void>}
 */
function timelineAdd(source, label, { w, h }) {
  const run = tape.run, slot = tape.next++;
  tape.pending += 1;
  return createImageBitmap(source, 0, 0, w, h).then((bitmap) => {
    tape.pending -= 1;
    if (run !== tape.run) { bitmap.close(); return; }
    tape.frames.push({ bitmap, label, slot });
    tape.frames.sort((x, y) => x.slot - y.slot);
    tape.bytes += 4 * w * h;
    while (tape.bytes > TIMELINE_CAP_BYTES && tape.frames.length > 1) {
      const old = tape.frames.shift();
      tape.bytes -= 4 * old.bitmap.width * old.bitmap.height;
      old.bitmap.close();
      tape.dropped += 1;
    }
    updateTimeline();
  }).catch(fail);
}

/**
 * Query. A bitmap's pixels at ERROR_PX² (bilinear), for the error curve.
 *
 * @param {ImageBitmap} bitmap
 * @returns {Uint8ClampedArray} (ERROR_PX, ERROR_PX, 4)
 */
function smallPixels(bitmap) {
  const c = new OffscreenCanvas(ERROR_PX, ERROR_PX).getContext('2d');
  c.drawImage(bitmap, 0, 0, ERROR_PX, ERROR_PX);
  return c.getImageData(0, 0, ERROR_PX, ERROR_PX).data;
}

/**
 * Pure function. Relative RGB error of an image against a reference: mean((a − ref)²) / mean((ref − 128)²), alpha ignored.
 *
 * @param {Uint8ClampedArray} a - RGBA bytes
 * @param {Uint8ClampedArray} ref - RGBA bytes, same size
 * @returns {number}
 * @example pixelError(new Uint8ClampedArray([128, 128, 128, 255]), new Uint8ClampedArray([228, 128, 128, 255])) // 1
 */
function pixelError(a, ref) {
  let num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) if (i % 4 !== 3) { num += (a[i] - ref[i]) ** 2; den += (ref[i] - 128) ** 2; }
  return num / (den || 1);
}

/** Command. After the final frame is in, each frame's pixel error vs the final image, drawn as a clickable curve under the scrubber. */
function timelineErrors() {
  const n = tape.frames.length;
  if (n < 2) return;
  const ref = smallPixels(tape.frames[n - 1].bitmap);
  tape.errors = tape.frames.map((f) => pixelError(smallPixels(f.bitmap), ref));
  drawErrors();
}

/** Command. Draws the error curve (log scale, steps marked) with the current frame's marker; a click jumps the scrubber. */
function drawErrors() {
  const e = tape.errors, box = $('tl-plot');
  box.hidden = !e || $('out-timeline').hidden;
  if (!e) return;
  const W = 640, H = 140, pad = 34, ys = e.filter((v) => v > 0);
  const lo = Math.log10(Math.min(...ys, 1e-4)), hi = Math.log10(Math.max(...ys, 1));
  const X = (i) => pad + ((W - pad - 6) * i) / Math.max(1, e.length - 1);
  const Y = (v) => 6 + ((H - 6 - pad) * (hi - Math.min(hi, Math.max(lo, Math.log10(Math.max(v, 1e-12)))))) / (hi - lo || 1);
  const steps = tape.frames.map((f, i) => (/end of step/.test(f.label) ? `<line class="step" x1="${X(i)}" x2="${X(i)}" y1="6" y2="${H - pad}"/>` : '')).join('');
  const at = Number($('tl-slider').value);
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%"><title>each preview's pixel error vs the final image (log); click to jump</title>${steps}
    <text x="2" y="${Y(10 ** hi) + 4}">${(10 ** hi).toExponential(0)}</text><text x="2" y="${Y(10 ** lo) + 4}">${(10 ** lo).toExponential(0)}</text>
    <path class="line-shown" d="${e.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('')}"/>
    <line class="marker" x1="${X(at)}" x2="${X(at)}" y1="6" y2="${H - pad}"/>
    <text x="${pad}" y="${H - 8}">error vs the final image, every preview (steps marked)</text></svg>`;
  box.onclick = (ev) => {
    const r = box.querySelector('svg').getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W;
    $('tl-slider').value = Math.round(((x - pad) / (W - pad - 6)) * (e.length - 1));
    timelinePlay(false);
    showFrame(Number($('tl-slider').value));
  };
}

/** Command. Syncs the slider with the frames (a slider at the end follows new frames); shows the tab once there is a frame. */
function updateTimeline() {
  const n = tape.frames.length, s = $('tl-slider'), atEnd = Number(s.value) >= Number(s.max);
  $('tab-timeline').hidden = n === 0;
  s.max = Math.max(0, n - 1);
  if (atEnd) s.value = s.max;
  if (!$('out-timeline').hidden) showFrame(Number(s.value));
}

/** Command. Draws frame i on #out-timeline at 1:1 (the canvas takes the frame's size) and labels it. */
function showFrame(i) {
  const f = tape.frames[i], n = tape.frames.length;
  if (!f) { $('tl-label').textContent = 'No previews yet: generate with Sneak peek on.'; return; }
  const c = $('out-timeline');
  if (c.width !== f.bitmap.width || c.height !== f.bitmap.height) Object.assign(c, { width: f.bitmap.width, height: f.bitmap.height });
  c.getContext('2d').drawImage(f.bitmap, 0, 0);
  $('tl-label').textContent = `${i + 1} / ${n} · ${f.label}` + (tape.dropped ? ` · ${tape.dropped} oldest dropped (memory cap)` : '');
  if (tape.errors) drawErrors();
}

/** Command. Starts or stops playing the frames from the slider's position; stops at the last frame. */
function timelinePlay(on = !tape.timer) {
  clearInterval(tape.timer);
  tape.timer = 0;
  $('tl-play').textContent = on ? 'Pause' : 'Play';
  if (!on) return;
  const s = $('tl-slider');
  if (Number(s.value) >= Number(s.max)) s.value = 0;
  tape.timer = setInterval(() => {
    if (Number(s.value) >= Number(s.max)) { timelinePlay(false); return; }
    s.value = Number(s.value) + 1;
    showFrame(Number(s.value));
  }, 1000 / TIMELINE_FPS);
}
$('tl-slider').addEventListener('input', () => { timelinePlay(false); showFrame(Number($('tl-slider').value)); });
$('tl-play').addEventListener('click', () => timelinePlay());

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

/** Pure function. The model family of a version ('sana' when its manifest has none). @example familyOf({manifest: {}}) // 'sana' */
const familyOf = (v) => v?.manifest.family ?? 'sana';

/** Query. The sampler settings in the UI: steps, guidance, the TrigFlow angles they give, and the output size. */
function samplerSettings() {
  const steps = Number($('steps').value);
  const sMid = steps === 2 ? Number($('s-mid').value) : null;
  const [width, height] = $('size').value.split('x').map(Number);
  return { steps, guidance: Number($('guidance').value), timesteps: sprintTimesteps(steps, Number($('s-max').value), sMid), width, height,
    adapterScale: Number($('lambda-v').value), gamma: Number($('gamma-v').value), mu: Number($('mu-v').value) };  // the typed value (may be outside 0-1, m00893)
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
function equationText(manifest, { steps, guidance, timesteps, width, height, adapterScale, gamma, mu }, seed) {
  const weights = manifest.weights ?? 'weight equation missing from manifest.json';
  const scaled = (manifest.adapter ? weights.replace('λ', adapterScale) : weights).replace('γ', manifest.splitter ? gamma : 'γ').replace('μ', manifest.matte ? mu : 'μ');
  const head = `${scaled} · ${manifest.precision} · `;
  const tail = `${width} × ${height} · seed ${seed}`;
  if (manifest.family === 'fibo2') {
    const sig = fibo2Sigmas(steps, fibo2Mu((width * height) / 256, manifest.sampler.shift_seq, manifest.sampler.shift_mu));
    return `${head}FIBO2 flow Euler, ${steps} step${steps > 1 ? 's' : ''} at σ = ${sig.slice(0, -1).map((v) => +v.toFixed(3)).join(', ')} · ` +
      `CFG ${guidance}${guidance > 1 ? ' (2 passes per step, null prompt)' : ' (off)'} · ${tail}`;
  }
  const s = timesteps.slice(0, -1).map((v) => +v.toFixed(4)).join(', ');
  return `${head}Sprint sampler, ${steps} step${steps > 1 ? 's' : ''} at s = ${s} · guidance ${guidance} (embedded) · ${tail}`;
}

/**
 * Command. Sets the steps choices and the guidance label for a family; with defaults (a newly loaded version), also its
 * default steps and guidance.
 *
 * @param {string} family - 'sana' | 'fibo2'
 * @param {{steps: number, guidance: number}|null} defaults
 */
function applyFamily(family, defaults) {
  const keep = $('steps').value;
  $('steps').innerHTML = STEP_CHOICES[family].map((n) => `<option value="${n}">${n}</option>`).join('');
  $('steps').value = defaults ? String(defaults.steps) : STEP_CHOICES[family].includes(Number(keep)) ? keep : String(STEP_CHOICES[family][1]);
  if (defaults) $('guidance').value = defaults.guidance;
}

/**
 * Command. Syncs the sampler controls: s_mid only applies to 2 steps; the guidance value label; sizes other than
 * 1024² need a version with dynamic-size graphs; the dashed frame shows the matte crop of a non-square size.
 */
function syncSamplerUi() {
  const fibo2 = familyOf(state.pipeline?.version ?? state.selected) === 'fibo2';
  for (const b of document.querySelectorAll('#pad-tabs [data-padview=alpha], #pad-tabs [data-padview=uncertainty]')) b.hidden = !state.pipeline?.splitter;
  const dec = state.pipeline?.decoderFiles ?? {};  // the sneak peek options only where a version ships them (m00880)
  $('every-layer').closest('label').hidden = !state.pipeline?.highway;
  $('peek-decoder').closest('label').hidden = Object.keys(dec).length === 0;
  for (const o of $('peek-decoder').options) o.disabled = !dec[o.value];
  if (dec[$('peek-decoder').value] === undefined && Object.keys(dec).length) $('peek-decoder').value = Object.keys(dec)[0];
  $('s-max').disabled = fibo2;
  $('s-mid').disabled = fibo2 || $('steps').value !== '2';
  $('guidance-v').textContent = $('guidance').value;
  $('lambda').disabled = $('lambda-v').disabled = !state.pipeline?.hasLambda;
  const splitter = state.pipeline?.splitter;  // manifest §17: γ and the Alpha / Uncertainty views
  $('gamma').disabled = $('gamma-v').disabled = !splitter;
  $('mu').disabled = $('mu-v').disabled = !state.pipeline?.matte;  // manifest §21
  $('tab-alpha').hidden = $('tab-uncertainty').hidden = !splitter;
  $('tab-gray').textContent = splitter ? `RGB (${splitter.heads.join(' · ')})` : 'RGB on gray';
  if (!splitter && ['alpha', 'uncertainty'].includes(document.querySelector('.tab.on')?.dataset.view)) $('tab-gray').click();
  const manifest = (state.pipeline?.version ?? state.selected)?.manifest, dynamic = manifest?.dynamic_size ?? true;
  const noCfg = fibo2 && manifest?.sampler?.cfg === false;  // a distilled few-step model (FIBO2 turbo) samples without CFG
  $('guidance').disabled = noCfg;
  if (noCfg) $('guidance').value = 1;
  $('guidance-label').textContent = fibo2 ? (noCfg ? 'Guidance (CFG off: distilled model)' : 'Guidance (CFG)') : 'Guidance (embedded)';
  for (const o of $('size').options) {
    const [w, h] = o.value.split('x').map(Number);
    o.disabled = !dynamic && o.value !== '1024x1024';
    o.textContent = `${w} × ${h}${w === h && manifest?.trained_sizes.includes(w) ? ' (trained)' : ''}`;
  }
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

/**
 * Command. Takes one progress event (util.js progressCounter) and returns at once: it only stores the event and asks
 * for one animation frame, where drawProgress paints the newest event (m00754: the UI must not slow generation down;
 * events between two frames are dropped, not queued).
 */
function showProgress(p) {
  state.progress = p;
  state.progressFrame ||= requestAnimationFrame(drawProgress);
}

/**
 * Command. Paints the newest progress event: the Steps bar (one segment per sampler step, filled by its transformer
 * parts), the Model bar (one segment per part of every graph: done, running, to do) and the status line.
 */
function drawProgress() {
  state.progressFrame = 0;
  const p = state.progress, steps = $('gen-steps'), model = $('gen-model');
  if (steps.children.length !== p.steps) steps.innerHTML = '<div><div></div></div>'.repeat(p.steps);
  if (model.children.length !== p.model.length) model.innerHTML = '<div></div>'.repeat(p.model.length);
  $('gen-progress').hidden = false;
  p.stepFill.forEach((f, i) => { steps.children[i].firstElementChild.style.width = `${100 * f}%`; });
  p.model.forEach((m, i) => {
    Object.assign(model.children[i], { className: m.state, title: m.where ? `${m.label} · ${m.where}` : m.label });
    model.children[i].style.flexGrow = m.weight;
  });
  $('gen-label').textContent = progressLine(p);
}

const peekFrame = { latest: null, scheduled: false };  // sneak peek: at most one paint per animation frame, newest preview only

/**
 * Command (paints #out-gray on the next frame). Shows a sneak peek preview (manifest §18), scaled to the output size.
 * Called from the fast decoder's queue, never from the compute loop; previews that arrive within one frame collapse to the newest.
 *
 * @param {{data: Float32Array, dims: number[]}} image - (1, 3, h, w) in [0, 1]
 * @param {string} label
 */
function showPeek(image, label) {
  const [, , h, w] = image.dims;
  timelineAdd(new ImageData(previewPixels(image), w, h), label, { w, h });
  peekFrame.latest = { image, label };
  if (peekFrame.scheduled) return;
  peekFrame.scheduled = true;
  requestAnimationFrame(() => {
    peekFrame.scheduled = false;
    if (!state.busy) return;  // the final image is already shown
    const { image, label: text } = peekFrame.latest;
    paintPreview($('out-gray'), image);
    $('out-gray').title = `Sneak peek · ${text}`;
    $('out-empty').hidden = true;
  });
}

/**
 * Pure function. A fast-decoder preview (1, 3, h, w) in [0, 1] as RGBA bytes (h, w, 4).
 *
 * @param {{data: Float32Array, dims: number[]}} image
 * @returns {Uint8ClampedArray}
 * @example previewPixels({data: new Float32Array([1, 0, 0.5]), dims: [1, 3, 1, 1]}) // [255, 0, 128, 255]
 */
function previewPixels({ data, dims: [, , h, w] }) {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) px[4 * i + c] = 255 * data[c * w * h + i];
    px[4 * i + 3] = 255;
  }
  return px;
}

/**
 * Command (draws on `canvas`). Paints a fast-decoder preview (1, 3, h, w) in [0, 1], scaled to the canvas.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{data: Float32Array, dims: number[]}} image
 */
function paintPreview(canvas, image) {
  const [, , h, w] = image.dims, px = previewPixels(image);
  const small = new OffscreenCanvas(w, h);
  small.getContext('2d').putImageData(new ImageData(px, w, h), 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
}

/**
 * Pure function. Relative error mean((a − ref)²) / mean(ref²).
 *
 * @example relError(new Float32Array([1, 1]), new Float32Array([2, 0])) // 1
 */
function relError(a, ref) {
  let num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) { num += (a[i] - ref[i]) ** 2; den += ref[i] ** 2; }
  return num / den;
}

/**
 * Command (fills #peek-timeline-body). The "Preview timeline" accordion (m00822): every preview of the last generation
 * and its error against the final x̂0 (the highway's own estimate and the shown one), step boundaries, and the blend
 * weight heatmap (step × block) of this schedule. Computed once, after the generation (it never slows it down).
 *
 * @param {{steps: number, blocks: number, points: object[], final: Float32Array, weights: object|null, stats: object}} t
 */
function showTimeline(t) {
  const W = 640, H = 200, pad = 34, pts = t.points.map((p) => ({ ...p, raw: relError(p.raw, t.final), shown: relError(p.shown, t.final) }));
  const ys = pts.flatMap((p) => [p.raw, p.shown]).filter((v) => v > 0);
  const lo = Math.log10(Math.min(...ys, 1e-4)), hi = Math.log10(Math.max(...ys, 1));
  const X = (i) => pad + ((W - pad - 6) * i) / Math.max(1, pts.length - 1);
  const Y = (v) => 6 + ((H - 6 - pad) * (hi - Math.min(hi, Math.max(lo, Math.log10(Math.max(v, 1e-12)))))) / (hi - lo || 1);  // clamped to the plot
  const path = (key) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(p[key]).toFixed(1)}`).join('');
  const steps = pts.map((p, i) => (p.block === t.blocks ? `<line class="step" x1="${X(i)}" x2="${X(i)}" y1="6" y2="${H - pad}"/>` : '')).join('');
  const ticks = [lo, hi].map((v) => `<text x="2" y="${Y(10 ** v) + 4}">${(10 ** v).toExponential(0)}</text>`).join('');
  let heat = '';
  if (t.weights && t.blocks) {
    const cw = 180 / t.steps, ch = 180 / t.blocks;
    for (let s = 0; s < t.steps; s++)
      for (let b = 0; b < t.blocks; b++) {
        const w = highwayWeight(t.weights, s, t.steps, b);
        heat += `<rect x="${s * cw}" y="${180 - (b + 1) * ch}" width="${cw + 0.5}" height="${ch + 0.5}" fill="var(--peek-shown)" fill-opacity="${w.toFixed(3)}"/>`;
      }
    heat = `<svg class="heat" viewBox="0 0 180 200"><title>blend weight w on the highway: step (x) × block (y)</title>${heat}<text x="0" y="196">step 1 → ${t.steps} · block 1 ↑ ${t.blocks}</text></svg>`;
  }
  $('peek-timeline-body').innerHTML = `<div class="peek-plot"><svg viewBox="0 0 ${W} ${H}" width="100%">
    <title>preview error vs the final image</title>${steps}${ticks}
    <path class="line-raw" d="${path('raw')}"/><path class="line-shown" d="${path('shown')}"/>
    <text x="${pad}" y="${H - 8}">${pts.length} previews (gray: highway estimate, blue: shown) · ${t.stats.shown} decoded, ${t.stats.dropped} dropped, ${(100 * t.stats.fraction).toFixed(1)}% of the time</text>
  </svg>${heat}</div>`;
  $('peek-timeline').hidden = false;
}

const single = { sched: new TurnScheduler(), runs: 0, key: null };  // one job at a time; real-time edits cancel it (§20, item 134)

/** Command. Runs one generation from the current editor state; if one is running, queues one more (and, in real time, cancels the running one). */
async function generate() {
  if (!state.pipeline) return;
  if (padMode()) return padRunAll();
  if (state.busy) {
    state.pending = true;
    if ($('realtime').checked) single.sched.cancel(single.key);  // drop the stale run between two graph parts; its buffers are freed
    return;
  }
  state.busy = true;
  single.key = `run${++single.runs}`;
  syncRunButtons();
  const sampler = samplerSettings(), seed = Number($('seed').value), { width, height } = sampler;
  status(`Generating ${width} × ${height}, ${sampler.steps} step${sampler.steps === 1 ? '' : 's'}: progress below.`);
  const matte = matteAtSize(width, height);
  $('gen-steps').innerHTML = $('gen-model').innerHTML = '';
  timelineReset();
  if ($('sneak-peek').checked && state.pipeline.decoderFiles) await state.pipeline.useDecoder($('peek-decoder').value).catch(fail);  // loads the chosen one once
  if ($('sneak-peek').checked && state.pipeline.fastDecoder) Object.assign($('out-gray'), { width, height });  // previews paint here
  const result = await state.pipeline.generate({
    prompt: $('prompt').value, matte: mattePixelsToTensor(matte, width * height), matteRev: editor.rev, seed, ...sampler,
    onPreview: $('sneak-peek').checked ? showPeek : null,
    everyLayer: $('every-layer').checked,
    gate: single.sched.gateFor(single.key),  // Cancel / Pause (m00894) and real-time edits act between graph parts
  }, showProgress).catch((e) => { if (e.name !== 'AbortError') fail(e); return null; }).finally(() => { state.busy = false; syncRunButtons(); });
  if (state.cancelled) {
    state.cancelled = false;
    status(`Cancelled at ${state.progress ? progressLine(state.progress) : 'the start'}. The previews so far stay in the Timeline tab.`);
  }
  if (!result) {  // cancelled (a newer edit is pending) or failed
    if (state.pending) { state.pending = false; generate(); }
    return;
  }
  const { image, images, timings, timeline } = result;
  $('equation').textContent = equationText(state.pipeline.version.manifest, sampler, seed);
  const views = outputViews(result, matte, width, height);
  for (const [id, pixels, w] of views) {
    Object.assign($(id), { width: w, height });
    $(id).getContext('2d').putImageData(new ImageData(pixels, w, height), 0, 0);
  }
  $('out-empty').hidden = true;
  $('dl-gray').disabled = $('dl-rgba').disabled = false;
  if (tape.next) timelineAdd($('out-gray'), 'final image', { w: width, h: height }).then(timelineErrors);  // the gray image (head 0)
  saveToGallery(views.map(([id]) => id), `${$('prompt').value.slice(0, 80)} · seed ${seed} · ${equationText(state.pipeline.version.manifest, sampler, seed)}`);
  state.last = `${state.pipeline.version.id}_seed${$('seed').value}`;
  showTimings(timings);
  if (timeline?.final) showTimeline(timeline);
  status(`Done in ${timings.total.toFixed(0)} ms (${state.pipeline.ep}).`);
  refreshCacheInfo();
  if (state.pending) { state.pending = false; generate(); }
}

/**
 * Query (uses the loaded splitter's backgrounds). A finished generation's output views as [canvas id, RGBA bytes, width]:
 * splitter versions: the heads side by side, RGBA, alpha (a matte version's matte head, splitFit), uncertainty; otherwise RGB on gray and RGBA (alpha from the mask).
 * Single mode and the four-way pad share it (m00886).
 *
 * @param {{image: object, images?: Float32Array[]}} result
 * @param {Uint8ClampedArray} matte - (height, width, 4) the mask at the output size
 * @returns {[string, Uint8ClampedArray, number][]}
 */
/**
 * Pure function. The RGBA fit of a splitter generation: the iterative least-squares fit over the background heads gives
 * color and α; a matte version (manifest §21) takes α from its matte head instead (the gray image's mean over RGB,
 * clipped to [0, 1]) and keeps the fit's color and uncertainty.
 *
 * @param {Float32Array[]} images - K heads, each (3, H, W) in [0, 1]; the matte head last when pipeline.matte
 * @param {{splitter: {backgrounds: number[]}, matte: object|null}} pipeline
 * @param {number} hw - H·W
 * @returns {{color: Float32Array, alpha: Float32Array, uncertainty: Float32Array}}
 *
 * @example splitFit([gray, white, black, Float32Array.of(0.2, 0.2, 0.2)], { splitter: { backgrounds: [0.5, 1, 0] }, matte: {} }, 1).alpha
 *   // Float32Array [0.2]: from the matte head, not the fit
 */
function splitFit(images, pipeline, hw) {
  const k = pipeline.splitter.backgrounds.length;
  const fit = rgbaFromComposites(images.slice(0, k), pipeline.splitter.backgrounds, hw);
  if (!pipeline.matte) return fit;
  const m = images[k], alpha = new Float32Array(hw);
  for (let i = 0; i < hw; i++) alpha[i] = Math.min(1, Math.max(0, (m[i] + m[hw + i] + m[2 * hw + i]) / 3));
  return { ...fit, alpha };
}

function outputViews({ image, images }, matte, width, height) {
  const { onGray, cutout } = decodeToPixels(image, matte, width * height);
  const split = images && splitterPixels(images, splitFit(images, state.pipeline, width * height), width, height);
  return split ? [['out-gray', split.strip, images.length * width], ['out-rgba', split.rgba, width], ['out-alpha', split.alpha, width],
                  ['out-uncertainty', split.uncertainty, width]]
    : [['out-gray', onGray, width], ['out-rgba', cutout, width]];  // without a splitter, RGBA takes its alpha from the input mask
}

/** Query. Whether a generation is running (single mode, or any pad slot). */
function running() { return padMode() ? pad.slots.some((sl) => sl.running) : state.busy; }

/** Query. The scheduler keys of what is running. */
function runKeys() { return padMode() ? pad.slots.filter((sl) => sl.running).map((sl) => sl.key) : state.busy ? [single.key] : []; }

/** Command. Cancels what is running (its next graph part throws; its buffers are freed); the previews and Timeline so far stay. */
function cancelRun() {
  if (padMode()) { pad.started = false; for (const sl of pad.slots) if (sl.running) { pad.sched.cancel(sl.key); sl.state.textContent = 'cancelled'; } }
  else { state.pending = false; single.sched.cancel(single.key); state.cancelled = true; }
  syncRunButtons();
}

/** Command. Generate ⇄ Cancel, and Pause shown only while something runs. */
function syncRunButtons() {
  const on = running();
  $('generate').textContent = on ? 'Cancel' : 'Generate';
  $('generate').classList.toggle('primary', !on);
  $('pause').hidden = !on;
  if (!on) $('pause').textContent = 'Pause';
}

/** Command. In real-time mode (always in the four-way pad), schedules a debounced generation after an edit. */
function scheduleRealtime() {
  if (!state.pipeline || !($('realtime').checked || (padMode() && pad.started))) return;
  clearTimeout(state.timer);
  state.timer = setTimeout(generate, REALTIME_DEBOUNCE_MS);
}

// ---------- four-way pad (manifest §20, m00848) ----------

const PAD_SLOTS = 4;
const PAD_EDIT_DEBOUNCE_MS = 300;  // a slot's prompt / seed edit restarts it after this pause
const pad = { sched: new TurnScheduler(), slots: [], started: false, view: 'gray' };

/** Query. Whether the four-way pad tab is the active mode. */
function padMode() { return !$('pad').hidden; }

/**
 * Command. Builds the pad's slot cards (once): image, prompt, seed. Slot 0 starts with the main prompt; seeds 0..3.
 *
 * @param {{prompt: string, seed: number, off: boolean}[]} [saved] - restored slots (persistence)
 */
function buildPad(saved) {
  $('pad').innerHTML = '';
  pad.slots = Array.from({ length: PAD_SLOTS }, (_, i) => {
    const s = saved?.[i] ?? { prompt: $('prompt').value, seed: i, off: false };
    const el = document.createElement('div');
    el.className = `slot${s.off ? ' off' : ''}`;
    el.innerHTML = `<div class="square"><canvas width="1024" height="1024"></canvas><p class="slot-state">idle</p></div>
      <input type="text" class="slot-prompt" aria-label="Prompt ${i + 1}">
      <div class="slot-row"><label>Seed <input type="number" min="0"></label><button class="link">New seed</button></div>`;
    $('pad').append(el);
    const slot = { ...s, el, canvas: el.querySelector('canvas'), state: el.querySelector('.slot-state'), key: null, gen: 0, timer: 0, t0: 0 };
    const [prompt, seed] = [el.querySelector('.slot-prompt'), el.querySelector('input[type=number]')];
    prompt.value = slot.prompt; seed.value = slot.seed;
    const edited = () => { slot.prompt = prompt.value; slot.seed = Number(seed.value); scheduleSave();
      clearTimeout(slot.timer); if (pad.started) slot.timer = setTimeout(() => padRestart(i), PAD_EDIT_DEBOUNCE_MS); };
    prompt.addEventListener('input', edited); seed.addEventListener('input', edited);
    el.querySelector('.slot-row button').addEventListener('click', () => { seed.value = Math.floor(Math.random() * SEED_RANGE); edited(); });
    el.querySelector('.square').addEventListener('click', () => {  // switch off / on: dims to 50% and pauses; the others get its turns
      slot.off = !slot.off; el.classList.toggle('off', slot.off);
      if (slot.key) pad.sched.setPaused(slot.key, slot.off);
      scheduleSave();
    });
    attachPanZoom(el.querySelector('.square'));  // after the click toggle: a pan drag swallows its click
    return slot;
  });
}

/** Command. Restarts every slot with the current mask and settings (Generate, or a mask / settings edit in the pad). */
function padRunAll() {
  pad.started = true;
  pad.slots.forEach((_, i) => padRestart(i));
}

/**
 * Command. Cancels slot i's running job (its next graph part throws, freeing its buffers) and starts it over.
 *
 * @param {number} i
 */
async function padRestart(i) {
  const slot = pad.slots[i];
  if (slot.key) pad.sched.cancel(slot.key);
  if (!slot.prompt.trim()) { slot.state.textContent = 'empty: no prompt'; slot.key = null; return; }
  const key = slot.key = `slot${i}:${++slot.gen}`;
  const sampler = samplerSettings(), { width, height } = sampler, matte = matteAtSize(width, height);
  Object.assign(slot.canvas, { width, height });
  slot.t0 = performance.now();
  slot.state.textContent = 'starting';
  slot.running = true;
  syncRunButtons();
  const live = () => slot.key === key;  // a restarted slot ignores its old job's late callbacks
  const res = await state.pipeline.generate({
    prompt: slot.prompt, matte: mattePixelsToTensor(matte, width * height), matteRev: editor.rev, seed: slot.seed, ...sampler,
    onPreview: $('sneak-peek')?.checked ? (image) => { if (live()) paintPreview(slot.canvas, image); } : null,
    gate: pad.sched.gateFor(key, slot.off),
  }, (p) => { if (live()) slot.state.textContent = progressLine(p); }).catch((e) => {
    if (e.name !== 'AbortError' && live()) { slot.state.textContent = `error: ${e.message}`; console.error(e); }
    return null;
  }).finally(() => { if (live()) { slot.running = false; syncRunButtons(); } });
  if (!res || !live()) return;
  slot.views = Object.fromEntries(outputViews(res, matte, width, height).map(([id, px, w]) => [id.replace('out-', ''), { px, w, h: height }]));
  paintSlot(slot);
  slot.state.textContent = `done · seed ${slot.seed} · ${((performance.now() - slot.t0) / 1000).toFixed(1)} s`;
}

/** Command. Paints a finished slot's chosen output view (the pad's view tabs, m00886); a missing view falls back to gray. */
function paintSlot(slot) {
  const v = slot.views?.[pad.view] ?? slot.views?.gray;
  if (!v) return;
  Object.assign(slot.canvas, { width: v.w, height: v.h });
  slot.canvas.getContext('2d').putImageData(new ImageData(v.px, v.w, v.h), 0, 0);
  slot.canvas.classList.toggle('checker', pad.view === 'rgba');
}
for (const b of document.querySelectorAll('#pad-tabs .tab')) b.addEventListener('click', () => {
  pad.view = b.dataset.padview;
  for (const x of document.querySelectorAll('#pad-tabs .tab')) x.classList.toggle('on', x === b);
  pad.slots.forEach(paintSlot);
});

/** Command. Switches between the single output and the four-way pad (the editor and settings are shared). */
function setMode(mode) {
  for (const b of $('mode-tabs').querySelectorAll('.tab')) b.classList.toggle('on', b.dataset.mode === mode);
  const gallery = mode === 'gallery';  // top-level (m00888): the gallery replaces the whole workbench; the editor state stays as it was
  document.querySelector('.workbench').hidden = gallery;
  $('out-gallery').hidden = !gallery;
  if (gallery) { renderGallery(); return; }
  const isPad = mode === 'pad';
  $('pad').hidden = $('pad-help').hidden = $('pad-tabs').hidden = !isPad;
  $('single-head').hidden = $('equation').hidden = isPad;
  document.querySelector('.output-panel > .square').hidden = isPad;
  if (isPad && !pad.slots.length) buildPad();
  if (!isPad) { for (const s of pad.slots) if (s.key) pad.sched.cancel(s.key); pad.started = false; }
  syncRunButtons();
  scheduleSave();
}
for (const b of $('mode-tabs').querySelectorAll('.tab')) b.addEventListener('click', () => setMode(b.dataset.mode));

/**
 * Command. Pairs a Greek-letter slider with its number box (m00893): the box takes any value (also < 0 or > 1) and is
 * what the graph gets; the slider follows it, pinned at its end when the value is outside its range; moving the slider
 * sets the box. One helper for λ, γ (and μ when it arrives).
 *
 * @param {string} id - the slider's id; its box is `${id}-v`
 */
function bindGreek(id) {
  const slider = $(id), box = $(`${id}-v`);
  slider.addEventListener('input', () => { box.value = slider.value; });
  box.addEventListener('input', () => { slider.value = Math.min(Number(slider.max), Math.max(Number(slider.min), Number(box.value))); scheduleRealtime(); scheduleSave(); });
}
['lambda', 'gamma', 'mu'].forEach(bindGreek);

// Generate turns into Cancel while it runs, with Pause / Resume beside it (m00894); Esc cancels. Both act between graph parts.
$('generate').addEventListener('click', () => (running() ? cancelRun() : generate()));
$('pause').addEventListener('click', () => {
  const on = $('pause').textContent === 'Pause';
  runKeys().forEach((k) => (padMode() ? pad.sched : single.sched).setPaused(k, on));
  $('pause').textContent = on ? 'Resume' : 'Pause';
  status(on ? 'Paused (between graph parts). Press Resume.' : 'Resumed.');
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && running() && $('gal-view').hidden) cancelRun(); });
$('prompt').addEventListener('input', scheduleRealtime);
$('seed').addEventListener('input', scheduleRealtime);
for (const id of ['steps', 'guidance', 's-max', 's-mid', 'size', 'lambda', 'gamma', 'mu']) $(id).addEventListener('input', () => { syncSamplerUi(); scheduleRealtime(); });
$('realtime').addEventListener('change', scheduleRealtime);
$('new-seed').addEventListener('click', () => { $('seed').value = Math.floor(Math.random() * SEED_RANGE); scheduleRealtime(); });
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate(); });

// ---------- model class: alpha matte or trimask ----------

// What the editor's values mean to each class (training: a matte is alpha; a trimask is 1 / 0.5 / 0, src/sana_alpha/trimask)
const COND_TEXT = {
  matte: ['Matte', 'white = object · gray = partly transparent · black = background'],
  trimask: ['Trimask', 'white = object · gray = unknown · black = background'],
};

/**
 * Pure function. The condition class of a version: its manifest's `cond` ("trimask"), else "matte" (the older exports have no `cond`).
 *
 * @example condOf({manifest: {cond: 'trimask'}}) // 'trimask'
 * @example condOf({manifest: {}}) // 'matte'
 */
const condOf = (v) => v.manifest.cond ?? 'matte';

/**
 * Command. Shows the editor as version v's class (the heading) and marks its trained sizes. Both classes keep the RGBA
 * cutout: its alpha is the editor image (for a trimask, gray = 0.5 alpha; m00687: wanted even if imperfect).
 */
function applyCond(v) {
  const cond = condOf(v);
  if (!COND_TEXT[cond]) throw new Error(`${v.id}: unknown manifest cond "${cond}"`);
  [$('cond-name').textContent, $('cond-text').textContent] = COND_TEXT[cond];
  syncSamplerUi();
}

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
  const allFlags = await Promise.all(state.versions.map((v) => cachedFlags(versionFiles(v)))); // all at once, then one DOM swap (m00851)
  const cards = state.versions.map((v, i) => {
    const files = versionFiles(v);
    const flags = allFlags[i];
    const total = files.reduce((a, f) => a + f.bytes, 0);
    const nCached = flags.filter(Boolean).length;
    const card = document.createElement('div');
    card.className = `version${state.selected === v ? ' selected' : ''}`;
    const cached = cacheState(flags);
    card.innerHTML = `
      <span class="version-head"><i class="cache-dot" data-state="${cached}" role="img"></i><span class="version-name"></span></span>
      <span class="label"></span>
      <span class="meta desc"></span>
      <span class="meta">${COND_TEXT[condOf(v)][0]} model · ${formatBytes(total)} · ${nCached === files.length ? 'cached' : `${nCached}/${files.length} files cached`}</span>
      <div class="row"><button>${state.pipeline?.version === v ? 'Loaded' : 'Load'}</button></div>`;
    const dot = card.querySelector('.cache-dot');
    dot.title = dot.ariaLabel = { none: 'Not cached', partial: `Partly cached: ${nCached} of ${files.length} files`, full: 'Fully cached' }[cached];
    card.querySelector('.version-name').textContent = v.id;
    card.querySelector('.label').textContent = v.manifest.label ?? '';
    card.querySelector('.desc').textContent = v.manifest.description ?? '';
    const btn = card.querySelector('button');
    btn.disabled = state.pipeline?.version === v || v.manifest.missing.length > 0;
    if (v.manifest.missing.length) btn.textContent = `Not ready: no ${v.manifest.missing.join(', ')}`;
    btn.addEventListener('click', () => loadVersion(v).catch(fail));
    return card;
  });
  box.replaceChildren(...cards);
}

/** Command. Creates one progress row per file in `box` and returns its updater. */
/**
 * Pure function. A progress row's key: role, file name and URL (content-addressed URLs alone collide for identical files).
 *
 * @param {{role: string, name: string, url: string}} f
 * @returns {string}
 *
 * @example rowKey({ role: 'transformer_chunks', name: 'transformer.part3.onnx', url: 'https://x/blobs/ab/abcd.000' })
 * // 'transformer_chunks/transformer.part3.onnx/https://x/blobs/ab/abcd.000'
 */
function rowKey(f) { return `${f.role}/${f.name}/${f.url}`; }

function progressRows(files, box = $('files')) {
  box.innerHTML = '';
  clearInterval(box.ticker);  // a previous load's ticker
  const t0 = performance.now();
  const rows = new Map();
  for (const f of files) {
    const row = document.createElement('div');
    row.className = 'file';
    row.innerHTML = '<span class="name"></span><div class="bar"><div></div></div><span class="info meta">queued</span>';
    row.querySelector('.name').textContent = f.name;
    box.append(row);
    const k = rowKey(f);  // two files with the same bytes share a content-addressed URL: never key rows by URL alone (rows stuck at 'waiting')
    rows.set(k, [...(rows.get(k) ?? []), { row, drawn: 0, base: null, busy: null }]);
  }
  // rows in use right now say so and count seconds (m00881: 'no indication if any are LOADING'); one tick per second
  const all = [...rows.values()].flat();
  const show = (r) => { r.row.querySelector('.info').textContent = `${r.base ?? formatBytes(r.total ?? 0)} · ${r.busy.label}… ${((performance.now() - r.busy.since) / 1000).toFixed(0)} s`; };
  box.ticker = setInterval(() => { for (const r of all) if (r.busy) show(r); }, PROGRESS_TICK_MS);
  return (f, done, total, state) => { for (const r of rows.get(rowKey(f)) ?? []) update(r, done, total, state); };
  /**
   * Command. One row's state: false = downloading (done of total), true = cached, 'reading' = being read from the
   * cache now, 'creating' = its session is being created now, null = in a session ('· loaded').
   */
  function update(r, done, total, state) {
    r.total = total;
    if (state === 'reading' || state === 'creating') {
      r.row.querySelector('.bar > div').style.width = '100%';
      r.busy = { label: state === 'reading' ? 'reading' : 'creating session', since: performance.now() };
      show(r);
      return;
    }
    if (state === null) {  // its bytes are in a session (Pipeline.load): the last state of every row (never a bare 'queued')
      r.busy = null;
      r.row.querySelector('.bar > div').style.width = '100%';
      r.row.querySelector('.info').textContent = `${r.base ?? formatBytes(total)} · loaded`;
      return;
    }
    const now = performance.now();
    if (done < total && now - r.drawn < PROGRESS_REDRAW_MS) return;
    r.drawn = now;
    r.row.querySelector('.bar > div').style.width = `${(100 * done) / total}%`;
    r.row.classList.toggle('cached', state);
    if (state) { r.base = `${formatBytes(total)} · cached`; r.row.querySelector('.info').textContent = r.base; return; }
    const seconds = (now - t0) / 1000;
    const { mbps, eta } = rate(done, total, seconds);
    if (done === total) r.base = `${formatBytes(total)} · downloaded in ${seconds.toFixed(1)} s · ${mbps.toFixed(1)} MB/s`;
    r.row.querySelector('.info').textContent = done === total ? r.base
      : `${formatBytes(done)} / ${formatBytes(total)} · ${mbps.toFixed(1)} MB/s · ETA ${formatEta(eta)}`;
  }
}

/** Command. Downloads (or reads from cache) and initializes a version. */
async function loadVersion(v) {
  if (!canLoadHere(v)) {  // the other model family's ONNX Runtime build is already running in this page (pipeline.js)
    const url = new URL(location.href);
    url.searchParams.set('version', v.id);
    location.assign(url);
    return;
  }
  if (state.loading) { status(`Still loading ${state.loading}; wait for it before loading another version.`, true); return; }
  state.loading = v.id;  // one load at a time: two overlapping loads create sessions concurrently in one ORT runtime
  try { await loadNow(v); } finally { state.loading = null; }
}

/** Command. loadVersion's body, run while state.loading holds the version id. */
async function loadNow(v) {
  if (state.pipeline) { status(`Releasing ${state.pipeline.version.id}…`); await state.pipeline.release(); }  // its sessions held GPU memory
  state.selected = v;
  state.pipeline = null;
  applyCond(v);
  $('generate').disabled = true;
  await renderVersions();
  const { ep, label } = await pickEp();
  $('ep').textContent = label;
  status(`Loading ${v.id}…`);
  const t0 = performance.now();
  const prepare = (k, n, role) => status(`Preparing ${v.id}: ${role.replace('_', ' ')} · part ${k + 1} of ${n} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
  const step = (what, seconds) => status(`Preparing ${v.id}: ${what} · ${seconds.toFixed(0)} s (${((performance.now() - t0) / 1000).toFixed(0)} s in all)`);
  state.pipeline = await Pipeline.load(v, ep, progressRows(versionFiles(v)), prepare, (msg) => status(msg, true), step);
  const url = new URL(location.href);
  url.searchParams.set('version', v.id);
  history.replaceState(null, '', url);
  $('generate').disabled = false;
  applyFamily(familyOf(v), familyOf(v) === 'fibo2' ? v.manifest.sampler : null);
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
  if (await restoreUiState()) status('Restored your last matte, prompt and settings ("Reset all" starts over).');
  syncSamplerUi();
  state.versions = await loadVersions(modelsBase);
  const wanted = params.get('version');
  const pre = state.versions.find((v) => v.id === wanted);
  if (wanted && !pre) status(`Unknown version "${wanted}"; pick one below.`, true);
  state.selected = pre ?? state.versions[0];
  applyFamily(familyOf(state.selected), null);
  applyCond(state.selected);
  await renderVersions();
  await refreshCacheInfo();
  const fully = (await cachedFlags(versionFiles(state.selected))).every(Boolean);
  if (state.selected.manifest.missing.length) return status(`${state.selected.id} is not ready: no ${state.selected.manifest.missing.join(', ')}.`, true);
  if (pre || fully) await loadVersion(state.selected);
}

initPanes(document.querySelector('.workbench'));
attachPanZoom($('out-gray').parentElement);  // the output views (RGB / RGBA / splitter views) share one view
attachPanZoom(document.querySelector('.gal-view-img'));  // the gallery full view (m00889)
init().catch(fail);
