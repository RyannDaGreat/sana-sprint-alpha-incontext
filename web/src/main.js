// UI wiring: theme, matte tools, model picker with cached downloads, generation, real-time mode.

import '@fontsource-variable/inter'; // the UI font
import '@fontsource-variable/jost';   // a built-in choice of the text tool (fonts.js)
import './style.css';
import { MatteEditor } from './matte.js';
import { initPanes } from './panes.js';
import { fillFontSelect, ensureFont } from './fonts.js';
import { MaskGallery } from './gallery.js';
import { buildTrimaskUi, DEFAULT_GUI } from './trimask_ui.js';
import { levelFractions, rgbaToMask } from './trimask_ops.js';
import { highwayWeight } from './sneakpeek.js';
import { loadVersions, versionFiles, cachedFlags, cacheState, cacheBytes, clearCache } from './models.js';
import { countEmbeddings, clearEmbeddings } from './embcache.js';
import { saveUiState, loadUiState, clearUiState, readInputs, writeInputs } from './persist.js';
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
const PERSIST_IDS = ['prompt', 'seed', 'steps', 'guidance', 's-max', 's-mid', 'size', 'lambda', 'gamma', 'realtime', 'sneak-peek'];
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
      await saveUiState({ raster: await editor.raster.convertToBlob({ type: 'image/png' }), objects: structuredClone(editor.objects),
                          invert: editor.invert, inputs: readInputs($, PERSIST_IDS), mode: padMode() ? 'pad' : 'single',
                          pad: pad.slots.map(({ prompt, seed, off }) => ({ prompt, seed, off })) });
    } catch (e) {
      if (!persist.warned) status(`Could not save your matte and prompt in this browser (${e.message}); they will not survive a reload.`, true);
      persist.warned = true;
    }
  }, SAVE_DEBOUNCE_MS);
}

/** Command. Restores the saved matte and inputs, if any; returns whether something was restored. */
async function restoreUiState() {
  let saved;
  try { saved = await loadUiState(); } catch (e) { status(`Could not read your saved matte and prompt (${e.message}).`, true); }
  persist.ready = true;
  if (!saved) return false;
  editor.rctx.drawImage(await createImageBitmap(saved.raster), 0, 0);
  editor.rasterRev++;
  editor.objects = saved.objects; editor.invert = saved.invert; editor.selected = null;
  editor.changed();
  editor.resetHistory();
  writeInputs($, saved.inputs);
  if (saved.pad?.length) buildPad(saved.pad);
  if (saved.mode === 'pad') setMode('pad');
  return true;
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

for (const t of document.querySelectorAll('.tab')) {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
    for (const view of ['gray', 'rgba', 'alpha', 'uncertainty']) $(`out-${view}`).hidden = t.dataset.view !== view;
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

/** Pure function. The model family of a version ('sana' when its manifest has none). @example familyOf({manifest: {}}) // 'sana' */
const familyOf = (v) => v?.manifest.family ?? 'sana';

/** Query. The sampler settings in the UI: steps, guidance, the TrigFlow angles they give, and the output size. */
function samplerSettings() {
  const steps = Number($('steps').value);
  const sMid = steps === 2 ? Number($('s-mid').value) : null;
  const [width, height] = $('size').value.split('x').map(Number);
  return { steps, guidance: Number($('guidance').value), timesteps: sprintTimesteps(steps, Number($('s-max').value), sMid), width, height,
    adapterScale: Number($('lambda').value), gamma: Number($('gamma').value) };
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
function equationText(manifest, { steps, guidance, timesteps, width, height, adapterScale, gamma }, seed) {
  const weights = manifest.weights ?? 'weight equation missing from manifest.json';
  const scaled = (manifest.adapter ? weights.replace('λ', adapterScale) : weights).replace('γ', manifest.splitter ? gamma : 'γ');
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
  $('s-max').disabled = fibo2;
  $('s-mid').disabled = fibo2 || $('steps').value !== '2';
  $('guidance-v').textContent = $('guidance').value;
  $('lambda-v').textContent = $('lambda').value;
  $('lambda').disabled = !state.pipeline?.hasLambda;
  const splitter = state.pipeline?.splitter;  // manifest §17: γ and the Alpha / Uncertainty views
  $('gamma-v').textContent = $('gamma').value;
  $('gamma').disabled = !splitter;
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
 * Command (draws on `canvas`). Paints a fast-decoder preview (1, 3, h, w) in [0, 1], scaled to the canvas.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{data: Float32Array, dims: number[]}} image
 */
function paintPreview(canvas, { data, dims: [, , h, w] }) {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) px[4 * i + c] = 255 * data[c * w * h + i];
    px[4 * i + 3] = 255;
  }
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
  const sampler = samplerSettings(), seed = Number($('seed').value), { width, height } = sampler;
  status(`Generating ${width} × ${height}, ${sampler.steps} step${sampler.steps === 1 ? '' : 's'}: progress below.`);
  const matte = matteAtSize(width, height);
  $('gen-steps').innerHTML = $('gen-model').innerHTML = '';
  if ($('sneak-peek').checked && state.pipeline.fastDecoder) Object.assign($('out-gray'), { width, height });  // previews paint here
  const result = await state.pipeline.generate({
    prompt: $('prompt').value, matte: mattePixelsToTensor(matte, width * height), matteRev: editor.rev, seed, ...sampler,
    onPreview: $('sneak-peek').checked ? showPeek : null,
    gate: $('realtime').checked ? single.sched.gateFor(single.key) : undefined,  // the turn gate only where a cancel can come (real time); a plain run is ungated (WebGPU readback failures, m00860)
  }, showProgress).catch((e) => { if (e.name !== 'AbortError') fail(e); return null; }).finally(() => { state.busy = false; });
  if (!result) {  // cancelled (a newer edit is pending) or failed
    if (state.pending) { state.pending = false; generate(); }
    return;
  }
  const { image, images, timings, timeline } = result;
  $('equation').textContent = equationText(state.pipeline.version.manifest, sampler, seed);
  const { onGray, cutout } = decodeToPixels(image, matte, width * height);
  const split = images && splitterPixels(images, rgbaFromComposites(images, state.pipeline.splitter.backgrounds, width * height), width, height);
  const views = split ? [['out-gray', split.strip, images.length * width], ['out-rgba', split.rgba, width], ['out-alpha', split.alpha, width],
                         ['out-uncertainty', split.uncertainty, width]]
    : [['out-gray', onGray, width], ['out-rgba', cutout, width]];  // without a splitter, RGBA takes its alpha from the input mask
  for (const [id, pixels, w] of views) {
    Object.assign($(id), { width: w, height });
    $(id).getContext('2d').putImageData(new ImageData(pixels, w, height), 0, 0);
  }
  $('out-empty').hidden = true;
  $('dl-gray').disabled = $('dl-rgba').disabled = false;
  state.last = `${state.pipeline.version.id}_seed${$('seed').value}`;
  showTimings(timings);
  if (timeline?.final) showTimeline(timeline);
  status(`Done in ${timings.total.toFixed(0)} ms (${state.pipeline.ep}).`);
  refreshCacheInfo();
  if (state.pending) { state.pending = false; generate(); }
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
const pad = { sched: new TurnScheduler(), slots: [], started: false };

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
      <textarea rows="2" aria-label="Prompt ${i + 1}"></textarea>
      <div class="slot-row"><label>Seed <input type="number" min="0"></label><button class="link">New seed</button></div>`;
    $('pad').append(el);
    const slot = { ...s, el, canvas: el.querySelector('canvas'), state: el.querySelector('.slot-state'), key: null, gen: 0, timer: 0, t0: 0 };
    const [prompt, seed] = [el.querySelector('textarea'), el.querySelector('input')];
    prompt.value = slot.prompt; seed.value = slot.seed;
    const edited = () => { slot.prompt = prompt.value; slot.seed = Number(seed.value); scheduleSave();
      clearTimeout(slot.timer); if (pad.started) slot.timer = setTimeout(() => padRestart(i), PAD_EDIT_DEBOUNCE_MS); };
    prompt.addEventListener('input', edited); seed.addEventListener('input', edited);
    el.querySelector('button').addEventListener('click', () => { seed.value = Math.floor(Math.random() * SEED_RANGE); edited(); });
    el.querySelector('.square').addEventListener('click', () => {  // switch off / on: dims to 50% and pauses; the others get its turns
      slot.off = !slot.off; el.classList.toggle('off', slot.off);
      if (slot.key) pad.sched.setPaused(slot.key, slot.off);
      scheduleSave();
    });
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
  const live = () => slot.key === key;  // a restarted slot ignores its old job's late callbacks
  const res = await state.pipeline.generate({
    prompt: slot.prompt, matte: mattePixelsToTensor(matte, width * height), matteRev: editor.rev, seed: slot.seed, ...sampler,
    onPreview: $('sneak-peek')?.checked ? (image) => { if (live()) paintPreview(slot.canvas, image); } : null,
    gate: pad.sched.gateFor(key, slot.off),
  }, (p) => { if (live()) slot.state.textContent = progressLine(p); }).catch((e) => {
    if (e.name !== 'AbortError' && live()) { slot.state.textContent = `error: ${e.message}`; console.error(e); }
    return null;
  });
  if (!res || !live()) return;
  const { onGray } = decodeToPixels(res.image, matte, width * height);
  slot.canvas.getContext('2d').putImageData(new ImageData(onGray, width, height), 0, 0);
  slot.state.textContent = `done · seed ${slot.seed} · ${((performance.now() - slot.t0) / 1000).toFixed(1)} s`;
}

/** Command. Switches between the single output and the four-way pad (the editor and settings are shared). */
function setMode(mode) {
  for (const b of $('mode-tabs').querySelectorAll('.tab')) b.classList.toggle('on', b.dataset.mode === mode);
  const isPad = mode === 'pad';
  $('pad').hidden = $('pad-help').hidden = !isPad;
  $('single-head').hidden = $('equation').hidden = isPad;
  document.querySelector('.output-panel > .square').hidden = isPad;
  if (isPad && !pad.slots.length) buildPad();
  if (!isPad) { for (const s of pad.slots) if (s.key) pad.sched.cancel(s.key); pad.started = false; }
  scheduleSave();
}
for (const b of $('mode-tabs').querySelectorAll('.tab')) b.addEventListener('click', () => setMode(b.dataset.mode));

$('generate').addEventListener('click', generate);
$('prompt').addEventListener('input', scheduleRealtime);
$('seed').addEventListener('input', scheduleRealtime);
for (const id of ['steps', 'guidance', 's-max', 's-mid', 'size', 'lambda', 'gamma']) $(id).addEventListener('input', () => { syncSamplerUi(); scheduleRealtime(); });
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
  const t0 = performance.now();
  const rows = new Map();
  for (const f of files) {
    const row = document.createElement('div');
    row.className = 'file';
    row.innerHTML = '<span class="name"></span><div class="bar"><div></div></div><span class="info meta">queued</span>';
    row.querySelector('.name').textContent = f.name;
    box.append(row);
    const k = rowKey(f);  // two files with the same bytes share a content-addressed URL: never key rows by URL alone (rows stuck at 'waiting')
    rows.set(k, [...(rows.get(k) ?? []), { row, drawn: 0 }]);
  }
  return (f, done, total, fromCache) => { for (const r of rows.get(rowKey(f)) ?? []) update(r, done, total, fromCache); };
  function update(r, done, total, fromCache) {
    if (fromCache === null) {  // the file's bytes are in hand (Pipeline.load): a row no progress call reached must not stay 'waiting' (m00861)
      r.row.querySelector('.bar > div').style.width = '100%';  // its session can use it: the last state of every row (never a bare 'queued')
      const info = r.row.querySelector('.info');
      if (!info.textContent.endsWith(' · loaded')) info.textContent = `${r.touched ? info.textContent : formatBytes(total)} · loaded`;
      return;
    }
    r.touched = true;
    const now = performance.now();
    if (done < total && now - r.drawn < PROGRESS_REDRAW_MS) return;
    r.drawn = now;
    r.row.querySelector('.bar > div').style.width = `${(100 * done) / total}%`;
    r.row.classList.toggle('cached', fromCache);
    if (fromCache) { r.row.querySelector('.info').textContent = `${formatBytes(total)} · cached`; return; }
    const seconds = (now - t0) / 1000;
    const { mbps, eta } = rate(done, total, seconds);
    r.row.querySelector('.info').textContent = done === total
      ? `${formatBytes(total)} · downloaded in ${seconds.toFixed(1)} s · ${mbps.toFixed(1)} MB/s`
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
  state.pipeline = await Pipeline.load(v, ep, progressRows(versionFiles(v)), prepare, (msg) => status(msg, true));
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
init().catch(fail);
