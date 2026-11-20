// Small helpers: formatting, seeded noise, image tensors, generation progress.

const MB = 1024 * 1024;
export const SIZE = 1024; // image and matte side, in pixels

/**
 * Pure function. Size and offset of a w × h image fitted inside the SIZE² square, centered.
 *
 * @example fitRect(2048, 1024) // {w: 1024, h: 512, x: 0, y: 256}
 * @example fitRect(500, 1000)  // {w: 512, h: 1024, x: 256, y: 0}
 */
export function fitRect(w, h) {
  const s = Math.min(SIZE / w, SIZE / h);
  const fw = Math.round(w * s), fh = Math.round(h * s);
  return { w: fw, h: fh, x: Math.round((SIZE - fw) / 2), y: Math.round((SIZE - fh) / 2) };
}

/**
 * Pure function. The centered w : h crop of the SIZE² matte editor, in editor pixels (the largest that fits).
 *
 * @param {number} w - output width
 * @param {number} h - output height
 * @returns {{x: number, y: number, w: number, h: number}}
 *
 * @example cropRect(1024, 768) // {x: 0, y: 128, w: 1024, h: 768}
 * @example cropRect(512, 512)  // {x: 0, y: 0, w: 1024, h: 1024}
 */
export function cropRect(w, h) {
  const s = SIZE / Math.max(w, h);
  const cw = Math.round(w * s), ch = Math.round(h * s);
  return { x: (SIZE - cw) / 2, y: (SIZE - ch) / 2, w: cw, h: ch };
}

/**
 * Pure function. Formats a byte count as KB below 1 MB, MB with one decimal, GB above 1024 MB.
 *
 * @param {number} bytes
 * @returns {string}
 *
 * @example formatBytes(46996) // '46 KB'
 * @example formatBytes(5 * 1024 * 1024) // '5.0 MB'
 * @example formatBytes(3 * 1024 ** 3) // '3.00 GB'
 */
export function formatBytes(bytes) {
  if (bytes < MB) return `${Math.round(bytes / 1024)} KB`;
  return bytes >= 1024 * MB ? `${(bytes / 1024 / MB).toFixed(2)} GB` : `${(bytes / MB).toFixed(1)} MB`;
}

/**
 * Pure function. Download speed and remaining time.
 *
 * @param {number} done - Bytes received
 * @param {number} total - Bytes expected
 * @param {number} seconds - Time since the download started
 * @returns {{mbps: number, eta: number}} MB/s and seconds left (Infinity before any data)
 *
 * @example rate(10 * 1024 * 1024, 30 * 1024 * 1024, 2) // {mbps: 5, eta: 4}
 */
export function rate(done, total, seconds) {
  const bps = seconds > 0 ? done / seconds : 0;
  return { mbps: bps / MB, eta: bps > 0 ? (total - done) / bps : Infinity };
}

/**
 * Pure function. Formats seconds as m:ss.
 *
 * @param {number} s
 * @returns {string}
 *
 * @example formatEta(75.2) // '1:16'
 * @example formatEta(Infinity) // '–'
 */
export function formatEta(s) {
  if (!Number.isFinite(s)) return '–';
  const t = Math.ceil(s);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

/**
 * Pure function. Standard normal samples from a seeded PRNG (mulberry32 + Box–Muller).
 *
 * @param {number} n - Number of samples
 * @param {number} seed - Integer seed
 * @returns {Float32Array} n samples of N(0, 1)
 *
 * @example
 * const a = gaussianNoise(4, 7); // e.g. Float32Array [0.41, -1.2, 0.03, 0.88]
 * gaussianNoise(4, 7).every((v, i) => v === a[i]) // true (same seed, same noise)
 */
export function gaussianNoise(n, seed) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 1) / 4294967297; // (0, 1)
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(uniform()));
    const a = 2 * Math.PI * uniform();
    out[i] = r * Math.cos(a);
    if (i + 1 < n) out[i + 1] = r * Math.sin(a);
  }
  return out;
}

/**
 * Pure function. Grayscale RGBA pixels → planar matte tensor in [−1, 1], replicated to 3 channels.
 *
 * @param {Uint8ClampedArray} rgba - (H, W, 4) uint8 RGBA; the red channel is the matte
 * @param {number} hw - H · W
 * @returns {Float32Array} (1, 3, H, W) float32 in [−1, 1]
 *
 * @example mattePixelsToTensor(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]), 2)
 * // Float32Array [1, -1, 1, -1, 1, -1]
 */
export function mattePixelsToTensor(rgba, hw) {
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    const v = rgba[4 * i] / 127.5 - 1;
    out[i] = v; out[hw + i] = v; out[2 * hw + i] = v;
  }
  return out;
}

/**
 * Pure function. Planar RGB in [−1, 1] → RGB-on-gray RGBA pixels and the RGBA cutout.
 * Un-compositing: fg = (rgb − (1 − a)·0.5) / a, with a = matte / 255; alpha = matte.
 *
 * @param {Float32Array} chw - (1, 3, H, W) float32 in [−1, 1], RGB on 0.5 gray
 * @param {Uint8ClampedArray} matte - (H, W, 4) uint8; red channel = alpha
 * @param {number} hw - H · W
 * @returns {{onGray: Uint8ClampedArray, cutout: Uint8ClampedArray}} Two (H, W, 4) uint8 RGBA
 *
 * @example
 * const {onGray, cutout} = decodeToPixels(new Float32Array([1, 0, -1]), new Uint8ClampedArray([255, 255, 255, 255]), 1);
 * // onGray  [255, 128, 0, 255]  (pixel values; rounded)
 * // cutout  [255, 128, 0, 255]  (alpha 1: fg = rgb)
 */
export function decodeToPixels(chw, matte, hw) {
  const onGray = new Uint8ClampedArray(4 * hw);
  const cutout = new Uint8ClampedArray(4 * hw);
  for (let i = 0; i < hw; i++) {
    const a = matte[4 * i] / 255;
    for (let c = 0; c < 3; c++) {
      const v = (chw[c * hw + i] + 1) / 2; // [0, 1]
      onGray[4 * i + c] = Math.round(v * 255);
      cutout[4 * i + c] = a > 0 ? Math.round(((v - (1 - a) * 0.5) / a) * 255) : 0;
    }
    onGray[4 * i + 3] = 255;
    cutout[4 * i + 3] = matte[4 * i];
  }
  return { onGray, cutout };
}

// Expected time of a stage as a fraction of one transformer pass, so the bar moves with time, not with stage count.
// From the first real WebGPU run (user's machine, FIBO2, m00722): pass ≈ 42.9 s, text encoder 5.1 s, trimask encoder 3.3 s,
// decoder 4.9 s. SANA's ratios are not measured; it uses the same ones.
const STAGE_PASS_FRACTION = { text: 0.12, matte: 0.076, decoder: 0.113 };

/**
 * Pure function. The 1-based units (blocks, layers) a part holds: web_chunks.split puts unit b of B, 0-based, in part
 * ⌊b·K/B⌋.
 *
 * @param {number} c - part index, 0-based
 * @param {number} k - parts
 * @param {number} b - units
 * @returns {[number, number]} first and last unit, 1-based
 *
 * @example partUnits(0, 11, 46) // [1, 5]
 * @example partUnits(10, 11, 46) // [43, 46]
 * @example partUnits(3, 20, 20) // [4, 4]
 */
export function partUnits(c, k, b) {
  return [Math.ceil((c * b) / k) + 1, Math.ceil(((c + 1) * b) / k)];
}

/**
 * Pure function. Where a part is, as text: its unit range out of the stage's units.
 *
 * @param {[number, number]} r - first and last unit, 1-based
 * @param {string} unit - the unit's name ('block', 'layer')
 * @param {number} units - the stage's unit count
 * @returns {string}
 *
 * @example unitText([10, 13], 'block', 46) // 'blocks 10–13 of 46'
 * @example unitText([14, 14], 'layer', 26) // 'layer 14 of 26'
 */
export function unitText([first, last], unit, units) {
  return `${first === last ? `${unit} ${first}` : `${unit}s ${first}–${last}`} of ${units}`;
}

/**
 * Command (each report calls onProgress; reads the clock). Progress of one generation for two bars and a status line.
 * A generation is a fixed list of parts: the stages before the sampler steps (`before`), steps × passes transformer
 * passes of `transformer.parts` parts each, the stages after (`after`); a stage of a split graph (web_chunks.split) is
 * one part per unit group, an unsplit one a single part. The caller reports each part as it STARTS (`start(i)`, or
 * `finish()` at the end): the parts before it are done, it is running; a report at or before the newest one is ignored. Nothing here waits: onProgress must only store
 * the event (main.js paints at most once per animation frame), so progress costs no compute time (m00754).
 * Time weights: a transformer part is 1, a stage STAGE_PASS_FRACTION[stage] × transformer.parts, split evenly over its
 * parts. Event: {running: label | null, where: text | null, step, steps, pass, passes, stepFill: [0..1] per step,
 * model: [{label, weight, state: 'done'|'running'|'todo'}] (the Model bar: before-stage parts, the current pass's parts,
 * after-stage parts), done, total, elapsed ms, remaining ms | null (units left × mean transformer-part time)}.
 *
 * @param {{steps: number, passes: number, transformer: {parts: number, units: number|null},
 *   before: {stage: string, label: string, parts: number, units: number|null, unit: string}[], after: same}} plan -
 *   units: the graph's unit count (manifest chunked[role].units; null when unsplit), unit: its name
 * @param {Function} onProgress
 * @returns {{start: (i: number) => void, stagePart: (j: number, c: number) => number,
 *   passPart: (step: number, pass: number, c: number) => number, afterPart: (j: number, c: number) => number, finish: () => void}}
 */
export function progressCounter({ steps, passes, transformer, before, after }, onProgress) {
  const k = transformer.parts, perStep = passes * k;
  const where = (c, g) => (g.units ? unitText(partUnits(c, g.parts, g.units), g.unit, g.units) : null);
  const expand = (list) => list.flatMap((g) => Array.from({ length: g.parts }, (_, c) => ({
    label: g.label, where: where(c, g), weight: (STAGE_PASS_FRACTION[g.stage] * k) / g.parts })));
  const pre = expand(before), post = expand(after), first = (list, j) => list.slice(0, j).reduce((s, g) => s + g.parts, 0);
  const tf = { ...transformer, unit: 'block' };
  const parts = [...pre, ...Array.from({ length: steps * perStep }, (_, i) => ({ label: 'transformer', where: where(i % k, tf), weight: 1 })), ...post];
  const total = parts.reduce((s, q) => s + q.weight, 0), t0 = performance.now(), passStart = pre.length, postStart = pre.length + steps * perStep;
  let partTime = 0, partCount = 0, last = null;  // last: [part index, start time]
  const report = (i) => {
    if (last && i <= last[0]) return;  // GPU completions can land after a later part (or finish) was reported
    const now = performance.now();
    if (last && last[0] >= passStart && last[0] < postStart) { partTime += now - last[1]; partCount += 1; }
    last = [i, now];
    const done = parts.slice(0, i).reduce((s, q) => s + q.weight, 0), x = i - passStart, inSteps = x >= 0 && x < steps * perStep;
    const step = inSteps ? Math.floor(x / perStep) + 1 : 0, pass = inSteps ? Math.floor((x % perStep) / k) + 1 : 0;
    const c = inSteps ? x % k : -1, state = (j) => (j < i ? 'done' : j === i ? 'running' : 'todo');
    const passState = (j) => (x < 0 ? 'todo' : x >= steps * perStep ? 'done' : j < c ? 'done' : j === c ? 'running' : 'todo');
    onProgress({
      running: i < parts.length ? parts[i].label : null, where: i < parts.length ? parts[i].where : null, step, steps, pass, passes,
      stepFill: Array.from({ length: steps }, (_, s) => Math.min(1, Math.max(0, (x - s * perStep) / perStep))),
      model: [...pre.map((q, j) => ({ ...q, state: state(j) })),
              ...Array.from({ length: k }, (_, j) => ({ label: where(j, tf) ?? 'transformer', weight: 1, state: passState(j) })),
              ...post.map((q, j) => ({ ...q, state: state(postStart + j) }))],
      done, total, elapsed: now - t0, remaining: partCount ? (total - done) * (partTime / partCount) : null,
    });
  };
  return { start: report, stagePart: (j, c) => first(before, j) + c, passPart: (st, ps, c) => passStart + (st - 1) * perStep + (ps - 1) * k + c,
           afterPart: (j, c) => postStart + first(after, j) + c, finish: () => report(parts.length) };
}

/**
 * Command. Yields one task to the browser (MessageChannel: unlike setTimeout it is not throttled, unlike
 * requestAnimationFrame it does not wait for a frame), so the page can paint and handle input between two blocking wasm
 * runs. Costs well under a millisecond.
 *
 * @returns {Promise<void>}
 * @example await yieldTask() // undefined, after one task
 */
export function yieldTask() {
  return new Promise((resolve) => { const c = new MessageChannel(); c.port1.onmessage = () => { c.port1.close(); resolve(); }; c.port2.postMessage(0); });
}

/**
 * Pure function. The status line of one progress event (progressCounter): what runs now, where, time and time left.
 *
 * @param {object} p - a progressCounter event
 * @returns {string}
 *
 * @example progressLine({running: 'transformer', where: 'blocks 10–13 of 46', step: 2, steps: 4, pass: 1, passes: 2, elapsed: 4200, remaining: 6100})
 * // 'Step 2 of 4 · pass 1 of 2 (CFG) · blocks 10–13 of 46 · 4.2 s · about 0:07 left'
 * @example progressLine({running: 'Text encoder (Gemma)', where: 'layer 3 of 26', step: 0, steps: 2, elapsed: 900, remaining: null})
 * // 'Text encoder (Gemma) · layer 3 of 26 · 0.9 s'
 * @example progressLine({running: 'Image decoder', where: null, step: 0, steps: 2, elapsed: 61000, remaining: 3000})
 * // 'Image decoder · 61.0 s · about 0:03 left'
 * @example progressLine({running: null, steps: 4, elapsed: 98000, remaining: 0})
 * // 'Done · 4 steps · 98.0 s'
 */
export function progressLine(p) {
  const seconds = `${(p.elapsed / 1000).toFixed(1)} s`;
  if (p.running === null) return `Done · ${p.steps} step${p.steps === 1 ? '' : 's'} · ${seconds}`;
  const left = p.remaining === null ? '' : ` · about ${formatEta(p.remaining / 1000)} left`;
  const where = p.where ? ` · ${p.where}` : p.step ? ' · transformer (one graph: no per-block progress)' : '';
  const head = p.step ? `Step ${p.step} of ${p.steps}${p.passes > 1 ? ` · pass ${p.pass} of ${p.passes} (CFG)` : ''}` : p.running;
  return `${head}${where} · ${seconds}${left}`;
}
