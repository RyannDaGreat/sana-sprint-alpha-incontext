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
 * Command (calls onProgress). Progress of one generation in time-weighted units: a transformer chunk is 1 unit, a
 * pass is `chunks` units, a stage is STAGE_PASS_FRACTION × chunks units. Each tick reports {done, total, label, step,
 * steps, stepDone, perStep, elapsed ms, remaining ms | null}: step 0 = before or after the sampler steps; stepDone =
 * chunks done in that step of perStep (passes × chunks); remaining = units left × the mean chunk time measured so far
 * in this run (null until a chunk has run).
 *
 * @param {number} steps - sampler steps
 * @param {number} passes - transformer passes per step (2 with CFG)
 * @param {number} chunks - chunks per pass (1 = unsplit graph)
 * @param {string[]} stages - the other stages of this generation (keys of STAGE_PASS_FRACTION, repeats allowed)
 * @param {Function} onProgress
 * @returns {(label: string, step?: number, stage?: string) => void} tick: step > 0 for a chunk, else stage
 */
export function progressCounter(steps, passes, chunks, stages, onProgress) {
  const weight = (stage) => STAGE_PASS_FRACTION[stage] * chunks;
  const perStep = passes * chunks, total = steps * perStep + stages.reduce((s, k) => s + weight(k), 0);
  const t0 = performance.now();
  let done = 0, stepDone = 0, current = 0, last = t0, chunkTime = 0, chunkCount = 0;
  return (label, step = 0, stage = null) => {
    const now = performance.now();
    if (step) { chunkTime += now - last; chunkCount += 1; }
    last = now;
    done += step ? 1 : weight(stage);
    stepDone = step === current ? stepDone + 1 : 1;
    current = step;
    onProgress({ done, total, label, step, steps, stepDone: step ? stepDone : 0, perStep, elapsed: now - t0,
                 remaining: chunkCount ? (total - done) * (chunkTime / chunkCount) : null });
  };
}

/**
 * Pure function. The progress line of one generation event (pipeline.js progressCounter).
 *
 * @param {{label: string, step: number, steps: number, stepDone: number, perStep: number, elapsed: number, remaining: number|null}} p
 * @param {number} chunks - transformer chunks per pass (1 = unsplit graph)
 * @returns {string}
 *
 * @example progressLine({label: 'transformer 2', step: 2, steps: 4, stepDone: 3, perStep: 5, elapsed: 4200, remaining: 6100}, 5)
 * // 'Step 2 of 4 · transformer 2 · blocks 3/5 done · 4.2 s · about 6 s left'
 * @example progressLine({label: 'text encoder', step: 0, steps: 4, stepDone: 0, perStep: 5, elapsed: 900, remaining: null}, 5)
 * // 'text encoder done · 0.9 s'
 */
export function progressLine(p, chunks) {
  const seconds = `${(p.elapsed / 1000).toFixed(1)} s`;
  const left = p.remaining === null ? '' : ` · about ${Math.round(p.remaining / 1000)} s left`;
  if (!p.step) return `${p.label} done · ${seconds}${left}`;
  const blocks = chunks > 1 ? ` · blocks ${((p.stepDone - 1) % chunks) + 1}/${chunks} done` : '';
  return `Step ${p.step} of ${p.steps} · ${p.label}${blocks} · ${seconds}${left}`;
}
