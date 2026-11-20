// The splitter (manifest §17): a FIBO2 version whose transformer returns one velocity per background head (gray, white,
// black), so the app gets the object on K known backgrounds and solves RGBA from them. Mirrors fibo2.sample's x0 rule
// and fibo2.rgba_from_composites (src/sana_alpha/fibo2.py), the iterative least squares RGBA fit (m00826).

/**
 * Pure function. Every head's clean latent from the last step (fibo2.sample): with the shared noise estimate
 * ε̂ = x + (1 − σ)·v_0, x0_k = ε̂ − v_k. Head 0 (gray) is the plain Euler step to σ = 0: x − σ·v_0.
 *
 * @param {Float32Array} x - (N,) latents at σ, N = 32·LH·LW
 * @param {Float32Array} v - (K·N,) the K heads' velocities, head-major (graph output (1, K, 32, LH, LW))
 * @param {number} sigma - σ of the last step
 * @param {number} heads - K
 * @returns {Float32Array[]} K arrays of (N,)
 *
 * @example splitX0(new Float32Array([1]), new Float32Array([2, 3, 1]), 0.5, 3).map((a) => a[0]) // [0, -1, 1]
 */
export function splitX0(x, v, sigma, heads) {
  const n = x.length;
  return Array.from({ length: heads }, (_, k) => x.map((xi, i) => xi + (1 - sigma) * v[i] - v[k * n + i]));
}

const ALPHA_ILS_ITERATIONS = 32;  // as fibo2.ALPHA_ILS_ITERATIONS: 20 reach the box-constrained optimum on every validated pixel
const ALPHA_ILS_TOLERANCE = 1e-7;  // a pixel stops iterating once a step moves α by less than this (float32 resolution near 1)
const ALPHA_ILS_FLOOR = 1e-9;  // as fibo2.ALPHA_ILS_FLOOR: the F step divides by max(α, this), so α can leave 0

/**
 * Pure function. RGBA with decontaminated color from one image on K known solid backgrounds (m00824, m00826), the same
 * iterative least squares as fibo2.rgba_from_composites: per pixel min over α ∈ [0, 1], F ∈ [0, 1]³ of
 * Σ_k,c (C_kc − α·F_c − (1 − α)·b_kc)², by alternating exact 1-D solves from the clipped unconstrained fit
 *   1 − α = Σ_k,c (C_kc − C̄_c)·(b_kc − b̄_c) / Σ_k,c (b_kc − b̄_c)²
 *   F_c ← clip(mean_k (C_kc − (1 − α)·b_kc) / α, 0, 1),  α ← clip(Σ (C − b)·(F − b) / Σ (F − b)², 0, 1)
 * F = 0 where α = 0. The uncertainty is the RMS residual of the fit: 0 where the K images agree with one RGBA.
 *
 * @param {Float32Array[]} composites - K planar (3, H, W) images in [0, 1] (vae_decoder outputs)
 * @param {Array<number|number[]>} backgrounds - (K,) gray levels, e.g. [0.5, 1, 0], or (K, 3) RGB colors
 * @param {number} hw - H·W
 * @param {number} [iterations] - alternating steps
 * @returns {{color: Float32Array, alpha: Float32Array, uncertainty: Float32Array}} color (3, H, W) planar straight
 *   (not premultiplied), alpha (H·W,), uncertainty (H·W,)
 *
 * @example
 * // one pixel: F = (0.9, 0.1, 0.4), α = 0.25 on gray, white, black
 * const C = [0.5, 1, 0].map((b) => Float32Array.from([0.9, 0.1, 0.4], (f) => 0.25 * f + 0.75 * b));
 * const r = rgbaFromComposites(C, [0.5, 1, 0], 1);
 * [...r.color, r.alpha[0]].map((v) => +v.toFixed(6))  // [0.9, 0.1, 0.4, 0.25]
 * r.uncertainty[0] < 1e-6  // true
 * // an opaque pixel whose gray image is 0.3 redder: the uncertainty rises (rgba_from_composites gives the same)
 * const D = [0.5, 1, 0].map(() => Float32Array.of(0.2, 0.6, 1)); D[0][0] += 0.3;
 * +rgbaFromComposites(D, [0.5, 1, 0], 1).uncertainty[0].toFixed(3)  // 0.082
 */
export function rgbaFromComposites(composites, backgrounds, hw, iterations = ALPHA_ILS_ITERATIONS) {
  const k = backgrounds.length;
  const b = backgrounds.map((v) => (typeof v === 'number' ? [v, v, v] : v));  // (K, 3)
  const bMean = [0, 1, 2].map((c) => b.reduce((s, bk) => s + bk[c], 0) / k);
  const bSum = bMean.map((m) => m * k);
  let bVar = 0, bSq = 0;
  for (const bk of b) for (let c = 0; c < 3; c++) { bVar += (bk[c] - bMean[c]) ** 2; bSq += bk[c] ** 2; }
  if (bVar === 0) throw new Error(`rgbaFromComposites needs at least two different backgrounds, got ${JSON.stringify(backgrounds)}`);
  const color = new Float32Array(3 * hw), alpha = new Float32Array(hw), uncertainty = new Float32Array(hw);
  const mean = [0, 0, 0], cMinusB = [0, 0, 0], f = [0, 0, 0];
  const solveColor = (a) => { for (let c = 0; c < 3; c++) f[c] = Math.min(1, Math.max(0, (mean[c] - (1 - a) * bMean[c]) / Math.max(a, ALPHA_ILS_FLOOR))); };
  for (let i = 0; i < hw; i++) {
    let beta = 0, cross = 0;
    for (let c = 0; c < 3; c++) {
      let m = 0, d = 0;
      for (let j = 0; j < k; j++) { const v = composites[j][c * hw + i]; m += v; d += v - b[j][c]; cross += (v - b[j][c]) * b[j][c]; }
      mean[c] = m / k; cMinusB[c] = d;
      for (let j = 0; j < k; j++) beta += (composites[j][c * hw + i] - mean[c]) * (b[j][c] - bMean[c]);
    }
    let a = Math.min(1, Math.max(0, 1 - beta / bVar));
    for (let t = 0; t < iterations; t++) {
      solveColor(a);
      const num = f[0] * cMinusB[0] + f[1] * cMinusB[1] + f[2] * cMinusB[2] - cross;
      const den = k * (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) - 2 * (f[0] * bSum[0] + f[1] * bSum[1] + f[2] * bSum[2]) + bSq;
      const next = den > 0 ? Math.min(1, Math.max(0, num / den)) : a;
      if (Math.abs(next - a) < ALPHA_ILS_TOLERANCE) { a = next; break; }  // converged: the remaining steps would not move α
      a = next;
    }
    solveColor(a);
    let sq = 0;
    for (let c = 0; c < 3; c++) {
      const fc = a > 0 ? f[c] : 0;
      for (let j = 0; j < k; j++) sq += (composites[j][c * hw + i] - a * fc - (1 - a) * b[j][c]) ** 2;
      color[c * hw + i] = fc;
    }
    alpha[i] = a;
    uncertainty[i] = Math.sqrt(sq / (3 * k));
  }
  return { color, alpha, uncertainty };
}

const UNCERTAINTY_FULL_SCALE = 0.25;  // RMS residual shown at full heat: about a quarter of the [0, 1] range is a clear disagreement

/**
 * Pure function. Canvas pixels of the splitter views: the K images side by side, the RGBA cutout, the alpha matte
 * (white = opaque) and the uncertainty heatmap (black = the K images agree, through red to yellow-white at
 * UNCERTAINTY_FULL_SCALE).
 *
 * @param {Float32Array[]} composites - K planar (3, H, W) images in [0, 1]
 * @param {{color: Float32Array, alpha: Float32Array, uncertainty: Float32Array}} fit - rgbaFromComposites
 * @param {number} width
 * @param {number} height
 * @returns {{strip: Uint8ClampedArray, rgba: Uint8ClampedArray, alpha: Uint8ClampedArray, uncertainty: Uint8ClampedArray}}
 *   strip (H, K·W, 4), the others (H, W, 4)
 *
 * @example splitterPixels([Float32Array.of(1, 1, 1)], {color: Float32Array.of(1, 0, 0), alpha: Float32Array.of(0.5),
 *   uncertainty: Float32Array.of(0)}, 1, 1).rgba  // Uint8ClampedArray [255, 0, 0, 128]
 */
export function splitterPixels(composites, fit, width, height) {
  const hw = width * height, k = composites.length, byte = (v) => Math.round(v * 255);
  const strip = new Uint8ClampedArray(4 * k * hw), rgba = new Uint8ClampedArray(4 * hw);
  const alpha = new Uint8ClampedArray(4 * hw), uncertainty = new Uint8ClampedArray(4 * hw);
  for (let i = 0; i < hw; i++) {
    const y = Math.floor(i / width), x = i % width;
    for (let j = 0; j < k; j++) {
      const o = 4 * (y * k * width + j * width + x);
      for (let c = 0; c < 3; c++) strip[o + c] = byte(composites[j][c * hw + i]);
      strip[o + 3] = 255;
    }
    for (let c = 0; c < 3; c++) rgba[4 * i + c] = byte(fit.color[c * hw + i]);
    rgba[4 * i + 3] = byte(fit.alpha[i]);
    alpha.fill(byte(fit.alpha[i]), 4 * i, 4 * i + 3);
    const u = Math.min(1, fit.uncertainty[i] / UNCERTAINTY_FULL_SCALE);
    uncertainty.set([byte(Math.min(1, 3 * u)), byte(Math.min(1, Math.max(0, 3 * u - 1))), byte(Math.max(0, 3 * u - 2))], 4 * i);
    alpha[4 * i + 3] = uncertainty[4 * i + 3] = 255;
  }
  return { strip, rgba, alpha, uncertainty };
}
