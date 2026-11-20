// The splitter (manifest §17): a FIBO2 version whose transformer returns one velocity per background head (gray, white,
// black), so the app gets the object on K known backgrounds and solves RGBA from them. Mirrors fibo2.sample's x0 rule
// and fibo2.rgba_from_composites (src/sana_alpha/fibo2.py).

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

/**
 * Pure function. RGBA from one image on K known solid backgrounds: C_k = α·F + (1 − α)·b_k, least squares per pixel
 * over the K backgrounds × 3 channels for α (shared) and α·F (per channel), as fibo2.rgba_from_composites:
 *   1 − α = Σ_k,c (C_kc − mean_k C_c)·(b_k − b̄) / (3·Σ_k (b_k − b̄)²),  α·F_c = mean_k C_kc − (1 − α)·b̄
 * α is clipped to [0, 1] first; F = α·F / α (0 where α = 0), clipped to [0, 1]. The uncertainty is the RMS residual of
 * the fit: 0 where the K images agree with one RGBA.
 *
 * @param {Float32Array[]} composites - K planar (3, H, W) images in [0, 1] (vae_decoder outputs)
 * @param {number[]} backgrounds - (K,) each image's background value, e.g. [0.5, 1, 0]
 * @param {number} hw - H·W
 * @returns {{color: Float32Array, alpha: Float32Array, uncertainty: Float32Array}} color (3, H, W) planar straight
 *   (not premultiplied), alpha (H·W,), uncertainty (H·W,)
 *
 * @example
 * // one pixel: F = (0.9, 0.1, 0.4), α = 0.25 on gray, white, black
 * const C = [0.5, 1, 0].map((b) => Float32Array.from([0.9, 0.1, 0.4], (f) => 0.25 * f + 0.75 * b));
 * const r = rgbaFromComposites(C, [0.5, 1, 0], 1);
 * [...r.color, r.alpha[0]].map((v) => +v.toFixed(6))  // [0.9, 0.1, 0.4, 0.25]
 * r.uncertainty[0] < 1e-6  // true
 * // the gray image disagrees by 0.3 in red: uncertainty rises (rgba_from_composites gives the same)
 * C[0][0] += 0.3; +rgbaFromComposites(C, [0.5, 1, 0], 1).uncertainty[0].toFixed(3)  // 0.082
 */
export function rgbaFromComposites(composites, backgrounds, hw) {
  const k = backgrounds.length, bMean = backgrounds.reduce((a, b) => a + b, 0) / k;
  const bDev = backgrounds.map((b) => b - bMean), bVar = bDev.reduce((a, d) => a + d * d, 0);
  const color = new Float32Array(3 * hw), alpha = new Float32Array(hw), uncertainty = new Float32Array(hw);
  const mean = [0, 0, 0];
  for (let i = 0; i < hw; i++) {
    let beta = 0;
    for (let c = 0; c < 3; c++) {
      let m = 0;
      for (let j = 0; j < k; j++) m += composites[j][c * hw + i];
      mean[c] = m / k;
      for (let j = 0; j < k; j++) beta += (composites[j][c * hw + i] - mean[c]) * bDev[j];
    }
    const a = Math.min(1, Math.max(0, 1 - beta / (3 * bVar)));
    let sq = 0;
    for (let c = 0; c < 3; c++) {
      const premul = mean[c] - (1 - a) * bMean;
      for (let j = 0; j < k; j++) sq += (composites[j][c * hw + i] - premul - (1 - a) * backgrounds[j]) ** 2;
      color[c * hw + i] = a > 0 ? Math.min(1, Math.max(0, premul / a)) : 0;
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
