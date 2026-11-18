// Trimask operations: pure functions on a mask = Float32Array of H·W values in [0, 1], row-major,
// 1 = surely object (white), 0 = surely background (black), 0.5 = unknown (gray); fuzzy values allowed.
// All take (mask, W, H, ...) and return a new mask; none mutates its input. Tested by
// agents/.scratchpad/web/test_trimask_ops.mjs, which runs every @example below.

export const GRAY = 0.5;
export const HARDEN_LO = 1 / 3; // default harden thresholds: below = black, above HARDEN_HI = white, between = gray
export const HARDEN_HI = 2 / 3;
const BOX_PASSES = 3;           // three box blurs approximate a Gaussian
const LEVEL_BLACK_BELOW = 0.25; // levelFractions: counts as black below this, white above LEVEL_WHITE_ABOVE
const LEVEL_WHITE_ABOVE = 0.75;

/**
 * Pure function. Mask from canvas RGBA pixels (red channel / 255).
 *
 * @param {Uint8ClampedArray} rgba - (H, W, 4) uint8, gray in R = G = B
 * @returns {Float32Array} (H, W) in [0, 1]
 *
 * @example rgbaToMask(new Uint8ClampedArray([255,255,255,255, 0,0,0,255])) // Float32Array [1, 0]
 */
export function rgbaToMask(rgba) {
  const out = new Float32Array(rgba.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = rgba[4 * i] / 255;
  return out;
}

/**
 * Pure function. Opaque gray RGBA pixels from a mask.
 *
 * @param {Float32Array} mask - (H, W) in [0, 1]
 * @returns {Uint8ClampedArray} (H, W, 4) uint8, R = G = B = round(255 · v), A = 255
 *
 * @example maskToRgba(new Float32Array([1, 0.5])) // [255,255,255,255, 128,128,128,255]
 */
export function maskToRgba(mask) {
  const out = new Uint8ClampedArray(4 * mask.length);
  for (let i = 0; i < mask.length; i++) {
    out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = Math.round(255 * mask[i]);
    out[4 * i + 3] = 255;
  }
  return out;
}

/**
 * Pure function. Remaps the three levels: black → b, gray → g, white → w; values in between are
 * interpolated linearly, so fuzzy masks stay fuzzy.
 *
 * @param {Float32Array} mask - (H, W)
 * @param {[number, number, number]} to - new values of [black, gray, white]
 * @returns {Float32Array} (H, W)
 *
 * @example remapLevels(new Float32Array([0, 0.5, 1]), [0, 0, 1])     // [0, 0, 1]     (gray → black)
 * @example remapLevels(new Float32Array([0, 0.5, 1]), [0, 0.5, 0.5]) // [0, 0.5, 0.5] (white → gray)
 * @example remapLevels(new Float32Array([0, 0.25, 1]), [1, 0.5, 0])  // [1, 0.75, 0]  (swap black and white)
 */
export function remapLevels(mask, [b, g, w]) {
  const out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    out[i] = v < GRAY ? b + (g - b) * 2 * v : g + (w - g) * (2 * v - 1);
  }
  return out;
}

/**
 * Pure function. Snaps every value to one of three levels: ≤ lo → 0, ≥ hi → 1, else 0.5.
 *
 * @param {Float32Array} mask - (H, W)
 * @param {number} [lo]
 * @param {number} [hi]
 * @returns {Float32Array} (H, W) with values in {0, 0.5, 1}
 *
 * @example harden(new Float32Array([0.1, 0.4, 0.6, 0.9])) // [0, 0.5, 0.5, 1]
 * @example harden(new Float32Array([0.1, 0.4, 0.6, 0.9]), 0.5, 0.5) // [0, 0, 1, 1] (lo = hi: binarize)
 */
export function harden(mask, lo = HARDEN_LO, hi = HARDEN_HI) {
  const out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = mask[i] >= hi ? 1 : mask[i] <= lo ? 0 : GRAY;
  return out;
}

/**
 * Command (writes only into the caller's scratch buffers g, h and out). Sliding-window maximum of
 * src[s0 .. s0 + n) into out[o0 ..), window [i − half, i + half] clipped to the segment
 * (van Herk / Gil-Werman on a −∞-padded copy: O(n), no allocation).
 */
function slidingMaxInto(src, s0, n, half, g, h, out, o0) {
  const k = 2 * half + 1, p = n + 2 * half;
  for (let j = 0; j < p; j++) {
    const v = j < half || j >= n + half ? -Infinity : src[s0 + j - half];
    g[j] = j % k === 0 ? v : (g[j - 1] > v ? g[j - 1] : v);
  }
  for (let j = p - 1; j >= 0; j--) {
    const v = j < half || j >= n + half ? -Infinity : src[s0 + j - half];
    h[j] = j === p - 1 || (j + 1) % k === 0 ? v : (h[j + 1] > v ? h[j + 1] : v);
  }
  for (let i = 0; i < n; i++) out[o0 + i] = h[i] > g[i + 2 * half] ? h[i] : g[i + 2 * half];
}

/**
 * Pure function. Sliding-window maximum of one row, window [i − half, i + half] clipped to the row.
 *
 * @param {Float32Array} src - input values
 * @param {number} half - window half-width
 * @returns {Float32Array} same length, out[i] = max(src[i − half .. i + half])
 *
 * @example slidingMax(new Float32Array([0, 0, 1, 0, 0]), 1) // [0, 1, 1, 1, 0]
 */
export function slidingMax(src, half) {
  const out = new Float32Array(src.length), p = src.length + 2 * half;
  slidingMaxInto(src, 0, src.length, half, new Float32Array(p), new Float32Array(p), out, 0);
  return out;
}

/**
 * Pure function. Grayscale dilation with a disk of radius r (max over the disk): white grows outward.
 * The disk is a stack of rows of half-width √(r² − d²): for each distinct half-width, the whole
 * image's row maxima are computed once and merged at vertical offsets ±d. O(r · W · H).
 *
 * @param {Float32Array} mask - (H, W)
 * @param {number} W
 * @param {number} H
 * @param {number} r - radius in pixels (rounded)
 * @returns {Float32Array} (H, W)
 *
 * @example dilate(new Float32Array([0,0,0, 0,1,0, 0,0,0]), 3, 3, 1) // [0,1,0, 1,1,1, 0,1,0] (a plus: the radius-1 disk)
 */
export function dilate(mask, W, H, r) {
  r = Math.round(r);
  if (r <= 0) return Float32Array.from(mask);
  const out = new Float32Array(mask.length).fill(-Infinity), rows = new Float32Array(mask.length);
  const g = new Float32Array(W + 2 * r), h = new Float32Array(W + 2 * r);
  let d = 0;
  while (d <= r) {
    const half = Math.floor(Math.sqrt(r * r - d * d));
    for (let y = 0; y < H; y++) slidingMaxInto(mask, y * W, W, half, g, h, rows, y * W);
    for (; d <= r && Math.floor(Math.sqrt(r * r - d * d)) === half; d++) {
      for (const dy of d ? [-d, d] : [0]) {
        for (let y = Math.max(0, -dy); y < Math.min(H, H - dy); y++) {
          const o = y * W, s = (y + dy) * W;
          for (let x = 0; x < W; x++) if (rows[s + x] > out[o + x]) out[o + x] = rows[s + x];
        }
      }
    }
  }
  return out;
}

/**
 * Pure function. Grayscale erosion with a disk of radius r (min over the disk): white shrinks.
 *
 * @example erode(new Float32Array([1,1,1, 1,1,1, 1,1,0]), 3, 3, 1) // [1,1,1, 1,1,0, 1,0,0]
 */
export function erode(mask, W, H, r) {
  return remapLevels(dilate(remapLevels(mask, [1, GRAY, 0]), W, H, r), [1, GRAY, 0]);
}

/**
 * Pure function. Dilate into gray: the region that dilation would add becomes gray (unknown);
 * existing values never drop. out = max(m, min(dilate(m, r), 0.5)).
 *
 * @example dilateIntoGray(new Float32Array([0, 0, 1, 1]), 4, 1, 1) // [0, 0.5, 1, 1]
 */
export function dilateIntoGray(mask, W, H, r) {
  const d = dilate(mask, W, H, r), out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = Math.max(mask[i], Math.min(d[i], GRAY));
  return out;
}

/**
 * Pure function. Erode into gray: the rim that erosion would remove becomes gray (unknown);
 * existing values never rise. out = min(m, max(erode(m, r), 0.5)).
 *
 * @example erodeIntoGray(new Float32Array([0, 0, 1, 1]), 4, 1, 1) // [0, 0, 0.5, 1]
 */
export function erodeIntoGray(mask, W, H, r) {
  const e = erode(mask, W, H, r), out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = Math.min(mask[i], Math.max(e[i], GRAY));
  return out;
}

/**
 * Pure function. A gray (unknown) band of the given total width centered on every white/black edge:
 * erode into gray by width/2, then dilate into gray by width/2.
 *
 * @example edgeBand(new Float32Array([0, 0, 0, 1, 1, 1]), 6, 1, 2) // [0, 0, 0.5, 0.5, 1, 1]
 */
export function edgeBand(mask, W, H, width) {
  return dilateIntoGray(erodeIntoGray(mask, W, H, width / 2), W, H, width / 2);
}

/**
 * Pure function. Box-blur widths whose three passes approximate a Gaussian of standard deviation sigma
 * (Kovesi's method); returns half-widths.
 *
 * @example boxRadii(5) // [4, 4, 5]
 */
export function boxRadii(sigma) {
  const ideal = Math.sqrt((12 * sigma * sigma) / BOX_PASSES + 1);
  let wl = Math.floor(ideal); if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const m = Math.round((12 * sigma * sigma - BOX_PASSES * wl * wl - 4 * BOX_PASSES * wl - 3 * BOX_PASSES) / (-4 * wl - 4));
  return Array.from({ length: BOX_PASSES }, (_, i) => ((i < m ? wl : wu) - 1) / 2);
}

/**
 * Pure function. 1D box blur along rows (step 1) or columns (step W) with clamped edges.
 *
 * @param {Float32Array} src - (H, W)
 * @param {number} W
 * @param {number} H
 * @param {number} r - half-width
 * @param {boolean} vertical
 * @returns {Float32Array} (H, W)
 *
 * @example boxBlur1d(new Float32Array([0, 0, 3, 0, 0]), 5, 1, 1, false) // [0, 1, 1, 1, 0]
 */
export function boxBlur1d(src, W, H, r, vertical) {
  const out = new Float32Array(src.length), n = vertical ? H : W, lines = vertical ? W : H;
  const step = vertical ? W : 1, start = vertical ? 1 : W, k = 2 * r + 1;
  for (let l = 0; l < lines; l++) {
    const base = l * start, at = (i) => src[base + Math.min(n - 1, Math.max(0, i)) * step];
    let s = 0;
    for (let i = -r; i <= r; i++) s += at(i);
    for (let i = 0; i < n; i++) { out[base + i * step] = s / k; s += at(i + r + 1) - at(i - r); }
  }
  return out;
}

/**
 * Pure function. Soften: Gaussian blur (three box passes) of standard deviation sigma, so hard
 * edges become gray ramps about 2.5 · sigma wide.
 *
 * @example blur(new Float32Array([0,0,0,0,1,1,1,1]), 8, 1, 2).map((v) => +v.toFixed(2)) // [0.02, 0.09, 0.22, 0.4, 0.6, 0.78, 0.91, 0.98]
 */
export function blur(mask, W, H, sigma) {
  if (sigma <= 0) return Float32Array.from(mask);
  let m = mask;
  for (const r of boxRadii(sigma)) m = boxBlur1d(boxBlur1d(m, W, H, r, false), W, H, r, true);
  return m;
}

/**
 * Pure function. Per-pixel blend: where sel = 1 take b, where sel = 0 keep a (an op applied inside a selection).
 *
 * @param {Float32Array} a - (H, W) original
 * @param {Float32Array} b - (H, W) edited
 * @param {Float32Array} sel - (H, W) selection weight in [0, 1]
 * @returns {Float32Array} (H, W)
 *
 * @example blendSelection(new Float32Array([0, 0]), new Float32Array([1, 1]), new Float32Array([1, 0])) // [1, 0]
 */
export function blendSelection(a, b, sel) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * sel[i];
  return out;
}

/**
 * Pure function. Fractions of pixels that are black (< LEVEL_BLACK_BELOW), white (> LEVEL_WHITE_ABOVE) and gray (between).
 *
 * @example levelFractions(new Float32Array([0, 0.5, 1, 1])) // {black: 0.25, gray: 0.25, white: 0.5}
 */
export function levelFractions(mask) {
  let b = 0, g = 0, w = 0;
  for (const v of mask) { if (v < LEVEL_BLACK_BELOW) b++; else if (v > LEVEL_WHITE_ABOVE) w++; else g++; }
  return { black: b / mask.length, gray: g / mask.length, white: w / mask.length };
}
