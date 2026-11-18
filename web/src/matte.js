// The matte editor: a 1024² canvas built from a raster layer (uploads, brush, eraser) and vector
// objects (text, rectangles, ellipses, polygons, lines) that can be dragged and resized. Values are a
// trimask: white = object, black = background, gray = unknown; every drawing tool paints `level`.
// Filters (trimask_ops.js) flatten the objects into the raster; Undo brings them back.

import { SIZE, fitRect } from './util.js';
import { rgbaToMask, maskToRgba, blendSelection } from './trimask_ops.js';

const HANDLE = 28;        // resize handle side, in matte pixels
const MIN_OBJECT = 8;     // smallest shape side, in matte pixels
const HISTORY_LIMIT = 40; // undo steps kept; each step with a raster change holds one 4 MB snapshot
const CLOSE_DIST = 24;    // a polygon click this close to its first vertex closes it, in matte pixels
const SAME_POINT = 3;     // consecutive path points closer than this are one point (double-click sends two)
const DOUBLE_CLICK_MS = 500; // a second press this soon at the same spot finishes a path (the dblclick event is not always sent)
const DOUBLE_CLICK_DIST = 8;
const VERTEX = 12;        // vertex marker side while drawing a path
const DASH = [12, 8];     // selection outline dash pattern
const PATH_TOOLS = new Set(['polygon', 'line']);
const CURSOR_TOOLS = new Set(['brush', 'eraser', 'line']);

/**
 * Pure function. Thresholds a gray matte at `level`; `softness` > 0 gives a linear ramp of that width
 * centered on level (a "levels" feather) instead of a hard step.
 *
 * @param {Uint8ClampedArray} rgba - (H, W, 4) uint8, gray in R = G = B
 * @param {number} level - 0..255
 * @param {number} softness - ramp width in gray levels, 0 = hard binarize
 * @returns {Uint8ClampedArray} (H, W, 4) uint8 gray, opaque
 *
 * @example thresholdGray(new Uint8ClampedArray([100,100,100,255, 200,200,200,255]), 128, 0) // [0,0,0,255, 255,255,255,255]
 * @example thresholdGray(new Uint8ClampedArray([128,128,128,255]), 128, 64) // [128,128,128,255] (middle of the ramp)
 */
export function thresholdGray(rgba, level, softness) {
  const out = new Uint8ClampedArray(rgba.length), lo = level - softness / 2;
  for (let i = 0; i < rgba.length; i += 4) {
    const v = softness > 0 ? (255 * (rgba[i] - lo)) / softness : rgba[i] >= level ? 255 : 0;
    out[i] = out[i + 1] = out[i + 2] = v;
    out[i + 3] = 255;
  }
  return out;
}

/**
 * Pure function. Deep copy of the vector objects (plain data), for history snapshots.
 *
 * @example cloneObjects([{type: 'polygon', x: 1, y: 2, w: 3, h: 4, pts: [[0, 0], [1, 1]]}]) // same data, new objects and point arrays
 */
export function cloneObjects(objects) {
  return objects.map((o) => ({ ...o, ...(o.pts ? { pts: o.pts.map((p) => [...p]) } : {}) }));
}

/**
 * Pure function. Axis-aligned box of an object (text boxes come from the measured width).
 *
 * @param {object} o - {type: 'rect'|'ellipse'|'polygon'|'line', x, y, w, h} or {type: 'text', x, y, size, width}
 * @returns {{x: number, y: number, w: number, h: number}}
 *
 * @example objectBox({type: 'rect', x: 10, y: 20, w: 30, h: 40}) // {x: 10, y: 20, w: 30, h: 40}
 * @example objectBox({type: 'text', x: 0, y: 0, size: 100, width: 250}) // {x: 0, y: 0, w: 250, h: 100}
 */
export function objectBox(o) {
  return o.type === 'text' ? { x: o.x, y: o.y, w: o.width, h: o.size } : { x: o.x, y: o.y, w: o.w, h: o.h };
}

/**
 * Pure function. Whether point (px, py) is inside box b, optionally grown by `pad`.
 *
 * @example inBox({x: 0, y: 0, w: 10, h: 10}, 5, 5) // true
 * @example inBox({x: 0, y: 0, w: 10, h: 10}, 12, 5) // false
 */
export function inBox(b, px, py, pad = 0) {
  return px >= b.x - pad && px <= b.x + b.w + pad && py >= b.y - pad && py <= b.y + b.h + pad;
}

/**
 * Pure function. The normalized box spanned by two corner points.
 *
 * @example spanBox(50, 60, 10, 20) // {x: 10, y: 20, w: 40, h: 40}
 */
export function spanBox(x0, y0, x1, y1) {
  return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
}

/**
 * Pure function. CSS color of a trimask level.
 *
 * @example levelColor(0.5) // 'rgb(128, 128, 128)'
 * @example levelColor(1)   // 'rgb(255, 255, 255)'
 */
export function levelColor(level) {
  const v = Math.round(255 * level);
  return `rgb(${v}, ${v}, ${v})`;
}

/**
 * Pure function. Drops consecutive points closer than `eps` (a double-click adds its point twice).
 *
 * @param {{x: number, y: number}[]} pts
 * @returns {{x: number, y: number}[]}
 *
 * @example dedupePath([{x: 0, y: 0}, {x: 1, y: 0}, {x: 50, y: 0}, {x: 50, y: 1}], 3) // [{x: 0, y: 0}, {x: 50, y: 0}]
 */
export function dedupePath(pts, eps = SAME_POINT) {
  return pts.filter((p, i) => i === 0 || Math.hypot(p.x - pts[i - 1].x, p.y - pts[i - 1].y) >= eps);
}

/**
 * Pure function. A path as a box plus points relative to it (u, v in [0, 1]), so moving and resizing
 * an object only changes its box. A flat side gets size 1 to stay invertible.
 *
 * @param {{x: number, y: number}[]} pts
 * @returns {{x: number, y: number, w: number, h: number, pts: number[][]}}
 *
 * @example pathObject([{x: 10, y: 10}, {x: 30, y: 10}, {x: 30, y: 50}]) // {x: 10, y: 10, w: 20, h: 40, pts: [[0, 0], [1, 0], [1, 1]]}
 */
export function pathObject(pts) {
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x = Math.min(...xs), y = Math.min(...ys);
  const w = Math.max(1, Math.max(...xs) - x), h = Math.max(1, Math.max(...ys) - y);
  return { x, y, w, h, pts: pts.map((p) => [(p.x - x) / w, (p.y - y) / h]) };
}

/**
 * Pure function. Grayscale matte value per pixel from an uploaded image: alpha if the image has
 * any transparency, else luminance (Rec. 601).
 *
 * @param {Uint8ClampedArray} rgba - (H, W, 4) uint8
 * @returns {Uint8ClampedArray} (H, W, 4) uint8 gray, opaque
 *
 * @example uploadToMatte(new Uint8ClampedArray([255, 0, 0, 128])) // [128, 128, 128, 255] (uses alpha)
 * @example uploadToMatte(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255])) // [255,255,255,255, 0,0,0,255]
 */
export function uploadToMatte(rgba) {
  let hasAlpha = false;
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] < 255) { hasAlpha = true; break; }
  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0; i < rgba.length; i += 4) {
    const v = hasAlpha ? rgba[i + 3] : 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    out[i] = out[i + 1] = out[i + 2] = v;
    out[i + 3] = 255;
  }
  return out;
}

export class MatteEditor {
  /**
   * Command. Binds the editor to a matte canvas and an overlay canvas (both SIZE × SIZE).
   *
   * @param {HTMLCanvasElement} canvas - The matte itself (gray levels)
   * @param {HTMLCanvasElement} overlay - Selection box, path in progress, lasso and brush cursor, drawn on top
   * @param {() => void} onChange - Called after every edit
   */
  constructor(canvas, overlay, onChange) {
    this.canvas = canvas; this.overlay = overlay; this.onChange = onChange;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.octx = overlay.getContext('2d');
    this.raster = new OffscreenCanvas(SIZE, SIZE);
    this.rctx = this.raster.getContext('2d', { willReadFrequently: true });
    this.objects = [];
    this.selected = null;
    this.invert = false;
    this.rev = 0;
    this.tool = 'brush';
    this.brushSize = 64;
    this.level = 1;             // value every drawing tool paints: 1 white, 0.5 gray, 0 black
    this.text = { value: 'HELLO', font: "'Jost Variable'", size: 220, weight: 700 };
    this.drag = null;
    this.cursor = null;
    this.path = null;           // polygon / line vertices being placed, or null
    this.lastDown = -Infinity;  // time of the last path click, for double-click detection
    this.selection = null;      // {mask: Float32Array (H, W) in [0, 1], pts} limiting filters, or null = whole matte
    this.preview = null;        // {canvas, flat}: threshold preview (replaces the raster) or filter preview (flat: replaces everything)
    this.rasterRev = 0;         // bumped on every raster edit, so snapshots share unchanged rasters
    this.history = []; this.index = -1;
    this.onHistory = () => {};  // called when undo/redo availability or the preview may change
    this.clear();
    overlay.addEventListener('pointerdown', (e) => this.down(e));
    overlay.addEventListener('pointermove', (e) => this.move(e));
    overlay.addEventListener('pointerup', (e) => this.up(e));
    overlay.addEventListener('dblclick', () => this.finishPath());
    overlay.addEventListener('pointerleave', () => { this.cursor = null; this.drawOverlay(); });
  }

  /** Command. Clears raster and objects to black (one undo step). */
  clear() {
    this.preview = null; this.path = null;
    this.rctx.fillStyle = '#000';
    this.rctx.fillRect(0, 0, SIZE, SIZE);
    this.rasterRev++;
    this.objects = []; this.selected = null; this.invert = false;
    this.changed(); this.commit();
  }

  /** Command. Toggles inversion of the whole matte (one undo step; gray stays gray). */
  toggleInvert() { this.invert = !this.invert; this.changed(); this.commit(); }

  /** Command. Sets the painting level; with the Move tool, the selected object takes it too (one undo step). */
  setLevel(level) {
    this.level = level;
    if (this.selected && this.tool === 'move') { this.selected.level = level; this.changed(); this.commit(); }
  }

  // ---------- history ----------

  /** Command. Records the current state as a new undo step (drops the redo branch; keeps HISTORY_LIMIT steps). */
  commit() {
    const prev = this.history[this.index];
    const raster = prev?.rasterRev === this.rasterRev ? prev.raster : this.rctx.getImageData(0, 0, SIZE, SIZE);
    this.history = this.history.slice(0, this.index + 1);
    this.history.push({ raster, rasterRev: this.rasterRev, objects: cloneObjects(this.objects), invert: this.invert });
    if (this.history.length > HISTORY_LIMIT) this.history.shift();
    this.index = this.history.length - 1;
    this.onHistory();
  }

  /** Command. Makes the current state the only history step (the starting point). */
  resetHistory() { this.history = []; this.index = -1; this.commit(); }

  /** Query. Whether undo / redo have a step to go to. */
  canUndo() { return this.index > 0; }
  canRedo() { return this.index < this.history.length - 1; }

  /** Command. Restores history step `i`. */
  restore(i) {
    const s = this.history[i];
    this.index = i; this.preview = null; this.drag = null; this.selected = null; this.path = null;
    if (s.rasterRev !== this.rasterRev) { this.rctx.putImageData(s.raster, 0, 0); this.rasterRev = s.rasterRev; }
    this.objects = cloneObjects(s.objects); this.invert = s.invert;
    this.changed(); this.onHistory();
  }

  /** Command. Steps back / forward in history (no-op at either end). */
  undo() { if (this.canUndo()) this.restore(this.index - 1); }
  redo() { if (this.canRedo()) this.restore(this.index + 1); }

  // ---------- previews: threshold (raster only) and filters (whole matte) ----------

  /** Command. Shows `pixels` (SIZE², RGBA) as a preview without committing it; flat = it replaces objects and inversion too. */
  showPreview(pixels, flat) {
    const canvas = this.preview?.canvas ?? new OffscreenCanvas(SIZE, SIZE);
    canvas.getContext('2d').putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
    this.preview = { canvas, flat };
    this.changed(); this.onHistory();
  }

  /** Command. Shows the raster thresholded at `level` (with `softness` ramp). Vector objects are already sharp. */
  previewThreshold(level, softness) {
    this.showPreview(thresholdGray(this.rctx.getImageData(0, 0, SIZE, SIZE).data, level, softness), false);
  }

  /** Query. The whole matte as it looks without any preview: (SIZE, SIZE) Float32 in [0, 1]. */
  flatMask() {
    const c = new OffscreenCanvas(SIZE, SIZE).getContext('2d', { willReadFrequently: true });
    this.composite(c, this.raster);
    return rgbaToMask(c.getImageData(0, 0, SIZE, SIZE).data);
  }

  /**
   * Command. Previews a filter on the whole matte (inside the selection only, when there is one).
   *
   * @param {(mask: Float32Array, W: number, H: number) => Float32Array} op - pure (H, W) → (H, W)
   */
  previewOp(op) {
    const base = this.flatMask(), out = op(base, SIZE, SIZE);
    this.showPreview(maskToRgba(this.selection ? blendSelection(base, out, this.selection.mask) : out), true);
  }

  /** Command. Applies a filter at once (one undo step). */
  applyOp(op) { this.previewOp(op); this.applyPreview(); }

  /** Command. Drops the preview. */
  cancelPreview() { if (this.preview) { this.preview = null; this.changed(); this.onHistory(); } }

  /** Command. Writes the preview into the raster (one undo step); a filter preview also flattens objects and inversion. */
  applyPreview() {
    if (!this.preview) return;
    this.rctx.drawImage(this.preview.canvas, 0, 0);
    if (this.preview.flat) { this.objects = []; this.selected = null; this.invert = false; }
    this.preview = null; this.rasterRev++;
    this.changed(); this.commit();
  }

  // ---------- tools ----------

  /** Command. Sets the active tool: finishes a path in progress, drops the selection box when it no longer applies. */
  setTool(tool) {
    if (this.path) this.finishPath();
    this.tool = tool;
    if (!['move', 'text', 'rect', 'ellipse'].includes(tool)) this.selected = null;
    this.drawOverlay();
  }

  /** Command. Ends the polygon / line being placed: adds it as an object (one undo step) if it has enough points. */
  finishPath() {
    if (!this.path) return;
    const pts = dedupePath(this.path), polygon = this.tool === 'polygon';
    this.path = null;
    if (pts.length >= (polygon ? 3 : 2)) {
      const o = { type: this.tool, level: this.level, ...pathObject(pts), ...(polygon ? {} : { width: this.brushSize }) };
      this.objects.push(o);
      this.changed(); this.commit();
    } else this.drawOverlay();
  }

  /** Command. Drops the polygon / line being placed. */
  cancelPath() { this.path = null; this.drawOverlay(); }

  /** Command. Sets the filter selection from a closed lasso path (null or < 3 points: whole matte). */
  setSelection(pts) {
    if (!pts || pts.length < 3) { this.selection = null; this.drawOverlay(); return; }
    const c = new OffscreenCanvas(SIZE, SIZE).getContext('2d', { willReadFrequently: true });
    c.fillStyle = '#000'; c.fillRect(0, 0, SIZE, SIZE);
    c.fillStyle = '#fff'; c.beginPath();
    pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
    c.closePath(); c.fill();
    this.selection = { mask: rgbaToMask(c.getImageData(0, 0, SIZE, SIZE).data), pts };
    this.drawOverlay();
  }

  /**
   * Command. Replaces the raster layer with an uploaded image, fitted inside the square.
   *
   * @param {CanvasImageSource} bitmap - ImageBitmap, or an <img> (SVGs rasterize crisply at the fitted size)
   * @param {number} [width] - source size (an SVG without intrinsic size needs it)
   * @param {number} [height]
   * @param {boolean} [fresh] - Also drop objects and inversion (see placeGray)
   */
  upload(bitmap, width = bitmap.width, height = bitmap.height, fresh = false) {
    const { w, h } = fitRect(width, height);
    const t = new OffscreenCanvas(w, h).getContext('2d'); // image-sized, so no letterbox reads as alpha
    t.drawImage(bitmap, 0, 0, w, h);
    this.placeGray(uploadToMatte(t.getImageData(0, 0, w, h).data), w, h, fresh);
  }

  /**
   * Command. Replaces the raster layer with a w × h gray image, centered on black (fitRect's letterbox). One undo step.
   *
   * @param {Uint8ClampedArray} gray - (h, w, 4) uint8, R = G = B
   * @param {boolean} fresh - Also drop objects and inversion (a new matte, e.g. from a photo)
   */
  placeGray(gray, w, h, fresh = false) {
    if (fresh) { this.objects = []; this.selected = null; this.invert = false; }
    const r = fitRect(w, h);
    this.rctx.fillStyle = '#000';
    this.rctx.fillRect(0, 0, SIZE, SIZE);
    this.rctx.putImageData(new ImageData(gray, w, h), r.x, r.y);
    this.preview = null; this.rasterRev++;
    this.changed(); this.commit();
  }

  /** Command. Applies `patch` to the selected text object (or the defaults when none is selected). The caller commits. */
  setText(patch) {
    Object.assign(this.text, patch);
    const o = this.selected;
    if (o?.type !== 'text') return;
    const { value, ...style } = patch;
    Object.assign(o, style, value === undefined ? {} : { text: value });
    this.changed();
  }

  /** Command. Deletes the selected object (one undo step). */
  deleteSelected() {
    if (!this.selected) return;
    this.objects = this.objects.filter((o) => o !== this.selected);
    this.selected = null;
    this.changed(); this.commit();
  }

  /** Query. The matte's RGBA pixels, (1024, 1024, 4) uint8, gray in R = G = B. */
  pixels() { return this.ctx.getImageData(0, 0, SIZE, SIZE).data; }

  /** Command. Re-renders and notifies; bumps the revision used by the latent cache. */
  changed() { this.rev++; this.render(); this.onChange(); }

  /** Pure function. CSS font string of a text object. @example editor.fontOf({weight: 700, size: 100, font: 'Jost'}) // '700 100px Jost' */
  fontOf(o) { return `${o.weight} ${o.size}px ${o.font}`; }

  /** Command. Draws `base` (raster or threshold preview), the objects and the inversion into context c. */
  composite(c, base) {
    c.globalCompositeOperation = 'source-over';
    c.drawImage(base, 0, 0);
    for (const o of this.objects) {
      c.fillStyle = c.strokeStyle = levelColor(o.level ?? 1);
      if (o.type === 'rect') c.fillRect(o.x, o.y, o.w, o.h);
      if (o.type === 'ellipse') { c.beginPath(); c.ellipse(o.x + o.w / 2, o.y + o.h / 2, o.w / 2, o.h / 2, 0, 0, 2 * Math.PI); c.fill(); }
      if (o.type === 'polygon' || o.type === 'line') {
        c.beginPath();
        o.pts.forEach(([u, v], i) => (i ? c.lineTo(o.x + u * o.w, o.y + v * o.h) : c.moveTo(o.x + u * o.w, o.y + v * o.h)));
        if (o.type === 'polygon') { c.closePath(); c.fill(); } else { c.lineWidth = o.width; c.lineCap = c.lineJoin = 'round'; c.stroke(); }
      }
      if (o.type === 'text') {
        c.font = this.fontOf(o); c.textBaseline = 'top';
        o.width = c.measureText(o.text).width;
        c.fillText(o.text, o.x, o.y);
      }
    }
    if (this.invert) { c.globalCompositeOperation = 'difference'; c.fillStyle = '#fff'; c.fillRect(0, 0, SIZE, SIZE); c.globalCompositeOperation = 'source-over'; }
  }

  /** Command. Draws the matte canvas (a filter preview replaces everything) and the overlay. */
  render() {
    if (this.preview?.flat) this.ctx.drawImage(this.preview.canvas, 0, 0);
    else this.composite(this.ctx, this.preview?.canvas ?? this.raster);
    this.drawOverlay();
  }

  /** Command. Draws the selection box, resize handle, path in progress, lasso selection and brush cursor on the overlay. */
  drawOverlay() {
    const c = this.octx, accent = getComputedStyle(this.overlay).getPropertyValue('--accent').trim();
    c.clearRect(0, 0, SIZE, SIZE);
    c.strokeStyle = accent; c.fillStyle = accent; c.lineWidth = 3;
    if (this.selected) {
      const b = objectBox(this.selected);
      c.strokeRect(b.x, b.y, b.w, b.h);
      c.fillRect(b.x + b.w - HANDLE / 2, b.y + b.h - HANDLE / 2, HANDLE, HANDLE);
    }
    const lasso = this.drag?.kind === 'lasso' ? this.drag.pts : this.selection?.pts;
    if (lasso) {
      c.setLineDash(DASH); c.beginPath();
      lasso.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
      c.closePath(); c.stroke(); c.setLineDash([]);
    }
    if (this.path) {
      const pts = this.cursor ? [...this.path, this.cursor] : this.path;
      c.beginPath();
      pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
      c.stroke();
      for (const p of this.path) c.fillRect(p.x - VERTEX / 2, p.y - VERTEX / 2, VERTEX, VERTEX);
    }
    if (this.cursor && CURSOR_TOOLS.has(this.tool)) {
      c.beginPath(); c.arc(this.cursor.x, this.cursor.y, this.brushSize / 2, 0, 2 * Math.PI); c.stroke();
    }
  }

  /** Query. Pointer position in matte pixels. */
  point(e) {
    const r = this.overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  }

  /** Query. Topmost object under a point (lines also within half their width), or null. */
  hit(p) {
    return [...this.objects].reverse().find((o) => inBox(objectBox(o), p.x, p.y, o.type === 'line' ? o.width / 2 : 0)) ?? null;
  }

  /** Command. Draws one brush (current level) / eraser (black) segment on the raster. */
  stroke(a, b) {
    const c = this.rctx;
    c.strokeStyle = this.tool === 'brush' ? levelColor(this.level) : '#000';
    c.lineWidth = this.brushSize; c.lineCap = 'round';
    c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    this.rasterRev++;
  }

  /** Command. Pointer down: start a stroke, a lasso, a path vertex, a drag, a resize or a new object. */
  down(e) {
    this.overlay.setPointerCapture(e.pointerId);
    this.cancelPreview();
    const p = this.point(e);
    if (this.tool === 'brush' || this.tool === 'eraser') {
      this.drag = { kind: 'stroke', last: p, dirty: true };
      this.stroke(p, p); this.changed();
      return;
    }
    if (this.tool === 'lasso') { this.drag = { kind: 'lasso', pts: [p] }; return; }
    if (PATH_TOOLS.has(this.tool)) {
      const first = this.path?.[0], last = this.path?.at(-1);
      const double = last && e.timeStamp - this.lastDown < DOUBLE_CLICK_MS && Math.hypot(p.x - last.x, p.y - last.y) < DOUBLE_CLICK_DIST;
      this.lastDown = e.timeStamp;
      if (double || (this.tool === 'polygon' && this.path?.length >= 3 && Math.hypot(p.x - first.x, p.y - first.y) < CLOSE_DIST)) this.finishPath();
      else { this.path = [...(this.path ?? []), p]; this.selected = null; this.drawOverlay(); }
      return;
    }
    const sel = this.selected;
    if (sel) {
      const b = objectBox(sel);
      if (inBox({ x: b.x + b.w - HANDLE / 2, y: b.y + b.h - HANDLE / 2, w: HANDLE, h: HANDLE }, p.x, p.y, HANDLE / 2)) {
        this.drag = { kind: 'resize', start: p, box: b, size: sel.size };
        return;
      }
    }
    const hit = this.hit(p);
    if (hit) { this.selected = hit; this.drag = { kind: 'move', start: p, x: hit.x, y: hit.y }; this.drawOverlay(); return; }
    if (this.tool === 'text') {
      const o = { type: 'text', text: this.text.value, font: this.text.font, size: this.text.size, weight: this.text.weight, level: this.level, x: p.x, y: p.y, width: 0 };
      this.objects.push(o); this.selected = o; this.changed();
      this.drag = { kind: 'move', start: p, x: o.x, y: o.y, dirty: true };
    } else if (this.tool === 'rect' || this.tool === 'ellipse') {
      const o = { type: this.tool, level: this.level, x: p.x, y: p.y, w: 0, h: 0 };
      this.objects.push(o); this.selected = o;
      this.drag = { kind: 'create', start: p, obj: o };
    } else {
      this.selected = null; this.drawOverlay();
    }
  }

  /** Command. Pointer move: continue the current gesture, or just move the cursor (and a path's rubber band). */
  move(e) {
    const p = this.point(e);
    this.cursor = p;
    const d = this.drag;
    if (!d) { this.drawOverlay(); return; }
    if (d.kind === 'lasso') { d.pts.push(p); this.drawOverlay(); return; }
    d.dirty = true;
    if (d.kind === 'stroke') { this.stroke(d.last, p); d.last = p; }
    if (d.kind === 'move') { this.selected.x = d.x + p.x - d.start.x; this.selected.y = d.y + p.y - d.start.y; }
    if (d.kind === 'create') Object.assign(d.obj, spanBox(d.start.x, d.start.y, p.x, p.y));
    if (d.kind === 'resize') {
      const o = this.selected;
      if (o.type === 'text') o.size = Math.max(MIN_OBJECT, d.size * (p.y - d.box.y) / d.box.h);
      else { o.w = Math.max(MIN_OBJECT, p.x - o.x); o.h = Math.max(MIN_OBJECT, p.y - o.y); }
    }
    this.changed();
  }

  /** Command. Pointer up: end the gesture (one undo step if it changed anything); drops shapes that were only clicked, not dragged. */
  up() {
    const d = this.drag;
    this.drag = null;
    if (d?.kind === 'lasso') { this.setSelection(dedupePath(d.pts)); return; }
    if (d?.kind === 'create' && (d.obj.w < MIN_OBJECT || d.obj.h < MIN_OBJECT)) {
      this.objects = this.objects.filter((o) => o !== d.obj);
      this.selected = null;
      this.changed();
      return;
    }
    if (d?.dirty) this.commit();
  }
}
