// The matte editor: a 1024² white-on-black canvas built from a raster layer (uploads, brush,
// eraser) and vector objects (text, rectangles, ellipses) that can be dragged and resized.

import { SIZE, fitRect } from './util.js';

const HANDLE = 28;        // resize handle side, in matte pixels
const MIN_OBJECT = 8;     // smallest shape side, in matte pixels
const HISTORY_LIMIT = 40; // undo steps kept; each step with a raster change holds one 4 MB snapshot

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
 * @example cloneObjects([{type: 'rect', x: 1, y: 2, w: 3, h: 4}]) // [{type: 'rect', x: 1, y: 2, w: 3, h: 4}] (new objects)
 */
export function cloneObjects(objects) {
  return objects.map((o) => ({ ...o }));
}

/**
 * Pure function. Axis-aligned box of an object (text boxes come from the measured width).
 *
 * @param {object} o - {type: 'rect'|'ellipse', x, y, w, h} or {type: 'text', x, y, size, width}
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
   * @param {HTMLCanvasElement} canvas - The matte itself (pure white-on-black)
   * @param {HTMLCanvasElement} overlay - Selection box and brush cursor, drawn on top
   * @param {() => void} onChange - Called after every edit
   */
  constructor(canvas, overlay, onChange) {
    this.canvas = canvas; this.overlay = overlay; this.onChange = onChange;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.octx = overlay.getContext('2d');
    this.raster = new OffscreenCanvas(SIZE, SIZE);
    this.rctx = this.raster.getContext('2d');
    this.objects = [];
    this.selected = null;
    this.invert = false;
    this.rev = 0;
    this.tool = 'brush';
    this.brushSize = 64;
    this.text = { value: 'HELLO', font: "'Jost Variable'", size: 220, weight: 700 };
    this.drag = null;
    this.cursor = null;
    this.preview = null;        // threshold preview: an OffscreenCanvas drawn instead of the raster
    this.rasterRev = 0;         // bumped on every raster edit, so snapshots share unchanged rasters
    this.history = []; this.index = -1;
    this.onHistory = () => {};  // called when undo/redo availability or the threshold preview may change
    this.clear();
    overlay.addEventListener('pointerdown', (e) => this.down(e));
    overlay.addEventListener('pointermove', (e) => this.move(e));
    overlay.addEventListener('pointerup', (e) => this.up(e));
    overlay.addEventListener('pointerleave', () => { this.cursor = null; this.drawOverlay(); });
  }

  /** Command. Clears raster and objects to black (one undo step). */
  clear() {
    this.preview = null;
    this.rctx.fillStyle = '#000';
    this.rctx.fillRect(0, 0, SIZE, SIZE);
    this.rasterRev++;
    this.objects = []; this.selected = null; this.invert = false;
    this.changed(); this.commit();
  }

  /** Command. Toggles inversion of the whole matte (one undo step). */
  toggleInvert() { this.invert = !this.invert; this.changed(); this.commit(); }

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
    this.index = i; this.preview = null; this.drag = null; this.selected = null;
    if (s.rasterRev !== this.rasterRev) { this.rctx.putImageData(s.raster, 0, 0); this.rasterRev = s.rasterRev; }
    this.objects = cloneObjects(s.objects); this.invert = s.invert;
    this.changed(); this.onHistory();
  }

  /** Command. Steps back / forward in history (no-op at either end). */
  undo() { if (this.canUndo()) this.restore(this.index - 1); }
  redo() { if (this.canRedo()) this.restore(this.index + 1); }

  // ---------- threshold ----------

  /** Command. Shows the raster thresholded at `level` (with `softness` ramp) without committing it. */
  previewThreshold(level, softness) {
    const gray = thresholdGray(this.rctx.getImageData(0, 0, SIZE, SIZE).data, level, softness);
    this.preview ??= new OffscreenCanvas(SIZE, SIZE);
    this.preview.getContext('2d').putImageData(new ImageData(gray, SIZE, SIZE), 0, 0);
    this.changed();
  }

  /** Command. Drops the threshold preview. */
  cancelThreshold() { if (this.preview) { this.preview = null; this.changed(); this.onHistory(); } }

  /** Command. Writes the threshold preview into the raster (one undo step). Vector objects are already binary. */
  applyThreshold() {
    if (!this.preview) return;
    this.rctx.drawImage(this.preview, 0, 0);
    this.preview = null; this.rasterRev++;
    this.changed(); this.commit();
  }

  /** Command. Sets the active tool and drops the selection when it no longer applies. */
  setTool(tool) { this.tool = tool; if (tool === 'brush' || tool === 'eraser') this.selected = null; this.drawOverlay(); }

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

  /** Command. Draws raster + objects (+ inversion) onto the matte canvas. */
  render() {
    const c = this.ctx;
    c.globalCompositeOperation = 'source-over';
    c.drawImage(this.preview ?? this.raster, 0, 0);
    c.fillStyle = '#fff';
    for (const o of this.objects) {
      if (o.type === 'rect') c.fillRect(o.x, o.y, o.w, o.h);
      if (o.type === 'ellipse') { c.beginPath(); c.ellipse(o.x + o.w / 2, o.y + o.h / 2, o.w / 2, o.h / 2, 0, 0, 2 * Math.PI); c.fill(); }
      if (o.type === 'text') {
        c.font = this.fontOf(o); c.textBaseline = 'top';
        o.width = c.measureText(o.text).width;
        c.fillText(o.text, o.x, o.y);
      }
    }
    if (this.invert) { c.globalCompositeOperation = 'difference'; c.fillRect(0, 0, SIZE, SIZE); c.globalCompositeOperation = 'source-over'; }
    this.drawOverlay();
  }

  /** Command. Draws the selection box, resize handle and brush cursor on the overlay. */
  drawOverlay() {
    const c = this.octx, accent = getComputedStyle(this.overlay).getPropertyValue('--accent').trim();
    c.clearRect(0, 0, SIZE, SIZE);
    c.strokeStyle = accent; c.fillStyle = accent; c.lineWidth = 3;
    if (this.selected) {
      const b = objectBox(this.selected);
      c.strokeRect(b.x, b.y, b.w, b.h);
      c.fillRect(b.x + b.w - HANDLE / 2, b.y + b.h - HANDLE / 2, HANDLE, HANDLE);
    }
    if (this.cursor && (this.tool === 'brush' || this.tool === 'eraser')) {
      c.beginPath(); c.arc(this.cursor.x, this.cursor.y, this.brushSize / 2, 0, 2 * Math.PI); c.stroke();
    }
  }

  /** Query. Pointer position in matte pixels. */
  point(e) {
    const r = this.overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  }

  /** Query. Topmost object under a point, or null. */
  hit(p) { return [...this.objects].reverse().find((o) => inBox(objectBox(o), p.x, p.y)) ?? null; }

  /** Command. Draws one brush/eraser segment on the raster. */
  stroke(a, b) {
    const c = this.rctx;
    c.strokeStyle = this.tool === 'brush' ? '#fff' : '#000';
    c.lineWidth = this.brushSize; c.lineCap = 'round';
    c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    this.rasterRev++;
  }

  /** Command. Pointer down: start a stroke, a drag, a resize or a new object. */
  down(e) {
    this.overlay.setPointerCapture(e.pointerId);
    this.cancelThreshold();
    const p = this.point(e);
    if (this.tool === 'brush' || this.tool === 'eraser') {
      this.drag = { kind: 'stroke', last: p, dirty: true };
      this.stroke(p, p); this.changed();
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
      const o = { type: 'text', text: this.text.value, font: this.text.font, size: this.text.size, weight: this.text.weight, x: p.x, y: p.y, width: 0 };
      this.objects.push(o); this.selected = o; this.changed();
      this.drag = { kind: 'move', start: p, x: o.x, y: o.y, dirty: true };
    } else if (this.tool === 'rect' || this.tool === 'ellipse') {
      const o = { type: this.tool, x: p.x, y: p.y, w: 0, h: 0 };
      this.objects.push(o); this.selected = o;
      this.drag = { kind: 'create', start: p, obj: o };
    } else {
      this.selected = null; this.drawOverlay();
    }
  }

  /** Command. Pointer move: continue the current gesture, or just move the brush cursor. */
  move(e) {
    const p = this.point(e);
    this.cursor = p;
    const d = this.drag;
    if (!d) { this.drawOverlay(); return; }
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
    if (d?.kind === 'create' && (d.obj.w < MIN_OBJECT || d.obj.h < MIN_OBJECT)) {
      this.objects = this.objects.filter((o) => o !== d.obj);
      this.selected = null;
      this.changed();
      return;
    }
    if (d?.dirty) this.commit();
  }
}
