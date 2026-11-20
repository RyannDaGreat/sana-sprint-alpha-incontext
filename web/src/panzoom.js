// Pan/zoom + minimap for the app's image displays (m00889, m00890). A vanilla-JS port of the user's SvelteLib components
// (github.com/RyannDaGreat/SvelteLib): src/lib/PanZoom.svelte (the viewport math: expZoomFactor, zoomTowards, pan,
// pinch, ease-out animation; wheel pans, ctrl+wheel / trackpad pinch zooms) and src/lib/MiniMap.svelte (the visible
// rect and world bounds in one overview). Additions for an image viewer: drag to pan, double press to fit, a 1:1
// button (one image pixel per device pixel; pixelated above), keys + − 0, and a draggable minimap box over a thumbnail.
// The Svelte port (manifest §12) can use the two original components directly.

const MIN_ZOOM = 1;           // the fitted image (zoom 1 = the container's own layout)
const MAX_ZOOM = 64;
const ZOOM_SENSITIVITY = 0.01;  // PanZoom.svelte's default
const KEY_ZOOM = 1.5;         // one + or − press
const ANIMATION_MS = 300;     // PanZoom.svelte's default
const DRAG_THRESHOLD_PX = 4;  // a smaller move is a click (pad slots toggle on click)
const MINIMAP_PX = 120;       // MiniMap.svelte's maxSize, a little smaller for the app's panels

/**
 * Pure function. Exponential zoom factor from a wheel delta (PanZoom.svelte).
 *
 * @param {number} deltaY
 * @returns {number}
 *
 * @example expZoomFactor(100) // 0.5 (zoom out)
 * @example expZoomFactor(-100) // 2 (zoom in)
 */
export function expZoomFactor(deltaY) {
  return Math.pow(2, -deltaY * ZOOM_SENSITIVITY);
}

/**
 * Pure function. The viewport zoomed to `zoom` with screen point (sx, sy) kept fixed (PanZoom.svelte zoomTowards).
 *
 * @param {number} sx
 * @param {number} sy
 * @param {number} zoom
 * @param {{zoom: number, panX: number, panY: number}} vp
 * @returns {{zoom: number, panX: number, panY: number}}
 *
 * @example zoomTowards(100, 100, 2, {zoom: 1, panX: 0, panY: 0}) // {zoom: 2, panX: -100, panY: -100}
 */
export function zoomTowards(sx, sy, zoom, vp) {
  const wx = (sx - vp.panX) / vp.zoom, wy = (sy - vp.panY) / vp.zoom;
  return { zoom, panX: sx - wx * zoom, panY: sy - wy * zoom };
}

/**
 * Pure function. Keeps the zoomed image covering the view: pan within [w − w·zoom, 0] (no empty margins once zoomed in).
 *
 * @param {{zoom: number, panX: number, panY: number}} vp
 * @param {number} w - view width
 * @param {number} h - view height
 * @returns {{zoom: number, panX: number, panY: number}}
 *
 * @example clampPan({zoom: 2, panX: 50, panY: -500}, 100, 100) // {zoom: 2, panX: 0, panY: -100}
 */
export function clampPan(vp, w, h) {
  const c = (p, lo) => Math.min(0, Math.max(lo, p));
  return { zoom: vp.zoom, panX: c(vp.panX, w - w * vp.zoom), panY: c(vp.panY, h - h * vp.zoom) };
}

/**
 * Pure function. The zoom at which one image pixel covers one device pixel, for an image `pixels` wide shown fitted in
 * a view `cssWidth` CSS pixels wide.
 *
 * @param {number} pixels - the image's width in pixels
 * @param {number} cssWidth - the fitted view's width in CSS pixels
 * @param {number} dpr - devicePixelRatio
 * @returns {number}
 *
 * @example oneToOneZoom(1024, 512, 1) // 2
 * @example oneToOneZoom(1024, 512, 2) // 1
 */
export function oneToOneZoom(pixels, cssWidth, dpr) {
  return pixels / (cssWidth * dpr);
}

/**
 * Pure function. Ease-out cubic (PanZoom.svelte).
 *
 * @param {number} t - in [0, 1]
 * @returns {number}
 *
 * @example easeOutCubic(0.5) // 0.875
 */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Command. Makes `view` (a positioned, overflow-hidden box whose canvas/img children fill it, e.g. `.square`) pan- and
 * zoomable, with fit / 1:1 / − / + buttons and a minimap that shows while zoomed in. The children get a CSS transform;
 * their pixels and layout are untouched, so drawing code keeps working. A drag longer than DRAG_THRESHOLD_PX swallows
 * the click that follows (pad slots toggle on click).
 *
 * @param {HTMLElement} view
 * @returns {{fit: () => void, oneToOne: () => void}}
 */
export function attachPanZoom(view) {
  let vp = { zoom: 1, panX: 0, panY: 0 }, anim = 0, drag = null, pinch = null;
  view.classList.add('pz');
  view.tabIndex = view.tabIndex >= 0 ? view.tabIndex : 0;
  const bar = document.createElement('div');
  bar.className = 'pz-bar';
  bar.innerHTML = '<button data-a="out" title="Zoom out (−)">−</button><button data-a="in" title="Zoom in (+)">+</button>'
    + '<button data-a="fit" title="Fit (0, double-click)">Fit</button><button data-a="one" title="One image pixel per screen pixel">1:1</button>';
  const map = document.createElement('canvas');
  map.className = 'pz-map';
  map.hidden = true;
  view.append(bar, map);
  const layers = () => [...view.children].filter((c) => c.matches('canvas:not(.pz-map), img'));
  const shown = () => layers().find((c) => !c.hidden);
  const size = () => ({ w: view.clientWidth, h: view.clientHeight });

  function apply(next) {
    const { w, h } = size();
    vp = next.zoom <= MIN_ZOOM ? { zoom: 1, panX: 0, panY: 0 } : clampPan(next, w, h);
    for (const c of layers()) {
      c.style.transformOrigin = '0 0';
      c.style.transform = vp.zoom === 1 ? '' : `translate(${vp.panX}px, ${vp.panY}px) scale(${vp.zoom})`;
      const img = c.width || c.naturalWidth || 0;
      c.style.imageRendering = img && vp.zoom > oneToOneZoom(img, w, devicePixelRatio) ? 'pixelated' : '';
    }
    drawMap();
  }
  function animateTo(target) {
    cancelAnimationFrame(anim);
    const start = vp, t0 = performance.now();
    const tick = (now) => {
      const e = easeOutCubic(Math.min((now - t0) / ANIMATION_MS, 1));
      apply({ zoom: start.zoom + (target.zoom - start.zoom) * e, panX: start.panX + (target.panX - start.panX) * e,
              panY: start.panY + (target.panY - start.panY) * e });
      if (e < 1) anim = requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }
  const zoomAt = (z, sx, sy) => animateTo(zoomTowards(sx, sy, Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z)), vp));
  const center = () => { const { w, h } = size(); return [w / 2, h / 2]; };
  const fit = () => animateTo({ zoom: 1, panX: 0, panY: 0 });
  const oneToOne = () => { const c = shown(); if (c) zoomAt(oneToOneZoom(c.width || c.naturalWidth, size().w, devicePixelRatio), ...center()); };

  // MiniMap.svelte's geometry over a thumbnail: world = the whole image, box = the visible rect (−pan/zoom, view/zoom)
  function drawMap() {
    map.hidden = vp.zoom === 1;
    if (map.hidden) return;
    const { w, h } = size(), s = MINIMAP_PX / Math.max(w, h);
    map.width = Math.round(w * s * devicePixelRatio); map.height = Math.round(h * s * devicePixelRatio);
    map.style.width = `${w * s}px`; map.style.height = `${h * s}px`;
    const g = map.getContext('2d'), k = s * devicePixelRatio, src = shown();
    if (src) g.drawImage(src, 0, 0, map.width, map.height);
    g.lineWidth = 1.5 * devicePixelRatio;
    g.strokeStyle = getComputedStyle(view).getPropertyValue('--accent') || 'red';
    g.strokeRect(-vp.panX / vp.zoom * k, -vp.panY / vp.zoom * k, w / vp.zoom * k, h / vp.zoom * k);
  }
  function mapTo(e) {  // centers the view on the minimap point under the pointer
    const r = map.getBoundingClientRect(), { w, h } = size();
    const wx = (e.clientX - r.left) / r.width * w, wy = (e.clientY - r.top) / r.height * h;
    apply({ zoom: vp.zoom, panX: w / 2 - wx * vp.zoom, panY: h / 2 - wy * vp.zoom });
  }

  bar.addEventListener('click', (e) => {
    const a = e.target.dataset?.a;
    if (!a) return;
    e.stopPropagation();
    if (a === 'fit') fit(); else if (a === 'one') oneToOne(); else zoomAt(vp.zoom * (a === 'in' ? KEY_ZOOM : 1 / KEY_ZOOM), ...center());
  });
  map.addEventListener('pointerdown', (e) => { e.stopPropagation(); map.setPointerCapture(e.pointerId); mapTo(e);
    map.onpointermove = mapTo; map.onpointerup = () => { map.onpointermove = null; }; });
  view.addEventListener('wheel', (e) => {
    const r = view.getBoundingClientRect();
    if (e.ctrlKey) { e.preventDefault(); apply(zoomTowards(e.clientX - r.left, e.clientY - r.top,
      Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, vp.zoom * expZoomFactor(e.deltaY))), vp)); }
    else if (vp.zoom > 1) { e.preventDefault(); apply({ zoom: vp.zoom, panX: vp.panX - e.deltaX, panY: vp.panY - e.deltaY }); }
  }, { passive: false });  // at fit, a plain wheel still scrolls the page
  view.addEventListener('mousedown', (e) => {  // a double press fits: mousedown's detail counts presses even with capture
    if (e.button === 0 && e.detail === 2 && vp.zoom !== 1 && !e.target.closest('.pz-bar, .pz-map')) { drag = null; fit(); }
  });
  view.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || vp.zoom === 1 || e.target.closest('.pz-bar, .pz-map')) return;
    drag = { x: e.clientX, y: e.clientY, vp, moved: false };
    view.setPointerCapture(e.pointerId);
  });
  view.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) drag.moved = true;
    if (drag.moved) apply({ zoom: drag.vp.zoom, panX: drag.vp.panX + dx, panY: drag.vp.panY + dy });
  });
  view.addEventListener('pointerup', () => { if (drag?.moved) view.addEventListener('click', (c) => c.stopPropagation(), { capture: true, once: true }); drag = null; });
  view.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea')) return;
    if (e.key === '+' || e.key === '=') zoomAt(vp.zoom * KEY_ZOOM, ...center());
    else if (e.key === '-' || e.key === '_') zoomAt(vp.zoom / KEY_ZOOM, ...center());
    else if (e.key === '0') fit();
    else return;
    e.preventDefault();
  });
  view.addEventListener('touchstart', (e) => {  // two-finger pinch (PanZoom.svelte calcPinchZoom)
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const r = view.getBoundingClientRect(), [a, b] = e.touches;
    pinch = { d: Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY), x: (a.clientX + b.clientX) / 2 - r.left,
              y: (a.clientY + b.clientY) / 2 - r.top, vp };
  }, { passive: false });
  view.addEventListener('touchmove', (e) => {
    if (!pinch || e.touches.length !== 2) return;
    e.preventDefault();
    const r = view.getBoundingClientRect(), [a, b] = e.touches;
    const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, pinch.vp.zoom * Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY) / pinch.d));
    const p = zoomTowards(pinch.x, pinch.y, z, pinch.vp);
    apply({ zoom: z, panX: p.panX + ((a.clientX + b.clientX) / 2 - r.left - pinch.x), panY: p.panY + ((a.clientY + b.clientY) / 2 - r.top - pinch.y) });
  }, { passive: false });
  view.addEventListener('touchend', () => { pinch = null; });
  new ResizeObserver(() => apply(vp)).observe(view);
  return { fit, oneToOne, redraw: drawMap };
}
