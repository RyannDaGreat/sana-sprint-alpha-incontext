// Resizable workbench panes (m00850): the 1px dividers between the tool sidebar | output over trimask | settings columns, and
// between the output and the trimask below it (m00900), are draggable. Behaves like SvelteLib's SplitPane (github.com/RyannDaGreat/SvelteLib,
// src/lib/SplitPane.svelte): a thin visible line, a wider invisible hit area, accent colour while hovered or dragged.
// Keyboard: arrow keys move a focused divider; double-click resets it. Sizes persist per viewer (localStorage).

const STORE_KEY = 'workbench-panes';
const KEY_STEP_PX = 16;  // one arrow-key press
const MIN_PANE_PX = 5;  // every pane can shrink to almost nothing: the user decides its size (m00902)
const DOUBLE_PRESS_MS = 400;  // two presses this close reset a divider (dblclick does not fire once pointerdown is cancelled)

/**
 * Pure function. Clamps x into [lo, hi]; when the range is empty (hi < lo), returns lo.
 *
 * @param {number} x
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 *
 * @example clamp(500, 220, 800) // 500
 * @example clamp(100, 220, 800) // 220
 * @example clamp(900, 220, 800) // 800
 */
export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(x, hi));
}

/**
 * Pure function. New pane sizes after dragging divider `which` to pointer offset `pos` (px from the workbench's
 * left edge for columns, from its top for the row divider), within the workbench box `w` × `h`.
 *
 * @param {{left: number, right: number, out: number}} sizes - tool sidebar width, inspector width, output panel height (px)
 * @param {'left'|'right'|'out'} which
 * @param {number} pos
 * @param {number} w - workbench width
 * @param {number} h - workbench height
 * @returns {{left: number, right: number, out: number}}
 *
 * @example dragSizes({left: 360, right: 360, out: 400}, 'left', 300, 1440, 850) // {left: 300, right: 360, out: 400}
 * @example dragSizes({left: 360, right: 360, out: 400}, 'left', 0, 1440, 850) // {left: 5, right: 360, out: 400}
 * @example dragSizes({left: 360, right: 360, out: 400}, 'right', 800, 1440, 850) // {left: 360, right: 640, out: 400}
 * @example dragSizes({left: 360, right: 360, out: 400}, 'right', 100, 1440, 850) // {left: 360, right: 1075, out: 400} (the middle keeps 5)
 * @example dragSizes({left: 360, right: 360, out: 400}, 'out', 849, 1440, 850) // {left: 360, right: 360, out: 845}
 */
export function dragSizes(sizes, which, pos, w, h) {
  if (which === 'left') return { ...sizes, left: clamp(pos, MIN_PANE_PX, w - sizes.right - MIN_PANE_PX) };
  if (which === 'right') return { ...sizes, right: clamp(w - pos, MIN_PANE_PX, w - sizes.left - MIN_PANE_PX) };
  return { ...sizes, out: clamp(pos, MIN_PANE_PX, h - MIN_PANE_PX) };
}

/**
 * Query. Saved sizes, or null (none saved, or storage unavailable).
 *
 * @returns {{left: number, right: number, out: number}|null}
 */
function loadSizes() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? 'null');
  } catch (e) {  // storage blocked (private mode, sandbox): defaults, said once in the console
    console.warn('workbench panes: sizes not restored', e);
    return null;
  }
}

/**
 * Command. Saves sizes (per viewer); storage failures are reported, not hidden.
 */
function saveSizes(sizes) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(sizes));
  } catch (e) {
    console.warn('workbench panes: sizes not saved', e);
  }
}

/**
 * Command. Makes the workbench's dividers draggable: adds three separator elements to `bench` and drives the CSS
 * variables --pane-left, --pane-right, --pane-out on it (style.css lays the grid out from them, desktop only).
 *
 * @param {HTMLElement} bench - the .workbench element
 */
export function initPanes(bench) {
  const css = getComputedStyle(document.documentElement);
  const side = parseFloat(css.getPropertyValue('--side-w')) || 360;
  const defaults = () => ({ left: side, right: side, out: Math.round(bench.clientHeight / 2) });
  let sizes = loadSizes() ?? defaults();
  const apply = () => {
    bench.style.setProperty('--pane-left', `${sizes.left}px`);
    bench.style.setProperty('--pane-right', `${sizes.right}px`);
    bench.style.setProperty('--pane-out', `${sizes.out}px`);
  };
  for (const which of ['left', 'right', 'out']) {
    const handle = document.createElement('div');
    handle.className = `pane-handle pane-${which}`;
    handle.tabIndex = 0;
    handle.setAttribute('role', 'separator');
    handle.setAttribute('aria-orientation', which === 'out' ? 'horizontal' : 'vertical');
    handle.title = 'Drag to resize · double-click to reset';
    const posOf = (e) => {
      const r = bench.getBoundingClientRect();
      return which === 'out' ? e.clientY - r.top : e.clientX - r.left;
    };
    const set = (pos) => { sizes = dragSizes(sizes, which, pos, bench.clientWidth, bench.clientHeight); apply(); };
    const reset = () => { sizes = { ...sizes, [which]: defaults()[which] }; apply(); saveSizes(sizes); };
    let lastPress = 0;
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      if (e.timeStamp - lastPress < DOUBLE_PRESS_MS) { lastPress = 0; reset(); return; }
      lastPress = e.timeStamp;
      handle.setPointerCapture(e.pointerId);
      handle.classList.add('dragging');
      const move = (ev) => { lastPress = 0; set(posOf(ev)); };  // a drag is not half of a double press
      const up = () => {
        handle.removeEventListener('pointermove', move);
        handle.classList.remove('dragging');
        saveSizes(sizes);
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', up, { once: true });
    });
    handle.addEventListener('keydown', (e) => {
      const sign = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[e.key];
      if (!sign) return;
      const cur = which === 'left' ? sizes.left : which === 'right' ? bench.clientWidth - sizes.right : sizes.out;
      set(cur + sign * KEY_STEP_PX); saveSizes(sizes);
      e.preventDefault();
    });
    bench.append(handle);
  }
  apply();
}
