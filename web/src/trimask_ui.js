// Trimask controls: the value palette (shared) and one filter panel, picked by ?gui=A|B|C|D
// (web/TRIMASK_GUI.md). A: one-click filter buttons. B: remap matrix + morphology with preview/Apply.
// C: plain-language verbs, optionally limited to a lasso selection. D (default): the user-study merge of A and C.
// All run trimask_ops.js through the editor.

import { blur, harden, dilate, erode, dilateIntoGray, erodeIntoGray, edgeBand, remapLevels, HARDEN_LO, HARDEN_HI } from './trimask_ops.js';

export const DEFAULT_GUI = 'D'; // the user-study winner (web/TRIMASK_GUI.md)
const SIGMA_PER_WIDTH = 1 / 2.5; // soften: a Gaussian ramp is about 2.5 σ wide, so σ = width / 2.5
const SIZE_DEFAULT = 20;         // px: grow/shrink distance, band and ramp width
const SIZE_MAX = 150;
const PERCENT = 100;
const LEVELS = [{ v: 1, name: 'White' }, { v: 0.5, name: 'Gray' }, { v: 0, name: 'Black' }];

/**
 * Pure function. The filter for an op id and its size: (mask, W, H) → mask.
 *
 * @param {string} id - grow | shrink | growGray | shrinkGray | band | soften | harden | remap
 * @param {{size?: number, to?: number[], lo?: number, hi?: number}} p - size in px; remap targets [black, gray, white]
 * @returns {(mask: Float32Array, W: number, H: number) => Float32Array}
 *
 * @example opFn('remap', {to: [0, 0, 1]})(new Float32Array([0, 0.5, 1]), 3, 1) // [0, 0, 1] (gray → black)
 * @example opFn('growGray', {size: 1})(new Float32Array([0, 0, 1]), 3, 1) // [0, 0.5, 1]
 */
export function opFn(id, p) {
  const fns = {
    grow: (m, W, H) => dilate(m, W, H, p.size),
    shrink: (m, W, H) => erode(m, W, H, p.size),
    growGray: (m, W, H) => dilateIntoGray(m, W, H, p.size),
    shrinkGray: (m, W, H) => erodeIntoGray(m, W, H, p.size),
    band: (m, W, H) => edgeBand(m, W, H, p.size),
    soften: (m, W, H) => blur(m, W, H, p.size * SIGMA_PER_WIDTH),
    harden: (m) => harden(m, p.lo ?? HARDEN_LO, p.hi ?? HARDEN_HI),
    remap: (m) => remapLevels(m, p.to),
  };
  if (!fns[id]) throw new Error(`unknown trimask op "${id}"`);
  return fns[id];
}

// Remaps as [black, gray, white] targets, shared by A and C.
const REMAPS = {
  whiteToGray: [0, 0.5, 0.5], grayToBlack: [0, 0, 1], grayToWhite: [0, 1, 1], blackToGray: [0.5, 0.5, 1],
  blackToWhite: [1, 0.5, 1], whiteToBlack: [0, 0.5, 0], swap: [1, 0.5, 0], allGray: [0.5, 0.5, 0.5],
};

/** Pure function. Element from an HTML string. @example html('<b>x</b>').tagName // 'B' */
function html(s) {
  const t = document.createElement('template');
  t.innerHTML = s.trim();
  return t.content.firstElementChild;
}

// Before/after glyphs on the filter buttons: two TILE-sized black tiles with an arrow between them.
const TILE = 21;
const TILE_GAP = 11;
const MID = TILE / 2;
const VALUE_NAME = { 0: 'black', 0.5: 'gray', 1: 'white' };
const SOFT_FADES = [[8, 0.3], [6.5, 0.4], [5, 0.6], [3.5, 1]]; // [radius, opacity]: a stepped blur, no <defs> needed

/** Pure function. SVG circle centered in a tile, filled with a matte value (CSS token --v-white|gray|black). @example disc(5, 'white') // '<circle cx="10.5" cy="10.5" r="5" fill="var(--v-white)" opacity="1"/>' */
const disc = (r, value, opacity = 1) => `<circle cx="${MID}" cy="${MID}" r="${r}" fill="var(--v-${value})" opacity="${opacity}"/>`;
const SOFT = SOFT_FADES.map(([r, o]) => disc(r, 'white', o)).join('');
const HARD = disc(8, 'gray') + disc(5, 'white');
const OP_TILES = { // op id → [before, after]
  grow: [disc(5, 'white'), disc(8, 'white')],
  shrink: [disc(7, 'white'), disc(4, 'white')],
  growGray: [disc(5, 'white'), disc(8, 'gray') + disc(5, 'white')],
  shrinkGray: [disc(7, 'white'), disc(7, 'gray') + disc(4, 'white')],
  band: [disc(6, 'white'), disc(8, 'gray') + disc(4, 'white')],
  soften: [disc(6, 'white'), SOFT],
  harden: [SOFT, HARD],
};

/** Pure function. Three vertical stripes (black | gray | white) with the given values, filling a tile. @example stripes([0, 0.5, 0.5]) // black, gray, gray stripes */
const stripes = (values) => values.map((v, i) => `<rect x="${i * TILE / 3}" width="${TILE / 3}" height="${TILE}" fill="var(--v-${VALUE_NAME[v]})"/>`).join('');

/**
 * Pure function. SVG string of a filter's before → after glyph. A remap shows its mapping as stripes
 * (before: black | gray | white; after: the targets [black, gray, white] it maps to).
 *
 * @param {string} id - op id (grow, shrink, growGray, shrinkGray, band, soften, harden, remap)
 * @param {{to?: number[]}} [p] - remap targets [black, gray, white]
 * @returns {string}
 *
 * @example glyph('remap', {to: [0, 0.5, 0.5]}) // '<svg class="glyph" ...>': stripes black|gray|white → black|gray|gray
 * @example glyph('grow') // '<svg class="glyph" ...>': a small white disc → a larger one
 */
function glyph(id, p) {
  const tiles = id === 'remap' ? [stripes([0, 0.5, 1]), stripes(p.to)] : OP_TILES[id];
  if (!tiles) throw new Error(`no glyph for trimask op "${id}"`);
  const tile = (x, shapes) => `<g transform="translate(${x})"><rect class="tile" width="${TILE}" height="${TILE}" fill="var(--v-black)"/>${shapes}</g>`;
  const arrowX = TILE + TILE_GAP / 2;
  return `<svg class="glyph" viewBox="0 0 ${2 * TILE + TILE_GAP} ${TILE}" aria-hidden="true">${tile(0, tiles[0])}`
    + `<path class="arrow" d="M${arrowX - 3} ${MID}h6m-2-2 2 2-2 2"/>${tile(TILE + TILE_GAP, tiles[1])}</svg>`;
}

/** Command. Builds the value palette into `box`; names = labels for white, gray, black. */
function palette(box, editor, names) {
  const row = html(`<div class="row group palette"><span class="field group-title">${names.title}</span><div class="ops"></div></div>`);
  for (const [i, l] of LEVELS.entries()) {
    const b = html(`<button class="swatch${l.v === editor.level ? ' on' : ''}" data-level="${l.v}" title="${names.tips?.[i] ?? ""}"><i style="--swatch: rgb(${255 * l.v}, ${255 * l.v}, ${255 * l.v})"></i>${names.levels[i]}</button>`);
    b.addEventListener('click', () => {
      row.querySelectorAll('.swatch').forEach((x) => x.classList.toggle('on', x === b));
      editor.setLevel(l.v);
    });
    row.lastChild.append(b);
  }
  box.append(row);
}

/** Command. A size slider row; returns the input. */
function sizeSlider(box, label, onInput) {
  const row = html(`<div class="row"><label class="field">${label} <input type="range" min="1" max="${SIZE_MAX}" value="${SIZE_DEFAULT}"><span class="meta">${SIZE_DEFAULT} px</span></label></div>`);
  const input = row.querySelector('input');
  input.addEventListener('input', () => { row.querySelector('.meta').textContent = `${input.value} px`; });
  input.addEventListener('change', () => onInput(Number(input.value))); // on release: a large radius takes ~1 s
  box.append(row);
  return input;
}

/**
 * Command. Immediate-apply buttons (designs A and C): a click applies at once (one undo step); moving the
 * size slider right after re-runs that op from the same starting point, replacing its step.
 */
function immediateButtons(box, editor, groups, sizeLabel, note) {
  let last = null; // {id, params, index}
  const status = html('<p class="meta hint"></p>');
  const run = (id, params, label) => {
    editor.applyOp(opFn(id, params));
    last = { id, params, label, index: editor.index };
    status.textContent = `${label}${params.size ? ` ${params.size} px` : ''} applied${editor.selection ? ' inside the lasso' : ''}. ${note}`;
  };
  const size = sizeSlider(box, sizeLabel, (v) => {
    if (!last?.params.size || editor.index !== last.index) return;
    editor.undo();
    run(last.id, { ...last.params, size: v }, last.label);
  });
  for (const [title, ops] of groups) {
    const row = html(`<div class="row group"><span class="field group-title">${title}</span><div class="ops tiles"></div></div>`);
    for (const [label, id, extra, tip] of ops) {
      const b = html(`<button class="op">${glyph(id, extra)}<span></span></button>`);
      b.lastChild.textContent = label;
      if (tip) b.title = tip;
      b.addEventListener('click', () => run(id, { ...(extra ?? {}), size: extra?.to ? undefined : Number(size.value) }, label));
      row.lastChild.append(b);
    }
    box.append(row);
  }
  box.append(status);
}

/** Command. Design A: technical one-click filter buttons. */
function designA(box, editor) {
  palette(box, editor, { title: 'Paint', levels: ['White', 'Gray', 'Black'] });
  box.append(html('<h3 class="sub-h">Filters</h3>'));
  immediateButtons(box, editor, [
    ['Edges', [['Soften', 'soften'], ['Harden', 'harden'], ['Edge → gray band', 'band']]],
    ['Shape', [['Dilate', 'grow'], ['Erode', 'shrink'], ['Dilate into gray', 'growGray'], ['Erode into gray', 'shrinkGray']]],
    ['Values', [['White → gray', 'remap', { to: REMAPS.whiteToGray }], ['Gray → black', 'remap', { to: REMAPS.grayToBlack }],
      ['Gray → white', 'remap', { to: REMAPS.grayToWhite }], ['Black → gray', 'remap', { to: REMAPS.blackToGray }],
      ['Black → white', 'remap', { to: REMAPS.blackToWhite }], ['White → black', 'remap', { to: REMAPS.whiteToBlack }],
      ['Swap black ↔ white', 'remap', { to: REMAPS.swap }]]],
  ], 'Size', 'Drag Size to adjust it; Undo reverts.');
}

/** Command. The "Where" row: filters act on the whole matte or only inside a lasso selection. */
function scopeRow(box, editor) {
  const scope = html(`<div class="row group"><span class="field group-title">Where</span><div class="ops"><span class="meta scope">Whole matte</span>
    <button class="lasso-tool" title="Pick the Lasso tool: drag around a region and the filters below change only inside it">Lasso</button>
    <button class="lasso-clear" title="Drop the lasso: filters change the whole matte again">Select all</button></div></div>`);
  const sync = () => { scope.querySelector('.scope').textContent = editor.selection ? 'Inside the lasso' : 'Whole matte'; };
  scope.querySelector('.lasso-tool').addEventListener('click', () => document.querySelector('[data-tool="lasso"]').click());
  scope.querySelector('.lasso-clear').addEventListener('click', () => { editor.setSelection(null); sync(); });
  editor.overlay.addEventListener('pointerup', () => setTimeout(sync));
  box.append(scope);
}

/** Command. Design C: plain-language verbs, scoped to a lasso selection when there is one. */
function designC(box, editor) {
  palette(box, editor, { title: 'Paint as', levels: ['Sure object', 'Unsure', 'Sure background'] });
  box.append(html('<h3 class="sub-h">Change the matte</h3>'));
  scopeRow(box, editor);
  immediateButtons(box, editor, [
    ['Edges', [['Blur edges', 'soften'], ['Snap to sure / unsure', 'harden'], ['Make edge unsure', 'band']]],
    ['Shape', [['Grow', 'grow'], ['Shrink', 'shrink'], ['Grow an unsure rim', 'growGray'], ['Shrink, rim unsure', 'shrinkGray']]],
    ['Certainty', [['Object → unsure', 'remap', { to: REMAPS.whiteToGray }], ['Unsure → background', 'remap', { to: REMAPS.grayToBlack }],
      ['Unsure → object', 'remap', { to: REMAPS.grayToWhite }], ['Background → unsure', 'remap', { to: REMAPS.blackToGray }],
      ['Background → object', 'remap', { to: REMAPS.blackToWhite }], ['Object → background', 'remap', { to: REMAPS.whiteToBlack }],
      ['Swap object ↔ background', 'remap', { to: REMAPS.swap }], ['Everything unsure', 'remap', { to: REMAPS.allGray }]]],
  ], 'Amount', 'Drag Amount to adjust it; Undo reverts.');
}

/**
 * Command. Design D (after the user study, the default): A's one-click buttons with technical names plus
 * their meaning, a tooltip on every button that says exactly what Size means, an explicit
 * "Harden → 3 levels", and C's lasso scope.
 */
function designD(box, editor) {
  palette(box, editor, { title: 'Paint', levels: ['Object', 'Unknown', 'Background'], tips: ['White = object: sure to be part of the foreground', 'Gray = unknown: partly transparent or not sure', 'Black = background: sure to be empty'] });
  box.append(html('<h3 class="sub-h">Filters <span class="meta">one click applies · Undo reverts</span></h3>'));
  scopeRow(box, editor);
  const remap = (label, to, tip) => [label, 'remap', { to }, `${tip} Fuzzy values in between are remapped proportionally.`];
  immediateButtons(box, editor, [
    ['Edges', [['Soften (blur)', 'soften', null, 'Blurs the matte: every hard edge becomes a smooth white → black ramp about Filter size px wide.'],
      ['Harden → 3 levels', 'harden', null, 'Snaps every pixel to one of three values: below 1/3 → black, above 2/3 → white, in between → gray.'],
      ['Edge → gray band', 'band', null, 'Puts a gray (unknown) band of total width Filter size, centered on every white/black edge: half inside, half outside.']]],
    ['Shape', [['Dilate (grow)', 'grow', null, 'Grows white outward by Filter size px.'],
      ['Erode (shrink)', 'shrink', null, 'Shrinks white inward by Filter size px.'],
      ['Dilate into gray', 'growGray', null, 'Grows the shape by Filter size px, but the added rim is gray (unknown), not white.'],
      ['Erode into gray', 'shrinkGray', null, 'Shrinks the shape by Filter size px, but the removed rim becomes gray (unknown), not black.']]],
    ['Values', [remap('White → gray', REMAPS.whiteToGray, 'Every white pixel becomes gray.'), remap('Gray → black', REMAPS.grayToBlack, 'Every gray pixel becomes black.'),
      remap('Gray → white', REMAPS.grayToWhite, 'Every gray pixel becomes white.'), remap('Black → gray', REMAPS.blackToGray, 'Every black pixel becomes gray.'),
      remap('Black → white', REMAPS.blackToWhite, 'Every black pixel becomes white.'), remap('White → black', REMAPS.whiteToBlack, 'Every white pixel becomes black.'),
      remap('Swap black ↔ white', REMAPS.swap, 'Black and white trade places; gray stays gray.'), remap('All → gray', REMAPS.allGray, 'Everything becomes gray (unknown).')]],
  ], 'Filter size', 'Drag Filter size to change it (re-runs this filter); Undo reverts.');
}

/** Command. Design B: a 3 × 3 remap matrix and morphology/soften/harden panels, each with live preview + Apply. */
function designB(box, editor) {
  palette(box, editor, { title: 'Paint', levels: ['White', 'Gray', 'Black'] });
  const active = html('<p class="meta hint">Change any control above to preview it on the matte, then press Apply.</p>');
  const bar = html('<div class="row"><button class="apply primary-lite" disabled>Apply</button><button class="cancel" disabled>Cancel</button></div>');
  let current = null; // {fn, label}
  const preview = (fn, label) => { current = { fn, label }; editor.previewOp(fn); active.textContent = `Previewing: ${label}`; sync(); };
  const sync = () => { bar.querySelector('.apply').disabled = bar.querySelector('.cancel').disabled = !editor.preview?.flat; };
  const done = (msg) => { current = null; active.textContent = msg; sync(); };
  bar.querySelector('.apply').addEventListener('click', () => { editor.applyPreview(); done(`Applied. Undo reverts.`); });
  bar.querySelector('.cancel').addEventListener('click', () => { editor.cancelPreview(); done('Cancelled.'); });
  editor.overlay.addEventListener('pointerdown', () => { if (current) done('Preview cancelled by drawing.'); });

  // Remap matrix: one row per source level, one radio column per target level.
  box.append(html('<h3 class="sub-h">Remap values</h3>'));
  const names = ['White', 'Gray', 'Black'];
  const grid = html(`<div class="matrix"><span></span>${names.map((n) => `<span class="meta">to ${n.toLowerCase()}</span>`).join('')}</div>`);
  for (const [r, from] of LEVELS.entries()) {
    grid.append(html(`<span class="field">${names[r]} →</span>`));
    for (const [c, to] of LEVELS.entries()) {
      const cell = html(`<label class="cell"><input type="radio" name="remap-${r}" value="${to.v}"${r === c ? ' checked' : ''}><span></span></label>`);
      cell.title = `${names[r]} becomes ${names[c].toLowerCase()}`;
      cell.querySelector('input').addEventListener('change', () => {
        const target = LEVELS.map((_, i) => Number(grid.querySelector(`input[name="remap-${i}"]:checked`).value)); // per row: white, gray, black
        const nameOf = (v) => LEVELS.find((l) => l.v === v).name.toLowerCase();
        const changes = LEVELS.filter((l, i) => target[i] !== l.v).map((l, i) => `${l.name.toLowerCase()} → ${nameOf(target[LEVELS.indexOf(l)])}`);
        preview(opFn('remap', { to: [target[2], target[1], target[0]] }), `remap ${changes.join(', ') || '(no change)'}`);
      });
      grid.append(cell);
    }
  }
  const reset = html('<button>Reset matrix</button>');
  reset.addEventListener('click', () => {
    grid.querySelectorAll('input').forEach((i) => { i.checked = Number(i.value) === LEVELS[Number(i.name.split('-')[1])].v; });
    if (current) { editor.cancelPreview(); done('Matrix reset.'); }
  });
  box.append(grid, html('<div class="row"></div>'));
  box.lastChild.append(reset);

  // Morphology: operation + into-gray toggle + size.
  box.append(html('<h3 class="sub-h">Morphology</h3>'));
  const morph = html(`<div class="row"><label class="field"><input type="radio" name="morph" value="grow" checked> Dilate</label>
    <label class="field"><input type="radio" name="morph" value="shrink"> Erode</label>
    <label class="field"><input type="radio" name="morph" value="band"> Edge band</label>
    <label class="switch"><input type="checkbox" class="into-gray"><span>Into gray</span></label></div>`);
  box.append(morph);
  const runMorph = () => {
    const kind = morph.querySelector('input[name="morph"]:checked').value, gray = morph.querySelector('.into-gray').checked;
    const id = kind === 'band' ? 'band' : gray ? `${kind}Gray` : kind;
    const label = `${{ grow: 'dilate', shrink: 'erode', band: 'edge band' }[kind]} ${size.value} px${gray && kind !== 'band' ? ' into gray' : ''}`;
    preview(opFn(id, { size: Number(size.value) }), label);
  };
  const size = sizeSlider(box, 'Size', runMorph);
  morph.querySelectorAll('input').forEach((i) => i.addEventListener('change', runMorph));

  // Soften and harden.
  box.append(html('<h3 class="sub-h">Soften / harden</h3>'));
  sizeSlider(box, 'Soften width', (v) => preview(opFn('soften', { size: v }), `soften ${v} px`));
  const hard = html(`<div class="row"><label class="field">Black below <input class="lo" type="range" min="0" max="${PERCENT}" value="${Math.round(PERCENT * HARDEN_LO)}"><span class="meta lo-v">${Math.round(PERCENT * HARDEN_LO)}%</span></label>
    <label class="field">White above <input class="hi" type="range" min="0" max="${PERCENT}" value="${Math.round(PERCENT * HARDEN_HI)}"><span class="meta hi-v">${Math.round(PERCENT * HARDEN_HI)}%</span></label>
    <button class="harden">Harden to 3 levels</button></div>`);
  const runHarden = () => {
    const lo = Number(hard.querySelector('.lo').value), hi = Number(hard.querySelector('.hi').value);
    hard.querySelector('.lo-v').textContent = `${lo}%`; hard.querySelector('.hi-v').textContent = `${hi}%`;
    preview(opFn('harden', { lo: lo / PERCENT, hi: hi / PERCENT }), `harden: black below ${lo}%, white above ${hi}%`);
  };
  hard.querySelectorAll('input').forEach((i) => i.addEventListener('input', runHarden));
  hard.querySelector('.harden').addEventListener('click', runHarden);
  box.append(hard, bar, active);
}

/**
 * Command. Builds the trimask controls for design `gui` into `box` and shows the tools it needs.
 *
 * @param {HTMLElement} box
 * @param {import('./matte.js').MatteEditor} editor
 * @param {string} gui - 'A' | 'B' | 'C'
 */
export function buildTrimaskUi(box, editor, gui) {
  const designs = { A: designA, B: designB, C: designC, D: designD };
  if (!designs[gui]) throw new Error(`?gui= must be A, B, C or D, not "${gui}"`);
  designs[gui](box, editor);
  document.querySelector('[data-tool="lasso"]').hidden = gui !== 'C' && gui !== 'D';
}
