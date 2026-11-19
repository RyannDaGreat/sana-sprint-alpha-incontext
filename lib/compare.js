// The SANA vs FIBO2 compare reports (manifest §15). Every page in journal/compare/ is a shell with
// <body data-view="...">; this module draws it from data/compare.json (written by src/sana_alpha/compare_build.py).

const MODELS = { sana: 'SANA-Sprint 1.6B', fibo2: 'FIBO2 8.1B' };
const VIEWS = [['index', 'Hub'], ['over_time', 'Over time'], ['scrub', 'Scrub'], ['matrix', 'Matrix'],
               ['strips', 'Strips'], ['pixel', 'Pixel'], ['fibo2', 'FIBO2 journal']];
const METRICS = {
  band_iou: ['IoU in the gray band', 'higher is better', true],
  white_miss: ['white miss', 'lower is better', false],
  black_hit: ['black hit', 'lower is better', false],
  iou: ['IoU, whole image', 'higher is better', true],
  exterior_gray_error: ['gray error outside the matte', 'lower is better', false],
};
const BASELINES = { band_iou: [['band_iou_bg', 'band as background'], ['band_iou_fg', 'band as object']] };
const BOOT_DRAWS = 1000;  // resamples per interval drawn in the browser (the per-eval ones come from Python)
const CHART = { w: 320, h: 220, left: 36, right: 8, top: 8, bottom: 28 };  // SVG user units; CSS scales the SVG
const TICKS = 4;
const DOT = 2.5;  // point radius, SVG user units
const WORST_SHOWN = 6;  // items per list in the matrix drill-down

/**
 * Pure function. Resolves a data path: absolute URLs (the published original PNGs) as they are, journal-relative paths from a page in journal/compare/.
 * @param {string} path
 * @returns {string}
 * @example url('data/compare.json') // '../data/compare.json'  (no image path here: publish_pages rewrites every literal one)
 * @example url('https://h/ab/cd.png') // 'https://h/ab/cd.png'
 */
export function url(path) { return /^https?:/.test(path) ? path : `../${path}`; }

/**
 * Query (reads location.hash). The view state, e.g. #method=scribbles&res=512.
 * @returns {Object<string, string>}
 */
function readHash() { return Object.fromEntries(new URLSearchParams(location.hash.slice(1))); }

/** Command. Merges `changes` into the URL hash without adding history entries. */
function writeHash(changes) {
  const state = { ...readHash(), ...changes };
  history.replaceState(null, '', `#${new URLSearchParams(state)}`);
}

/**
 * Pure function. Seeded PRNG (mulberry32): the same seed gives the same intervals on every load.
 * @param {number} seed
 * @returns {() => number} uniform in [0, 1)
 * @example rng(1)() // 0.6270739405881613
 */
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/**
 * Near-pure function (seeded RNG). Mean and 95% percentile-bootstrap interval of values (null dropped).
 * @param {(number|null)[]} values
 * @returns {number[]} [mean, low, high]
 * @example bootstrap([0.5, 0.5]) // [0.5, 0.5, 0.5]
 */
export function bootstrap(values) {
  const v = values.filter((x) => x !== null && !Number.isNaN(x));
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const random = rng(v.length);
  const means = Array.from({ length: BOOT_DRAWS }, () => mean(Array.from(v, () => v[Math.floor(random() * v.length)]))).sort((a, b) => a - b);
  return [mean(v), means[Math.floor(BOOT_DRAWS * 0.025)], means[Math.floor(BOOT_DRAWS * 0.975)]];
}

/**
 * Pure function. Per-item mean over several evals (e.g. all methods of one step), ignoring missing values.
 * @param {Object[]} evals - records of compare.json
 * @param {string} metric
 * @param {boolean[]} keep - items to keep
 * @returns {(number|null)[]}
 * @example itemMeans([{per_item: {m: [1, null]}}, {per_item: {m: [0, 0.5]}}], 'm', [true, true]) // [0.5, 0.5]
 */
export function itemMeans(evals, metric, keep) {
  return keep.map((k, i) => {
    const v = evals.map((e) => e.per_item[metric][i]).filter((x) => x !== null);
    return k && v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
  }).filter((_, i) => keep[i]);
}

/** Pure function. Evals matching every given field. @example pick([{res: 1}, {res: 2}], {res: 2}).length // 1 */
export function pick(evals, where) { return evals.filter((e) => Object.entries(where).every(([k, v]) => v === undefined || e[k] === v)); }

/** Pure function. Sorted unique values. @example unique([3, 1, 3]) // [1, 3] */
export function unique(values) { return [...new Set(values)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)); }

/**
 * Pure function. Over-time series of one model: per step, the bootstrap of item means over the chosen methods.
 * x = 'step' (training step) or 'samples' (training images seen: step × global batch, the fair axis across models).
 * @returns {number[][]} [[x, mean, low, high], ...]
 */
function series(report, model, metric, res, methods, keep, x = 'step') {
  const done = report.evals.filter((e) => e.complete);  // a curve point only from finished evals: partial ones would bias it
  return unique(pick(done, { model, res }).map((e) => e.step)).map((step) => {
    const evals = pick(done, { model, res, step }).filter((e) => methods.includes(e.method));
    return evals.length ? [evals[0][x], ...bootstrap(itemMeans(evals, metric, keep))] : null;
  }).filter(Boolean);
}

/** Pure function. Element helper: tag, attributes, children (strings become text). */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === 'style' ? Object.assign(node.style, v) : k.startsWith('on') ? node.addEventListener(k.slice(2), v) : node.setAttribute(k, v);
  node.append(...children.flat().filter((c) => c !== null && c !== undefined));
  return node;
}

/** Pure function. A <select> over options [[value, label]] bound to the hash key. */
function select(key, options, value, onChange) {
  return el('label', {}, `${key} `, el('select', { onchange: (e) => { writeHash({ [key]: e.target.value }); onChange(); } },
    options.map(([v, label]) => { const o = el('option', { value: v }, label); o.selected = String(v) === String(value); return o; })));
}

/**
 * Pure function. An SVG line chart with interval bands. Series: {cls, label, points: [[x, mean, low, high]]}, baselines: {label, points: [[x, y]]}.
 * @returns {SVGElement}
 */
function lineChart(lines, baselines = [], yMax = 1) {
  const NS = 'http://www.w3.org/2000/svg';
  const all = lines.flatMap((s) => s.points.map((p) => p[0]));
  const [x0, x1] = all.length ? [Math.min(...all), Math.max(...all)] : [0, 1];
  const X = (x) => CHART.left + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (CHART.w - CHART.left - CHART.right);
  const Y = (y) => CHART.h - CHART.bottom - (y / yMax) * (CHART.h - CHART.top - CHART.bottom);
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${CHART.w} ${CHART.h}`);
  const add = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
  add('line', { class: 'axis', x1: CHART.left, x2: CHART.w - CHART.right, y1: Y(0), y2: Y(0) });
  for (let k = 0; k <= TICKS; k++) add('text', { x: 2, y: Y((yMax * k) / TICKS) + 4 }, ((yMax * k) / TICKS).toFixed(2));
  const xs = unique(all), every = Math.ceil(xs.length / (TICKS + 2));
  for (const x of xs.filter((_, i) => i % every === 0)) add('text', { x: X(x), y: CHART.h - 8, 'text-anchor': 'middle' }, x >= 1000 ? `${x / 1000}k` : x);
  for (const b of baselines) add('path', { class: 'baseline', d: b.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0])},${Y(p[1])}`).join('') }).append(Object.assign(document.createElementNS(NS, 'title'), { textContent: b.label }));
  for (const s of lines) {
    if (!s.points.length) continue;
    const band = s.points.map((p) => `${X(p[0])},${Y(p[3])}`).concat(s.points.slice().reverse().map((p) => `${X(p[0])},${Y(p[2])}`));
    add('polygon', { class: `band ${s.cls}`, points: band.join(' ') });
    add('path', { class: `line ${s.cls}`, d: s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0])},${Y(p[1])}`).join('') });
    for (const p of s.points) add('circle', { class: s.cls, cx: X(p[0]), cy: Y(p[1]), r: DOT }).append(Object.assign(document.createElementNS(NS, 'title'), { textContent: `${s.label} step ${p[0]}: ${p[1].toFixed(3)} [${p[2].toFixed(3)}, ${p[3].toFixed(3)}]` }));
  }
  return svg;
}

/**
 * Pure function. A cell showing one original image (lazy-loaded, never downscaled on the server; a click opens the
 * file at full resolution), or a gray "rendering" cell when `src` is null (an item its eval has not reached yet).
 * @example imageCell('img/a.png', 'sana') // <a class="cell"><img src="../img/a.png"><span class="tag">sana</span></a>  (real paths are rewritten by publish_pages)
 */
function imageCell(src, label) {
  if (!src) return el('div', { class: 'empty', title: `${label}: rendering` }, `${label}: rendering`);
  return el('a', { class: 'cell', href: url(src), target: '_blank', title: `${label} (original file)` },
    el('img', { src: url(src), loading: 'lazy', decoding: 'async', alt: label }), el('span', { class: 'tag' }, label));
}

/**
 * Pure function. Placeholder for a model with no eval at this (method, resolution, step): "not trained yet" when the
 * model has no evals at all, "rendering" when it has evals at this step (this one is queued), else "no eval at step N".
 * @example empty('fibo2') // <div class="empty">FIBO2 8.1B: not trained yet</div>
 */
function empty(model, report, step) {
  const evals = report ? pick(report.evals, { model }) : [];
  const text = !evals.length ? 'not trained yet' : pick(evals, { step }).length ? 'rendering' : `no eval at step ${step}`;
  return el('div', { class: 'empty' }, `${MODELS[model]}: ${text}`);
}

/** Pure function. Mask of the items in the chosen split ("held" = held-out only). */
function keepOf(report, split) { return report.items.map((i) => split === 'all' || i.split === 'held-out'); }

/** Pure function. Formats [mean, low, high]. @example fmt([0.5, 0.4, 0.6]) // '0.500 [0.400, 0.600]' */
export function fmt(ci) { return ci ? `${ci[0].toFixed(3)} [${ci[1].toFixed(3)}, ${ci[2].toFixed(3)}]` : 'n/a'; }

/** Command. Hub: verdict, status strip, headline chart, glossary. */
function hub(report, root) {
  const keep = keepOf(report, 'held');
  const lines = Object.keys(MODELS).map((m) => ({ cls: m === 'sana' ? 'sana' : 'fibo', label: MODELS[m], points: series(report, m, 'band_iou', 1024, report.methods, keep, 'samples') }));
  const last = (m) => lines.find((l) => l.label === MODELS[m]).points.at(-1);
  const sana = last('sana'), fibo = last('fibo2');
  const near = fibo && lines[0].points.reduce((a, p) => (Math.abs(p[0] - fibo[0]) < Math.abs(a[0] - fibo[0]) ? p : a));  // SANA at the closest training amount
  const k = (n) => `${Math.round(n / 1000)}k`;
  const verdict = fibo ? `IoU in the gray band (held-out, 1024²) after ${k(fibo[0])} training images: FIBO2 ${fmt(fibo.slice(1))}; SANA after ${k(near[0])}: ${fmt(near.slice(1))}. SANA's latest (${k(sana[0])} images): ${fmt(sana.slice(1))}. Intervals are per model (not a paired difference): no "better" claim yet.`
    : `FIBO2 is not trained yet: there is nothing to compare. SANA's IoU in the gray band (held-out, 1024², ${report.methods.length} methods) is ${sana ? `${fmt(sana.slice(1))} at step ${sana[0]}` : 'not evaluated yet'}.`;
  const count = (m) => pick(report.evals, { model: m }).length;
  root.append(
    el('p', { class: 'verdict' }, verdict),
    el('div', { class: 'status' }, Object.keys(MODELS).map((m) => el('div', {}, el('b', {}, MODELS[m]), `: ${count(m)} evals, steps ${unique(pick(report.evals, { model: m }).map((e) => e.step)).join(', ') || 'none'}`)),
      el('div', {}, `${report.items.length} items (${keep.filter(Boolean).length} held-out), methods: ${report.methods.join(', ')}, resolutions: ${report.resolutions.join(', ')}`)),
    el('div', { class: 'chart' }, el('h3', {}, 'IoU in the gray band vs training images seen (held-out, 1024², 95% bootstrap bands)'),
      lineChart(lines, BASELINES.band_iou.map(([b, label]) => ({ label, points: series(report, 'sana', b, 1024, report.methods, keep, 'samples').map((p) => [p[0], p[1]]) }))),
      el('p', { class: 'legend' }, el('span', { class: 'sana' }, '● SANA'), el('span', { class: 'fibo' }, '● FIBO2'), el('span', {}, '- - trivial baselines (band filled as background / as object)'))),
    el('h2', {}, 'Glossary'),
    el('dl', { class: 'glossary' }, [
      ['trimask', 'the condition: white = surely object, black = surely background, gray = unknown (the model decides).'],
      ['gray band', 'the trimask\'s unknown pixels.'],
      ['IoU in the gray band', 'overlap of the painted object with the true matte, only inside the gray band (1 = exact). The primary metric: elsewhere the trimask gives the answer away.'],
      ['white miss / black hit', 'share of the white left unpainted / of the black painted (0 = obeyed).'],
      ['method', 'one of the 19 ways a trimask is made from the true matte (manifest §13).'],
      ['held-out', 'items never seen in training. Headlines use only these.'],
      ['λ = 0', 'full subtraction of the style adapter (the shipped model).'],
      ['FIBO2', 'Bria Fibo 2.0, 8.1B, trained the same way as SANA (manifest §14).'],
    ].flatMap(([t, d]) => [el('dt', {}, t), el('dd', {}, d)])));
}

/** Command. Small multiples of every metric over steps, per model, with per-method lines on demand. */
function overTime(report, root) {
  const draw = () => {
    const h = readHash(), res = Number(h.res || 1024), split = h.split || 'held', perMethod = h.per_method === '1', x = h.x || 'samples';
    const keep = keepOf(report, split);
    body.replaceChildren(...Object.entries(METRICS).map(([metric, [title, sense]]) => {
      const lines = Object.keys(MODELS).flatMap((m) => (perMethod ? report.methods.map((method) => [m, [method], `${MODELS[m]} ${method}`]) : [[m, report.methods, MODELS[m]]])
        .map(([model, methods, label]) => ({ cls: model === 'sana' ? 'sana' : 'fibo', label, points: series(report, model, metric, res, methods, keep, x) })));
      const baselines = (BASELINES[metric] || []).map(([k, label]) => ({ label, points: series(report, 'sana', k, res, report.methods, keep, x).map((p) => [p[0], p[1]]) }));
      const yMax = Math.max(METRICS[metric][2] ? 1 : 0.1, ...lines.flatMap((l) => l.points.map((p) => p[3])));
      return el('div', { class: 'chart' }, el('h3', {}, `${title} (${sense})`), lineChart(lines, baselines, yMax));
    }));
  };
  const h = readHash();
  const body = el('div', { class: 'multiples' });
  root.append(el('div', { class: 'controls' },
    select('res', report.resolutions.map((r) => [r, `${r}²`]), h.res || 1024, draw),
    select('split', [['held', 'held-out'], ['all', 'all items']], h.split || 'held', draw),
    select('per_method', [['0', 'mean over methods'], ['1', 'one line per method']], h.per_method || '0', draw),
    select('x', [['samples', 'training images seen'], ['step', 'training step']], h.x || 'samples', draw)),
  el('p', { class: 'note' }, 'Each point: mean over items of the per-item mean over methods; band: 95% bootstrap over items. Dashed: trivial baselines. y starts at 0.'), body);
  draw();
}

/** Command. Checkpoint slider: every item's trimask, SANA and FIBO2 at the chosen step, method and resolution. */
function scrub(report, root) {
  const h = readHash();
  const steps = unique(report.evals.map((e) => e.step));
  const slider = el('input', { type: 'range', min: 0, max: Math.max(steps.length - 1, 0), value: Math.max(steps.indexOf(Number(h.step)), steps.length - 1) });
  const label = el('b', {});
  const grid = el('div', { class: 'grid' });
  const draw = () => {
    const s = readHash(), method = s.method || report.methods[0], res = Number(s.res || 1024), step = steps[Number(slider.value)];
    label.textContent = `step ${step}`;
    writeHash({ step });
    const cond = report.conditions.trimask[`${method}@${res}`];
    grid.replaceChildren(...report.items.map((item, i) => ((s.split || 'all') === 'all' || item.split === 'held-out' ? el('div', { class: 'item' },
      el('div', { class: 'cells' }, cond ? imageCell(cond.images[i], 'trimask') : empty('sana', report, step),
        ...Object.keys(MODELS).map((m) => { const e = pick(report.evals, { model: m, method, res, step })[0]; return e ? imageCell(e.images[i], m) : empty(m, report, step); })),
      el('a', { href: `pixel.html#page_id=${item.page_id}&method=${method}&res=${res}&step=${step}` }, `${item.split} · ${item.caption.slice(0, 60)}`)) : null)).filter(Boolean));
  };
  slider.addEventListener('input', draw);
  root.append(el('div', { class: 'controls' }, select('method', report.methods.map((m) => [m, m]), h.method || report.methods[0], draw),
    select('res', report.resolutions.map((r) => [r, `${r}²`]), h.res || 1024, draw),
    select('split', [['all', 'all items'], ['held', 'held-out']], h.split || 'all', draw), label, slider),
  el('p', { class: 'note' }, 'Cells: trimask, SANA, FIBO2 (same item, same noise). Click a caption to compare at full resolution.'), grid);
  draw();
}

/** Command. Method × resolution heatmap of one metric at one step; a cell opens its worst, best and random items. */
function matrix(report, root) {
  const h = readHash();
  const steps = unique(report.evals.map((e) => e.step));
  const table = el('div', { class: 'scroller' }), drill = el('div', {});
  const draw = () => {
    const s = readHash(), metric = s.metric || 'band_iou', step = Number(s.step || steps.at(-1)), keep = keepOf(report, 'held');
    const shade = (v) => `color-mix(in srgb, color-mix(in srgb, var(--good) ${Math.round((METRICS[metric][2] ? v : 1 - Math.min(v * 10, 1)) * 100)}%, var(--bad)) var(--heat-strength), var(--bg))`;
    table.replaceChildren(el('table', { class: 'heat' }, el('tr', {}, el('th', {}, 'method'), report.resolutions.flatMap((r) => Object.keys(MODELS).map((m) => el('th', {}, `${m} ${r}²`)))),
      report.methods.map((method) => el('tr', {}, el('th', {}, method), report.resolutions.flatMap((res) => Object.keys(MODELS).map((m) => {
        const e = pick(report.evals, { model: m, method, res, step })[0];
        if (!e) return el('td', {}, '—');
        if (!e.complete) return el('td', { class: 'click', onclick: () => showDrill(e, metric) }, `rendering ${e.images.filter(Boolean).length}/${e.images.length}`);
        const v = bootstrap(itemMeans([e], metric, keep))[0];
        return el('td', { class: 'click', style: { background: shade(v) }, onclick: () => showDrill(e, metric) }, v.toFixed(3));
      }))))));
  };
  const showDrill = (e, metric) => {
    const order = report.items.map((_, i) => i).filter((i) => e.per_item[metric][i] !== null).sort((a, b) => e.per_item[metric][a] - e.per_item[metric][b]);
    const worstFirst = METRICS[metric][2] ? order : order.slice().reverse();
    const random = rng(e.step)();
    const lists = [['worst', worstFirst.slice(0, WORST_SHOWN)], ['best', worstFirst.slice(-WORST_SHOWN).reverse()],
      ['random', order.map((i) => [i, rng(i + random * 1e6)()]).sort((a, b) => a[1] - b[1]).slice(0, WORST_SHOWN).map((x) => x[0])]];
    const cond = report.conditions.trimask[`${e.method}@${e.res}`];
    drill.replaceChildren(el('h2', {}, `${e.model} ${e.method} ${e.res}² step ${e.step}: ${METRICS[metric][0]}`),
      el('p', { class: 'note' }, 'worst / best: sorted by the metric over all items; random: a seeded draw. Each pair: trimask, generated.'),
      ...lists.flatMap(([name, idx]) => [el('h3', {}, name), el('div', { class: 'grid' }, idx.map((i) => el('div', { class: 'item' },
        el('div', { class: 'cells' }, imageCell(cond.images[i], 'trimask'), imageCell(e.images[i], e.model)),
        el('a', { href: `pixel.html#page_id=${report.items[i].page_id}&method=${e.method}&res=${e.res}&step=${e.step}` }, `${e.per_item[metric][i].toFixed(3)} · ${report.items[i].split}`))))]));
  };
  root.append(el('div', { class: 'controls' }, select('metric', Object.entries(METRICS).map(([k, v]) => [k, v[0]]), h.metric || 'band_iou', draw),
    select('step', steps.map((s) => [s, s]), h.step || steps.at(-1), draw)), el('p', { class: 'note' }, 'Held-out mean per cell. Green = good, red = bad.'), table, drill);
  draw();
}

/** Command. One item's filmstrip: per method and resolution, the image at every evaluated step. */
function strips(report, root) {
  const h = readHash();
  const body = el('div', {});
  const draw = () => {
    const s = readHash(), index = Math.max(report.items.findIndex((i) => i.page_id === s.page_id), 0), res = Number(s.res || 1024);
    body.replaceChildren(...report.methods.flatMap((method) => Object.keys(MODELS).map((m) => {
      const evals = pick(report.evals, { model: m, method, res }).sort((a, b) => a.step - b.step);
      return el('div', {}, el('h3', {}, `${MODELS[m]} · ${method} · ${res}²`), evals.length ? el('div', { class: 'strip' },
        imageCell(report.conditions.trimask[`${method}@${res}`].images[index], 'trimask'), evals.map((e) => imageCell(e.images[index], `${e.step}`))) : empty(m, report));
    })));
  };
  root.append(el('div', { class: 'controls' }, select('page_id', report.items.map((i) => [i.page_id, `${i.split} · ${i.caption.slice(0, 50)}`]), h.page_id || report.items[0].page_id, draw),
    select('res', report.resolutions.map((r) => [r, `${r}²`]), h.res || 1024, draw)), body);
  draw();
}

/** Command. Full-resolution images of one item side by side, zoomed together (wheel or slider; the pointer sets the zoom center). */
function pixel(report, root) {
  const h = readHash();
  const steps = unique(report.evals.map((e) => e.step));
  const zoom = el('input', { type: 'range', min: 1, max: 8, step: 0.5, value: 1 });
  const body = el('div', { class: 'pixel' });
  const apply = () => body.querySelectorAll('img').forEach((img) => { img.style.transform = `scale(${zoom.value})`; });
  const draw = () => {
    const s = readHash(), index = Math.max(report.items.findIndex((i) => i.page_id === s.page_id), 0);
    const method = s.method || report.methods[0], res = Number(s.res || 1024), step = Number(s.step || steps.at(-1));
    const figure = (src, caption) => el('figure', {}, el('div', { class: 'zoom' }, el('img', { src: url(src), loading: 'lazy', alt: caption })), el('figcaption', {}, caption));
    body.replaceChildren(figure(report.conditions.trimask[`${method}@${res}`].images[index], `trimask (${method})`),
      figure(report.conditions.matte[res].images[index], 'true matte'), figure(report.conditions.reference[res].images[index], 'MAGICK on gray (target)'),
      ...Object.keys(MODELS).map((m) => { const e = pick(report.evals, { model: m, method, res, step })[0]; return !e ? el('figure', {}, empty(m, report, step)) : e.images[index] ? figure(e.images[index], `${MODELS[m]} step ${step}`) : el('figure', {}, el('div', { class: 'empty' }, `${MODELS[m]} step ${step}: rendering`)); }));
    apply();
  };
  zoom.addEventListener('input', apply);
  body.addEventListener('pointermove', (e) => { const img = e.target.closest('img'); if (!img) return; const r = img.getBoundingClientRect();
    body.style.setProperty('--ox', `${((e.clientX - r.left) / r.width) * 100}%`); body.style.setProperty('--oy', `${((e.clientY - r.top) / r.height) * 100}%`); });
  root.append(el('div', { class: 'controls' }, select('page_id', report.items.map((i) => [i.page_id, `${i.split} · ${i.caption.slice(0, 50)}`]), h.page_id || report.items[0].page_id, draw),
    select('method', report.methods.map((m) => [m, m]), h.method || report.methods[0], draw), select('res', report.resolutions.map((r) => [r, `${r}²`]), h.res || 1024, draw),
    select('step', steps.map((x) => [x, x]), h.step || steps.at(-1), draw), el('label', {}, 'zoom ', zoom)), body);
  draw();
}

/** Command. The FIBO2 journal: recipe card and training loss (SANA's beside it, on its own axis: not comparable). */
async function fibo2(report, root) {
  const loss = async (run) => { const r = await fetch(`data/loss_${run}.json`); return r.ok ? r.json() : null; };
  const runs = Object.entries(report.runs);
  const charts = await Promise.all(runs.map(async ([run, spec]) => {
    const points = await loss(run);
    const yMax = points ? Math.max(...points.map((p) => p[1])) : 1;
    return el('div', { class: 'chart' }, el('h3', {}, `${spec.label} (${run}) training loss, window means`),
      points ? lineChart([{ cls: spec.family === 'sana' ? 'sana' : 'fibo', label: run, points: points.map((p) => [p[0], p[1], p[1], p[1]]) }], [], yMax) : empty(spec.family));
  }));
  const trained = runs.some(([, spec]) => spec.family === 'fibo2');
  const recipe = [['', 'SANA-Sprint (control_tri_scratch)', 'FIBO2'], ['parameters', '1.6B', '8.1B'],
    ['VAE latent', 'DC-AE: 32 ch at H/32', 'FLUX.2: 32 ch at H/8, 2×2 patches'], ['tokens at 1024²', '1,024', '4,096'],
    ['text encoder', 'Gemma-2-2B', 'Qwen3-VL-4B'], ['training', 'style adapter, then GateControl LoRA + gates', 'the same, through FIBO2\'s context latents'],
    ['data', '19 trimask methods, 1024 (50%) / 768 / 512 / 256', 'the same'], ['shipped model', 'stock Sprint + control, λ = 0, 2 steps', 'not decided (distillation pending)']];
  root.append(el('h2', {}, 'Recipe card'), el('div', { class: 'scroller' }, el('table', { class: 'heat' }, recipe.map((row, r) => el('tr', {}, row.map((c) => el(r ? 'td' : 'th', {}, c)))))));
  root.append(el('p', { class: 'verdict' }, trained ? 'FIBO2 training runs are listed below.' : 'FIBO2: not trained yet. Its latents are being encoded (manifest §14).'),
    el('div', { class: 'multiples' }, charts, trained ? null : el('div', { class: 'chart' }, el('h3', {}, 'FIBO2 training loss'), empty('fibo2'))),
    el('p', { class: 'note' }, 'Losses are not comparable across models: different VAE, latent size, sampler and loss weighting.'));
}

const DRAW = { index: hub, over_time: overTime, scrub, matrix, strips, pixel, fibo2 };

/** Command. Draws the page's view: navigation, fairness banner, then the view itself. */
async function main() {
  const view = document.body.dataset.view;
  const root = document.querySelector('main');
  root.prepend(el('nav', { class: 'views' }, el('a', { href: '../index.html' }, '← journal'),
    VIEWS.map(([v, label]) => el('a', { href: `${v}.html`, ...(v === view ? { 'aria-current': 'page' } : {}) }, label))),
  el('p', { class: 'banner' }, 'This compares training recipes, not architectures. Headlines use held-out items and every evaluated step (never the best one). Same eval sets, noise and image path for both models. No "better" until an interval excludes 0 at a matched training amount.'));
  const response = await fetch('data/compare.json');
  if (!response.ok) throw new Error(`data/compare.json: HTTP ${response.status}`);
  await DRAW[view](await response.json(), root);
}

main();
