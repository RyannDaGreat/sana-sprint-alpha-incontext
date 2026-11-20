// Text-tool fonts: the bundled/system ones plus a curated Google Fonts list, loaded on demand (Jost stays the UI font).

const GOOGLE_CSS = 'https://fonts.googleapis.com/css2';
const PROBE_SIZE = 100; // px; any size works for document.fonts.load

// weights: the static weights to request (omit for single-weight fonts; the weight menu then falls back to synthesis).
export const FONTS = [
  { group: 'Built in', name: 'Jost', family: "'Jost Variable'" },
  { group: 'Built in', name: 'System serif', family: 'Georgia, serif' },
  { group: 'Built in', name: 'System mono', family: "'Courier New', monospace" },
  { group: 'Built in', name: 'Impact', family: "Impact, 'Arial Black', sans-serif" },
  { group: 'Sans', name: 'Inter', weights: [300, 500, 700, 900] },
  { group: 'Sans', name: 'Montserrat', weights: [300, 500, 700, 900] },
  { group: 'Sans', name: 'Oswald', weights: [300, 500, 700] },
  { group: 'Sans', name: 'Bebas Neue' },
  { group: 'Sans', name: 'Anton' },
  { group: 'Sans', name: 'Archivo Black' },
  { group: 'Serif', name: 'Playfair Display', weights: [400, 700, 900] },
  { group: 'Serif', name: 'EB Garamond', weights: [400, 500, 700, 800] },
  { group: 'Serif', name: 'Cinzel', weights: [400, 700, 900] },
  { group: 'Serif', name: 'DM Serif Display' },
  { group: 'Serif', name: 'Abril Fatface' },
  { group: 'Serif', name: 'Ultra' },
  { group: 'Script', name: 'Pacifico' },
  { group: 'Script', name: 'Lobster' },
  { group: 'Script', name: 'Dancing Script', weights: [400, 700] },
  { group: 'Script', name: 'Great Vibes' },
  { group: 'Script', name: 'Permanent Marker' },
  { group: 'Blackletter', name: 'UnifrakturMaguntia' },
  { group: 'Blackletter', name: 'Pirata One' },
  { group: 'Blackletter', name: 'Grenze Gotisch', weights: [300, 500, 700, 900] },
  { group: 'Monospace', name: 'JetBrains Mono', weights: [300, 500, 700, 800] },
  { group: 'Monospace', name: 'Space Mono', weights: [400, 700] },
  { group: 'Display', name: 'Bungee' },
  { group: 'Display', name: 'Rubik Mono One' },
  { group: 'Display', name: 'Black Ops One' },
  { group: 'Display', name: 'Alfa Slab One' },
  { group: 'Display', name: 'Luckiest Guy' },
  { group: 'Display', name: 'Monoton' },
];

/**
 * Pure function. CSS font-family of a FONTS entry (Google fonts are quoted by name).
 *
 * @param {{name: string, family?: string}} f
 * @returns {string}
 *
 * @example fontFamily({name: 'Bebas Neue'}) // "'Bebas Neue'"
 * @example fontFamily({name: 'Jost', family: "'Jost Variable'"}) // "'Jost Variable'"
 */
export function fontFamily(f) {
  return f.family ?? `'${f.name}'`;
}

/**
 * Pure function. Google Fonts CSS API URL of a FONTS entry.
 *
 * @param {{name: string, weights?: number[]}} f
 * @returns {string}
 *
 * @example googleCssUrl({name: 'Bebas Neue'}) // 'https://fonts.googleapis.com/css2?family=Bebas+Neue&display=block'
 * @example googleCssUrl({name: 'Space Mono', weights: [400, 700]}) // 'https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=block'
 */
export function googleCssUrl(f) {
  const axis = f.weights ? `:wght@${f.weights.join(';')}` : '';
  return `${GOOGLE_CSS}?family=${f.name.replace(/ /g, '+')}${axis}&display=block`;
}

/**
 * Command. Fills a <select> with FONTS in <optgroup>s; each option's value is the CSS family.
 *
 * @param {HTMLSelectElement} select
 */
export function fillFontSelect(select) {
  select.innerHTML = '';
  for (const group of [...new Set(FONTS.map((f) => f.group))]) {
    const og = document.createElement('optgroup');
    og.label = group;
    for (const f of FONTS.filter((x) => x.group === group)) og.append(new Option(f.name, fontFamily(f)));
    select.append(og);
  }
  enhanceFontSelect(select);
}

const PREVIEW_SUFFIX = ' preview'; // preview faces get their own family name: they hold only a few glyphs and must never shadow the full font
const PREVIEW_ROOT_MARGIN = '100px';
const TYPEAHEAD_MS = 700;
const previews = new Map(); // font name -> Promise<css family>

/**
 * Pure function. Cheap preview request for a font: default weight, only the glyphs of the font's own name.
 *
 * @param {{name: string}} f
 * @returns {string}
 *
 * @example previewCssUrl({name: 'Bebas Neue'}) // 'https://fonts.googleapis.com/css2?family=Bebas+Neue&display=block&text=Bebas%20Neue'
 */
export const previewCssUrl = (f) => `${googleCssUrl({ name: f.name })}&text=${encodeURIComponent(f.name)}`;

/**
 * Command. Loads a font's preview face once (appends a <style> to <head>); built-in fonts need nothing.
 * Rejects when the CSS request fails.
 *
 * @param {{name: string, family?: string}} f
 * @returns {Promise<string>} CSS font-family to preview with
 */
function ensurePreview(f) {
  if (f.family) return Promise.resolve(f.family);
  const family = `'${f.name}${PREVIEW_SUFFIX}'`;
  if (!previews.has(f.name)) {
    previews.set(f.name, fetch(previewCssUrl(f))
      .then((r) => { if (!r.ok) throw new Error(`Preview of ${f.name}: HTTP ${r.status}`); return r.text(); })
      .then((css) => {
        document.head.append(Object.assign(document.createElement('style'), { textContent: css.replaceAll(`'${f.name}'`, family) }));
        return document.fonts.load(`16px ${family}`, f.name);
      })
      .then(() => family));
  }
  return previews.get(f.name);
}

/**
 * Command. Hides `select` (it stays the source of truth: .value reads and writes, 'change' events) and puts a
 * combobox button plus a listbox popup beside it. Each option is drawn in its own font, loaded when it scrolls
 * into view. Choosing sets select.value and dispatches a bubbling 'change'. Keys: Up/Down/Home/End/Enter/Space/Esc
 * and type-ahead. Setting select.value from code refreshes the button.
 *
 * @param {HTMLSelectElement} select - filled by fillFontSelect
 */
export function enhanceFontSelect(select) {
  const el = (tag, cls, attrs = {}) => Object.assign(document.createElement(tag), { className: cls }, attrs);
  const root = el('div', 'fontpick'), button = el('button', 'fontpick-button', { type: 'button' });
  const list = el('ul', 'fontpick-list', { role: 'listbox', id: `fontpick-${select.id}`, hidden: true });
  button.setAttribute('role', 'combobox');
  button.setAttribute('aria-haspopup', 'listbox');
  button.setAttribute('aria-controls', list.id);
  const opts = [];
  for (const group of [...new Set(FONTS.map((f) => f.group))]) {
    list.append(el('li', 'fontpick-group', { textContent: group, role: 'presentation' }));
    for (const f of FONTS.filter((x) => x.group === group)) {
      const li = el('li', 'fontpick-option', { textContent: f.name, role: 'option', id: `${list.id}-${opts.length}` });
      li.font = f;
      opts.push(li);
      list.append(li);
    }
  }
  const seen = new IntersectionObserver((entries) => entries.filter((e) => e.isIntersecting).forEach(({ target }) => {
    seen.unobserve(target);
    ensurePreview(target.font).then((family) => { target.style.fontFamily = family; }).catch((e) => {
      target.dataset.failed = '';
      target.title = e.message;
      throw e; // marked in the list, and reported in the console
    });
  }), { root: list, rootMargin: PREVIEW_ROOT_MARGIN });
  const current = () => opts.findIndex((o) => fontFamily(o.font) === select.value);
  const activeIndex = () => opts.findIndex((o) => o.classList.contains('active'));
  const sync = () => {
    const selected = opts[current()];
    button.textContent = selected.font.name;
    ensurePreview(selected.font).then((family) => { button.style.fontFamily = family; });
    opts.forEach((o) => o.setAttribute('aria-selected', o === selected));
  };
  const activate = (i) => {
    i = Math.max(0, Math.min(opts.length - 1, i));
    opts.forEach((o, k) => o.classList.toggle('active', k === i));
    button.setAttribute('aria-activedescendant', opts[i].id);
    opts[i].scrollIntoView({ block: 'nearest' });
  };
  const setOpen = (open) => {
    list.hidden = !open;
    button.setAttribute('aria-expanded', open);
    if (open) { opts.forEach((o) => seen.observe(o)); activate(Math.max(0, current())); }
  };
  const choose = (i) => {
    select.value = fontFamily(opts[i].font);
    select.dispatchEvent(new Event('change', { bubbles: true }));
    setOpen(false);
  };
  let typed = '', typedAt = 0;
  button.addEventListener('click', () => setOpen(list.hidden));
  button.addEventListener('blur', () => setOpen(false));
  list.addEventListener('mousedown', (e) => { // keep focus on the button
    e.preventDefault();
    const o = e.target.closest('.fontpick-option');
    if (o) choose(opts.indexOf(o));
  });
  button.addEventListener('keydown', (e) => {
    const step = { ArrowDown: 1, ArrowUp: -1 }[e.key], jump = { Home: 0, End: opts.length - 1 }[e.key];
    if (step || jump !== undefined) { e.preventDefault(); list.hidden ? setOpen(true) : activate(jump ?? activeIndex() + step); }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); list.hidden ? setOpen(true) : choose(activeIndex()); }
    else if (e.key === 'Escape') setOpen(false);
    else if (e.key.length === 1) { // type-ahead: a prefix typed within TYPEAHEAD_MS
      typed = (e.timeStamp - typedAt > TYPEAHEAD_MS ? '' : typed) + e.key.toLowerCase();
      typedAt = e.timeStamp;
      const hit = opts.findIndex((o) => o.font.name.toLowerCase().startsWith(typed));
      if (hit >= 0) { setOpen(true); activate(hit); }
    }
  });
  const valueProp = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
  Object.defineProperty(select, 'value', { get() { return valueProp.get.call(this); }, set(v) { valueProp.set.call(this, v); sync(); } });
  select.hidden = true;
  root.append(button, list);
  select.before(root);
  sync();
}

const loaded = new Map(); // family -> Promise

/**
 * Command. Makes a font family usable on canvas: adds its Google stylesheet once (built-in fonts need nothing)
 * and waits until the given weight is loaded. Rejects when the stylesheet fails.
 *
 * @param {string} family - a value from fillFontSelect
 * @param {number} weight
 * @returns {Promise<void>}
 */
export function ensureFont(family, weight) {
  const f = FONTS.find((x) => fontFamily(x) === family);
  if (!f) throw new Error(`Unknown font ${family}`);
  if (!f.family && !loaded.has(family)) {
    loaded.set(family, new Promise((resolve, reject) => {
      // CORS mode: the local server sends COEP require-corp, which blocks no-cors cross-origin stylesheets.
      const link = Object.assign(document.createElement('link'), { rel: 'stylesheet', crossOrigin: 'anonymous', href: googleCssUrl(f) });
      link.onload = resolve;
      link.onerror = () => reject(new Error(`Could not load font ${f.name} from Google Fonts`));
      document.head.append(link);
    }));
  }
  return Promise.resolve(loaded.get(family)).then(() => document.fonts.load(`${weight} ${PROBE_SIZE}px ${family}`)).then(() => {});
}
