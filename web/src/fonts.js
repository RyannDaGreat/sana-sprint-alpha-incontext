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
