// The mask gallery: a full-screen <dialog> of mask thumbnails by category (masks/categories.json +
// masks/<id>/index.json, written by scripts/mask_index.mjs and the mask collectors). Clicking one hands its URL to onPick.

const MASKS = './masks/';
const ALL = 'all';

/**
 * Pure function. Short credit line of a mask item.
 *
 * @param {{name: string, author?: string, license?: string}} item
 * @returns {string}
 *
 * @example credit({name: 'fox', author: 'Jane', license: 'CC0'}) // 'fox · Jane · CC0'
 * @example credit({name: 'fox'}) // 'fox'
 */
export function credit(item) {
  return [item.name, item.author, item.license].filter(Boolean).join(' · ');
}

/** Query. Fetches JSON, throwing on a non-OK response. */
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

export class MaskGallery {
  /**
   * Command. Binds the gallery to its dialog (with #gallery-filter, #gallery-grid, #gallery-close inside).
   *
   * @param {HTMLDialogElement} dialog
   * @param {(url: string, item: object) => Promise<void>} onPick - loads a mask; the dialog closes after it resolves
   */
  constructor(dialog, onPick) {
    this.dialog = dialog; this.onPick = onPick;
    this.categories = null; this.filter = ALL; this.query = '';
    this.search = dialog.querySelector('#gallery-search');
    this.search.addEventListener('input', () => { this.query = this.search.value.trim().toLowerCase(); if (this.categories) this.render(); });
    dialog.querySelector('#gallery-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); }); // backdrop click
  }

  /** Command. Opens the dialog (search field focused); the first time, loads every category index. */
  async open() {
    this.dialog.showModal();
    this.search.focus();
    if (this.categories) return;
    const { categories } = await getJson(`${MASKS}categories.json`);
    this.categories = await Promise.all(categories.map(async (c) => ({ ...c, items: (await getJson(`${MASKS}${c.id}/index.json`)).items })));
    this.render();
  }

  /** Command. Draws the category filter and the grid for the current filter. */
  render() {
    const filter = this.dialog.querySelector('#gallery-filter'), grid = this.dialog.querySelector('#gallery-grid');
    const total = this.categories.reduce((a, c) => a + c.items.length, 0);
    filter.innerHTML = '';
    for (const c of [{ id: ALL, name: 'All', count: total }, ...this.categories]) {
      const b = document.createElement('button');
      b.className = `tab${c.id === this.filter ? ' on' : ''}`;
      b.textContent = `${c.name} ${c.count}`;
      b.addEventListener('click', () => { this.filter = c.id; this.render(); });
      filter.append(b);
    }
    grid.innerHTML = '';
    for (const c of this.categories.filter((x) => this.filter === ALL || x.id === this.filter)) {
      const items = c.items.filter((item) => item.name.toLowerCase().includes(this.query));
      if (!items.length) continue;
      const h = document.createElement('h3');
      h.textContent = c.name;
      grid.append(h);
      for (const item of items) grid.append(this.tile(c, item));
    }
    if (!grid.children.length) grid.innerHTML = '<p class="meta">No mask matches.</p>';
  }

  /** Command. Builds one thumbnail tile (pure DOM construction, no I/O until the lazy image loads). */
  tile(category, item) {
    const url = `${MASKS}${category.id}/${item.file}`;
    const fig = document.createElement('figure');
    fig.className = 'mask-tile';
    fig.title = credit(item);
    fig.innerHTML = '<img loading="lazy" alt=""><div class="mask-info"><span class="mask-credit"></span><a target="_blank" rel="noopener">source</a></div><figcaption class="mask-name"></figcaption>';
    Object.assign(fig.querySelector('img'), { src: url, alt: item.name });
    fig.querySelector('.mask-name').textContent = item.name;
    fig.querySelector('.mask-credit').textContent = [item.author, item.license].filter(Boolean).join(' · ');
    const a = fig.querySelector('a');
    if (item.source_url) a.href = item.source_url; else a.remove();
    a.addEventListener('click', (e) => e.stopPropagation());
    fig.addEventListener('click', async () => { await this.onPick(url, item); this.dialog.close(); });
    return fig;
  }
}
