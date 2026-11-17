// Writes public/masks/categories.json from the category folders that have an index.json (run by `npm run build`).
// Each public/masks/<id>/index.json is {"name"?: str, "items": [{file, name, source_url, license, author}]}.
import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';

const MASKS = new URL('../public/masks/', import.meta.url);
const FIRST = ['magick']; // shown first in the gallery; the rest are alphabetical

/**
 * Pure function. Display name of a category folder.
 *
 * @param {string} id - folder name
 * @returns {string}
 *
 * @example categoryName('people_poses') // 'People poses'
 */
function categoryName(id) {
  const words = id.replace(/_/g, ' ');
  return words[0].toUpperCase() + words.slice(1);
}

const ids = readdirSync(MASKS, { withFileTypes: true })
  .filter((d) => d.isDirectory() && existsSync(new URL(`${d.name}/index.json`, MASKS)))
  .map((d) => d.name)
  .sort((a, b) => (FIRST.includes(b) - FIRST.includes(a)) || a.localeCompare(b));
const categories = ids.map((id) => {
  const index = JSON.parse(readFileSync(new URL(`${id}/index.json`, MASKS)));
  return { id, name: index.name ?? categoryName(id), count: index.items.length };
}).filter((c) => c.count > 0);
writeFileSync(new URL('categories.json', MASKS), JSON.stringify({ categories }, null, 1));
console.log(`masks: ${categories.map((c) => `${c.id} ${c.count}`).join(', ')}`);
