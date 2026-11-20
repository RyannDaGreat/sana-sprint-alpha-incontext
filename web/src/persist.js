// UI persistence (manifest §20, m00848): the drawn matte, the prompt, the seed and the settings survive a reload, until
// "Reset all". One IndexedDB record; IndexedDB because the matte raster (a 1024² PNG) is too big for localStorage.

const DB_NAME = 'sana-ui', STORE = 'state', KEY = 'v1';

/**
 * Query. Opens (creating on first use) the UI state database.
 *
 * @returns {Promise<IDBDatabase>}
 */
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error(`IndexedDB ${DB_NAME}: blocked by another open tab of this app; close it and reload`));
  });
}

/**
 * Command (IndexedDB). Runs one request on the state store and resolves with its result.
 *
 * @param {'readonly'|'readwrite'} mode
 * @param {(store: IDBObjectStore) => IDBRequest} make
 * @returns {Promise<any>}
 */
async function request(mode, make) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = make(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Command. Saves the UI state.
 *
 * @param {{raster: Blob, objects: object[], invert: boolean, inputs: Record<string, string|boolean>, pad?: object[]}} state
 * @returns {Promise<void>}
 */
export const saveUiState = (state) => request('readwrite', (s) => s.put(state, KEY));

/**
 * Query. The saved UI state, or undefined when nothing is saved.
 *
 * @returns {Promise<object|undefined>}
 */
export const loadUiState = () => request('readonly', (s) => s.get(KEY));

/**
 * Command. Deletes the saved UI state ("Reset all").
 *
 * @returns {Promise<void>}
 */
export const clearUiState = () => request('readwrite', (s) => s.delete(KEY));

/**
 * Query. The persisted values of the given inputs: `checked` for checkboxes, `value` otherwise (ids not on this page are skipped).
 *
 * @param {(id: string) => HTMLInputElement} $
 * @param {string[]} ids
 * @returns {Record<string, string|boolean>}
 * @example readInputs($, ['seed', 'realtime']) // {seed: '0', realtime: false}
 */
export function readInputs($, ids) {
  return Object.fromEntries(ids.filter((id) => $(id)).map((id) => [id, $(id).type === 'checkbox' ? $(id).checked : $(id).value]));
}

/**
 * Command. Writes saved values back into the inputs (ids that no longer exist, or values a select no longer offers, are skipped).
 *
 * @param {(id: string) => HTMLInputElement} $
 * @param {Record<string, string|boolean>} values
 */
export function writeInputs($, values) {
  for (const [id, v] of Object.entries(values)) {
    const el = $(id);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = Boolean(v);
    else if (el.tagName !== 'SELECT' || [...el.options].some((o) => o.value === v)) el.value = v;
  }
}
