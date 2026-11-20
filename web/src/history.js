// Gallery of past generations (the Gallery output tab; web/src/gallery.js is the unrelated mask gallery) (m00876, m00878: KISS): every finished generation's outputs as lossless PNG Blobs plus its
// full UI state (matte, prompt, seed, settings), in its own IndexedDB database: never the model cache, never evicted by
// model downloads. Writes are asynchronous and happen after a generation, never on the compute path.

const DB_NAME = 'sana-gallery', STORE = 'items';

/**
 * Query. Opens (creating on first use) the gallery database.
 *
 * @returns {Promise<IDBDatabase>}
 */
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error(`IndexedDB ${DB_NAME}: blocked by another open tab of this app; close it and reload`));
  });
}

/**
 * Command (IndexedDB). Runs one request on the item store and resolves with its result.
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
 * Command. Stores one generation.
 *
 * @param {{ts: number, version: string, info: object, state: object, outputs: {name: string, blob: Blob}[], timeline: {label: string, blob: Blob}[], thumb: Blob}} item
 * @returns {Promise<number>} its id
 */
export const addItem = (item) => request('readwrite', (s) => s.add(item));

/**
 * Query. Every stored generation, newest first.
 *
 * @returns {Promise<object[]>}
 */
export const listItems = async () => (await request('readonly', (s) => s.getAll())).reverse();

/**
 * Command. Deletes one generation.
 *
 * @param {number} id
 * @returns {Promise<void>}
 */
export const deleteItem = (id) => request('readwrite', (s) => s.delete(id));

/**
 * Command. Deletes every generation ("Clear gallery").
 *
 * @returns {Promise<void>}
 */
export const clearItems = () => request('readwrite', (s) => s.clear());

/**
 * Pure function. Bytes an item takes (its PNGs, its timeline frames and the matte raster). Items saved before m00906 have no timeline.
 *
 * @param {{outputs: {blob: Blob}[], timeline?: {blob: Blob}[], thumb: Blob, state: {raster: Blob}}} item
 * @returns {number}
 * @example itemBytes({outputs: [{blob: new Blob(['ab'])}], thumb: new Blob(['c']), state: {raster: new Blob(['d'])}}) // 4
 * @example itemBytes({outputs: [{blob: new Blob(['ab'])}], timeline: [{blob: new Blob(['xyz'])}], thumb: new Blob(['c']), state: {raster: new Blob(['d'])}}) // 7
 */
export const itemBytes = (item) => [...item.outputs, ...(item.timeline ?? [])].reduce((n, o) => n + o.blob.size, 0) + item.thumb.size + item.state.raster.size;
