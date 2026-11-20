// Gemma prompt-embedding cache in IndexedDB, keyed by text encoder + prompt text.

const DB_NAME = 'sana-web';
const STORE = 'embeddings';

/**
 * Command. Opens (and on first use creates) the IndexedDB database.
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
 * Command. Runs one request in a transaction and resolves with its result.
 *
 * @param {'readonly'|'readwrite'} mode
 * @param {(store: IDBObjectStore) => IDBRequest} op
 * @returns {Promise<any>}
 */
async function run(mode, op) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = op(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Pure function. The cache key: the text encoder's identity plus the exact prompt.
 *
 * @param {string} encoderId - e.g. the text encoder URL + byte size
 * @param {string} prompt
 * @returns {string}
 *
 * @example embeddingKey('https://h/v1/text_encoder.onnx:123', 'a red fox') // 'https://h/v1/text_encoder.onnx:123\na red fox'
 */
export function embeddingKey(encoderId, prompt) {
  return `${encoderId}\n${prompt}`;
}

/**
 * Query. A cached embedding, or undefined.
 *
 * @param {string} key
 * @returns {Promise<{type: string, data: ArrayBufferView, dims: number[], mask: BigInt64Array} | undefined>}
 */
export function getEmbedding(key) {
  return run('readonly', (s) => s.get(key));
}

/** Command. Stores an embedding under `key`. */
export function putEmbedding(key, value) {
  return run('readwrite', (s) => s.put(value, key));
}

/** Query. Number of cached prompt embeddings. */
export function countEmbeddings() {
  return run('readonly', (s) => s.count());
}

/** Command. Deletes all cached prompt embeddings. */
export function clearEmbeddings() {
  return run('readwrite', (s) => s.clear());
}
