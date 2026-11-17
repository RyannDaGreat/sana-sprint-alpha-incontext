// Model versions: index, manifests, cached downloads (OPFS) with progress.
//
// SEMANTIC BINDING: web-model-contract
// models/web/index.json      = {"versions": ["<version>", ...]}
// models/web/<v>/manifest.json = {
//   "label": str, "description": str,
//   "scaling_factor": 0.41407, "timestep": float, "guidance": float, "max_sequence_length": 300,
//   "weights": str, "precision": str,                (weight equation and precision label, shown under the output)
//   "adapter": {source, rank, ...} | null,           (runtime λ: the transformer takes adapter_scale; weights contain "λ")
//   "dynamic_size": bool,                            (graphs take any H × W (multiples of 32); false: 1024² only)
//   "complex_human_instruction": [str, ...]          (optional; default = SANA's, see prompt.js)
//   "files": {                                        (paths are relative to the version folder)
//     "tokenizer": {path, bytes}, "tokenizer_config": {path, bytes},
//     "text_encoder" | "vae_encoder" | "transformer" | "vae_decoder": {
//        path, bytes, sha256, dtype,                  (sha256 optional, cache key; dtype informational: fp32 / fp16 / int8 / int4)
//        "parts": [{url, bytes}],                     (optional; shards concatenated in order, URLs relative to the folder)
//        "external_data": [{path, bytes, sha256, parts}],  (optional; ONNX external weights)
//        "names": {role: onnx_name}                   (optional; overrides the default names in pipeline.js)
//     }}}
//   "latent_scaling_in_graph": bool,                 (optional, default false: JS does × / ÷ scaling_factor)
//   "decoder_output_range": "-1,1" | "0,1",          (optional, default "-1,1")
// The exporter's own manifest format (flat "files": {name: bytes} + "graphs", optional flat
// "sha256": {name: hex} and "parts": {name: [{url, bytes}]}) is accepted too and converted by normalizeManifest.
// Input dtypes (float32 / float16 / int64 / int32) are read from each session's metadata.
// The transformer gets N(0, 1) noise and the scaled matte latent and returns the denoised,
// scaled latent: all Sprint one-step math (sigma_data, TrigFlow) lives inside the graph; pipeline.js loops it
// for 1–4 steps (sprintTimesteps) and adds fresh noise between steps.

export const CACHE_NAME = 'sana-web-models';
const PART_SUFFIX = '.part';  // OPFS name of a download in progress (renamed when complete)
const CACHE_READ_CHUNK_BYTES = 256 * 1024 * 1024;  // Reads cached files in 256 MB slices: one read of a >2 GB Blob fails in Chrome
export const MODEL_FILES = ['tokenizer', 'tokenizer_config', 'text_encoder', 'vae_encoder', 'transformer', 'vae_decoder'];

/**
 * Query. Fetches JSON and fails loudly on a non-200 status.
 *
 * @param {string} url
 * @returns {Promise<any>}
 */
export async function fetchJson(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`GET ${url}: HTTP ${res.status}`);
  return res.json();
}

/**
 * Query. Loads models/web/index.json and every version's manifest.
 *
 * @param {string} base - URL of the models/web/ folder, ending in '/'
 * @returns {Promise<{id: string, url: string, manifest: object}[]>}
 */
export async function loadVersions(base) {
  const index = await fetchJson(new URL('index.json', base).href);
  return Promise.all(index.versions.map(async (id) => {
    const url = new URL(`${id}/`, base).href;
    return { id, url, manifest: normalizeManifest(await fetchJson(new URL('manifest.json', url).href)) };
  }));
}

// Exporter graph input/output names → this app's role keys (see DEFAULT_NAMES in pipeline.js).
const EXPORT_NAMES = {
  transformer: { noise: 'latents', matte_latent: 'cond_latents', encoder_hidden_states: 'prompt_embeds',
    encoder_attention_mask: 'prompt_mask', timestep: 'timestep', guidance: 'guidance', adapter_scale: 'adapter_scale',
    output: 'denoised' },
  vae_encoder: { image: 'image', output: 'latent' },
  vae_decoder: { latent: 'latent', output: 'image' },
  text_encoder: { input_ids: 'input_ids', attention_mask: 'attention_mask', output: 'last_hidden_state' },
};

/**
 * Pure function. Converts the exporter's manifest (flat "files": {name: bytes}) to this app's
 * schema, and lists the roles that have no file yet in `missing`. App-schema manifests pass through.
 *
 * @param {object} raw - Parsed manifest.json
 * @returns {object} App-schema manifest with `missing: string[]`
 *
 * @example
 * normalizeManifest({version: 'v', files: {'vae_encoder.onnx': 5, 'vae_encoder.onnx.data': 50},
 *   sampler: {timestep: 1.5708, guidance: 4.5}, vae: {scaling_factor: 0.41407}})
 * // {label: 'v', ..., latent_scaling_in_graph: true, decoder_output_range: '0,1',
 * //  files: {vae_encoder: {path: 'vae_encoder.onnx', bytes: 5, external_data: [{path: 'vae_encoder.onnx.data', bytes: 50}], names: {...}}},
 * //  missing: ['tokenizer', 'tokenizer_config', 'text_encoder', 'transformer', 'vae_decoder']}
 */
export function normalizeManifest(raw) {
  const exported = Object.values(raw.files).every((v) => typeof v === 'number');
  if (!exported) return { ...raw, missing: MODEL_FILES.filter((r) => !raw.files[r]) };
  const files = {};
  for (const [name, bytes] of Object.entries(raw.files)) {
    const role = name.replace(/\.onnx(\.data)?$|\.json$/, '');
    if (name.endsWith('.data')) continue;
    const entry = (n, b) => ({ path: n, bytes: b, sha256: raw.sha256?.[n], parts: raw.parts?.[n] });
    files[role] = { ...entry(name, bytes), names: EXPORT_NAMES[role] };
    // external weights: <name>.data, plus extra files <name>.<tag>.data (the -lambda delta: transformer.onnx.lambda.data)
    const data = Object.keys(raw.files).filter((n) => n.startsWith(`${name}.`) && n.endsWith('.data')).sort();
    if (data.length) files[role].external_data = data.map((n) => entry(n, raw.files[n]));
  }
  return {
    label: raw.version, description: raw.source ? `Exported from ${raw.source}.` : '',
    scaling_factor: raw.vae.scaling_factor, timestep: raw.sampler.timestep, guidance: raw.sampler.guidance,
    weights: raw.weights, precision: precisionLabel(raw.quantization), adapter: raw.adapter ?? null,
    dynamic_size: typeof raw.graphs.transformer.inputs[0][1][2] === 'string',  // [1, 32, "LH", "LW"]: any size
    max_sequence_length: 300, latent_scaling_in_graph: true, decoder_output_range: '0,1',
    files, missing: MODEL_FILES.filter((r) => !files[r]),
  };
}

/**
 * Pure function. Short precision label of an exporter manifest's `quantization` (transformer and text encoder).
 *
 * @param {object} q - manifest.quantization: {graph: {bits, lowrank_rank, ...}} when quantized, {graph: {fp16: file}} when not
 * @returns {string}
 *
 * @example precisionLabel({transformer: {bits: 4, lowrank_rank: 32}, text_encoder: {bits: 8, lowrank_rank: 0}}) // 'transformer int4 SVDQuant r32 · Gemma int8'
 * @example precisionLabel({transformer: {fp16: 'transformer.onnx'}, text_encoder: {fp16: 'text_encoder.onnx'}}) // 'transformer fp16 · Gemma fp16'
 */
export function precisionLabel(q) {
  const one = (g) => (g?.bits ? `int${g.bits}${g.lowrank_rank ? ` SVDQuant r${g.lowrank_rank}` : ''}` : 'fp16');
  return `transformer ${one(q.transformer)} · Gemma ${one(q.text_encoder)}`;
}

/**
 * Pure function. Every file a version needs, as absolute URLs with sizes. The cache key is the file's
 * sha256 (shared by versions that share a file), or without one its URL + size, so a re-exported file
 * is downloaded again.
 *
 * @param {{url: string, manifest: object}} version
 * @returns {{role: string, name: string, url: string, bytes: number}[]}
 *
 * @example
 * versionFiles({url: 'https://h/models/web/v1/', manifest: {files: {transformer: {path: 'transformer.onnx', bytes: 9,
 *   external_data: [{path: 'transformer.onnx.data', bytes: 99}]}}}})
 * // [{role: 'transformer', name: 'transformer.onnx', url: 'https://h/models/web/v1/transformer.onnx', bytes: 9,
 * //   key: 'https://h/models/web/v1/transformer.onnx?v=9'},
 * //  {role: 'transformer', name: 'transformer.onnx.data', url: '…/transformer.onnx.data', bytes: 99, key: '…/transformer.onnx.data?v=99'}]
 */
export function versionFiles(version) {
  const out = [];
  for (const [role, f] of Object.entries(version.manifest.files)) {
    for (const g of [f, ...(f.external_data ?? [])]) {
      const url = new URL(g.path, version.url).href;
      const parts = g.parts?.map((p) => ({ url: new URL(p.url, version.url).href, bytes: p.bytes }));
      out.push({ role, name: g.path.split('/').pop(), url, bytes: g.bytes, key: g.sha256 ? `sha256:${g.sha256}` : `${url}?v=${g.bytes}`, parts });
    }
  }
  return out;
}

/**
 * Query. The OPFS folder that holds the model cache (created on first use).
 * OPFS, not Cache Storage: Chrome's Cache.put fails on multi-GB responses (seen with a 3.2 GB file).
 *
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
async function cacheDir() {
  return (await navigator.storage.getDirectory()).getDirectoryHandle(CACHE_NAME, { create: true });
}

/**
 * Query. OPFS file name for a cache key: hex SHA-256 (keys are URLs, too long for file names).
 *
 * @param {string} key
 * @returns {Promise<string>} 64 hex chars
 */
async function cacheName(key) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(key));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Query. The cached file for `key`, or null when it is not cached.
 *
 * @returns {Promise<File|null>}
 */
async function cachedFile(key) {
  const name = await cacheName(key);
  for await (const entry of (await cacheDir()).values()) if (entry.name === name) return entry.getFile();
  return null;
}

/**
 * Query. Which of these files are already in the cache.
 *
 * @param {{key: string}[]} files
 * @returns {Promise<boolean[]>}
 */
export async function cachedFlags(files) {
  return Promise.all(files.map(async (f) => (await cachedFile(f.key)) !== null));
}

const keyQueue = new Map(); // cache key → promise of the last fetchCached on it

/**
 * Command. Returns a file's bytes from the OPFS cache, or downloads it (streamed, with progress) while writing it
 * to OPFS. Calls on the same key run one after another (versions share Gemma and DC-AE files: a read must never
 * overlap the write of the same file). Fails loudly if the size differs from the manifest.
 *
 * @param {{url: string, key: string, bytes: number, parts?: {url: string, bytes: number}[]}} file - parts: shards to concatenate
 * @param {(done: number, total: number, fromCache: boolean) => void} onProgress
 * @returns {Promise<Uint8Array>}
 */
export function fetchCached(file, onProgress) {
  // the previous call's error is reported to that call's own caller; here it only orders the queue
  const run = (keyQueue.get(file.key) ?? Promise.resolve()).catch(() => {}).then(() => fetchCachedNow(file, onProgress));
  keyQueue.set(file.key, run);
  return run;
}

/**
 * Command. fetchCached without the per-key queue. A download is written to `<name>.part` and renamed to `<name>`
 * only when complete, so an interrupted download (closed tab, network error) never looks cached.
 */
async function fetchCachedNow(file, onProgress) {
  let hit = await cachedFile(file.key);
  if (hit && hit.size !== file.bytes) {  // builds before 2026-11-17 left a 0-byte file when a download was interrupted
    console.warn(`cached ${file.url}: ${hit.size} bytes, manifest says ${file.bytes}; deleting and downloading again`);
    await (await cacheDir()).removeEntry(await cacheName(file.key));
    hit = null;
  }
  if (hit) {
    const data = new Uint8Array(file.bytes);
    for (let at = 0; at < file.bytes; at += CACHE_READ_CHUNK_BYTES) {  // Chrome throws NotReadableError reading a >2 GB Blob at once
      data.set(new Uint8Array(await hit.slice(at, at + CACHE_READ_CHUNK_BYTES).arrayBuffer()), at);
      onProgress(Math.min(at + CACHE_READ_CHUNK_BYTES, file.bytes), file.bytes, true);
    }
    return data;
  }
  const name = await cacheName(file.key);
  const handle = await (await cacheDir()).getFileHandle(`${name}${PART_SUFFIX}`, { create: true });
  const writer = await handle.createWritable();
  const data = new Uint8Array(file.bytes);
  let done = 0;
  for (const part of file.parts ?? [{ url: file.url }]) {  // shards (≤ 95 MB, GitHub's file limit) are concatenated
    const res = await fetch(part.url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`GET ${part.url}: HTTP ${res.status}`);
    const reader = res.body.getReader();
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      if (done + value.length > file.bytes) throw new Error(`${file.url}: more than the manifest's ${file.bytes} bytes`);
      data.set(value, done);
      await writer.write(value);
      done += value.length;
      onProgress(done, file.bytes, false);
    }
  }
  if (done !== file.bytes) { await writer.abort(); throw new Error(`${file.url}: got ${done} bytes, manifest says ${file.bytes}`); }
  await writer.close();
  await handle.move(name);
  return data;
}

/**
 * Query. Total bytes in this app's model cache.
 *
 * @returns {Promise<number>}
 */
export async function cacheBytes() {
  let total = 0;
  for await (const entry of (await cacheDir()).values()) total += (await entry.getFile()).size;
  return total;
}

/** Command. Deletes the whole model cache. */
export async function clearCache() {
  await (await navigator.storage.getDirectory()).removeEntry(CACHE_NAME, { recursive: true });
}
