// The matte / trimask → image pipeline on ONNX Runtime Web (WebGPU EP, or WASM), for both model families:
// SANA-Sprint (manifest §12: Sprint's few-step sampler inside the transformer graph) and FIBO2 (§16: one velocity pass per
// graph call; the flow Euler sampler with CFG runs here). Stages: tokenizer → text_encoder → (cache) ; matte → vae_encoder ;
// transformer ; vae_decoder.

import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { versionFiles, fetchCached } from './models.js';
import { gemmaInputs, selectIndex, COMPLEX_HUMAN_INSTRUCTION } from './prompt.js';
import { embeddingKey, getEmbedding, putEmbedding } from './embcache.js';
import { gaussianNoise, progressCounter } from './util.js';

const LATENT_CHANNELS = 32;          // both VAEs: (B, LC, LH, LW); LH = H / 32 (DC-AE, SANA) or H / 8 (FLUX.2, FIBO2)
const VAE_FACTOR = { sana: 32, fibo2: 8 };
const GEMMA_PAD_ID = 0;
// FIBO2's sessions need more than wasm32's 4 GB, so they run on the Memory64 build (web/scripts/build_ort_memory64.sh,
// served from public/ort-memory64/); SANA keeps npm's build. ORT initializes its wasm once per page: one runtime per page.
const MEMORY64_PATH = 'ort-memory64/';
let runtime = null; // 'default' | 'memory64', fixed by the first Pipeline.load of this page

/**
 * Pure function. The ORT runtime a version needs.
 *
 * @example runtimeOf({manifest: {family: 'fibo2'}}) // 'memory64'
 * @example runtimeOf({manifest: {family: 'sana'}}) // 'default'
 */
export const runtimeOf = (version) => (version.manifest.family === 'fibo2' ? 'memory64' : 'default');

/**
 * Query. Whether this page can still load `version` without a reload (its runtime matches the one already started).
 *
 * @returns {boolean}
 */
export const canLoadHere = (version) => runtime === null || runtime === runtimeOf(version);

// Default ONNX names per role; a manifest file entry's "names" overrides them (web-model-contract).
const DEFAULT_NAMES = {
  text_encoder: { input_ids: 'input_ids', attention_mask: 'attention_mask', output: 'last_hidden_state' },
  vae_encoder: { image: 'image', output: 'latent' },
  transformer: {
    noise: 'noise', matte_latent: 'matte_latent', encoder_hidden_states: 'encoder_hidden_states',
    encoder_attention_mask: 'encoder_attention_mask', timestep: 'timestep', guidance: 'guidance',
    adapter_scale: 'adapter_scale', output: 'latent',
  },
  vae_decoder: { latent: 'latent', output: 'image' },
};
// FIBO2's graph names (src/sana_alpha/fibo2_web.py graph_specs)
const FIBO2_NAMES = {
  text_encoder: { input_ids: 'input_ids', attention_mask: 'attention_mask', output: 'prompt_embeds' },
  vae_encoder: { image: 'image', output: 'latent' },
  transformer: { latents: 'latents', cond_latents: 'cond_latents', sigma: 'sigma', prompt_embeds: 'prompt_embeds',
    adapter_scale: 'adapter_scale', output: 'velocity' },
  vae_decoder: { latent: 'latent', output: 'image' },
};

/**
 * Pure function. FIBO2's resolution shift mu for an image of this many tokens (fibo2.shift_mu: linear between
 * (shift_seq[0], shift_mu[0]) and (shift_seq[1], shift_mu[1])).
 *
 * @example +fibo2Mu(4096, [256, 4096], [0.5, 1.15]).toFixed(3) // 1.15
 * @example +fibo2Mu(1024, [256, 4096], [0.5, 1.15]).toFixed(3) // 0.63
 */
export function fibo2Mu(tokens, seq, mu) {
  return mu[0] + ((mu[1] - mu[0]) / (seq[1] - seq[0])) * (tokens - seq[0]);
}

/**
 * Pure function. FIBO2's noise levels (fibo2.sample): u_k = linspace(1, 1/steps, steps), σ_k = e^mu / (e^mu + 1/u_k − 1),
 * then a final 0. Step k moves x from σ_k to σ_{k+1}.
 *
 * @param {number} steps
 * @param {number} mu - fibo2Mu of the image's token count (LH·LW/4)
 * @returns {number[]} steps + 1 values, from 1 down to 0
 *
 * @example fibo2Sigmas(2, 1.15).map((v) => +v.toFixed(3)) // [1, 0.76, 0]
 */
export function fibo2Sigmas(steps, mu) {
  const e = Math.exp(mu);
  const u = Array.from({ length: steps }, (_, k) => (steps === 1 ? 1 : 1 - (k * (1 - 1 / steps)) / (steps - 1)));
  return [...u.map((x) => e / (e + 1 / x - 1)), 0];
}

/**
 * Pure function. A FIBO2 prompt from the prompt box: a JSON object as typed, else a caption wrapped as FIBO2's JSON
 * caption (fibo2.caption_prompt: {"short_description": …}, no spaces).
 *
 * @example fibo2Prompt('red apple') // '{"short_description":"red apple"}'
 * @example fibo2Prompt(' {"short_description":"x","objects":[]} ') // '{"short_description":"x","objects":[]}'
 */
export function fibo2Prompt(text, key = 'short_description') {
  const t = text.trim();
  return t.startsWith('{') ? JSON.stringify(JSON.parse(t)) : JSON.stringify({ [key]: t });
}

/**
 * Pure function. float32 → IEEE half bits (round to nearest, overflow → ±Inf).
 *
 * @param {Float32Array} src
 * @returns {Uint16Array}
 *
 * @example toHalf(new Float32Array([1, -2, 0.5, 70000])) // Uint16Array [15360, 49152, 14336, 31744]
 */
export function toHalf(src) {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i];
    const x = u[0], sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, man = x & 0x7fffff;
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (man ? 0x200 : 0); continue; }
    const e = exp - 127 + 15;
    if (e >= 0x1f) out[i] = sign | 0x7c00;
    else if (e <= 0) out[i] = e < -10 ? sign : sign | (((man | 0x800000) >> (1 - e)) + 0x1000) >> 13;
    else out[i] = sign | (e << 10) | ((man + 0x1000) >> 13);
  }
  return out;
}

/**
 * Pure function. IEEE half bits → float32.
 *
 * @param {Uint16Array} src
 * @returns {Float32Array}
 *
 * @example fromHalf(new Uint16Array([15360, 49152, 14336])) // Float32Array [1, -2, 0.5]
 */
export function fromHalf(src) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 0x1f ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

/**
 * Pure function. Builds an ORT tensor of the session's expected type from float32 or integer data.
 *
 * @param {string} type - 'float32' | 'float16' | 'int64' | 'int32'
 * @param {ArrayLike<number>} values
 * @param {number[]} dims
 * @returns {ort.Tensor}
 *
 * @example makeTensor('int64', [1, 0], [1, 2]).data // BigInt64Array [1n, 0n]
 */
export function makeTensor(type, values, dims) {
  const data = {
    float32: () => Float32Array.from(values),
    float16: () => toHalf(Float32Array.from(values)),
    int64: () => BigInt64Array.from(values, (v) => BigInt(v)),
    int32: () => Int32Array.from(values),
  }[type];
  if (!data) throw new Error(`unsupported input type ${type}`);
  return new ort.Tensor(type, data(), dims);
}

/**
 * Pure function. An output tensor's data as float32.
 *
 * @param {ort.Tensor} t - float32 or float16 tensor
 * @returns {Float32Array}
 *
 * @example asFloat32(new ort.Tensor('float16', new Uint16Array([15360]), [1])) // Float32Array [1]
 */
export function asFloat32(t) {
  if (t.type === 'float32') return t.data;
  if (t.type === 'float16') return t.data instanceof Uint16Array ? fromHalf(t.data) : Float32Array.from(t.data);
  throw new Error(`unexpected output type ${t.type}`);
}

/**
 * Pure function. Rows `index` of a (1, L, D) array → (1, index.length, D).
 *
 * @param {Float32Array} src - (1, L, D)
 * @param {number[]} index
 * @param {number} d - D
 * @returns {Float32Array}
 *
 * @example selectRows(new Float32Array([1, 2, 3, 4, 5, 6]), [0, 2], 2) // Float32Array [1, 2, 5, 6]
 */
export function selectRows(src, index, d) {
  const out = new Float32Array(index.length * d);
  index.forEach((r, i) => out.set(src.subarray(r * d, (r + 1) * d), i * d));
  return out;
}

/**
 * Pure function. Sprint's TrigFlow angles for `steps` steps, as diffusers SCMScheduler.set_timesteps:
 * 2 steps with sMid → [sMax, sMid, 0], else linspace(sMax, 0, steps + 1). Step k runs at s_k.
 *
 * @param {number} steps - 1–4
 * @param {number} sMax - first angle (Sprint: 1.5708 ≈ π/2, pure noise)
 * @param {number|null} sMid - second angle, only for 2 steps (Sprint: 1.3)
 * @returns {number[]} steps + 1 angles, ending in 0
 *
 * @example sprintTimesteps(2, 1.5708, 1.3) // [1.5708, 1.3, 0]
 * @example sprintTimesteps(4, 1.5708, null) // [1.5708, 1.1781, 0.7854, 0.3927, 0]
 */
export function sprintTimesteps(steps, sMax, sMid) {
  if (steps === 2 && sMid !== null) return [sMax, sMid, 0];
  return Array.from({ length: steps + 1 }, (_, k) => sMax - (sMax * k) / steps);
}

/**
 * Pure function. Seed of the fresh noise ε_k added before step k (k ≥ 1); step 0 uses `seed` itself.
 *
 * @param {number} seed
 * @param {number} k - step index
 * @returns {number} uint32
 *
 * @example stepSeed(0, 0) // 0
 * @example stepSeed(0, 1) // 2654435769
 */
export function stepSeed(seed, k) {
  return (seed ^ Math.imul(k, 0x9e3779b9)) >>> 0;
}

/**
 * Query. Input element type per input name, from the session's metadata.
 *
 * @param {ort.InferenceSession} session
 * @returns {Record<string, string>}
 */
function inputTypes(session) {
  if (!session.inputMetadata) throw new Error('this onnxruntime-web build has no session.inputMetadata');
  return Object.fromEntries(session.inputMetadata.map((m) => [m.name, m.type]));
}

/**
 * Command. Runs a session and records its wall time under `stage` in `timings`.
 *
 * @returns {Promise<Record<string, ort.Tensor>>}
 */
async function timed(timings, stage, session, feeds) {
  const t0 = performance.now();
  const out = await session.run(feeds);
  timings[stage] = performance.now() - t0;
  return out;
}

export class Pipeline {
  /**
   * Command. Downloads (or reads from cache) every file of a version and creates the ORT sessions.
   *
   * @param {{id: string, url: string, manifest: object}} version
   * @param {'webgpu'|'wasm'} ep
   * @param {(file: object, done: number, total: number, fromCache: boolean) => void} onProgress
   * @returns {Promise<Pipeline>}
   */
  static async load(version, ep, onProgress) {
    if (!canLoadHere(version)) throw new Error(`${version.id} needs the ${runtimeOf(version)} runtime; this page started ${runtime}: reload`);
    if (runtime === null && runtimeOf(version) === 'memory64') ort.env.wasm.wasmPaths = new URL(MEMORY64_PATH, document.baseURI).href;
    runtime = runtimeOf(version);
    const files = versionFiles(version);
    const bytes = await Promise.all(files.map((f) => fetchCached(f, (d, t, c) => onProgress(f, d, t, c))));
    const byRole = {};
    files.forEach((f, i) => ((byRole[f.role] ??= []).push({ ...f, data: bytes[i] })));
    const json = (role) => JSON.parse(new TextDecoder().decode(byRole[role][0].data));
    const p = new Pipeline();
    p.version = version;
    p.ep = ep;
    p.tokenizer = new Tokenizer(json('tokenizer'), json('tokenizer_config'));
    p.sessions = {};
    p.types = {};
    p.chunks = null;  // the transformer's chained chunk sessions when the version has them (web_chunks.split)
    for (const role of Object.keys(DEFAULT_NAMES)) {
      const chunked = byRole[`${role}_chunks`];
      const [graphs, external] = chunked ? [chunked, byRole[role] ?? []] : [[byRole[role][0]], byRole[role].slice(1)];
      const sessions = [];
      for (const [i, g] of graphs.entries()) {
        const onGpu = chunked && ep === 'webgpu' && i < graphs.length - 1;  // intermediate float tensors stay on the GPU
        sessions.push(await ort.InferenceSession.create(g.data, {
          executionProviders: [ep],
          externalData: external.map((e) => ({ path: e.name, data: e.data })),
          ...(onGpu ? { preferredOutputLocation: Object.fromEntries(g.gpu_outputs.map((n) => [n, 'gpu-buffer'])) } : {}),
        }));
      }
      if (chunked) p.chunks = sessions;
      else p.sessions[role] = sessions[0];
      p.types[role] = Object.assign({}, ...sessions.map(inputTypes));
    }
    const tf = version.manifest.files.text_encoder;
    p.encoderId = `${new URL(tf.path, version.url).href}:${tf.bytes}`;
    p.memory = new Map(); // embedding key → value (this page's lifetime)
    p.matte = { rev: -1, latent: null };
    p.family = version.manifest.family ?? 'sana';
    p.hasLambda = p.name('transformer', 'adapter_scale') in p.types.transformer;  // runtime-λ graph (manifest "adapter")
    return p;
  }

  /** Pure function. Progress units of one transformer pass: its chunk count, or 1 for an unsplit graph. @example p.passUnits() // 5 */
  passUnits() { return this.chunks?.length ?? 1; }

  /**
   * Command. One transformer pass: the unsplit session, or the chunks in order (each reads its inputs from the graph
   * inputs and earlier chunks' outputs; GPU-resident intermediates are released after the pass). Calls `tick` after
   * each chunk and records the pass's wall time under `label`.
   *
   * @param {Record<string, ort.Tensor>} feeds - the unsplit graph's inputs
   * @param {Record<string, number|string>} timings
   * @param {string} label
   * @param {() => void} tick
   * @returns {Promise<Record<string, ort.Tensor>>} {output_name: tensor}
   */
  async runTransformer(feeds, timings, label, tick) {
    if (!this.chunks) { const out = await timed(timings, label, this.sessions.transformer, feeds); await tick(); return out; }
    const t0 = performance.now(), values = { ...feeds }, made = [];
    for (const s of this.chunks) {
      const out = await s.run(Object.fromEntries(s.inputNames.map((n) => [n, values[n]])));
      for (const [name, tensor] of Object.entries(out)) { values[name] = tensor; made.push(tensor); }
      await tick();
    }
    const name = this.name('transformer', 'output');
    for (const t of made) if (t !== values[name] && t.location === 'gpu-buffer') t.dispose();
    timings[label] = performance.now() - t0;
    return { [name]: values[name] };
  }

  /** Pure function. The ONNX name of `key` for `role`. @example p.name('vae_decoder', 'output') // 'image' */
  name(role, key) {
    return this.version.manifest.files[role].names?.[key] ?? (this.family === 'fibo2' ? FIBO2_NAMES : DEFAULT_NAMES)[role][key];
  }

  /**
   * Command. FIBO2 prompt → (1, 6, T, 7680) embedding (every token: batch 1, nothing padded); memory → IndexedDB →
   * text encoder. Fails loudly over the manifest's max_tokens (the graph's gist count is fixed for those lengths).
   *
   * @param {string} prompt - a JSON prompt (fibo2Prompt)
   * @param {Record<string, number|string>} timings
   * @returns {Promise<{data: Float32Array, dims: number[]}>}
   */
  async embedFibo2(prompt, timings) {
    const key = embeddingKey(this.encoderId, prompt);
    if (this.memory.has(key)) { timings.text_encoder = 'cached (memory)'; return this.memory.get(key); }
    const stored = await getEmbedding(key);
    if (stored) { timings.text_encoder = 'cached (IndexedDB)'; this.memory.set(key, stored); return stored; }
    const text = this.version.manifest.text_encoder;
    const t0 = performance.now();
    const ids = this.tokenizer.encode(text.template.replace('{}', prompt), { add_special_tokens: false }).ids;
    timings.tokenize = performance.now() - t0;
    if (ids.length > text.max_tokens) throw new Error(`prompt is ${ids.length} tokens; this version takes at most ${text.max_tokens}`);
    const types = this.types.text_encoder, n = (k) => this.name('text_encoder', k);
    const out = await timed(timings, 'text_encoder', this.sessions.text_encoder, {
      [n('input_ids')]: makeTensor(types[n('input_ids')], ids, [1, ids.length]),
      [n('attention_mask')]: makeTensor(types[n('attention_mask')], ids.map(() => 1), [1, ids.length]),
    });
    const hidden = out[n('output')];
    const value = { data: asFloat32(hidden), dims: hidden.dims };
    this.memory.set(key, value);
    await putEmbedding(key, value);
    return value;
  }

  /**
   * Command. One FIBO2 generation (fibo2.sample): noise x ~ N(0, 1), then for each σ_k → σ_{k+1}:
   * v = transformer(x, σ_k, prompt), with CFG v = v_null + guidance·(v − v_null) against the null prompt, x += (σ_{k+1} − σ_k)·v.
   *
   * @param {{prompt: string, matte: Float32Array, matteRev: number, seed: number, steps: number, guidance: number,
   *   width: number, height: number, adapterScale: number}} job - matte (1, 3, H, W) in [−1, 1]; width, height: multiples of 16
   * @param {Function} onProgress - see progressCounter
   * @returns {Promise<{image: Float32Array, timings: Record<string, number|string>}>} image (1, 3, H, W) in [−1, 1]
   */
  async generateFibo2({ prompt, matte, matteRev, seed, steps, guidance, width, height, adapterScale }, onProgress) {
    const timings = {};
    const t0 = performance.now();
    const m = this.version.manifest, f = VAE_FACTOR.fibo2, cfg = guidance > 1;
    const dims = [1, LATENT_CHANNELS, height / f, width / f];
    const stages = cfg ? ['text', 'text', 'matte', 'decoder'] : ['text', 'matte', 'decoder'];  // text (+ null prompt), trimask, decoder
    const tick = progressCounter(steps, cfg ? 2 : 1, this.passUnits(), stages, onProgress);
    const emb = await this.embedFibo2(fibo2Prompt(prompt, m.text_encoder.caption_key), timings);
    await tick('text encoder', 0, 'text');
    const nullEmb = cfg ? await this.embedFibo2(m.text_encoder.null_prompt, {}) : null;
    if (cfg) await tick('text encoder (null prompt)', 0, 'text');
    const cond = await this.encodeMatte(matte, matteRev, width, height, timings);
    await tick('trimask encoder', 0, 'matte');
    const types = this.types.transformer, n = (k) => this.name('transformer', k);
    const t = (k, values, d) => makeTensor(types[n(k)], values, d);
    const sigmas = fibo2Sigmas(steps, fibo2Mu((dims[2] * dims[3]) / 4, m.sampler.shift_seq, m.sampler.shift_mu));
    const fixed = { [n('cond_latents')]: t('cond_latents', cond, dims), ...(this.hasLambda ? { [n('adapter_scale')]: t('adapter_scale', [adapterScale], [1]) } : {}) };
    const promptT = t('prompt_embeds', emb.data, emb.dims), nullT = nullEmb && t('prompt_embeds', nullEmb.data, nullEmb.dims);
    let x = gaussianNoise(dims.reduce((a, b) => a * b), seed);
    for (let k = 0; k < steps; k++) {
      const pass = async (e, label) => asFloat32((await this.runTransformer({
        ...fixed, [n('latents')]: t('latents', x, dims), [n('sigma')]: t('sigma', [sigmas[k]], [1]), [n('prompt_embeds')]: e,
      }, timings, label, () => tick(label, k + 1)))[n('output')]);
      let v = await pass(promptT, `transformer ${k + 1}`);
      if (nullT) {
        const vNull = await pass(nullT, `transformer ${k + 1} (null)`);
        v = v.map((c, i) => vNull[i] + guidance * (c - vNull[i]));
      }
      const ds = sigmas[k + 1] - sigmas[k];
      x = x.map((xi, i) => xi + ds * v[i]);
    }
    const nLat = this.name('vae_decoder', 'latent');
    const dec = await timed(timings, 'vae_decoder', this.sessions.vae_decoder, { [nLat]: makeTensor(this.types.vae_decoder[nLat], x, dims) });
    await tick('image decoder', 0, 'decoder');
    timings.total = performance.now() - t0;
    return { image: asFloat32(dec[this.name('vae_decoder', 'output')]).map((v) => 2 * v - 1), timings };
  }

  /**
   * Command. Prompt → (1, 300, 2304) embedding + (1, 300) mask; memory → IndexedDB → text encoder.
   *
   * @param {string} prompt
   * @param {Record<string, number|string>} timings - receives text_encoder ms or 'cached'
   * @returns {Promise<{data: Float32Array, dims: number[], mask: number[]}>}
   */
  async embed(prompt, timings) {
    const key = embeddingKey(this.encoderId, prompt);
    if (this.memory.has(key)) { timings.text_encoder = 'cached (memory)'; return this.memory.get(key); }
    const stored = await getEmbedding(key);
    if (stored) { timings.text_encoder = 'cached (IndexedDB)'; this.memory.set(key, stored); return stored; }
    const m = this.version.manifest;
    const maxLen = m.max_sequence_length;
    const encode = (t) => this.tokenizer.encode(t).ids;
    const t0 = performance.now();
    const { ids, mask } = gemmaInputs(encode, prompt, m.complex_human_instruction ?? COMPLEX_HUMAN_INSTRUCTION, maxLen, GEMMA_PAD_ID);
    timings.tokenize = performance.now() - t0;
    const types = this.types.text_encoder, s = this.sessions.text_encoder;
    const [nIds, nMask] = [this.name('text_encoder', 'input_ids'), this.name('text_encoder', 'attention_mask')];
    const out = await timed(timings, 'text_encoder', s, {
      [nIds]: makeTensor(types[nIds], ids, [1, ids.length]),
      [nMask]: makeTensor(types[nMask], mask, [1, mask.length]),
    });
    const hidden = out[this.name('text_encoder', 'output')];
    const d = hidden.dims[2];
    const sel = selectIndex(ids.length, maxLen);
    const value = { data: selectRows(asFloat32(hidden), sel, d), dims: [1, maxLen, d], mask: sel.map((i) => mask[i]) };
    this.memory.set(key, value);
    await putEmbedding(key, value);
    return value;
  }

  /**
   * Command. Matte tensor → scaled latent, reused while the matte revision is unchanged.
   *
   * @param {Float32Array} matte - (1, 3, H, W) in [−1, 1]
   * @param {number} rev - Matte editor revision
   * @param {number} width - W
   * @param {number} height - H
   * @param {Record<string, number|string>} timings
   * @returns {Promise<Float32Array>} (1, 32, H/32, W/32), × scaling_factor
   */
  async encodeMatte(matte, rev, width, height, timings) {
    rev = `${rev}:${width}x${height}`;
    if (this.matte.rev === rev) { timings.vae_encoder = 'cached (same matte)'; return this.matte.latent; }
    const nImg = this.name('vae_encoder', 'image');
    const out = await timed(timings, 'vae_encoder', this.sessions.vae_encoder,
      { [nImg]: makeTensor(this.types.vae_encoder[nImg], matte, [1, 3, height, width]) });
    const m = this.version.manifest, sf = m.latent_scaling_in_graph ? 1 : m.scaling_factor;
    const latent = asFloat32(out[this.name('vae_encoder', 'output')]).map((v) => v * sf);
    this.matte = { rev, latent };
    return latent;
  }

  /**
   * Command. One generation: Sprint's multi-step sampler (SanaSprintPipeline + SCMScheduler, unit-scale latents).
   * Step k: denoised = transformer(latents, s_k, guidance); then latents = cos(s_{k+1})·denoised + sin(s_{k+1})·ε_{k+1}.
   *
   * @param {{prompt: string, matte: Float32Array, matteRev: number, seed: number, timesteps: number[], guidance: number,
   *   width: number, height: number}} job - matte (1, 3, H, W); timesteps: sprintTimesteps(...) (steps + 1 angles);
   *   guidance: Sprint's embedded guidance scale (one pass, not CFG); width, height: multiples of 32;
   *   adapterScale: λ for runtime-λ versions (hasLambda), ignored otherwise
   * @param {Function} [onProgress] - called after each stage and each transformer chunk (see progressCounter)
   * @returns {Promise<{image: Float32Array, timings: Record<string, number|string>}>} image (1, 3, H, W) in [−1, 1]
   */
  async generate(job, onProgress = () => {}) {
    if (this.family === 'fibo2') return this.generateFibo2(job, onProgress);
    const { prompt, matte, matteRev, seed, timesteps, guidance, width, height, adapterScale } = job;
    const timings = {};
    const t0 = performance.now();
    const dims = [1, LATENT_CHANNELS, height / VAE_FACTOR.sana, width / VAE_FACTOR.sana];
    const steps = timesteps.length - 1;
    const tick = progressCounter(steps, 1, this.passUnits(), ['text', 'matte', 'decoder'], onProgress);
    const emb = await this.embed(prompt, timings);
    await tick('text encoder', 0, 'text');
    const matteLatent = await this.encodeMatte(matte, matteRev, width, height, timings);
    await tick('matte encoder', 0, 'matte');
    const m = this.version.manifest, types = this.types.transformer;
    const n = (k) => this.name('transformer', k);
    const t = (k, values, dims) => makeTensor(types[n(k)], values, dims);
    const numel = dims.reduce((a, b) => a * b);
    const fixed = {
      [n('matte_latent')]: t('matte_latent', matteLatent, dims),
      [n('encoder_hidden_states')]: t('encoder_hidden_states', emb.data, emb.dims),
      [n('encoder_attention_mask')]: t('encoder_attention_mask', emb.mask, [1, emb.mask.length]),
      [n('guidance')]: t('guidance', [guidance], [1]),
      ...(this.hasLambda ? { [n('adapter_scale')]: t('adapter_scale', [adapterScale], [1]) } : {}),
    };
    let latents = gaussianNoise(numel, seed), denoised;
    for (let k = 0; k + 1 < timesteps.length; k++) {
      const label = `transformer ${k + 1}`;
      const out = await this.runTransformer({
        ...fixed, [n('noise')]: t('noise', latents, dims), [n('timestep')]: t('timestep', [timesteps[k]], [1]),
      }, timings, label, () => tick(label, k + 1));
      denoised = asFloat32(out[n('output')]);
      const s = timesteps[k + 1];
      if (s > 0) {
        const eps = gaussianNoise(numel, stepSeed(seed, k + 1));
        latents = denoised.map((v, i) => Math.cos(s) * v + Math.sin(s) * eps[i]);
      }
    }
    const latent = denoised.map((v) => v / (m.latent_scaling_in_graph ? 1 : m.scaling_factor));
    const nLat = this.name('vae_decoder', 'latent');
    const dec = await timed(timings, 'vae_decoder', this.sessions.vae_decoder,
      { [nLat]: makeTensor(this.types.vae_decoder[nLat], latent, dims) });
    await tick('image decoder', 0, 'decoder');
    timings.total = performance.now() - t0;
    const image = asFloat32(dec[this.name('vae_decoder', 'output')]);
    return { image: m.decoder_output_range === '0,1' ? image.map((v) => 2 * v - 1) : image, timings };
  }
}
