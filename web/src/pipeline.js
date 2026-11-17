// The one-step matte → image pipeline on ONNX Runtime Web (WebGPU EP, or WASM).
// Stages: tokenizer → text_encoder → (cache) ; matte → vae_encoder ; transformer ; vae_decoder.

import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { versionFiles, fetchCached } from './models.js';
import { gemmaInputs, selectIndex, COMPLEX_HUMAN_INSTRUCTION } from './prompt.js';
import { embeddingKey, getEmbedding, putEmbedding } from './embcache.js';
import { gaussianNoise } from './util.js';

const LATENT_CHANNELS = 32;          // DC-AE latent (B, LC, LH, LW), LH = H / 32
const VAE_FACTOR = 32;
const GEMMA_PAD_ID = 0;

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
    for (const role of Object.keys(DEFAULT_NAMES)) {
      const [model, ...external] = byRole[role];
      p.sessions[role] = await ort.InferenceSession.create(model.data, {
        executionProviders: [ep],
        externalData: external.map((e) => ({ path: e.name, data: e.data })),
      });
      p.types[role] = inputTypes(p.sessions[role]);
    }
    const tf = version.manifest.files.text_encoder;
    p.encoderId = `${new URL(tf.path, version.url).href}:${tf.bytes}`;
    p.memory = new Map(); // embedding key → value (this page's lifetime)
    p.matte = { rev: -1, latent: null };
    p.hasLambda = p.name('transformer', 'adapter_scale') in p.types.transformer;  // runtime-λ graph (manifest "adapter")
    return p;
  }

  /** Pure function. The ONNX name of `key` for `role`. @example p.name('vae_decoder', 'output') // 'image' */
  name(role, key) {
    return this.version.manifest.files[role].names?.[key] ?? DEFAULT_NAMES[role][key];
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
   * @returns {Promise<{image: Float32Array, timings: Record<string, number|string>}>} image (1, 3, H, W) in [−1, 1]
   */
  async generate({ prompt, matte, matteRev, seed, timesteps, guidance, width, height, adapterScale }) {
    const timings = {};
    const t0 = performance.now();
    const dims = [1, LATENT_CHANNELS, height / VAE_FACTOR, width / VAE_FACTOR];
    const emb = await this.embed(prompt, timings);
    const matteLatent = await this.encodeMatte(matte, matteRev, width, height, timings);
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
      const out = await timed(timings, `transformer ${k + 1}`, this.sessions.transformer, {
        ...fixed, [n('noise')]: t('noise', latents, dims), [n('timestep')]: t('timestep', [timesteps[k]], [1]),
      });
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
    timings.total = performance.now() - t0;
    const image = asFloat32(dec[this.name('vae_decoder', 'output')]);
    return { image: m.decoder_output_range === '0,1' ? image.map((v) => 2 * v - 1) : image, timings };
  }
}
