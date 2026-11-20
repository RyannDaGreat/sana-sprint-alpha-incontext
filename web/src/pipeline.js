// The matte / trimask → image pipeline on ONNX Runtime Web (WebGPU EP, or WASM), for both model families:
// SANA-Sprint (manifest §12: Sprint's few-step sampler inside the transformer graph) and FIBO2 (§16: one velocity pass per
// graph call; the flow Euler sampler with CFG runs here). Stages: tokenizer → text_encoder → (cache) ; matte → vae_encoder ;
// transformer ; vae_decoder.

import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { versionFiles, fetchCached } from './models.js';
import { gemmaInputs, selectIndex, COMPLEX_HUMAN_INSTRUCTION } from './prompt.js';
import { embeddingKey, getEmbedding, putEmbedding } from './embcache.js';
import { gaussianNoise, progressCounter, yieldTask } from './util.js';
import { splitX0 } from './splitter.js';
import { SneakPeek, tweedieX0, blendX0, poolLatent, highwayWeight, tableAt } from './sneakpeek.js';

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
    adapter_scale: 'adapter_scale', gamma: 'gamma', output: 'velocity' },
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
 * Pure function. For every tensor name that a chain's parts consume, the index of the last part that reads it; the
 * chain's output is kept (Infinity).
 *
 * @param {{inputNames: string[]}[]} chain - sessions in run order
 * @param {string} output - the chain's output name
 * @returns {Record<string, number>}
 *
 * @example lastUses([{ inputNames: ['x'] }, { inputNames: ['a', 'x'] }, { inputNames: ['b'] }], 'y')
 * // { x: 1, a: 1, b: 2, y: Infinity }
 */
export function lastUses(chain, output) {
  const last = {};
  chain.forEach((s, c) => s.inputNames.forEach((n) => { last[n] = c; }));
  last[output] = Infinity;
  return last;
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
  static async load(version, ep, onProgress, onPrepare = () => {}) {
    if (!canLoadHere(version)) throw new Error(`${version.id} needs the ${runtimeOf(version)} runtime; this page started ${runtime}: reload`);
    if (runtime === null && runtimeOf(version) === 'memory64') {
      ort.env.wasm.wasmPaths = new URL(MEMORY64_PATH, document.baseURI).href;
      // wasm EP only: one thread (under the PWA's cross-origin isolation the Memory64 build's threaded wasm stalled session
      // creation, frozen at "text encoder · part 1 of 113"). Not for WebGPU: FIBO2 on WebGPU worked with the default
      // threads and failed readbacks ("Failed to wait for the operation") after numThreads = 1 (2026-11-20, m00858-m00860)
      if (ep === 'wasm') ort.env.wasm.numThreads = 1;
    }
    runtime = runtimeOf(version);
    const files = versionFiles(version);
    const bytes = await Promise.all(files.map((f) => fetchCached(f, (d, t, c) => onProgress(f, d, t, c))
      .then((data) => { onProgress(f, f.bytes, f.bytes, null); return data; })));  // null: done, whichever way (no row stays 'waiting')
    const byRole = {};
    files.forEach((f, i) => ((byRole[f.role] ??= []).push({ ...f, data: bytes[i] })));
    const json = (role) => JSON.parse(new TextDecoder().decode(byRole[role][0].data));
    const p = new Pipeline();
    p.version = version;
    p.ep = ep;
    p.tokenizer = new Tokenizer(json('tokenizer'), json('tokenizer_config'));
    p.types = {};
    p.chains = {};  // role → its sessions in run order: one, or the parts of a split graph (web_chunks.split)
    const graphsOf = (role) => byRole[`${role}_chunks`] ?? [byRole[role][0]];
    const nGraphs = Object.keys(DEFAULT_NAMES).reduce((n, role) => n + graphsOf(role).length, 0);
    let made = 0;  // session creation after the downloads is its own phase (~90-115 per-part sessions): reported, m00852
    for (const role of Object.keys(DEFAULT_NAMES)) {
      const chunked = byRole[`${role}_chunks`];
      const [graphs, external] = chunked ? [chunked, byRole[role] ?? []] : [[byRole[role][0]], byRole[role].slice(1)];
      const sessions = [];
      for (const [i, g] of graphs.entries()) {
        const onGpu = chunked && ep === 'webgpu' && i < graphs.length - 1;  // intermediate float tensors stay on the GPU
        const own = chunked ? external.filter((e) => g.external.includes(e.name)) : external;  // a part reads only its weight files
        onPrepare(made, nGraphs, role);
        await new Promise((r) => setTimeout(r, 0));  // one task per part so the status can paint (session creation blocks the page; ≈ 0.5 s over 113 parts vs ≈ 30 s of creation)
        sessions.push(await ort.InferenceSession.create(g.data, {
          executionProviders: [ep],
          externalData: own.map((e) => ({ path: e.name, data: e.data })),
          ...(onGpu ? { preferredOutputLocation: Object.fromEntries(g.gpu_outputs.map((n) => [n, 'gpu-buffer'])) } : {}),
        }));
        made += 1;
      }
      p.chains[role] = sessions;
      p.types[role] = Object.assign({}, ...sessions.map(inputTypes));
    }
    p.fastDecoder = null;  // sneak peek (manifest §18): optional; float32 latent (1, 32, LH, LW) → float32 image (1, 3, h, w) in [0, 1]
    if (byRole.fast_decoder) {
      const fd = byRole.fast_decoder[0];
      p.fastDecoder = { session: await ort.InferenceSession.create(fd.data, { executionProviders: [ep] }),
                        names: { latent: 'latent', image: 'image', ...version.manifest.files.fast_decoder.names } };
    }
    p.highway = null;  // sneak peek highway (§18): optional; one graph for every block's head, on a chunk part's main stream
    if (byRole.highway) {
      if (!version.manifest.highway?.stream) throw new Error(`${version.id}: a highway file but no manifest.highway (stream names, c, start)`);
      p.highway = { session: await ort.InferenceSession.create(byRole.highway[0].data, { executionProviders: [ep] }), ...version.manifest.highway };
    }
    const tf = version.manifest.files.text_encoder;
    p.encoderId = `${new URL(tf.path, version.url).href}:${tf.bytes}`;
    p.memory = new Map(); // embedding key → value (this page's lifetime)
    p.matte = { rev: -1, latent: null };
    p.family = version.manifest.family ?? 'sana';
    p.hasLambda = p.name('transformer', 'adapter_scale') in p.types.transformer;  // runtime-λ graph (manifest "adapter")
    p.gpu = ep === 'webgpu' ? await ort.env.webgpu.device : null;  // the GPUDevice ORT runs on: progress follows its queue (run)
    p.splitter = version.manifest.splitter ?? null;  // a splitter version (manifest §17): one velocity per background head, runtime γ
    if (p.splitter && !(p.name('transformer', 'gamma') in p.types.transformer)) throw new Error(`${version.id}: splitter manifest but no gamma input`);
    return p;
  }

  /**
   * Query (reads this version's sessions and manifest). One stage of the progress plan: graph `role`'s parts and unit count (manifest chunked; null units
   * for an unsplit graph).
   *
   * @example p.stagePlan('text_encoder', 'text', 'Text encoder (Gemma)', 'layer') // {stage: 'text', label: 'Text encoder (Gemma)', unit: 'layer', parts: 26, units: 26}
   */
  stagePlan(role, stage, label, unit) {
    return { stage, label, unit, parts: this.chains[role].length, units: this.version.manifest.files[role].units ?? null };
  }

  /**
   * Command (the counter calls onProgress). The progress counter of one generation (util.js
   * progressCounter) with this version's parts.
   *
   * @param {number} steps
   * @param {number} passes - transformer passes per step (2 with CFG)
   * @param {object[]} before - stagePlan()s of the stages before the sampler steps
   * @param {object[]} after - stagePlan()s of the stages after
   * @param {Function} onProgress
   * @returns {ReturnType<typeof progressCounter>}
   */
  progress(steps, passes, before, after, onProgress) {
    const { parts, units } = this.stagePlan('transformer', null, null, 'block');
    return progressCounter({ steps, passes, transformer: { parts, units }, before, after }, onProgress);
  }

  /**
   * Command. Releases every ORT session of this pipeline (its weights' GPU and wasm memory). Switching versions without
   * this kept each loaded FIBO2 (≈ 14 GB) alive: memory piled up until WebGPU readbacks failed (2026-11-20).
   */
  async release() {
    const all = [...Object.values(this.chains).flat(), this.fastDecoder?.session, this.highway?.session].filter(Boolean);
    for (const s of all) await s.release();
    this.chains = {};
    this.broken = 'released';
  }

  /**
   * Command. Runs graph `role`: its one session, or its parts in order (each reads its inputs from the graph inputs and
   * earlier parts' outputs; a GPU-resident output is released right after the last part that reads it, lastUses), and
   * records the wall time under `label`. Progress never makes the computation wait (m00754): under WebGPU, `run` only
   * submits GPU work, so part c is reported when the GPU has finished the work submitted before it
   * (`queue.onSubmittedWorkDone`, a promise nobody awaits). Under wasm the computation blocks the page, so before each
   * part it yields one task (yieldTask, no frame wait) so the page can paint between parts.
   * After a failed run (not a cancel) the pipeline is marked broken: ORT's runtime is unusable, later runs fail loudly.
   *
   * @param {string} role
   * @param {Record<string, ort.Tensor>} feeds - the unsplit graph's inputs
   * @param {Record<string, number|string>} timings
   * @param {string} label
   * @param {(c: number) => void} startPart
   * @param {(c: number, values: Record<string, ort.Tensor>) => Promise<void>} [afterPart] - after part c's outputs exist
   *   (the sneak peek reads a part's main stream there; it returns at once unless it takes a preview)
   * @param {() => Promise<() => void>} [gate] - awaited before every part (manifest §20): resolves with a release function
   *   once this job may run one part (the four-way pad's turn-taking), or throws an AbortError to cancel the job between parts
   * @param {() => Promise<void>} [between] - awaited after every part, inside the job's turn (the sneak peek's decoder)
   * @returns {Promise<Record<string, ort.Tensor>>} {output_name: tensor}
   */
  async run(role, feeds, timings, label, startPart, afterPart, gate, between) {
    if (this.broken) throw new Error(`the model failed earlier (${this.broken}); reload the page`);
    const t0 = performance.now(), values = { ...feeds }, made = new Map();  // made: name → GPU-resident tensor this run created
    const chain = this.chains[role], name = this.name(role, 'output'), last = lastUses(chain, name);
    const free = (c) => {  // a part's GPU outputs are released as soon as no later part reads them (not at the chain's end:
      for (const [k, t] of made) if ((last[k] ?? -1) <= c) { t.dispose(); made.delete(k); }  // ~46 kept streams overflowed GPU memory)
    };
    try {
      for (const [c, s] of chain.entries()) {
        const release = gate ? await gate() : null;
        try {
          if (this.gpu) this.gpu.queue.onSubmittedWorkDone().then(() => startPart(c));
          else { startPart(c); await yieldTask(); }
          const out = await s.run(Object.fromEntries(s.inputNames.map((n) => [n, values[n]])));
          for (const [k, tensor] of Object.entries(out)) { values[k] = tensor; if (tensor.location === 'gpu-buffer') made.set(k, tensor); }
          if (afterPart) await afterPart(c, values);
          free(c);
          if (between) await between();  // this job's sneak peek decodes here, inside its turn: never concurrently with a part
        } finally { release?.(); }
      }
    } catch (e) {  // a cancelled (or failed) job frees every GPU buffer it made, the output too, then rethrows
      for (const t of made.values()) t.dispose();
      if (e.name !== 'AbortError') this.broken = e.message.slice(0, 160);  // ORT's runtime is unusable after a failed run
      throw e;
    }
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
   * @param {(c: number) => void} startPart - progress, called before each text-encoder part
   * @returns {Promise<{data: Float32Array, dims: number[]}>}
   */
  async embedFibo2(prompt, timings, startPart, gate) {
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
    const out = await this.run('text_encoder', {
      [n('input_ids')]: makeTensor(types[n('input_ids')], ids, [1, ids.length]),
      [n('attention_mask')]: makeTensor(types[n('attention_mask')], ids.map(() => 1), [1, ids.length]),
    }, timings, 'text_encoder', startPart, undefined, gate);
    const hidden = out[n('output')];
    const value = { data: asFloat32(hidden), dims: hidden.dims };
    this.memory.set(key, value);
    await putEmbedding(key, value);
    return value;
  }

  /**
   * Command. One FIBO2 generation (fibo2.sample): noise x ~ N(0, 1), then for each σ_k → σ_{k+1}:
   * v = transformer(x, σ_k, prompt), with CFG v = v_null + guidance·(v − v_null) against the null prompt, x += (σ_{k+1} − σ_k)·v.
   * A splitter version (manifest §17) returns K velocities; x follows head 0 (gray), and at the last step every head's
   * clean latent is splitX0's x0_k, each decoded by vae_decoder.
   *
   * @param {{prompt: string, matte: Float32Array, matteRev: number, seed: number, steps: number, guidance: number,
   *   width: number, height: number, adapterScale: number, gamma: number}} job - matte (1, 3, H, W) in [−1, 1]; width,
   *   height: multiples of 16; gamma: the splitter strength γ (splitter versions only)
   * @param {Function} onProgress - see progressCounter
   * @returns {Promise<{image: Float32Array, images: Float32Array[]|null, timings: Record<string, number|string>}>} image
   *   (1, 3, H, W) in [−1, 1] (head 0 for a splitter); images: a splitter's K heads (1, 3, H, W) in [0, 1], else null
   */
  async generateFibo2({ prompt, matte, matteRev, seed, steps, guidance, width, height, adapterScale, gamma, onPreview, gate }, onProgress) {
    const timings = {};
    const t0 = performance.now();
    const m = this.version.manifest, f = VAE_FACTOR.fibo2, cfg = guidance > 1;
    const dims = [1, LATENT_CHANNELS, height / f, width / f];
    const text = (label) => this.stagePlan('text_encoder', 'text', label, 'layer');
    const before = [text('Text encoder (Qwen3-VL)'), ...(cfg ? [text('Text encoder (null prompt)')] : []),
                    this.stagePlan('vae_encoder', 'matte', 'Trimask encoder (VAE)', 'block')];
    const after = (this.splitter?.heads ?? [null]).map((h) => this.stagePlan('vae_decoder', 'decoder', h ? `Image decoder (${h})` : 'Image decoder', 'block'));
    const progress = this.progress(steps, cfg ? 2 : 1, before, after, onProgress);
    const emb = await this.embedFibo2(fibo2Prompt(prompt, m.text_encoder.caption_key), timings, (c) => progress.start(progress.stagePart(0, c)), gate);
    const nullEmb = cfg ? await this.embedFibo2(m.text_encoder.null_prompt, {}, (c) => progress.start(progress.stagePart(1, c)), gate) : null;
    const cond = await this.encodeMatte(matte, matteRev, width, height, timings, (c) => progress.start(progress.stagePart(before.length - 1, c)), gate);
    const types = this.types.transformer, n = (k) => this.name('transformer', k);
    const t = (k, values, d) => makeTensor(types[n(k)], values, d);
    const sigmas = fibo2Sigmas(steps, fibo2Mu((dims[2] * dims[3]) / 4, m.sampler.shift_seq, m.sampler.shift_mu));
    const fixed = { [n('cond_latents')]: t('cond_latents', cond, dims), ...(this.hasLambda ? { [n('adapter_scale')]: t('adapter_scale', [adapterScale], [1]) } : {}),
                    ...(this.splitter ? { [n('gamma')]: t('gamma', [gamma], [1]) } : {}) };
    const promptT = t('prompt_embeds', emb.data, emb.dims), nullT = nullEmb && t('prompt_embeds', nullEmb.data, nullEmb.dims);
    let x = gaussianNoise(dims.reduce((a, b) => a * b), seed), heads = null;
    const fd = this.fastDecoder, hw = this.highway, peek = fd && onPreview && new SneakPeek(fd.session,
      (z, d) => ({ [fd.names.latent]: new ort.Tensor('float32', z, d) }), fd.names.image, onPreview);
    const between = peek ? () => peek.drain() : undefined;  // previews decode between this job's parts (one ORT run at a time)
    const timeline = peek ? { steps, blocks: hw?.blocks ?? 0, points: [], weights: hw?.weights ?? null } : null;  // the accordion's plot data
    let prevX0 = null, lastExpected = Infinity;  // the previous step's full-model x̂0; the expected error of the last shown preview
    const blockOfPart = hw ? Object.fromEntries(Object.entries(hw.stream).map(([name, b]) => [this.partProducing(name), [b, name]])) : {};
    // Per-block previews (§18): after a part that ends main block b, if the preview queue is free, the highway head b estimates
    // the velocity from that block's stream; x̂0 = x − σ·v̂, blended with the previous step's x̂0 (w from the manifest
    // table), then the fast decoder. Only the conditional pass is used (a CFG preview would need both passes' heads).
    const blockPeek = (k) => async (c, values) => {
      const hit = blockOfPart[c];
      if (!hit || !peek.ready()) return;
      const [b, name] = hit, t1 = performance.now();
      // remedy (c) of §18: never show a preview that is expected (held-out table) to be worse than the last one shown
      const expected = hw.expected ? tableAt(hw.expected, k, steps, b) : -Infinity;
      if (expected >= lastExpected) return;
      const out = await hw.session.run({ stream: values[name], c: values[hw.c], latents: new ort.Tensor('float32', x, dims),
                                         block: new ort.Tensor('int64', BigInt64Array.from([BigInt(b)]), []),
                                         start: new ort.Tensor('int64', BigInt64Array.from([BigInt(hw.start)]), []) });
      const raw = tweedieX0(x, asFloat32(out.velocity), sigmas[k]);
      const w = highwayWeight(hw.weights, k, steps, b);
      const shown = prevX0 && w < 1 ? blendX0(raw, prevX0, w) : raw;
      const queued = peek.offer(shown, dims, `step ${k + 1} of ${steps} · block ${b + 1} of ${hw.blocks}`);
      peek.charge(performance.now() - t1);  // after the offer: the head's time delays the NEXT preview, not this one
      if (!queued) return;
      if (hw.expected) lastExpected = expected;
      timeline.points.push({ step: k, block: b, w, raw: poolLatent(raw, dims), shown: poolLatent(shown, dims) });
    };
    for (let k = 0; k < steps; k++) {
      const pass = async (e, label, j) => asFloat32((await this.run('transformer', {
        ...fixed, [n('latents')]: t('latents', x, dims), [n('sigma')]: t('sigma', [sigmas[k]], [1]), [n('prompt_embeds')]: e,
      }, timings, label, (c) => progress.start(progress.passPart(k + 1, j, c)), hw && peek && j === 1 ? blockPeek(k) : undefined, gate, between))[n('output')]);
      let v = await pass(promptT, `transformer ${k + 1}`, 1);
      if (nullT) {
        const vNull = await pass(nullT, `transformer ${k + 1} (null)`, 2);
        v = v.map((c, i) => vNull[i] + guidance * (c - vNull[i]));
      }
      if (peek) {  // the step's own Tweedie estimate (full model): always offered, it is the best preview so far (§18)
        prevX0 = tweedieX0(x, v, sigmas[k]);
        if (hw?.expected_step_end) lastExpected = tableAt(Object.fromEntries(Object.entries(hw.expected_step_end).map(([n, e]) => [n, e.map((x) => [x])])), k, steps, 0);
        peek.offer(prevX0, dims, `step ${k + 1} of ${steps}`, true);
        timeline.points.push({ step: k, block: timeline.blocks, w: 1, raw: poolLatent(prevX0, dims), shown: poolLatent(prevX0, dims) });
      }
      if (this.splitter && k === steps - 1) heads = splitX0(x, v, sigmas[k], this.splitter.heads.length);  // v: head-major (1, K, 32, LH, LW)
      const ds = sigmas[k + 1] - sigmas[k];
      x = x.map((xi, i) => xi + ds * v[i]);  // head 0's velocity: the first 32·LH·LW values
    }
    const nLat = this.name('vae_decoder', 'latent');
    const decode = async (z, label, j) => asFloat32((await this.run('vae_decoder', { [nLat]: makeTensor(this.types.vae_decoder[nLat], z, dims) },
      timings, label, (c) => progress.start(progress.afterPart(j, c)), undefined, gate, between))[this.name('vae_decoder', 'output')]);
    const images = [];
    for (const [j, z] of (heads ?? [x]).entries()) images.push(await decode(z, heads ? `vae_decoder ${this.splitter.heads[j]}` : 'vae_decoder', j));
    progress.finish();
    timings.total = performance.now() - t0;
    if (timeline) { timeline.final = poolLatent(prevX0, dims); timeline.stats = peek.stats(); }
    return { image: images[0].map((v) => 2 * v - 1), images: heads ? images : null, timings, timeline };
  }

  /**
   * Query. The index of the transformer part that outputs tensor `name` (a chunked version's main-stream tensors).
   *
   * @param {string} name - e.g. 'add_101'
   * @returns {number}
   */
  partProducing(name) {
    const c = this.chains.transformer.findIndex((s) => s.outputNames.includes(name));
    if (c < 0) throw new Error(`no transformer part outputs ${name} (manifest.highway.stream)`);
    return c;
  }

  /**
   * Command. Prompt → (1, 300, 2304) embedding + (1, 300) mask; memory → IndexedDB → text encoder.
   *
   * @param {string} prompt
   * @param {Record<string, number|string>} timings - receives text_encoder ms or 'cached'
   * @returns {Promise<{data: Float32Array, dims: number[], mask: number[]}>}
   */
  async embed(prompt, timings, startPart, gate) {
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
    const types = this.types.text_encoder;
    const [nIds, nMask] = [this.name('text_encoder', 'input_ids'), this.name('text_encoder', 'attention_mask')];
    const out = await this.run('text_encoder', {
      [nIds]: makeTensor(types[nIds], ids, [1, ids.length]),
      [nMask]: makeTensor(types[nMask], mask, [1, mask.length]),
    }, timings, 'text_encoder', startPart, undefined, gate);
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
   * @param {(c: number) => void} startPart - progress, called before each VAE-encoder part
   * @returns {Promise<Float32Array>} (1, 32, H/32, W/32), × scaling_factor
   */
  async encodeMatte(matte, rev, width, height, timings, startPart, gate) {
    rev = `${rev}:${width}x${height}`;
    if (this.matte.rev === rev) { timings.vae_encoder = 'cached (same matte)'; return this.matte.latent; }
    const nImg = this.name('vae_encoder', 'image');
    const out = await this.run('vae_encoder', { [nImg]: makeTensor(this.types.vae_encoder[nImg], matte, [1, 3, height, width]) },
      timings, 'vae_encoder', startPart, undefined, gate);
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
   * @param {Function} [onProgress] - called as each part of each graph starts, and at the end (see progressCounter)
   * @returns {Promise<{image: Float32Array, timings: Record<string, number|string>}>} image (1, 3, H, W) in [−1, 1]
   */
  async generate(job, onProgress = () => {}) {
    if (this.family === 'fibo2') return this.generateFibo2(job, onProgress);
    const { prompt, matte, matteRev, seed, timesteps, guidance, width, height, adapterScale, gate } = job;
    const timings = {};
    const t0 = performance.now();
    const dims = [1, LATENT_CHANNELS, height / VAE_FACTOR.sana, width / VAE_FACTOR.sana];
    const steps = timesteps.length - 1;
    const progress = this.progress(steps, 1, [this.stagePlan('text_encoder', 'text', 'Text encoder (Gemma)', 'layer'),
      this.stagePlan('vae_encoder', 'matte', 'Matte encoder (VAE)', 'block')], [this.stagePlan('vae_decoder', 'decoder', 'Image decoder', 'block')], onProgress);
    const emb = await this.embed(prompt, timings, (c) => progress.start(progress.stagePart(0, c)), gate);
    const matteLatent = await this.encodeMatte(matte, matteRev, width, height, timings, (c) => progress.start(progress.stagePart(1, c)), gate);
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
      const out = await this.run('transformer', {
        ...fixed, [n('noise')]: t('noise', latents, dims), [n('timestep')]: t('timestep', [timesteps[k]], [1]),
      }, timings, label, (c) => progress.start(progress.passPart(k + 1, 1, c)), undefined, gate);
      denoised = asFloat32(out[n('output')]);
      const s = timesteps[k + 1];
      if (s > 0) {
        const eps = gaussianNoise(numel, stepSeed(seed, k + 1));
        latents = denoised.map((v, i) => Math.cos(s) * v + Math.sin(s) * eps[i]);
      }
    }
    const latent = denoised.map((v) => v / (m.latent_scaling_in_graph ? 1 : m.scaling_factor));
    const nLat = this.name('vae_decoder', 'latent');
    const dec = await this.run('vae_decoder', { [nLat]: makeTensor(this.types.vae_decoder[nLat], latent, dims) },
      timings, 'vae_decoder', (c) => progress.start(progress.afterPart(0, c)), undefined, gate);
    progress.finish();
    timings.total = performance.now() - t0;
    const image = asFloat32(dec[this.name('vae_decoder', 'output')]);
    return { image: m.decoder_output_range === '0,1' ? image.map((v) => 2 * v - 1) : image, timings };
  }
}
