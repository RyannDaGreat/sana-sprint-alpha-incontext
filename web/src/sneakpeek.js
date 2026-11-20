// Sneak peek (manifest §18): live previews while a generation runs. After every sampler step (and, with a sneak peek
// highway, after every transformer block) the Tweedie estimate x̂0 = x_t − σ·v goes through the fast decoder. Previews
// never slow the generation down on purpose: the compute loop only calls offer(), which returns at once; at most one
// fast-decoder run is in flight, and a newer estimate replaces an older one still waiting (stale previews are dropped).
// A duty cycle keeps previews (highway heads + fast decoder) to DUTY of the wall time.

const DUTY = 0.1;  // m00755: previews may cost at most 10% of the generation
const sessionTail = new WeakMap();  // session → the promise of its last queued run: the four-way pad's slots share one fast decoder (§20)

/**
 * Command. Runs `session` on `feeds` after every run already queued on it (one ONNX Runtime session runs one call at a time).
 *
 * @param {ort.InferenceSession} session
 * @param {Record<string, ort.Tensor>} feeds
 * @returns {Promise<Record<string, ort.Tensor>>}
 */
export function serialRun(session, feeds) {
  const run = (sessionTail.get(session) ?? Promise.resolve()).then(() => session.run(feeds));
  sessionTail.set(session, run.catch(() => {}));  // the next caller waits for this run, whatever its outcome; this caller still sees the error
  return run;
}

/**
 * Pure function. The rectified-flow clean-latent estimate x̂0 = x − σ·v (Tweedie's formula for x_t = (1 − σ)·x0 + σ·ε,
 * v = ε − x0). Uses the first x.length values of v (a splitter's head 0).
 *
 * @param {Float32Array} x - (1, 32, LH, LW) latents at noise level σ
 * @param {Float32Array} v - velocity, at least x.length values
 * @param {number} sigma
 * @returns {Float32Array} (1, 32, LH, LW)
 *
 * @example tweedieX0(new Float32Array([1, 2]), new Float32Array([4, 6]), 0.5) // Float32Array [-1, -1]
 */
export function tweedieX0(x, v, sigma) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - sigma * v[i];
  return out;
}

/**
 * Pure function. The shown preview of remedy (b) in §18: w·x̂0_highway + (1 − w)·x̂0_previous step (no sawtooth).
 *
 * @param {Float32Array} raw - the highway's x̂0
 * @param {Float32Array} prev - the previous step's full-model x̂0
 * @param {number} w - weight on the highway, in [0, 1]
 * @returns {Float32Array}
 *
 * @example blendX0(new Float32Array([4]), new Float32Array([0]), 0.25) // Float32Array [1]
 */
export function blendX0(raw, prev, w) {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = w * raw[i] + (1 - w) * prev[i];
  return out;
}

/**
 * Pure function. An entry of a per-schedule (step × block) table, e.g. the expected preview error, for step k of `steps`,
 * block b, from a manifest table {"<steps>": [[value per block] per step]} (fitted on held-out generations, §19). A schedule
 * without its own table uses the table of the nearest step count, at the same relative position in the schedule.
 *
 * @param {Record<string, number[][]>} table
 * @param {number} k
 * @param {number} steps
 * @param {number} b
 * @returns {number}
 *
 * @example tableAt({"4": [[0.9, 0.5], [0.2, 0.1], [0.1, 0.05], [0.1, 0.01]]}, 1, 4, 0) // 0.2
 * @example tableAt({"4": [[0.9, 0.5], [0.2, 0.1], [0.1, 0.05], [0.1, 0.01]]}, 2, 8, 0) // 0.2 (step 2 of 8 ≈ step 1 of 4)
 */
export function tableAt(table, k, steps, b) {
  const counts = Object.keys(table).map(Number);
  const n = counts.reduce((best, c) => (Math.abs(c - steps) < Math.abs(best - steps) ? c : best), counts[0]);
  const s = Math.round((k * (n - 1)) / Math.max(1, steps - 1));
  return table[String(n)][s][b];
}

/**
 * Pure function. The calibrated highway weight w (remedy (b) of §18): 1 at step 0 (nothing earlier) or without a table.
 *
 * @example highwayWeight({"4": [[1, 1], [0.2, 1], [0.1, 1], [0.1, 1]]}, 1, 4, 0) // 0.2
 * @example highwayWeight({"4": [[1, 1], [0.2, 1], [0.1, 1], [0.1, 1]]}, 0, 4, 0) // 1
 */
export function highwayWeight(table, k, steps, b) {
  return k === 0 || !table ? 1 : tableAt(table, k, steps, b);
}

/**
 * Pure function. A latent averaged over 4×4 patches, for the timeline plot (16× fewer values to keep per preview).
 *
 * @param {Float32Array} z - (1, C, H, W)
 * @param {number[]} dims - [1, C, H, W], H and W multiples of 4
 * @returns {Float32Array} (1, C, H/4, W/4)
 *
 * @example poolLatent(new Float32Array(16).fill(2), [1, 1, 4, 4]) // Float32Array [2]
 */
export function poolLatent(z, dims) {
  const [, c, h, w] = dims, H = h / 4, W = w / 4, out = new Float32Array(c * H * W);
  for (let ch = 0; ch < c; ch++)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) out[(ch * H + (y >> 2)) * W + (x >> 2)] += z[(ch * h + y) * w + x] / 16;
  return out;
}

export class SneakPeek {
  /**
   * Command (keeps the session). A preview queue around one fast-decoder session.
   *
   * @param {ort.InferenceSession} session - float32 latent (1, 32, LH, LW) → float32 image (1, 3, h, w) in [0, 1] (the
   *   fast-decoder contract: fp16 weights inside, float32 at both ends)
   * @param {(x0: Float32Array, dims: number[]) => Record<string, ort.Tensor>} feed - the session's inputs (dtype per graph)
   * @param {string} output - the image output's name
   * @param {(image: {data: Float32Array, dims: number[]}, label: string) => void} onPreview
   */
  constructor(session, feed, output, onPreview) {
    Object.assign(this, { session, feed, output, onPreview, busy: false, pending: null, enabled: true, shown: 0, dropped: 0,
                          nextAt: 0, spentMs: 0, t0: performance.now() });
  }

  /**
   * Query. Whether a per-block preview may start now: the decoder is idle and the duty cycle allows it. Previews may use at
   * most DUTY of the time (m00755: "should not make the model any more than 10% slower"): after a preview that cost d ms
   * (highway + decoder), the next one waits d·(1 − DUTY)/DUTY ms.
   *
   * @returns {boolean}
   */
  ready() { return this.enabled && !this.busy && !this.pending && performance.now() >= this.nextAt; }

  /** Command. Counts ms the caller spent for a preview (the highway head) against the duty cycle. */
  charge(ms) { this.spentMs += ms; this.nextAt = Math.max(this.nextAt, performance.now() + (ms * (1 - DUTY)) / DUTY); }

  /** Query. {shown, dropped, spentMs, fraction of the wall time spent on previews}. */
  stats() { return { shown: this.shown, dropped: this.dropped, spentMs: this.spentMs, fraction: this.spentMs / (performance.now() - this.t0) }; }

  /**
   * Command. Queues x̂0 for a preview and returns at once. A queued estimate that has not been decoded yet is replaced:
   * only the newest one is decoded. Decoding happens in drain(), which the pipeline awaits between graph parts: one ORT
   * runtime must never run two sessions at once (the asyncify build; concurrent runs broke WebGPU readbacks, 2026-11-20).
   *
   * @param {Float32Array} x0 - (1, 32, LH, LW) normalized latents
   * @param {number[]} dims - [1, 32, LH, LW]
   * @param {string} label - e.g. 'step 2 · block 30'
   * @param {boolean} [always] - the step's full-model estimate: queued even inside the duty-cycle wait
   * @returns {boolean} whether it was queued
   */
  offer(x0, dims, label, always = false) {
    if (!this.enabled || (!always && performance.now() < this.nextAt)) return false;  // the step's own estimate (always) skips the duty wait
    if (this.pending) this.dropped++;
    this.pending = { x0, dims, label };
    return true;
  }

  /** Command. Decodes the queued estimate, if any (awaited by the pipeline between graph parts and before returning). */
  async drain() {
    this.busy = true;
    while (this.pending) {
      const { x0, dims, label } = this.pending;
      this.pending = null;
      const t1 = performance.now();
      const image = (await serialRun(this.session, this.feed(x0, dims)))[this.output];
      const data = await image.getData();
      this.charge(performance.now() - t1);
      this.onPreview({ data, dims: image.dims }, label);
      this.shown++;
    }
    this.busy = false;
  }
}
