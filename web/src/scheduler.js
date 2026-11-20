// Turn-taking for several generations on one set of ONNX Runtime sessions (manifest §20, m00848): the four-way pad runs up
// to four jobs "at once" by granting one graph part at a time, round-robin over the jobs that are not paused. A paused job
// simply gets no turns, so the others run faster. A cancelled job's next gate throws an AbortError, and Pipeline.run frees
// its GPU buffers. One job with gateFor() is also how the single mode's real-time edits cancel a run partway (item 134).

/** Pure function. The error a cancelled job's gate throws. @example abortError().name // 'AbortError' */
export const abortError = () => new DOMException('cancelled', 'AbortError');

/**
 * Pure function. The next job to run after `last` in round-robin order among the ready ids, or null.
 *
 * @param {string[]} order - every job id, in slot order
 * @param {Set<string>} ready - ids waiting for a turn and not paused
 * @param {string|null} last - the id that ran last
 * @returns {string|null}
 * @example nextTurn(['a', 'b', 'c'], new Set(['a', 'c']), 'a') // 'c'
 * @example nextTurn(['a', 'b', 'c'], new Set(['a', 'c']), 'c') // 'a'
 * @example nextTurn(['a', 'b'], new Set(), 'a') // null
 */
export function nextTurn(order, ready, last) {
  const start = last === null ? 0 : order.indexOf(last) + 1;
  for (let i = 0; i < order.length; i++) {
    const id = order[(start + i) % order.length];
    if (ready.has(id)) return id;
  }
  return null;
}

export class TurnScheduler {
  /** Command. An empty scheduler: no jobs, no part running. */
  constructor() {
    this.jobs = new Map();  // id → {paused, cancelled, waiting: {resolve, reject} | null, parts}
    this.order = [];
    this.running = false;
    this.last = null;
  }

  /**
   * Command. Registers job `id` (replacing a previous one with the same id) and returns its gate for Pipeline.generate.
   *
   * @param {string} id
   * @param {boolean} [paused]
   * @returns {() => Promise<() => void>} gate: resolves with a release function when it is this job's turn
   */
  gateFor(id, paused = false) {
    const job = { paused, cancelled: false, waiting: null, parts: 0 };
    this.jobs.set(id, job);
    if (!this.order.includes(id)) this.order.push(id);
    return () => {
      if (job.cancelled) return Promise.reject(abortError());
      return new Promise((resolve, reject) => { job.waiting = { resolve, reject }; this.pump(); });
    };
  }

  /** Command. Grants the next turn when no part is running. */
  pump() {
    if (this.running) return;
    const ready = new Set([...this.jobs].filter(([, j]) => j.waiting && !j.paused && !j.cancelled).map(([id]) => id));
    const id = nextTurn(this.order, ready, this.last);
    if (id === null) return;
    const job = this.jobs.get(id), { resolve } = job.waiting;
    job.waiting = null; job.parts++;
    this.running = true; this.last = id;
    resolve(() => { this.running = false; this.pump(); });
  }

  /** Command. Pauses or resumes job `id` (a paused job keeps its state and gets no turns). */
  setPaused(id, paused) {
    const job = this.jobs.get(id);
    if (!job) return;
    job.paused = paused;
    this.pump();
  }

  /** Command. Cancels job `id`: a waiting gate rejects now, a running part finishes and the next gate rejects. */
  cancel(id) {
    const job = this.jobs.get(id);
    if (!job || job.cancelled) return;
    job.cancelled = true;
    if (job.waiting) { job.waiting.reject(abortError()); job.waiting = null; }
    this.pump();
  }

  /** Query. Turns granted to job `id` so far (the pad's per-slot rate). */
  parts(id) { return this.jobs.get(id)?.parts ?? 0; }
}
