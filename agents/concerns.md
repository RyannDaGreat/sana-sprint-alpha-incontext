# Concerns — append-only history

## 2026-11-17 03:27 — Project start
- Dump renamed from "Untitled Dump Tue Nov 17, 2026 at 3-09-36AM" to "SanaSprint Alpha InContext Plan" (user request, m00018). A symlink with the old name was left in `~/CleanCode/Dumps/` so the running session's cwd stays valid. The user may delete it.
- Planning only on the Mac. Training will run on an 8× A100 80GB node via a GitHub repo.
- Sonnet sub-frenzy launched (5 agents): MAGICK dataset, Sprint fine-tuning, AnimateDiff adapter, in-context conditioning, RGBA prior art. Outputs: `agents/.scratchpad/frenzy/`.

## Known risks
- Fine-tuning the Sprint student with a plain diffusion loss may destroy its few-step ability (unverified). The alternative is training the teacher and re-distilling.
- bf16 weight arithmetic (`base − adapter + control`) loses precision. Do the subtraction in fp32.
- MAGICK is CC BY-NC 4.0. Fine for research.

## 2026-11-17 03:30 — Dump renamed again (no spaces)
- The user said (m00168) dump names must not contain spaces. Renamed "SanaSprint Alpha InContext Plan" → `sana-sprint-alpha-incontext`, which matches the GitHub repo name. Lesson: never put spaces in dump names.
- Symlinks with the old names (`SanaSprint Alpha InContext Plan`, `Untitled Dump Tue Nov 17, 2026 at 3-09-36AM`) are still in `~/CleanCode/Dumps/` so the running session's cwd stays valid. The user may delete them.

## 2026-11-17 03:30 — Sub-frenzy results (4 of 5 in)
- MAGICK: 148,193 RGBA PNGs at 1024², 191 GB, straight alpha. Captions = `subject` column of `magick_index.tsv` (read with quoting=3). `load_dataset` does not work; use `hf download`. Some green spill in hair → filter by `sim_score`.
- Sprint: no successful plain flow-matching LoRA reported. The maintainer has "never tried" a diffusion loss on Sprint (Sana issue #256). The diffusers sCM+LADD script (`train_sana_sprint_diffusers.py`) is merged, full fine-tune only, and inits the student from the teacher. → The teacher-first route is favored.
- AnimateDiff: the adapter is a LoRA on attention layers, MSE loss, scale 0 at inference; evidence is qualitative only. Closest precedent for base − adapter + control: lllyasviel's ControlNet transfer. No direct report found. Do the arithmetic in fp32.
- Prior art: Zippo (matte → RGB by latent concat) is the closest. No model trained on MAGICK has been published. Eval: exterior gray purity, IoU vs matte, VLM rubric, paired seeds.

## 2026-11-17 03:40 — Decisions: GateControl, 512 first, BlueJay, 12 h target
- User picked GateControl (arXiv 2603.27666) as the control method (m00231) after reading the paper I opened for them. Research agent 6 read the official code. Key fit: the base is frozen and only LoRA + gates train, so `W_control − W_adapter` = the control LoRA delta, which suits our weight arithmetic.
- Risk: GateControl's script is for stock SANA flow matching and never passes `guidance`. The Sprint teacher (TrigFlow) and student (guidance-embedded) need changes there. Not yet checked: whether diffusers' teacher checkpoint already behaves like a flow-matching SANA (it loads with `SanaPipeline`).
- Risk: gates start at 0.5, not 0, so adding GateControl modules changes the base output before training. This is the paper's choice; keep it.
- 512×512 end to end first (m00222, m00245). The dtab was cut to 512-only columns (latents (32,16,16), noise_512).
- Data now goes through BlueJay (m00196), which replaces the "no framework, plain subfolders" rule in the manifest.
- Lesson: `aviary bluejay initialize` starts a database server that keeps running. Delete its lance folder and re-initialize, and the old server answers "Dataset ... was not found". Kill the server first (`pkill -f "bluejay.local_database 0.0.0.0 <port>"`).

## 2026-11-17 03:54 — 1024 only; eval out of BlueJay; download split; initial_rows bug
- 512-first (m00222, m00245) dropped for 1024 only (m00295): the user prefers one resolution, and Sprint/teacher are 1024-native. Teacher first, distill last, confirmed (m00295).
- Mistake: I put eval (eval_noise, checkpoints, generations; plan_generations, generate, score_generation, vlm_judge) into BlueJay. The user (m00371) saw no reason for them: the agent runs eval itself. Removed. BlueJay only curates training data.
- Mistake: download_magick did two jobs in one opaque box (download the TSV, then a 191 GB snapshot to a folder). The Albatross graph made this obvious to the user (m00397). Split: import_index (TSV → one row per image) → download_image (per-image PNG blob from https://huggingface.co/datasets/OneOverZero/MAGICK/resolve/main/images/<2>/<id>.png, verified 1.5 MB). No more magick_raw folder.
- Bug found (user's suspicion, m00295): the old multi-field initial_rows (`,\tcheckpoint x` + indented fields) parsed as ONE field whose value swallowed the other lines. Multi-field rows need `,` alone on its line with the fields indented beneath. Our only seed now is single-field `,\tjob_id magick`. Verified by initialize + read back: 1 row, job_id "magick", import_claim null.
- MAGICK index facts: 148,193 rows but 140,955 unique page_ids. 7,238 images are listed twice (once picked=hand, once picked=auto; other fields equal). `picked` is text, not int. tag_idk is always 0. import_index must dedupe.
- fish's `wait` returns 0 even when a child fails. run_proc_curate.fish uses a FAIL_FLAG file that each failing loop touches; tested with a fake failing loop (exits 1, next step does not run).

## 2026-11-17 03:56 — dtab misaligned: wrong tab width
- What happened: the user saw the rows.parquet table out of line in their editor.
- Root cause: my scratch renderer aligned columns for 8-wide tabs. The house standard (MulticamDailies) is 4-wide tabs. Also, nested `input.json` blocks had the header on the `materialize` line; it belongs on the `input.json` line.
- Fix: copied MulticamDailies' `agents/dtab_format.py` (checker) and `agents/dtab_fold.py`; ran `fix`; `check` passes; `bluejay initialize` passes. The rules are now in AGENTS.md and the dtab (semantic binding `dtab-rules`).
- Lesson: `data/project.dtab` is edited by hand and checked with `dtab_format.py`. No generator: a generator overwrites the user's hand alignment.

## 2026-11-17 04:11 — processor queries were not inline
- What happened: the user asked (m00295) for the dtab style "especially how it inlined all the queries". I kept each processor query as a named `pending_` query in the `queries` section.
- Root cause: I copied MulticamDailies' current project.dtab, which names all its processor queries (0 inline). I did not act on the user's words.
- Fix: each processor now has its query inline (`query sql` block, or one line when tiny). The `queries` section holds only `view_dataset`, for Sparrow. Rule added to the dtab RULES block. `dtab_format.py check` and `bluejay initialize` pass.
- Lesson: when the user names a convention, follow the user's words, not only the reference file.

## 2026-11-17 04:55 — Step 1 baseline + eval harness (subagent 2)
- Eval set: sha256(page_id) mod 4000 == 0 over the deduped index gives 29 images, 0 nsfw. No second rule needed. Same rule as `import_index.py` (`int(sha256(page_id.encode()).hexdigest(), 16) % 4000`). Files: `resources/eval/` (mattes/, references/, prompts.json, noise.safetensors seed 20261117).
- Teacher sampling facts (from the checkpoint files, not the papers): the diffusers teacher loads with `SanaPipeline` + `DPMSolverMultistepScheduler` (flow_prediction, flow_shift 3.0). It is a plain flow (velocity) model; `timestep_scale` 0.001 is applied by the pipeline. TrigFlow appears only as a training-time wrapper in Sprint distillation. Sampling needs no TrigFlow code.
- Pitfall: SanaSprintPipeline's 2-step SCM sampler draws fresh noise at the intermediate step. Fixed `latents` alone do not make it reproducible; `generate()` passes one CPU generator per image (seed 777 + eval index).
- Pitfall: `rp.labeled_image` / `rp.resize_image` need cv2, which the root env does not have; `rp.resize_image` then silently returned a list. Montage uses PIL instead (no new dependency).
- Pitfall: the dtab changed under me (the data agent edits it at the same time). Anchored string inserts failed loudly (assert count == 1), which is the wanted behavior; re-read before every edit.
- Lesson: `dtab_format.py fix` can only align rows whose cells are separated by 2+ spaces; with single spaces it cannot tell DESCRIPTION from EXAMPLE and refuses.
- Baseline result: both models look modern and clean. The gray suffix gives a studio gray that is too light (median border 0.59 Sprint / 0.64 teacher, target 0.50), with gradients and shadows; 4/29 and 3/29 images have a neutral border within 0.05 of 0.5. Sprint 0.27 s/image, teacher 1.7–2.2 s/image (batch 1, A100). Journal: `journal/step1_baseline.html`.
- Eval moved into Bluejay (user m00606, m00611, m00612): `eval_models` (initial_rows) -> `plan_eval` (rows per model x eval image) -> `generate_eval` (claims rows; a per-GPU rp.web_evaluator server keeps the models loaded). Claims need existing rows, and the work is a cross product (model x image), so a plan step must create the rows; settings cannot come from environ because initial_rows do not substitute settings. Bluejay outputs match the plain-script outputs within 1.3–1.6/255 mean abs (bf16, batch 4 vs 1).
- Measured: one generate_eval run costs ~12 s besides GPU time (uv start, claim, PNG blob commit); 8 Sprint rows took ~15 s. EVAL_BATCH_SIZE raised to 16.
- Model weights are localized automatically to /models/sana-sprint-alpha-incontext/<repo path> by `models.localize()` on first load (atomic rename; race-safe).

## 2026-11-17 09:55 — Step 0 curation running (data subagent)
- Aviary: `data/Aviary` = corp/Aviary bluejay2 @ edd536d5. `setup.sh` clones it and builds the full nflx-aviary wheel (stage.sh: npm builds of Sparrow/Albatross, Rust/maturin+zig for the two DataFusion libs; ~10 min the first time) into `data/Aviary/.wheel/`. `data/aviary.sh` = `uvx --from <that wheel> aviary`. Rejected: `install_local.sh` (installs a global uv tool outside the dump); `uv run --project Bluejay` (no `aviary` module, so `$AVIARY` = `python -P -m aviary` would break). Re-run `setup.sh` after a `git pull` in data/Aviary (the wheel name carries the commit).
- Mistake: started with 32 download loops x 16 threads. Hugging Face's "resolvers" limit is a fixed window of 5,000 requests per 300 s per token (headers `ratelimit-policy`, `retry-after`). The burst got 429s; 680 Samples ran out of 8 retries and their Claims errored. Fix: wait the 429's Retry-After (+jitter), give up only after 1 h; 8 download loops. Long-run cap is ~16.7 images/s, so downloading 141k images takes >= 2.3 h whatever the loop count.
- Lesson: killing the loops leaves Claims `started` forever (Bluejay never releases Claims). Recover with `bluejay unclaim` on rows with a pointer but no output, only when no loop runs; while loops run, unclaim only Claims whose claim_status is 'errored'.
- Data fact: some MAGICK PNGs are RGB with no alpha channel: the raw screen image, never keyed (seen: a yellow-screen letter). split_alpha raises for them, so their Claim errors and they get no rgb_on_gray (excluded from training). 3 of the first 128 index rows; fewer in random batches.
- DC-AE on A100 bf16: encode 13 img/s, decode 9 img/s at batch 8; slower at 32 (10.6 / 6.5). VAE round trip PSNR ~35 dB mean. Latents are stored unscaled (multiply by scaling_factor 0.41407 to train).
- Risk: other agents run on the same GPUs. One encode loop died with CUDA OOM on GPU 0 (another process held 42 GB + 19 GB). Its 128 Claims were unclaimed. A dead loop is not restarted (fail loudly); check `grep FAILED agents/.scratchpad/logs/curate.log`.
- Pitfall: the eval agent added tables to the dtab while I ran; `bluejay run` then refused to open until `bluejay initialize` ran again. Initialize is additive and safe to repeat.
- Model localization: encode_latents copies `models/<dir>` to `/models/sana-sprint-alpha-incontext/models/<dir>` (same path as `src/sana_alpha/models.localize`), under an flock, temp folder then rename. A first version used `/models/sana-sprint-alpha-incontext/<dir>` (no `models/`), a duplicate 9 GB copy; deleted.
