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
