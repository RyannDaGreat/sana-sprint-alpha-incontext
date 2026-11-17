# Manifest — SANA-Sprint In-Context Alpha → RGB-on-Gray

> STATUS: **executing on the node** (8× A100 80GB, 96 CPUs, 1.1 TB RAM) since 2026-11-17 4:16 AM ET, in bulldog mode + fork mode (m00600–m00605). The agent's name is **Sana**.

## 0. Agenda (node session, 13 h budget: 4:16 AM → 5:16 PM ET, m00600)

| By (ET) | Milestone |
|---|---|
| 5:15 AM | Aviary checkout in `data/Aviary`, `data/aviary.sh` runs it; `setup.sh`; processors written; curation running |
| 6:00 AM | Models downloaded; eval set + fixed noises fixed; Step 1 baseline grid |
| 7:00 AM | Curation done (download is capped by Hugging Face at ~16.7 img/s). Adapter + control code already proven on toy data (done 5:45 AM) |
| 7:05 AM | Adapter run launched on all 8 GPUs (`./train_adapter.sh adapter_v1`, babysat) |
| 10:30 AM | **Checkpoint 1** (adapter passes VLM check) → notification |
| 11:00 AM | Control (GateControl) run launched (babysat) |
| 2:30 PM | Control passes IoU/VLM; Step 4 `base − adapter + control` grid |
| 4:30 PM | Journal written; web UI if time allows |
| 5:16 PM | Hard stop: final notification |

**Agent deadlines (m00608).** Each subagent gets an expected finish time. After it, Sana babysits the agent (checks files, logs, GPUs every ~20 min) and steps in if it stalls. Wave 1 (launched 4:25 AM): data pipeline → 5:15 AM, eval set + baseline → 5:30 AM, adapter training code → 6:00 AM, GateControl port → 6:30 AM.

**Talk out loud: once per request (m00620, m00621).** Reply in chat first, then speak 1–2 plain-English sentences: `curl -s -d "MESSAGE" ntfy.sh/stark_comma > /dev/null &`. "Notify me" = `rp call ntfy_send` (phone); "talk to me" = this curl (audio). They are different. Done 1:18 PM: gallery ready (phone + one spoken message). Pending: one phone notification when the WebGPU demo runs and is tested (m00625).

**Notifications (max 6, m00604).** Each starts with "Sana:". Sent: (1) 8:26 AM Checkpoint 1. (2) 10:38 AM control works; subtraction loses the gray; λ = 0.75 fix; adapter_v2 on random colors. (3) 1:18 PM gallery_v1 ready (the message said 1:25 PM by mistake). Plan: (1) Checkpoint 1, (2) control model follows mattes, (3) Step 4 result, (4) final wrap-up. Two are kept in reserve for blockers that only the user can fix.

**Aviary (m00600).** `data/Aviary` is a git checkout of `corp/Aviary` branch `bluejay2` (gitignored; `setup.sh` clones it). `data/aviary.sh` runs Aviary from that checkout. It plays the same role as `aviary.sh` in MulticamDailies: every script calls Aviary only through it. The dtab uses `lineage` keys, which only the bluejay2 source has, not the PyPI release.

## 1. Goal (read this first)

**Problem.** We want a fast text-to-image model that takes an **alpha matte** (a grayscale image: white = opaque, black = transparent) plus a caption, and generates the **foreground RGB, composited on a flat 50% gray background** (RGB = 128,128,128 / 0.5 in [0,1]), so that the generated object exactly fills the matte. Combined with the input matte, that output gives a full RGBA image (the gray background is known, so the foreground color can be recovered by un-compositing).

**Base model.** SANA-Sprint (NVlabs, arXiv 2503.09641), a 1–4 step distilled DiT. We pick it because it is very fast, so many eval runs are cheap.

**Data.** MAGICK (Burgert et al., CVPR; https://ryanndagreat.github.io/MAGICK/, https://huggingface.co/datasets/OneOverZero/MAGICK). This is a large captioned RGBA dataset. A 2023-era diffusion model generated it, so the image quality and captions are dated ("very 2023 quality").

**Method: the AnimateDiff domain-adapter trick, used for in-context control.**
1. **Adapter model**: fine-tune SANA-Sprint (LoRA at the highest rank that fits VRAM, or full fine-tune) on MAGICK images composited on gray, with MAGICK captions. This absorbs the *domain*: gray backgrounds, MAGICK's dated look.
2. **Control model**: bake the adapter into the weights. On top of that, train matte control **based on GateControl** (arXiv 2603.27666, "Gated Condition Injection without Multimodal Attention"; user chose it, m00231: "it's official"). GateControl targets linear-attention models like SANA. The matte is a clean (noise-free) condition; the loss is on the generated image only. Exact architecture: see §5 (from research agent 6).
3. **Final model (inference only)**: `W_final = 1·W_base − 1·W_adapter + 1·W_control`, which is `W_base + (W_control − W_adapter)`. Only the control delta is added to the clean base. In theory this subtracts MAGICK's dated domain "junk" and keeps SANA-Sprint's quality. This is what AnimateDiff did when it adapted video from image data.

**Rationale.** Training the control behavior directly on dated data would also teach the dated look. The adapter soaks up the domain shift first, so the control delta contains (ideally) only "follow the matte".

## 2. Glossary

| Term | Meaning |
|---|---|
| **SANA-Sprint** | NVlabs' few-step (1–4 step) text-to-image model, distilled from SANA with sCM + LADD. 0.6B and 1.6B, 1024px. |
| **Teacher** | The multi-step SANA variant (TrigFlow, QK-norm) that Sprint is distilled from. HF: `Efficient-Large-Model/Sana_Sprint_{0.6B,1.6B}_1024px_teacher_diffusers`. |
| **sCM / LADD** | Continuous-time consistency distillation / latent adversarial diffusion distillation. Sprint's training recipe. |
| **MAGICK** | The user's CVPR dataset of generated RGBA images with captions, matted via chroma keying. |
| **Alpha / matte** | Grayscale image. White = opaque, black = transparent. |
| **Gray composite** | `RGB_on_gray = alpha·RGB + (1−alpha)·0.5`. |
| **Adapter model / adapter LoRA / adapter weights** | Step-2 model: Sprint fine-tuned to unconditionally (text-only) generate MAGICK-style gray composites. Named after AnimateDiff's domain adapter. |
| **Bake in** | Merge LoRA weights into base weights (`W ← W + BA·scale`) so the LoRA no longer needs to be held in memory. |
| **Control model** | Step-3 model: trained on top of the baked adapter to do in-context `[alpha | RGB-on-gray]` generation. |
| **Final model / difference model** | `W_base − W_adapter + W_control`. |
| **GateControl** | arXiv 2603.27666 (Liu, Tan, Hu, S. Liu, X. Wang, Mar 2026). Control for linear-attention DiTs like SANA via a gated condition-injection module. Our control method. |
| **BlueJay / Aviary** | The user's data-processing framework. Processors run commands on rows picked by SQL and commit results back to tables. Configured by `data/project.dtab`. On the node it lives at a path the user will give. |
| **Sparrow** | BlueJay's browser viewer for tables and image grids (user's voice transcript said "spyro"). |
| **project.dtab** | BlueJay's project config (tables, processors, queries, views). Ours is `data/project.dtab`. |
| **Babysit** | Watch a long job without stopping: sleep, check, report status, repeat, until the user says stop. Every training run is babysat. |
| **In-context generation** | Condition and target sit in one canvas/token sequence. The condition is clean, and the model learns to fill in the target. |
| **Eval set** | Fixed ≥20 prompts (+ their mattes) and **fixed initial noises**, reused for every model, so comparisons are apples to apples. |
| **Checkpoint 1** | The adapter model works (no artifacts/ugliness, VLM-judged quality ≥ MAGICK). Notify the user at this point. |
| **Journal** | A human-facing, distill.pub-style HTML write-up with images in `journal/`. It covers the presentable parts of the journey. Scrappy notes go in `agents/.scratchpad/journal/`. |
| **Node** | The remote training machine: 8× A100 80GB. |
| **Frenzy / sub-frenzy** | Parallel research subagents (here: Sonnet). Outputs go in `agents/.scratchpad/frenzy/`. |
| **Dump** | A self-contained, portable project folder. All paths are relative to the repo root. |
| **Manifest / concerns** | This file (what the project is) / `agents/concerns.md` (append-only history of how it got here). |

## 3. Experiment Plan

All steps use the **same eval set** (same prompts, same mattes, same noise tensors, same step count). Every step gets a journal entry with image grids.

| Step | What | Output | Pass criterion |
|---|---|---|---|
| 0 | **Curate data with BlueJay** (`data/project.dtab`). Download all of MAGICK. Processors write alpha, RGB-on-gray, quality stats and **precomputed DC-AE latents at 1024²** (the teacher's own VAE) into tables. Eval set = images where sha256(page_id) mod 4000 is 0 (about 35), never trained on. View it all in Sparrow. Run: `data/run_proc_curate.fish`. | BlueJay tables on the node | Spot-check composites in Sparrow. Alpha convention verified. VAE round-trip PSNR. |
| 1 | **Baseline.** Run stock SANA-Sprint (and the teacher) on the eval prompts. Plain prompts, and prompts with "on a gray background". | journal grid | — (reference only) |
| 2 | **Adapter.** Fine-tune Sprint on gray composites + MAGICK captions, text-only. LoRA at max rank that fits on A100 80GB, or full fine-tune. | adapter weights | VLM: no artifacts. Quality comparable to MAGICK. Flat 0.5 gray bg. → **Checkpoint 1, notify user.** |
| 3 | **Control (GateControl-based).** Bake the adapter into the weights. Train matte control: clean matte condition, loss on the generated image only. | control weights | Output silhouette matches the matte (IoU metric + VLM). |
| 4 | **Final (inference only).** `W_base − W_adapter + W_control`. | journal grid | Follows the matte AND looks like base-Sprint quality, not MAGICK quality. |
| 5 | **Distill** (only if training used the teacher) the final model to a 1–4 step Sprint student. | Sprint-speed final model | Same eval set, quality ≈ Step 4. |
| 6 | **Web UI** so people can play with it: upload or draw a matte, type a caption, get RGB-on-gray + RGBA. | `run_server.sh` | Works end-to-end from a browser. |

If a step fails, go back, try again, and journal it.

**Run rules (user, m00222, m00245, m00257):**
- **1024×1024 only** (m00295): Sprint and its teacher are trained at 1024, and one resolution keeps the pipeline simple.
- **Teacher first, distill at the very end** (m00295), after the teacher-based model works.
- **Eval inference is a BlueJay processor** (m00606, m00611): `generate_eval` reads its job (checkpoint, steps, CFG, noise file) from environment variables; `data/run_proc_eval.fish` launches one loop per GPU. Prompts are rows of an inference-set table (m00612). The model stays loaded across batches, and all GPUs stay busy (m00612). Each generation row records checkpoint, noise and seed, so Sparrow can show them. Judging (VLM, metrics write-up) is done by the agent (m00371). The journal is a static page; its images are hard links to the files BlueJay stored.
- **All 8 GPUs at full use** (m00611, m00612), in curation (encoding) as well as training and eval. **Keep net lines to a minimum** (m00612).
- **Every image label states its weight equation** (m00622), e.g. `W_teacher + 0.75·(W_adapter − W_teacher) + ΔW_control + gates, 20 steps` or `W_sprint (stock), 2 steps`. Journal pages use collapsible sections (m00622).
- **Show full subtraction vs not, and partial subtractions** (m00615): every control run is shown at λ ∈ {1, 0.75, 0.5, 0.25, 0} (W_base + λ(W_adapter − W_base) + control delta; λ = 0 is the final model) and as the Sprint transfer. A perfect gray background is not required. **≥50 examples** in the journal: a showcase set of 29 held-out + ≥35 train items, each labeled.
- **Journal images at full resolution** (m00616): every generated image appears at its full 1024×1024 resolution, many images per page, no downscaling and no page-size cap ("The reports should not baby my browser"). Images are hard links to the files BlueJay stored (m00606), so they cost no disk.
- **Big batch sizes** when they fit in GPU memory. Up to **1 hour** of run optimization (throughput, batch size, data loading) is allowed before the real run.
- **Target: something working within 12 hours** of 2026-11-17 03:38 EST, so by about 15:38 EST, if possible.
- **Every training run is babysat.**

## 4. Open Questions (to resolve with user / research)

1. **Sprint student vs teacher.** Decided: train the teacher, distill last (m00295). In diffusers the teacher is a plain rectified-flow (velocity) model: σ = 3u/(1+2u), u = sigmoid(N(0,1)), x_t = (1−σ)x0 + σε, target ε − x0, timestep = σ. TrigFlow is only for distillation. Sample: SanaPipeline + DPM-Solver (flow_prediction, flow_shift 3), 20 steps, CFG 4.5.
2. **LoRA vs full fine-tune.** Decided: adapter = full fine-tune (the limit of "max rank"; 24 GB for weights+grads+AdamW), fp32 master weights, so `W_base − W_adapter + W_control` is exact and nothing needs baking. Control = GateControl LoRA rank 16 + gates on the frozen adapter.
3. **Control mechanics.** Decided: GateControl-based (m00231). Exact layers, gate and rank: from research agent 6 (§5).
4. **Resolution.** Decided: 1024×1024 only (m00295).
5. **The weight-arithmetic step assumes LoRA/full deltas are compatible.** If the adapter is a LoRA, `W_control − W_adapter_baked` is exact in fp32 but lossy in bf16. Keep fp32 master copies for the subtraction.

## 5. Research Findings

From the 2026-11-17 sub-frenzy (5 Sonnet agents). The raw reports are in `agents/.scratchpad/frenzy/`, which is not saved, so the key facts are copied here.

**MAGICK dataset.** 148,193 RGBA PNGs, 1024², 191 GB, straight (not premultiplied) alpha. Path `images/{page_id[:2]}/{page_id}.png`. Index `magick_index.tsv` (read with `quoting=3`); columns `page_id, sim_score, color, nsfw, subject, tag_idk, picked`. Caption = `subject` (short phrase). 3,000 are nsfw. `load_dataset` fails; use `hf download OneOverZero/MAGICK --repo-type dataset --local-dir MAGICK`. Some green spill in hair: filter by `sim_score`.

**Training SANA-Sprint.** No report of a working plain flow-matching LoRA on the Sprint student. The maintainer "never tried" a diffusion loss on Sprint (Sana issue #256). The diffusers script `examples/research_projects/sana/train_sana_sprint_diffusers.py` (sCM+LADD) is merged, full fine-tune only, and inits the student from the teacher. Teacher differs from stock SANA: TrigFlow, dense time embedding, QK RMSNorm. Teachers: `Efficient-Large-Model/Sana_Sprint_{0.6B,1.6B}_1024px_teacher_diffusers`. Inference defaults: 2 steps, `max_timesteps=1.5708`, `intermediate_timesteps=1.3`, guidance 4.5. Fixed noise: pass `latents` of shape (B, 32, H/32, W/32) with `use_resolution_binning=False`. **→ Teacher-first, distill last.**

**AnimateDiff adapter.** LoRA on self- and cross-attention, MSE loss on single frames, base frozen, scale 0 at inference. Evidence is qualitative (paper Fig. 6). Closest precedent for `base − adapter + control`: lllyasviel's ControlNet transfer `Any3 + SD15_control − SD15`; task arithmetic (Ilharco et al.). No direct report of this exact recipe. **Do all weight arithmetic in fp32 from stored originals; round to bf16 once at the end.**

**In-context conditioning on SANA.**
- SANA has **no positional encoding** (position comes from the 3×3 depthwise conv in Mix-FFN). RoPE-shift sequence concat (Kontext, OminiControl, UNO) does not apply. Use a **wide canvas, clean matte on the left**.
- Timestep is one scalar per sample. Per-token timestep is not required: the Kontext pipeline gives clean context tokens the same scalar timestep and slices the output. LTX and Diffusion Forcing use per-token t, clean = 0 (upgrade path).
- **Loss on target (right-half) tokens only**, as in LTX and Kontext.
- GateControl (arXiv 2603.27666): condition control on SANA-1.6B, LoRA rank 16. Plain concat converges slowly on spatially aligned tasks; a 0.09M-param gate speeds it up >10×. **Chosen as our control method (m00231).**
- **GateControl details (agent 6, from the official code https://github.com/Carol-lyh/GateControl, Apache-2.0, no weights):**
  - The condition is a **second token stream**: same VAE, same `patch_embed`, same 20 blocks and weights, its own residuals. No canvas, no positional trick; alignment is learned.
  - Both streams get the same adaLN modulation and the **same timestep** as the noisy latent (condition is clean, not t=0). Text cross-attention and FFN run per stream. Only linear self-attention sees `[cond; x]` (bidirectional).
  - Gate per block: two `Linear(2240, 1)`, ~0.09M params total. `g_x = sigmoid(W_x(LN(x_in)))`, `g_c = sigmoid(W_c(LN(c_in)))`, LN non-affine on the modulated pre-attention input. x stream gets `g_x·attn_x + g_c·attn_c`; cond stream gets its attention ungated. Init zero → 0.5 each (not identity to the base).
  - Base frozen. Trained: LoRA rank 16, alpha 16, on `proj, linear_1, linear_2, linear, to_q, to_k, to_v, to_out.0, conv_point, conv_inverted, proj_out` + gates.
  - Loss: flow matching, target `noise − x0`, logit-normal weighting, target stream only. Condition dropout 0.1 (cond K/V zeroed, gate term ×0), text dropout 0.1.
  - Training: 4× H200, batch 16/GPU (64), bf16, Prodigy lr 1, wd 0.01 (0 on gates), 500 warmup, 10K steps for aligned tasks, 10K images. Inference 20 steps, CFG 4.5; ~2× tokens per forward.
  - Paper: ">10× faster convergence on spatial tasks" (curves, not a table). Canny@512 F1 0.26 vs OminiControl-SANA 0.23. Ablation: gating after cross-attn or plain addition is unstable.
  - Reuse: `train_sana_gatecontrol.py` works on SANA. Deps: diffusers, peft, accelerate, datasets, prodigyopt, transformers, cv2 (diffusers version unpinned). It never passes `guidance`, so Sprint (guidance-embedded, TrigFlow) needs changes.
  - **Fit with our weight arithmetic:** the adapter stays frozen while the control trains, so `W_control − W_adapter` is exactly the control LoRA delta. The final model = base + control LoRA + gates. The gates are new modules, not weight deltas; they are kept as they are.
- SDEdit-style re-pasting on a model not trained with a clean left half is weak (In-Context LoRA). Use re-pasting only as a diagnostic.
- Reported LoRA ranks: 4 (OminiControl) to 512 (UNO); 16 is typical. Steps: 5k (In-Context LoRA) to 100k (EasyControl); aligned tasks ~50k (OminiControl).
- Agent's suggested first design: 1024×512 canvas (16×32 latent tokens) for fast iteration, matte replicated to 3 channels, each half VAE-encoded separately, shared scalar t, rank-16 LoRA on all linears. Seam behavior through the Mix-FFN conv is unverified.

**Step 5 plan (research agent, 2026-11-17 6:40 AM; full report was in `agents/.scratchpad/frenzy/distill_plan.md`).** Teacher and Sprint student have the same architecture except `guidance_embeds` (student only) and `timestep_scale`. Both transformers take the flow σ ∈ [0,1] as `timestep` and predict velocity; TrigFlow lives only in the Sprint pipeline. Key rename: teacher `time_embed.emb.timestep_embedder.*` = student `time_embed.timestep_embedder.*`. Student − teacher weight distance is 1.54% (largest: timestep MLP `linear_2` 28%, `proj_out` 6%). Plan: (b) **transfer** — merge the control LoRA delta + gates onto the stock Sprint student in fp32 (remap keys; merge, do not attach PEFT, which would also hit `guidance_embedder`); eval at 1/2/4 steps. (b′) if the matte is followed too weakly: train only the control LoRA + gates on the frozen student against teacher-final outputs (~16–24 GPU-h). Full sCM+LADD (diffusers script) only as a last resort: its discriminator hooks `block.forward`, which the gated path never calls. Main risk: Sprint sets the layout in one pass at σ = 1, and its guidance is in the timestep embedding that the gates never saw. **(b) is implemented as an eval model type:** an `eval_models` row with `pipeline sprint` whose `checkpoint` is a control step folder = student + merged control LoRA + gates. Tested on overfit_test (06:47 AM): IoU 0.84/0.95/0.94 at 1/2/4 steps against 0.82 for the teacher control, so the main risk did not happen there (numbers and observations are in concerns.md).

**Prior art and eval.** Zippo (arXiv 2403.11077, matte latent concat, SD2.1) is the closest. No published model trained on MAGICK. Eval: ≥20 held-out MAGICK items × fixed seeds; metrics = exterior gray purity, IoU vs matte, edge-band L1 / color fringe, CLIP/SigLIP, VLM rubric + A/B, montage. FID is meaningless at this size; use KID if needed.

## 6. Hardware & Paths

- Planning machine: Mac (no CUDA). No training here.
- Training node: **8× A100 80GB** (user, m00040–m00041).
- All paths relative to repo root. Model weights are stored on FSx in `models/` (durable) and **localized automatically** (m00613, m00614): the loading code copies them to node-local `/models/sana-sprint-alpha-incontext/<path>` the first time a machine loads them (temp dir + atomic rename, so 8 loops at once are safe). Never a manual step. Pattern from MulticamDailies `processors/muse/serve.sh`.

## 7. File Structure

**Top-level rule (user, m00049):** the top level stays very clean. Anyone who opens the folder should know how to use it without reading the README. Only good, proven source goes to the top level. The bulk of the AI's work goes in `agents/.scratchpad/`, which is **not saved**. Everything runs through `uv run` (a `pyproject.toml` at the root, like any other dump).

Planned top level (target state; only `AGENTS.md`, `agents/`, `journal/` exist now):

```text
run_server.sh      # launch the web UI (always visible at top level)
run_sparrow.fish   # Sparrow viewer of the BlueJay tables, port 26400 (m00630)
run_diffusion.sh   # CLI inference: matte + caption -> RGB-on-gray / RGBA
pyproject.toml     # uv project; all deps
README.md
src/               # proven source: data curation, training, inference, weight arithmetic, server
models/            # model weights (gitignored; large)
resources/         # eval prompts, eval mattes, fixed noise tensors, small assets
journal/           # human-facing distill.pub-style journal
agents/            # manifest, concerns, todo, transcripts, .scratchpad/
AGENTS.md
```

Training entry points (e.g. `train_adapter.sh`, `train_control.sh`) may also live at the top level, but only once they are proven.

`data/` (user, m00196) holds the BlueJay project: `data/project.dtab` (the graph: tables, processors, queries, Sparrow views) and `data/processors/*.py` (one `uv run --script` per processor, except `generate_eval.py`, which runs in the root uv project to share `src/` and torch with training). Only `data/lance/` (the database, which holds the PNGs as blobs, about 191 GB plus derived columns) is gitignored. Fish scripts follow MulticamDailies (m00285, m00295): `data/aviary.sh` runs Aviary from the wheel that `setup.sh` builds out of the `data/Aviary` checkout (re-run `setup.sh` after pulling it), `data/uv.env` puts uv's cache on /localize, `data/run_proc_curate.fish` runs the pipeline (stages run at the same time; tmux session `curate`), `data/run_proc_eval.fish [GPU...]` runs eval inference, `data/run_sparrow.fish` opens the viewer, `data/recipes/loops/until_empty.fish` runs one processor until it has no work, `stage.fish` waits for the rows of the stage before it.

| Path | Purpose |
|---|---|
| `AGENTS.md` | Agent entry point. Says to read `agents/`. |
| `agents/manifest.md` | This file. |
| `agents/concerns.md` | Append-only history, mistakes, progress log. |
| `agents/todo.md` | TODO list, synced with the agent's list. |
| `agents/transcripts.md` | Verbatim (courtroom-style) transcripts of user inputs. |
| `agents/.scratchpad/` | AI workspace (frenzy output, logs, scratch journal). |
| `journal/` | Human-facing distill.pub-style journal (HTML + images). |
| `data/project.dtab` | BlueJay graph, MulticamDailies house style. Tables: `import_jobs` (one seeded row), `magick_samples` (one row per image). Processors: `import_index` (TSV → rows) → `download_image` (PNG blob) → `split_alpha` (matte + RGB on 0.5 gray + stats) → `encode_latents` (DC-AE latents (32,32,32) at 1024 + PSNR). View: `view_dataset`. |
| `data/processors/` | One script per processor (not yet written). |

## 8. Requirements (from user)

- Journal the real journey for human consumption (distill.pub style, with images). Subagents may write it.
- Same noises + same ≥20 prompts across all observations.
- Use a VLM to check quality against MAGICK.
- Notify the user at Checkpoint 1.
- LoRA rank: as high as possible without OOM.
- Data processing: BlueJay (m00196). Precompute latents and conditions as BlueJay tables; visualize them in Sparrow.
- Control method: GateControl-based (m00231).
- 1024 only (m00295); teacher first, distill last (m00295); big batches; ≤1 h run optimization; aim for something working in 12 h (m00245).
- dtab house style copied from MulticamDailies: annotated TYPE/DESCRIPTION/EXAMPLE, annotated SQL, alignment (m00295). Eval inference is a BlueJay processor; judging is done by the agent (m00371, m00606, m00611).
- Babysit every training run (m00257).
- Transcribe all user inputs (`agents/transcripts.md`).
- Planning happens here. Execution happens on the node via the GitHub repo.
- Node session (m00600–m00605): Aviary checkout in `data/`, called only through `data/aviary.sh`; bulldog mode, 13 h; fork mode with plain subagents (mostly Sonnet, Opus for hard parts); babysit once runs start; max 6 phone notifications, signed "Sana"; keep an agenda.
- **User transcripts are part of this manifest.** They live in `agents/transcripts.md` (verbatim, courtroom style), which is §11 of this manifest kept in its own file because it is long.

## 9. Semantic Bindings

### dtab-rules
The DTAB formatting rules. Change all members together.
- `AGENTS.md` section "DTAB formatting"
- `data/project.dtab` block "RULES FOR MAINTAINING THIS DTAB"
- `agents/dtab_format.py` module docstring (the checker's rules)

### web-model-contract
The browser app's model folder format (manifest §12). Change all members together.
- `agents/manifest.md` §12
- `web/src/models.js` header comment (app schema) and `normalizeManifest` (exporter format → app schema)
- `src/sana_alpha/web_export.py` `export` (graph_specs, write_manifest)

## 10. Success Criteria

The final model (`W_base − W_adapter + W_control`, full subtraction, λ = 0) generates an image that (a) matches the input matte's silhouette and (b) looks like stock SANA-Sprint quality rather than MAGICK's 2023 quality. A flat 0.5 gray background is NOT required (m00628: "It's okay if it does, it's okay if it doesn't ... The whole point is we're subtracting the adapter"); partial subtractions (λ > 0) are shown for curiosity (m00615). Judged on the fixed eval set and the showcase set against Steps 1–3. Because IoU here counts "object vs 0.5 gray", it understates silhouette match when the background is not gray; judge silhouette visually (or with a background-agnostic metric).

## 12. WebGPU browser app (m00624, m00625)

Goal: the one-step matte model runs in a browser on WebGPU, on a 16–32 GB MacBook (integrated GPU). Upload or draw a silhouette (brush, text, draggable shapes), type a caption, generate. Pick among model versions in the UI; each downloads with a loading bar and is cached in the browser (Cache Storage / OPFS) so the second load is fast. Gemma prompt embeddings are cached too (m00625), so mask edits never re-run the text encoder. Real-time toggle (default off: press Generate). Simplified Swiss style, full dark and light modes, Jost font (thin for titles, regular for text), loading bars with speeds.

Model contract (SEMANTIC BINDING: web-model-contract), one folder per version under `models/web/<version>/` (gitignored, served by the app's server) with `manifest.json`:
Exact form (written by `src/sana_alpha/web_export.py`; all graph I/O float32 except token ids; ONNX IR 13, opset 21, weights in `<file>.onnx.data`; `manifest.json` has flat `files: {name: bytes}`, `graphs` (I/O names, shapes, dtypes), `sampler`, `vae`, `text_encoder`, `quantization`):
- `text_encoder` (+ `tokenizer.json`, `tokenizer_config.json`): Gemma-2 2B fp16, `input_ids`, `attention_mask` (1, T) int64 → `last_hidden_state` (1, T, 2304), T dynamic. JS applies SANA's template (instruction + caption.lower().strip(), right-pad to len(instruction tokens) + 298) and keeps token 0 + the last 299 → (1, 300, 2304) + mask (1, 300).
- `vae_encoder`: `image` = matte copied to 3 channels, 2·m − 1, (1, 3, 1024, 1024) → `latent` (1, 32, 32, 32), × 0.41407 already applied inside.
- `transformer`: Sprint student + control delta (+ λ adapter delta) + gates. Inputs `latents` (1, 32, 32, 32) = standard-normal noise, `cond_latents` = vae_encoder output, `timestep` (1,) = TrigFlow angle s = 1.5708, `guidance` (1,) = 4.5 (× 0.1 inside), `prompt_embeds` (1, 300, 2304), `prompt_mask` (1, 300) float 1/0 → `denoised` (1, 32, 32, 32). All Sprint math is inside: t = sin s/(cos s + sin s), n = √(t² + (1−t)²), v = model(n·latents, t), F = ((1−2t)·n·latents + (1−2t+2t²)·v)/n, denoised = cos s·latents − sin s·F. More steps: next latents = cos s'·denoised + sin s'·ε.
- `vae_decoder`: `latent` = denoised (÷ 0.41407 inside) → `image` (1, 3, 1024, 1024) RGB in [0, 1].
- fp16 everywhere, with two float32 spots (cross-attention `to_q`, `caption_projection.linear_2`) that overflow fp16; quantized transformers live in sibling folders `<version>-int4` / `-int8` whose other files are hard links. Quantization details per file are in `manifest.json` `quantization`.
- **Quantize properly** (m00629): no naive round-to-nearest int4. Use SANA's own scheme if one exists (MIT Han Lab's SVDQuant / DeepCompressor / Nunchaku has SANA 4-bit models), adapted to what ONNX Runtime Web can run; otherwise a researched, calibrated method. Every quantized variant is judged against fp16 on the eval set.

**GitHub Pages deploy (m00633, m00634).** Once the browser app works with a real model, publish it on GitHub Pages; heavy model files go in a separate branch (not LFS), sharded below GitHub's 100 MB per-file limit only if needed, and the app fetches them by URL. Never read the token in the git remote. Notify the user once it is live. Blocker found 1:40 PM: the repo RyannDaGreat/sana-sprint-alpha-incontext is private (the unauthenticated API returns 404), so Pages and raw.githubusercontent.com model URLs are not reachable by browsers; the user chose (m00635) to make this repo public. Licenses (checked): SANA-Sprint 1.6B is Apache-2.0; its Gemma-2-2B text encoder is under the Gemma Terms of Use (redistribution allowed with the terms and use policy attached); our control and adapter deltas are trained on MAGICK (CC BY-NC 4.0), so the published demo should say non-commercial research use. **Journal is live (m00637, 2026-11-17):** https://ryanndagreat.github.io/sana-sprint-alpha-incontext/ is the `gh-pages` branch (orphan), built and pushed by `src/sana_alpha/publish_pages.py` (worktree in `agents/.scratchpad/gh-pages`, never touches main's tree). The journal PNGs are 2.5 GB, over the 1 GB Pages limit, so the Pages copy is full-resolution WebP q95 (684 MB, same pixel sizes; each page's footer says so); the PNG originals stay in BlueJay. The built app is at `webgpu/`; `models/web/index.json` there is an empty placeholder until the `models` branch exists.
