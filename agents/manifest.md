# Manifest — SANA-Sprint In-Context Alpha → RGB-on-Gray

> STATUS: **planning repo.** No training happens here. This repo is pushed to GitHub and cloned onto the training node (8× A100 80GB).
> The user has not yet confirmed the plan summary (see "Distilled Plan"). Do not start Step 1+ until they confirm.

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
| 0 | **Curate data with BlueJay** (`data/project.dtab`). Download all of MAGICK. Processors write alpha, RGB-on-gray, quality stats and **precomputed DC-AE latents at 512²** into tables. Pick the eval set: ≥20 held-out prompts + mattes + saved noise tensors. View it all in Sparrow. | BlueJay tables on the node | Spot-check composites in Sparrow. Alpha convention verified. VAE round-trip PSNR. |
| 1 | **Baseline.** Run stock SANA-Sprint (and the teacher) on the eval prompts. Plain prompts, and prompts with "on a gray background". | journal grid | — (reference only) |
| 2 | **Adapter.** Fine-tune Sprint on gray composites + MAGICK captions, text-only. LoRA at max rank that fits on A100 80GB, or full fine-tune. | adapter weights | VLM: no artifacts. Quality comparable to MAGICK. Flat 0.5 gray bg. → **Checkpoint 1, notify user.** |
| 3 | **Control (GateControl-based).** Bake the adapter into the weights. Train matte control: clean matte condition, loss on the generated image only. | control weights | Output silhouette matches the matte (IoU metric + VLM). |
| 4 | **Final (inference only).** `W_base − W_adapter + W_control`. | journal grid | Follows the matte AND looks like base-Sprint quality, not MAGICK quality. |
| 5 | **Distill** (only if training used the teacher) the final model to a 1–4 step Sprint student. | Sprint-speed final model | Same eval set, quality ≈ Step 4. |
| 6 | **Web UI** so people can play with it: upload or draw a matte, type a caption, get RGB-on-gray + RGBA. | `run_server.sh` | Works end-to-end from a browser. |

If a step fails, go back, try again, and journal it.

**Run rules (user, m00222, m00245, m00257):**
- **512×512 end to end first** (all steps 0–4, eval included). Upgrade to 1024 only after that works.
- **Big batch sizes** when they fit in GPU memory. Up to **1 hour** of run optimization (throughput, batch size, data loading) is allowed before the real run.
- **Target: something working within 12 hours** of 2026-11-17 03:38 EST, so by about 15:38 EST, if possible.
- **Every training run is babysat.**

## 4. Open Questions (to resolve with user / research)

1. **Sprint student vs teacher.** Train directly on the Sprint student (fast eval, but a plain diffusion loss may break few-step ability)? Or train on the teacher and distill to Sprint later? User said (m00018, paraphrased): "maybe we could just start with the teacher model and distill it a little later."
2. **LoRA vs full fine-tune** for the adapter and the control. With 8× A100 80GB, full fine-tune of 1.6B is feasible.
3. **Control mechanics.** Decided: GateControl-based (m00231). Exact layers, gate and rank: from research agent 6 (§5).
4. **Resolution.** Decided: 512×512 end to end first, then upgrade (m00222, m00245).
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

**Prior art and eval.** Zippo (arXiv 2403.11077, matte latent concat, SD2.1) is the closest. No published model trained on MAGICK. Eval: ≥20 held-out MAGICK items × fixed seeds; metrics = exterior gray purity, IoU vs matte, edge-band L1 / color fringe, CLIP/SigLIP, VLM rubric + A/B, montage. FID is meaningless at this size; use KID if needed.

## 6. Hardware & Paths

- Planning machine: Mac (no CUDA). No training here.
- Training node: **8× A100 80GB** (user, m00040–m00041).
- All paths relative to repo root. Base models may live under `/models/` (system-wide convention).

## 7. File Structure

**Top-level rule (user, m00049):** the top level stays very clean. Anyone who opens the folder should know how to use it without reading the README. Only good, proven source goes to the top level. The bulk of the AI's work goes in `agents/.scratchpad/`, which is **not saved**. Everything runs through `uv run` (a `pyproject.toml` at the root, like any other dump).

Planned top level (target state; only `AGENTS.md`, `agents/`, `journal/` exist now):

```text
run_server.sh      # launch the web UI (always visible at top level)
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

`data/` (user, m00196) holds the BlueJay project: `data/project.dtab` (the graph: tables, processors, queries, Sparrow views) and `data/processors/*.py` (one `uv run --script` per processor). Only `data/lance/` (the database) and `data/magick_raw/` (the 191 GB download) are gitignored.

| Path | Purpose |
|---|---|
| `AGENTS.md` | Agent entry point. Says to read `agents/`. |
| `agents/manifest.md` | This file. |
| `agents/concerns.md` | Append-only history, mistakes, progress log. |
| `agents/todo.md` | TODO list, synced with the agent's list. |
| `agents/transcripts.md` | Verbatim (courtroom-style) transcripts of user inputs. |
| `agents/.scratchpad/` | AI workspace (frenzy output, logs, scratch journal). |
| `journal/` | Human-facing distill.pub-style journal (HTML + images). |
| `data/project.dtab` | BlueJay graph. Tables: `import_jobs`, `magick_samples`, `eval_noise`, `checkpoints`, `generations`. Processors: `download_magick` → `split_alpha` → `encode_latents`; `make_eval_noise`; `plan_generations` → `generate` → `score_generation`, `vlm_judge`. Views: `view_dataset`, `view_eval_grid`, `view_checkpoint_scores`. |
| `data/processors/` | One script per processor (not yet written). |

## 8. Requirements (from user)

- Journal the real journey for human consumption (distill.pub style, with images). Subagents may write it.
- Same noises + same ≥20 prompts across all observations.
- Use a VLM to check quality against MAGICK.
- Notify the user at Checkpoint 1.
- LoRA rank: as high as possible without OOM.
- Data processing: BlueJay (m00196). Precompute latents and conditions as BlueJay tables; visualize them in Sparrow.
- Control method: GateControl-based (m00231).
- 512×512 end to end first; big batches; ≤1 h run optimization; aim for something working in 12 h (m00222, m00245).
- Babysit every training run (m00257).
- Transcribe all user inputs (`agents/transcripts.md`).
- Planning happens here. Execution happens on the node via the GitHub repo.

## 9. Semantic Bindings

None yet.

## 10. Success Criteria

The final model (`W_base − W_adapter + W_control`) generates the right-half image so that it (a) matches the input matte's silhouette, (b) has a flat 0.5 gray background, and (c) looks like stock SANA-Sprint quality rather than MAGICK's 2023 quality. It is judged on the fixed eval set against Steps 1–3.
