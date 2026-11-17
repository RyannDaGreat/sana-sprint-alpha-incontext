# Manifest — SANA-Sprint In-Context Alpha → RGB-on-Gray

> STATUS: **planning repo.** No training happens here. This repo is pushed to GitHub and cloned onto the training node (8× A100 80GB).
> The user has not yet confirmed the plan summary (see "Distilled Plan"). Do not start Step 1+ until they confirm.

## 1. Goal (read this first)

**Problem.** We want a fast text-to-image model that takes an **alpha matte** (a grayscale image: white = opaque, black = transparent) plus a caption, and generates the **foreground RGB, composited on a flat 50% gray background** (RGB = 128,128,128 / 0.5 in [0,1]), so that the generated object exactly fills the matte. Combined with the input matte, that output gives a full RGBA image (the gray background is known, so the foreground color can be recovered by un-compositing).

**Base model.** SANA-Sprint (NVlabs, arXiv 2503.09641), a 1–4 step distilled DiT. We pick it because it is very fast, so many eval runs are cheap.

**Data.** MAGICK (Burgert et al., CVPR; https://ryanndagreat.github.io/MAGICK/, https://huggingface.co/datasets/OneOverZero/MAGICK). This is a large captioned RGBA dataset. A 2023-era diffusion model generated it, so the image quality and captions are dated ("very 2023 quality").

**Method: the AnimateDiff domain-adapter trick, used for in-context control.**
1. **Adapter model**: fine-tune SANA-Sprint (LoRA at the highest rank that fits VRAM, or full fine-tune) on MAGICK images composited on gray, with MAGICK captions. This absorbs the *domain*: gray backgrounds, MAGICK's dated look.
2. **Control model**: bake the adapter into the weights. On top of that, train in-context generation: the canvas is `[alpha matte | RGB-on-gray]` side by side. The left half is clean (no noise). The model denoises the right half.
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
| 0 | **Curate data.** Download all of MAGICK. Write plain subfolders: `alpha/` (matte), `rgb_on_gray/` (composite), `captions/`. No framework, just a simple script. Pick the eval set: ≥20 held-out prompts + mattes + saved noise tensors. | `data/` on the node | Spot-check composites. Alpha convention verified. |
| 1 | **Baseline.** Run stock SANA-Sprint (and the teacher) on the eval prompts. Plain prompts, and prompts with "on a gray background". | journal grid | — (reference only) |
| 2 | **Adapter.** Fine-tune Sprint on gray composites + MAGICK captions, text-only. LoRA at max rank that fits on A100 80GB, or full fine-tune. | adapter weights | VLM: no artifacts. Quality comparable to MAGICK. Flat 0.5 gray bg. → **Checkpoint 1, notify user.** |
| 3 | **Control.** Bake the adapter into the weights. Train in-context `[alpha | RGB-on-gray]`: left half clean, loss on the right half. | control weights | Output silhouette matches the matte (IoU metric + VLM). |
| 4 | **Final (inference only).** `W_base − W_adapter + W_control`. | journal grid | Follows the matte AND looks like base-Sprint quality, not MAGICK quality. |

If a step fails, go back, try again, and journal it.

## 4. Open Questions (to resolve with user / research)

1. **Sprint student vs teacher.** Train directly on the Sprint student (fast eval, but a plain diffusion loss may break few-step ability)? Or train on the teacher and distill to Sprint later? User said (m00018, paraphrased): "maybe we could just start with the teacher model and distill it a little later."
2. **LoRA vs full fine-tune** for the adapter and the control. With 8× A100 80GB, full fine-tune of 1.6B is feasible.
3. **In-context mechanics on SANA.** Wide canvas (2048×1024) vs sequence concat. Clean left half via per-token timestep 0, vs re-pasting the clean latent each step. Masked loss. Positional encoding. (Research pending; LTX IC-LoRA was mentioned by the user.)
4. **Resolution.** Sprint is 1024px. Use 1024×1024 per half (2048-wide canvas) or 512 per half?
5. **The weight-arithmetic step assumes LoRA/full deltas are compatible.** If the adapter is a LoRA, `W_control − W_adapter_baked` is exact in fp32 but lossy in bf16. Keep fp32 master copies for the subtraction.

## 5. Research Findings

(Filled in from the sub-frenzy: `agents/.scratchpad/frenzy/agent_*.md`.)

## 6. Hardware & Paths

- Planning machine: Mac (no CUDA). No training here.
- Training node: **8× A100 80GB** (user, m00040–m00041).
- All paths relative to repo root. Base models may live under `/models/` (system-wide convention).

## 7. File Structure

| Path | Purpose |
|---|---|
| `AGENTS.md` | Agent entry point. Says to read `agents/`. |
| `agents/manifest.md` | This file. |
| `agents/concerns.md` | Append-only history, mistakes, progress log. |
| `agents/todo.md` | TODO list, synced with the agent's list. |
| `agents/transcripts.md` | Verbatim (courtroom-style) transcripts of user inputs. |
| `agents/.scratchpad/` | AI workspace (frenzy output, logs, scratch journal). |
| `journal/` | Human-facing distill.pub-style journal (HTML + images). |

## 8. Requirements (from user)

- Journal the real journey for human consumption (distill.pub style, with images). Subagents may write it.
- Same noises + same ≥20 prompts across all observations.
- Use a VLM to check quality against MAGICK.
- Notify the user at Checkpoint 1.
- LoRA rank: as high as possible without OOM.
- Data processing: simple scripts into subfolders, no framework.
- Transcribe all user inputs (`agents/transcripts.md`).
- Planning happens here. Execution happens on the node via the GitHub repo.

## 9. Semantic Bindings

None yet.

## 10. Success Criteria

The final model (`W_base − W_adapter + W_control`) generates the right-half image so that it (a) matches the input matte's silhouette, (b) has a flat 0.5 gray background, and (c) looks like stock SANA-Sprint quality rather than MAGICK's 2023 quality. It is judged on the fixed eval set against Steps 1–3.
