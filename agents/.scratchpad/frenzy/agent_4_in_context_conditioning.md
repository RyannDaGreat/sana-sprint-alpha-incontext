# Agent 4: In-context (token-concat) conditioning of DiTs, applied to SANA / SANA-Sprint alpha-matte -> RGB-on-gray

Method: web research only (WebSearch + WebFetch of arXiv HTML / docs). Items marked **[UNVERIFIED]** were not confirmed from a primary source in this session.

## 1. Facts about SANA that drive the design

- **No positional encoding (NoPE).** SANA paper: "We are surprised that we can remove Positional Embedding without any loss in performance"; position info comes from the 3x3 depthwise conv in Mix-FFN. (https://arxiv.org/html/2410.10629v3 , summary via search; paper text quoted in search result.)
  - Consequence: there are NO position ids to shift. "Sequence concat with shifted RoPE ids" (Kontext/OminiControl/UNO) does not apply. The only sensible geometry is **spatial concat on a wider canvas**, where the left/right identity is carried by the content + the conv seam + (optionally) a learned signal.
  - Mix-FFN 3x3 depthwise conv sees across the seam: left clean tokens leak into right-edge tokens at every layer. This is a feature (spatial locality across the seam), but note the conv is zero-padded at image borders during base pretraining; at the seam it now sees real neighbors. LoRA must learn this (cheap).
- **Attention is ReLU linear attention**, O(N). Bidirectional, global. Every right token sees a global KV summary that includes the clean left tokens, and left tokens also query that same summary (so noise in the right half leaks into left-half features unless masked; see section 5).
- **DC-AE f32c32, patch 1**: 1 token = 32x32 px. A 1024x512 canvas = 32x16 = 512 tokens; 512x512 per half = 16x16 = 256 tokens each. Latent grid is cheap, so the wide canvas costs little.
- **Timestep is a single scalar per sample** (SANA-Sprint paper: conditioning is scalar t, CFG scale added "as an additional condition to the time embedding"; Sprint changes c_noise(t) from 1000t to t, "dense time-embedding"). Per-token timestep is NOT built in. Whether SANA's AdaLN-single modulation (t_block) broadcasts over tokens trivially is my recollection of the code **[UNVERIFIED: read Sana/diffusion/model/nets/sana_blocks.py]**. Source: https://arxiv.org/html/2503.09641
- **SANA-Sprint** = sCM (TrigFlow) + LADD hybrid, 1-4 steps, unified model; timestep schedule for 2 steps [arctan(200/0.5), 1.3, 0], 4 steps [arctan(200/0.5), 1.3, 1.1, 0.6, 0]; max-time t=pi/2. Training of the distilled model: 32 A100, lr 2e-6, batch 512, 20k iters. Linear attention kept because PyTorch has no Flash-Attn JVP kernel (sCM needs JVP). Paper says nothing about inpainting / LoRA / mixed timesteps. A SANA-Sprint-ControlNet exists (HED scribble, 1024^2; teacher SANA-ControlNet trained first, then distilled). https://arxiv.org/html/2503.09641
- **SANA ControlNet** (NVlabs): ControlNet-style transformer copy, HED only in docs, training details "coming soon"; no mention of in-context/concat. https://github.com/NVlabs/Sana/blob/main/asset/docs/sana_controlnet.md
- **Directly relevant SANA result: GateControl** ("Gated Condition Injection without Multimodal Attention: Towards Controllable Linear-Attention Transformers", Liu, Tan, Hu, Liu, Wang, arXiv 2603.27666, https://arxiv.org/html/2603.27666v1). Same group as OminiControl.
  - Uses SANA-1.0 1.6B, **token concat [X; C_T; C_I] through shared backbone, LoRA rank 16 (~18.9M params)**, Prodigy lr=1, wd 0.01, 4xH200 batch 16/GPU; spatial tasks 10K images from Text-to-Image-2M for 10K steps; subject task Subjects200K-1024 for 20K steps.
  - Finding: plain concat in linear attention WORKS ("bidirectional linear attention is sufficient for condition injection") but **converges much slower on spatially aligned tasks** (FID 22.6 / SSIM 0.36 without gate vs 19.0 / 0.42 with). They add a 0.09M-param token-wise sigmoid gate after linear attention; gated model at 1k steps beats ungated at 10k steps on Canny.
  - Paper does NOT discuss positional encoding or condition timestep (stated by the fetch; consistent with NoPE). Compute/latency of longer sequences not quantified.
  - Caveat: their condition is spatially aligned but injected as a separate token block (concat in sequence, no spatial adjacency), NOT side-by-side canvas; so their slow convergence may or may not transfer. Our alpha->RGB task is also spatially aligned (matte silhouette = foreground silhouette), so expect the same slow-convergence risk.

## 2. Survey table

| Method | Concat type | Position handling | Condition timestep / noise | Loss | LoRA rank | Data / steps |
|---|---|---|---|---|---|---|
| **LTX-Video IC-LoRA** (Lightricks; https://docs.ltx.io/open-source-model/ltx-trainer/training-modes , LTX-Video paper https://arxiv.org/pdf/2501.00103) | Sequence concat of pre-encoded clean reference latents appended to target seq; bidirectional self-attn | Same grid as target ("overlap") in classic IC-LoRA per community notes; scale-adjusted RoPE when reference is downscaled (`--reference-downscale-factor`). AVControl paper (https://arxiv.org/html/2603.24793v1) states LTX IC-LoRA does NOT use RoPE offsets, unlike Kontext | **Clean, timestep 0.** Needs per-token timestep: LTX-Video paper says it relaxes the DiT single-timestep restriction, "conditioning tokens can have diffusion timesteps t_c = 0". Trainer: fully-masked tokens get clean latents + timestep 0 | Reference tokens **excluded from loss** | Not on the trainer page **[UNVERIFIED rank/steps/dataset]** | n/a from fetched sources |
| **In-Context LoRA** (Huang et al., https://arxiv.org/html/2410.23775 , repo https://github.com/ali-vilab/In-Context-LoRA) | **Spatial concat**: panels merged into one big image (2-4 panels, e.g. [LEFT]/[RIGHT] tags in a joint caption); "approximately equivalent to token concat" | Plain FLUX RoPE over the merged canvas, no change | Training: **no special handling**, whole canvas noised and denoised (generation of the full set). Conditioning at inference only via training-free **SDEdit/inpaint** (mask panels to generate); authors say it "often fails to preserve identity" due to one-way vs two-way dependency; suggest trainable inpainting as future work | Not masked (whole composite) | **16** | 20-100 sets per task, **5,000 steps**, batch 4, 1 A100, FLUX.1-dev; lr/resolution not given |
| **OminiControl** (Tan et al., https://arxiv.org/html/2411.15098v3) | Sequence concat [C_T; X; C_I], full bidirectional MM-attention; no hard mask | Aligned tasks: condition reuses the same (i,j) as image tokens. Non-aligned: shifted by fixed offset (value not given in paper; EasyControl says OminiControl-style shift is 64 on height) | **Not stated** in paper | **Not stated** | default **4** (ablation 1-16; 16 best FID); 14.5M params; LoRA scale 0 on non-condition tokens | Subjects200K (111,767 pairs kept of 211k). Aligned tasks 50,000 iters on FLUX.1-dev, subject 15,000 on schnell; batch 1x8 accum, Prodigy, 2xH100 |
| **FLUX.1 Kontext** (https://arxiv.org/html/2506.15742v2) | Sequence concat of context tokens after target tokens; tested channel concat and found it "worse" (no ablation detail) | 3D RoPE: target (0,h,w), context (1,h,w), the offset as "virtual time step"; same h/w grid | Paper silent. **diffusers code** (https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py): clean image latents are concatenated, and the **same scalar timestep t** is used for all tokens (no per-token timestep); output is sliced `noise_pred[:, :latents.size(1)]` | Implicitly target-only (output sliced at inference; training code not public **[UNVERIFIED]**) | Full finetune, no LoRA | "millions of pairs", step counts not given |
| **EasyControl** (https://arxiv.org/html/2503.07027) | Sequence concat; condition branch isolated | Spatial: position-aware interpolation of RoPE (condition resized to 512^2 and positions scaled to match the generated grid). Subject: fixed shift dh=64 | Condition K/V computed once, no t argument ("agnostic to denoising timesteps"); condition effectively clean/timestep-independent; modulation details not given | Flow-matching on image only | **not given** (r << d), ~15M params/condition, LoRA only on condition-token Q/K/V | MultiGen-20M, Subjects200K, LAION-Face subset; **100k steps**/condition, 4xA100 batch 1, lr 1e-4; **causal block mask: condition tokens attend only to themselves** (-inf for cond->noise/text), enabling KV cache (~16.3s vs 38.9s at 1024^2) |
| **UNO** (https://arxiv.org/html/2504.02160) | Sequence concat [text; z_t; z_ref...] | UnoPE: ref indices start at max W/H of the noisy latent (diagonal offset); diagonal beat none/width-only/height-only in DINO ablation | References are clean noise-free tokens; timestep treatment not stated | Not stated (presumably standard) | **512** (sweep 4-512, gains flatten after ~128) | 230k single + 15k multi-subject synthetic pairs; lr 1e-5, batch 16, 8xA100, 5k steps stage I + 5k stage II |
| **VisualCloze** (https://arxiv.org/html/2504.07960, extra) | Spatial grid concat, infill (FLUX.1-Fill) | Grid RoPE | Context unmasked in infill input (clean) | Infill objective over masked cell | **256** | Graph200K, 20k iters, batch 64, 8xA100, lr 1e-4 |
| **Diffusion Forcing / DFoT** (https://arxiv.org/html/2407.01392v4 , https://arxiv.org/html/2502.06764v1) | Frames in one sequence | n/a | **Independent noise level per token/frame**; clean = level 0 (zero noise = unmasked). DFoT applies AdaLN **per token** with that frame's noise level; can fine-tune a full-sequence model into history-conditional one by varying per-frame noise | Standard, over noised tokens | n/a (video from-scratch/FT) | n/a |

Other: Edit Transfer (https://arxiv.org/html/2503.13327v1) uses 4-panel composite + LoRA; SkyReels-V2 DF transformer fine-tuned from full-sequence model (https://arxiv.org/pdf/2504.13074). 360Anything shifts condition time index (1 for images, 0.1 for video) per Kontext (search snippet only).

### Cross-cutting takeaways
1. **Per-token timestep is NOT required** for clean context to work. Kontext (the strongest production example) feeds clean context tokens with the SAME scalar t as the noisy target. LTX, DFoT, and the LTX trainer do use per-token timesteps (clean = 0). EasyControl avoids timestep entirely by isolating the condition branch.
2. **Loss is on target tokens only** wherever stated (LTX explicitly; Kontext implicitly; IC-LoRA is the exception: full canvas, no mask).
3. **LoRA rank spans 4 (Omini) to 512 (UNO)**; 16 is the most common choice at small data (IC-LoRA, GateControl). For a task as simple as matte->RGB, 16-64 is a reasonable start; no source gives evidence on this specific task.
4. **Steps/data**: tiny (20-100 sets, 5k steps) for IC-LoRA style where the model only has to follow a layout; 10k-100k steps for faithful spatial control (Omini 50k, Easy 100k, GateControl 10k with gate).
5. **Slow convergence on spatially aligned tasks** is repeatedly reported (OminiControl 50k vs 15k; GateControl "much more slowly"). Our task is aligned, so budget for this.
6. Everyone finds **bidirectional attention between condition and target** necessary for good control ("letting condition influence the entire stack is crucial", OminiControl).

## 3. Does "keep left clean, re-paste each step" (inpainting-style) work?

- Evidence for/against: IC-LoRA tried exactly this (training-free SDEdit on a model trained with all panels noised) and reports weak identity preservation. That model, however, was trained with noisy panels, so the clean-left vs noisy-left mismatch is the cause the authors name (one-way dependency). It does not test a model TRAINED with clean left + loss on right.
- Training with the left clean (timestep-consistent) and inference with the left clean is exactly the Kontext / LTX recipe. So the pure inference-time paste on an untrained model is not recommended; the trained version is the known-good design.
- Within a trained model, whether to re-paste: if the left half is clean at every step in both train and inference (no re-noising), you are just conditioning, no repaste step needed; the model's left output is discarded and the real condition is concatenated again every step. This is the Kontext implementation (`latent_model_input = cat([latents, image_latents])`, slice output).
- Alternative RePaint/inpaint style: at each step, set left = alpha * cos t + noise * sin t (i.e. noise the known region to the current noise level) so left matches the nominal timestep. This is the in-distribution choice if training used "left noised at same level, loss on right" (DFoT with shared-level sampling). **Not verified for SANA/Sprint; do not assume it is better.**
- **Few-step/consistency caveat [my reasoning, UNVERIFIED]:** Sprint has 1-4 steps with large jumps (t from pi/2 down). With 1-2 steps, any re-paste scheme has almost no opportunity to correct the seam; the model must be right in one shot. So the conditioning must be learned (trained), not enforced by iteration. This favors the trained clean-left design over inference-time inpainting tricks.

## 4. Fine-tuning a consistency-distilled model [UNVERIFIED, reasoning only]
- Sprint is trained with sCM + LADD (JVP-based); a naive flow-matching LoRA loss on it is not the objective it was distilled with. No source I found shows LoRA fine-tuning of SANA-Sprint with a new input structure. ControlNet-Sprint was done by training the ControlNet teacher then re-distilling (full pipeline), which is heavy.
- Options, cheapest first:
  1. Train the in-context LoRA on the **non-distilled SANA (1.6B / 1.5)** with flow matching, shared scalar t, target-only loss. This is the Kontext/GateControl recipe and the only one with direct evidence on SANA (GateControl).
  2. Try loading that LoRA onto SANA-Sprint (same architecture; Sprint is fine-tuned from the SANA teacher with modified time embedding c_noise(t)=t, so weights are close but the timestep embedding scale differs: LoRA transfer is a gamble).
  3. Fine-tune Sprint directly with an x0-regression loss at its discrete schedule (e.g. t in {pi/2, 1.3, 1.1, 0.6}), supervising only the right half, starting from t=pi/2 for 1-step. Cheap approximation of the distilled objective; no source validates it.
  4. Re-distill (sCM+LADD) from the in-context teacher: correct but expensive (Sprint used 32 A100, 20k iters, batch 512).

## 5. Recommended concrete design

**Canvas / geometry**
- Wide canvas, spatial concat (no RoPE exists to shift). Pixel canvas 1024x512 (left 512x512 clean matte, right 512x512 target) -> DC-AE latent 16x32 tokens (HxW), left = columns 0-15. Replicate the grayscale matte to 3 channels before DC-AE. Encode each half separately through DC-AE (avoid conv bleed at the seam in the AE; encoding the full wide image also works but mixes halves) then concatenate latents along W. **[Choice rationale: separate encode gives exact "clean condition latent"; unverified which is better.]**
- Alternative if you want 1024^2 outputs: 2048x1024 canvas = 32x64 tokens (2048 tokens); still cheap for linear attention.

**Timestep**
- Start with Kontext-style: **single scalar t shared by all tokens**, left half = clean latent (never noised), loss on right half only. Needs zero architecture change and has production evidence.
- Upgrade (if left-half features look timestep-contaminated or training is slow): per-token modulation where left tokens use t=0 (LTX/DFoT style): compute the AdaLN-single modulation twice (t and 0) and select by a token mask. Small code change in SANA blocks **[UNVERIFIED: need to read the SANA block code to confirm shape of modulation]**. In Sprint, t=0 under TrigFlow is the clean point, so "clean at t=0" is well defined.
- Do not add noise to the condition half at all in v1.

**Attention**
- Plain bidirectional (as in GateControl, Kontext). Optional EasyControl-style isolation is almost free with linear attention [my idea, unverified]: left queries use a KV summary built from left keys only, right queries use the full KV summary. This makes left features independent of the noise level, i.e. cacheable and equal to the t=0 behaviour without per-token timesteps. But it changes the pretrained computation for the left half and requires custom kernels; defer to v2.
- Consider GateControl's token-wise gate (0.09M params, +10x convergence speed on aligned tasks) if the plain LoRA converges slowly. Code not verified to be released.

**LoRA**
- All linear layers in the transformer blocks (attention qkv/o, cross-attn, FFN incl. depthwise conv optionally), rank 16 to start (matches IC-LoRA and GateControl-on-SANA); try 64 if underfitting. Prodigy lr=1 (GateControl) or AdamW 1e-4 (EasyControl, VisualCloze).
- Text prompt: short constant caption (e.g. "left: alpha matte, right: foreground on gray") or empty; IC-LoRA uses joint captions per panel.

**Loss**
- Flow-matching/velocity loss on right-half tokens only (mask by token columns). Left-half output discarded.

**Data / steps**
- Task is aligned and low-entropy on the left (silhouette -> RGB must be invented). Budget: ~10k-50k pairs and 10k-30k steps on SANA-1.6B as a starting point by analogy with GateControl (10K images / 10K steps with gate) and OminiControl (50k iter). IC-LoRA's 5k steps/20-100 samples is the floor, adequate only for a very narrow domain.

**Sprint path**
- Train on non-distilled SANA first (Option 1), evaluate 20-step quality. Then test transferring the LoRA to Sprint (Option 2) at 2-4 steps, and if that fails, Option 3 (x0 regression at Sprint timesteps on right half). Treat inference-time left re-pasting as a diagnostic, not the main mechanism.

## 6. Open / unverified items (do before committing)
- SANA code: does the timestep modulation support per-token shapes? Does SANA/Sprint use any non-trivial attention mask support in linear attention kernels?
- LTX IC-LoRA default LoRA rank, steps, dataset size (not on trainer page).
- OminiControl condition timestep and loss (paper silent; check github train code https://github.com/Yuanshi9815/OminiControl).
- Kontext training code is unpublished; the "same scalar t for context tokens" claim is from the diffusers inference pipeline, not from training.
- Whether the Sprint ControlNet released weights/code allows reuse as an additional conditioning path (docs only mention HED).
- GateControl code availability and whether its gate is needed for a LoRA at our scale.
- No source evaluated side-by-side clean-left canvas on a NoPE model; the seam behavior through Mix-FFN conv is untested, my reasoning only.
