# Agent 5: RGBA / alpha-matte -> RGB prior art (web research, 2026-11-17)

Legend: [V] = verified from a fetched page; [S] = from a search-result snippet only; [U] = unverified / from memory.

## Headline finding
No paper found that does exactly "grayscale alpha matte + caption -> RGB foreground on flat 50% gray, trained on MAGICK". Closest: **Zippo** (matte-to-RGB, SD2.1, full UNet finetune, matte encoded as 3-channel image through the plain RGB VAE). MAGICK has no published model trained on it that I could find; it was used as training data by PrismLayers (VAE decoder) and a community alpha-estimation derivative.
Nothing found using a flat gray background for generation; the dominant patterns are (a) multiscale color padding of transparent pixels (LayerDiffuse, AlphaAnything), (b) random solid background colors (Wan-Alpha), (c) light/white (Alfie), (d) black+white composites for evaluation (AlphaAnything). A fixed 50% gray is a defensible, simple choice (equidistant from black and white; fringe errors visible in both polarities) but is the experimenter's choice, not literature-backed.

## Per-method table

### LayerDiffuse (Zhang & Agrawala) https://arxiv.org/abs/2402.17113 , https://github.com/layerdiffusion/LayerDiffuse
- Alpha: "latent transparency" = small latent offset added to the pretrained VAE latent of the RGB image so the original latent distribution barely changes; a separate decoder recovers RGBA from (offset latent + RGB-decoded image). Alpha is NOT a diffusion channel. [V abstract]
- Base: SDXL (and SD1.5 variants in released Forge/diffusers code) [U for exact; abstract only says "any latent diffusion model"]. Data: 1M transparent layer pairs via human-in-the-loop collection [V]. Also layer-conditioned generation (FG given BG / BG given FG), and ControlNet-style structural control [V abstract].
- Background color: the RGB under transparent pixels is filled by multiscale color padding (coarse-to-fine bleed of foreground color) rather than a flat color [S: stated by AlphaAnything https://arxiv.org/html/2610.32413 as "LayerDiffuse's multiscale color padding"].
- Eval: user study, 97% preferred native transparency over generate-then-matte [V]. Details of other metrics [U].
- Relevance: padding trick is the thing to compare against flat gray; their released weights are usable as baseline for text->RGBA.

### Alfie https://arxiv.org/html/2408.14826v1 (code https://github.com/aimagelab/Alfie)
- Training-free. PixArt-Sigma-512 (0.6B DiT), 30 Euler steps [V]. Alpha is derived post-hoc from cross+self attention maps of the last 10 steps, GrabCut cleanup on a 4-class quantised map; optional opacity knob k=0.5 [V].
- Background: prompt-controlled, kept "uniform and light rather than perfectly white" (border threshold 0.8); centered-square latent blending to force a subject centered with empty borders [V]. Exact bg color/prompt not stated.
- Eval: CLIP score, "empty border rate" (4px margin, all pixels >0.8), 10-person user study vs ViTMatte (63% prefer), no FID/KID (no RGBA reference set), no VLM judge [V]. 3000 PixArt-Sigma training prompts.
- Cannot do semi-transparency [S].

### LayerFusion https://arxiv.org/abs/2412.04460 (https://layerfusion.github.io/)
- Training-free; built on LayerDiffuse's FG model + a BG model, blends at attention layers to produce FG RGBA, BG RGB and composite jointly. Alpha comes from LayerDiffuse's transparent-FG pathway; composite masks use attention-derived structure/content confidence priors [S].
- Eval: CLIP score, user study (50 people, 40 triplets) vs LayerDiffuse-style sequential baseline [S]. Adobe/Virginia Tech. Not an alpha->RGB method.

### ART (Anonymous Region Transformer) https://arxiv.org/abs/2502.18364 , https://github.com/microsoft/art-msra
- Multi-layer RGBA via FLUX.1-dev (LoRA) with a layer-sparse "anonymous region" attention, global caption + bbox layout, 50+ layers, own multi-layer transparent autoencoder (joint RGBA latent) [S]. CVPR 2025.
- Weights/checkpoints REMOVED by authors 2025-07-23 (data provenance concerns) [S]; do not plan to depend on it. Training-data details [U].

### TransPixar / TransPixeler https://arxiv.org/abs/2501.03006
- Text/image -> RGBA video. DiT (CogVideoX, Mochi) + alpha tokens appended to sequence, reinitialised alpha positional embeddings, LoRA on alpha-specific params, channel-adaptive attention; ~484 RGBA videos for training [S]. Cost: quadratic in longer sequence [S].
- Background color for RGB under transparency, and metrics: UNVERIFIED (PDF fetch returned binary). Open weights only partly released [S].

### Zippo https://arxiv.org/abs/2403.11077 (html v2 https://arxiv.org/html/2403.11077v2)  <-- closest to our task
- Alpha representation: single-channel matte replicated to 3 channels, encoded with the unchanged SD VAE; RGB and alpha share one latent space/UNet; conv_in widened, first/last layers replicated per modality, learnable task-identifier embedding added to timestep embedding [V].
- Matte->RGB: P(x | a, c): noisy RGB latent concatenated with CLEAN matte latent, input cat(z_t^x, z^a); trained jointly with RGB->matte and joint generation (modality-aware noise reassignment) [V].
- Base SD v2.1, full UNet finetune, lr 1e-5 cosine, bs 6, 768 crops [V]. Data: AM2K (~2k animals) + VITON-HD (~10k humans) with pseudo-mattes (P3M -> SAM IoU>0.9 filter -> trimap -> MatteFormer) [V]. Tiny data vs MAGICK's 140k.
- Background: not stated; RGB targets are original photos with natural backgrounds (so NOT a flat-bg setup) [V: "does not say"].
- Eval: matte->RGB only qualitative (vs ControlNet: better hair-detail alignment); no FID/CLIP at all; matting metrics SAD/MSE/MAD/CONN only for RGB->matte (AM2K SAD 17.49 full) [V].
- Takeaway: matte-as-3ch-image-through-VAE-as-condition is proven; matte->RGB metric precedent is absent, we must define our own.

### DreamLayer https://arxiv.org/abs/2503.12838 (https://ll3rd.github.io/DreamLayer/)
- ICCV 2025. Multi-layer text->layers with layer-assign embeddings, cross-layer attention, "layer decoder" outputs alpha; 400k-sample dataset (GPT-4 prompt splitting + segmentation/detection masks); baselines SD1.5 and LayerDiffuse; user study 20 subjects x 200 samples [S]. Base model / bg color [U].

### PSDiffusion https://arxiv.org/abs/2505.11468
- Single diffusion framework, all layers from global + per-layer prompts; Layer Cross-Attention Reweighting + Partial Joint Self-Attention; 30K "Inter-Layer" dataset (3-6 layers, artist-quality mattes). Eval on composites vs LayerDiffuse and ART: CLIP Score, FID, Layout Harmony, Interaction Plausibility [S]. Base model [U]. Relevant as precedent for FID-on-composites.

### MAGICK https://huggingface.co/datasets/OneOverZero/MAGICK , https://github.com/RyannDaGreat/MAGICK , CVPR 2024 paper (PDF fetch 403)
- 140k (paper says ~150k) captioned single-subject 1024x1024 RGBA PNGs; index `magick_index.tsv` cols `page_id`, `subject` (caption); URL `.../resolve/main/images/{page_id[:2]}/{page_id}.png`; ~191-200 GB; HF viewer shows only one validation row [V].
- Made by generating objects on a single-color background and chroma-keying to extract alpha (title + search snippet) [V/S]. Generator (reportedly SDXL per a citing paper [S]), exact key color and extraction math: UNVERIFIED, read the paper PDF if it matters (relevant: the RGB near edges may carry spill from the key color; compositing on gray will expose it).
- No model trained on it announced [V]. Used by: PrismLayers (transparency VAE decoder) https://arxiv.org/html/2505.22523v1 ; Trans-Adapter https://arxiv.org/abs/2508.01098 calls it best scale/quality balance [S]; community derivative World-Snapshot/MAGICK_Transparency (RGB-over-random-bg -> alpha, uses LayerDiffuse_Transparency key naming, FLUX.2 latents) https://huggingface.co/datasets/World-Snapshot/MAGICK_Transparency [V].

### Other related (not requested but useful)
- Wan-Alpha https://arxiv.org/html/2509.24979 : RGB hard-rendered onto RANDOM solid color from 8 (black, blue, green, cyan, red, magenta, yellow, white) so the encoder does not confuse bg color with transparency; soft render V*a + c*(1-a); evaluates composites on WHITE; GPT-4o VLM judge for text alignment (4.00) and naturalness (3.19) following OpenS2V; user study for transparency; notes no existing metric supports RGBA [V].
- AlphaAnything https://arxiv.org/html/2610.32413 : RGBA in the frozen RGB latent space, alpha branch on VAE decoder; evaluates on BLACK AND WHITE composites averaged; metrics RGB L1 (over GT alpha>0), Soft IoU = sum min / sum max, SAD, Alpha Grad (Gaussian-derivative sigma 1.4), Alpha Conn (Rhemann et al., MMEval 0.2.1), LPIPS; text->RGBA CLIPScore, Aesthetic, ImageReward, FID; seeds 42/43/44 [V].
- DRIP (NeurIPS 2024): SD2-based joint FG+alpha, conditioned on image+trimap, domain-switcher embedding [S] https://proceedings.neurips.cc/paper_files/paper/2024/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf
- Lightricks LTX-2.5 IC-LoRA Alpha-Gen (RGB->alpha video) [S].

## Design implications for the experiment
1. Flat 50% gray: consistent with MAGICK being single-subject with soft edges; target = alpha*F + 0.5*(1-alpha). Because the matte is given as input, the model never has to invent transparency; the task is "color the region" and the background is deterministic. Risk: model leaks gray into the interior (low-contrast gray objects) or fails to keep the exterior exactly 0.5 (checkable, see metrics). Wan-Alpha's reasoning (fixed bg color can be confused with foreground color) applies: gray objects are the hard case, include some in the prompts.
2. Semi-transparent pixels on a gray bg are ambiguous w.r.t. true F (RGB under a<1 is mixed). Composite-on-gray loss is fine for training; do not claim foreground recovery.
3. Zippo suggests conditioning by concatenating the clean matte latent (matte replicated to 3ch through the RGB VAE). For in-context (token-concat) conditioning in a DiT the same thing is done by appending matte tokens.

## Evaluation methods (what exists)
- VLM judge: used for RGBA video (Wan-Alpha, GPT-4o, text alignment + naturalness, 1-5) [V]; none in Alfie [V]. For us: VLM judge on the composite for caption fidelity and artifacts (halo, gray bleeding, lost detail); cheap and a reasonable proxy, but noisy; use fixed rubric, 2+ samples, report mean.
- FID/KID on composites: PSDiffusion, AlphaAnything use FID; Alfie avoided it for lack of RGBA reference set [V]. For us the reference set EXISTS (MAGICK composited on 50% gray), so FID/KID vs held-out MAGICK composites is valid, but with only ~20 prompts x seeds FID is meaningless (needs >=1-5k samples); use KID or skip at small scale, or use at a larger offline eval of ~1-2k captions.
- CLIP score on composite (text alignment): Alfie, LayerFusion, AlphaAnything [V/S].
- Alpha adherence: no precedent for matte->RGB. Define it ourselves (see below). Related precedent: Soft IoU, SAD, Grad, Conn (AlphaAnything) all measure alpha-vs-alpha; apply them after re-extracting a matte from the generated image.
- User study: LayerDiffuse, Alfie, LayerFusion, DreamLayer, Wan-Alpha; not cheap.

## Recommended cheap protocol (~20 prompts, fixed seeds)
Set: 20 held-out MAGICK items (never trained on), (matte, caption, GT composite on 0.5 gray). Include ~4 hard cases each: soft/fine edges (hair, fur), semi-transparent (glass, smoke), gray-colored object, thin structures, tiny object. Fixed seeds 42, 43, 44 (AlphaAnything uses the same) x 20 = 60 images; fixed steps/CFG/resolution; same sampler noise per (prompt, seed) across checkpoints so runs are paired.
Metrics (all scripted, minutes per run):
1. Exterior purity (alpha-adherence, no matting model needed): mean |gen - 0.5| over pixels where input alpha == 0 (eroded by a few px to avoid edge ambiguity); report mean and 99th percentile. Catches background leakage.
2. Interior content: mean abs deviation from 0.5 inside alpha==1 (eroded) = "did it paint something". Catch gray-fill failure (ratio of pixels within 0.02 of 0.5 should be similar to GT).
3. Region IoU / soft IoU: re-derive a foreground map from the generated image as |gen - 0.5| (max over channels) > tau (e.g. 0.05), IoU against matte>0.5; fails for genuinely gray object pixels, so also report same statistic on GT composite as the achievable ceiling and use ratio to ceiling. Optionally run an off-the-shelf matting/segmentation model (e.g. SAM or ViTMatte) on the output instead and compute Soft IoU (sum min/sum max) and SAD vs input matte (AlphaAnything definitions).
4. Edge quality: (a) band metric: L1 vs GT composite within a dilated-minus-eroded edge band of the matte (e.g. 8 px); (b) fringe/halo check: mean chroma of edge band pixels relative to GT (key-color spill from MAGICK chroma keying will show here); (c) optional gradient/connectivity metrics (Grad, Conn).
5. Full-reference vs GT composite (paired, since the GT exists): LPIPS and PSNR are only weakly meaningful (caption does not determine colors) so use LPIPS/DINO-feature distance as sanity, not headline.
6. Text alignment: CLIP score (or SigLIP) between caption and composite; compare to GT composites' CLIP score as ceiling.
7. VLM judge (one call per image, fixed rubric, temperature 0, JSON out): scores 1-5 for (a) matches caption, (b) edges/halo free, (c) no gray bleed/foreground clearly distinct from backdrop, (d) overall; plus pairwise A/B vs baseline checkpoint with position swapped. Also feed contact sheet (matte | gen | GT) for fidelity to shape.
8. Montage: matte | gen | GT per seed for human eyeballing; the human check remains the primary gate.
Report: per-metric mean over 60 images with paired bootstrap CI between checkpoints; track on ceiling normalisation from the GT composites.
Offline occasional (not per-step): KID/FID vs 1-2k held-out MAGICK gray composites.

## Unverified / flagged
- LayerDiffuse base model, other metrics, release details (abstract only fetched; paper PDF >10MB failed).
- MAGICK exact chroma key color, generator, matting algorithm (paper PDF 403; GitHub README is thin).
- TransPixar background color and metrics (PDF unreadable).
- ART training data and weight availability (weights reportedly removed 2025-07-23 per search snippet).
- DreamLayer / PSDiffusion base models and background handling (snippets only).
- LayerFusion details from snippets.
- Several cited papers have 2026 arXiv ids (e.g. 2610.32413, 2608.08487, 2605.14552) discovered via search; content for 2610.32413 was fetched [V], the others not.
- Claim that no flat-gray-background method exists: absence of evidence from a limited search, not proof.
