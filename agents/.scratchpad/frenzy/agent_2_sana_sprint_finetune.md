# SANA-Sprint fine-tuning research (agent 2)

Legend: [V] = read in a source this session. [U] = unverified / my inference.

## Short answers
1. LoRA directly on Sprint student: no official recipe, no positive report found. Maintainer has never tried a diffusion loss on it. Only closest empirical data (reward-LoRA, not FM loss) shows quality loss. [V]/[U]
2. Teacher = SANA-1.5-derived flow model converted to TrigFlow, with c_noise(t)=t (no x1000), QK-RMSNorm in self+cross attn, 5k-iter finetune. [V]
3. No published VRAM numbers anywhere. Estimates below are [U].
4. diffusers train_dreambooth_lora_sana.py: documented only for Sana_1600M_1024px_BF16_diffusers. Sprint support not documented; on Sprint it errored (issue #256). Teacher: untested [U].
5. Official sCM+LADD scripts init from a teacher/pretrained ckpt, not from an existing Sprint student (diffusers script has no init-from-student arg). Custom data possible; formats below.
6. Inference: see section 6.

## 1. LoRA on the Sprint student
- NVlabs/Sana issue #256 "NOT ABLE TO LORA FINE TUNE ON SANA SPRINT MODELS" (open, Apr 2025) https://github.com/NVlabs/Sana/issues/256 [V]
  - Sana's sana_lora_dreambooth.md script, with Sprint swapped in, errored ("invalid parameter ... batch size").
  - lawrence-cj (maintainer, May 2025): Sprint distilled with scm-loss (non-diffusion loss); unsure the standard diffusion loss still works; "never tried".
  - nbardy (Jul 2025): theoretically should work with standard loss; untested.
  - Another user (Nov 2025) hit same failure; also on SANA1.5_4.8B.
- No Reddit/HF-discussion/blog report of a successful FM-loss LoRA on Sprint found in my searches. [absence of evidence only]
- Issue #537 (maintainer comment): Sprint is guidance-distilled; embedded CFG trained only for scales ~4-5, no unconditional branch; ComfyUI recipe cfg_scale 4.5, KSampler cfg 1.0, 2 steps. https://github.com/NVlabs/Sana/issues/537 [V via search snippet]
- Closest empirical evidence: "Noise Hypernetworks" arXiv 2508.09968 (https://arxiv.org/abs/2508.09968), Table 10 [V]. Direct LoRA on SANA-Sprint 0.6B with a REWARD loss (ImageReward/HPSv2.1/PickScore/CLIP; SGD lr 1e-3, rank128/alpha256, bs18x3 accum, ~25 epochs, 70k prompts, 6 H100):
  - GenEval 1-step: base 0.70; r128 0.67; r64 0.68; r16 0.65; r8 0.59. 2-step: 0.72 -> 0.66. 4-step: 0.73 -> 0.62. Degradation grows with step count. Authors attribute to reward hacking, not to LoRA per se. It is NOT a plain FM loss, so it does not directly answer your question.
- Why plain FM/diffusion loss is likely risky [U, inference]: the student is a consistency function in TrigFlow parametrization with extra guidance embedding (CFG baked in) and a logvar head; an FM velocity loss trains a different objective (ODE tangent at all t) and does not preserve the consistency property across t, so few-step sampling drifts. nbardy's view is the opposite (could work). Test empirically: LoRA FM loss, then compare 1/2/4-step vs base.
- Safer paths [U]: (a) keep the sCM loss (+ optional LADD) and attach LoRA to the student in a modified diffusers distill script (script is full-FT only, no LoRA args); (b) train LoRA on the teacher with FM/TrigFlow loss then re-distill; (c) frozen generator + conditioning/noise method (HyperNoise, https://huggingface.co/lucaeyring/HyperNoise_Sana_Sprint_0.6B).
- Img2img-type SanaSprintImg2ImgPipeline exists in diffusers (inference only).

## 2. Teacher vs vanilla SANA (paper arXiv 2503.09641 HTML v4, https://arxiv.org/html/2503.09641v4) [V]
- Base: pruned/finetuned from SANA-1.5 4.8B into 0.6B/1.6B teachers; architecture/data otherwise as SANA-1.5.
- Training-free FM -> TrigFlow transform (Prop 3.1, Alg 1): t_FM = sin(t)/(sin(t)+cos(t)); input x_FM = (x_trig/sigma_d)*sqrt(t_FM^2+(1-t_FM)^2); output F = [(1-2t_FM)x_FM + (1-2t_FM+2t_FM^2) v_theta]/sqrt(t_FM^2+(1-t_FM)^2). sigma_d=0.5. Lossless-ish (FID 5.73 vs 5.81 flow, Euler 50 steps). Wrapper in diffusers script: SanaTrigFlow.
- Dense time embedding: c_noise(t)=t instead of 1000t (1000x blew gradient norms >1e3). Teacher finetuned 5k iters for this.
- QK-normalization: RMSNorm on Q and K in self- and cross-attention; added because 1.6B diverged (grad norm >1e3). Config has qk_norm true, cross_norm true ("allqknorm").
- Phase 1: teacher finetune lr 2e-5, global bs 1024, 5k iters. Phase 2: distill lr 2e-6, bs 512, 20k iters, 32 A100, DDP. Student initialized from finetuned teacher (Alg 2).
- Student adds CFG embedding: scale (4/4.5/5 uniform in training) *0.1 added to time embedding; zero-init in diffusers script. Student also has logvar head (adaptive loss weight).
- Linear attention kept; no flash-attn JVP kernel, so diffusers training uses SanaVanillaAttnProcessor (plain attention) because F.scaled_dot_product_attention lacks torch.jvp support.
- Issue #240 (https://github.com/NVlabs/Sana/issues/240): repo code still has pretrain_timestep = t*1000 at sana_multi_scale.py:426 but only for original Sana consistency; real scaling done by timestep/timestep_norm_scale_factor (line 282). Config: timestep_norm_scale_factor 1000. [V]

## 3. VRAM (all [U] unless stated)
- No source gives numbers. What is known [V]: the diffusers script holds student + frozen teacher copy (discriminator shares teacher), fp32 VAE, Gemma2 on GPU, explicit (non-SDPA) attention, JVP pass; its comment: "Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs". README example: bs 1/GPU, grad checkpointing, bf16, lr 1e-6, 30k steps. Official YAML: bs 16, CAME optimizer, grad clip 0.1.
- Rough arithmetic (weights bf16, AdamW fp32 states; 1024px = 32x32=1024 latent tokens, small by DiT standards):
  - 1.6B full FT sCM+LADD: student 3.2GB + teacher 3.2GB + grads 3.2GB + Adam 12.8GB (fp32 m,v) (+ fp32 master if used) ~ 25-35GB before activations/JVP/Gemma(5GB)/VAE; plausible 40-60GB -> 48/80GB class; 8-bit Adam can fit 24-32GB-ish.
  - 0.6B: roughly 1/2.7 of the above; ~20-30GB total plausible, fits 24GB with checkpointing.
  - LoRA (frozen bf16 base): 1.6B base 3.2GB + Gemma 5GB + VAE; activations with grad checkpointing at 1024px bs1 a few GB. Would fit 24GB comfortably; 0.6B fits easily.
- LoRA rank: parameter cost is trivial compared to activations. Rank 64-128 is feasible on 24GB at bs1 (rank 128 on all linears of 1.6B ~ tens-of-millions params, <1GB with Adam) [U]. Reward-LoRA paper used rank 128 on 6 H100 at bs18, not VRAM-limited info. Practical ceiling on any GPU is set by activations/batch, not rank. Measure with torch.cuda.max_memory_allocated.
- Cache text embeddings and VAE latents (official YAML loads precomputed VAE feats; diffusers dreambooth script has --offload / cache_latents) to free Gemma/VAE.

## 4. diffusers train_dreambooth_lora_sana.py with teacher
- README_sana.md https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md [V]: model Sana_1600M_1024px_BF16_diffusers; flags rank/lora_alpha/lora_layers (e.g. to_k,to_q,to_v)/use_8bit_adam/offload/cache_latents; peft>=0.14.0; bf16 (fp16 gave NaN per PR #10234; VAE fp32). No mention of Sprint or teacher.
- Teacher (Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher_diffusers, card: loads with SanaPipeline, 20 steps; https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher_diffusers) is a SanaTransformer2DModel w/o guidance embeds [V from training script: teacher loaded with guidance_embeds=False].
- Whether the dreambooth script trains it correctly [U]: loads as SanaTransformer2DModel, so LoRA attach likely works mechanically. Pitfalls: (1) teacher expects TrigFlow/c_noise=t conventions, whereas the script uses standard flow-matching timesteps and presumably t*1000 scaling via the scheduler. (2) the teacher's time_embed key layout differs (script remaps time_embed.emb.timestep_embedder.* -> time_embed.timestep_embedder.*). Verify the pipeline's own timestep scaling for the teacher before trusting any loss. Could not verify what SanaPipeline does for this checkpoint.
- Sprint student in that script: user reports errors (#256).

## 5. sCM+LADD scripts, init, data
- Diffusers: examples/research_projects/sana/train_sana_sprint_diffusers.py (+ .sh, README), PR #11514 MERGED to main May 8, 2025 (commit 784db0e) https://github.com/huggingface/diffusers/pull/11514 [V]. Post-merge, a user (Jun 2025) reported errors in log_validation; scxue acknowledged and promised a fix; fix status unverified.
  - Full fine-tune only: no LoRA/rank args. No init-from-student arg: student = copy of --pretrained_model_name_or_path (the TEACHER dir; must be a local dir since it loads transformer/diffusion_pytorch_model.safetensors directly) + zero-init guidance embedder. Only --resume_from_checkpoint restores training states. Hack: point pretrained path at a Sprint student dir? Not supported as-is (teacher is built from same weights with guidance_embeds=False, so would distill from a student) [U].
  - Data: HF load_dataset(dataset_name, data_files=file_path) parquet; hardcoded item["llava"] = caption, item["image"] = raw image bytes (PIL decode). Edit these keys for custom data; Resize/CenterCrop/Normalize at --resolution (default 512, use 1024). Unused args: image_column, caption_column, cache_latents, offload. persistent_workers=True requires num_workers>0.
  - Known script quirks (from reading): D-phase fills t with largest_timestep_prob (0.5) instead of largest_timestep (likely bug); `pred_x_0.detach()` no-op; save_model_card crashes with --push_to_hub (instance_prompt); only disc heads saved as disc_heads.pt.
  - Hyperparams: lr 1e-6 (README), logit_mean/std 0.2/1.6 (G), -0.6/1.0 (D), adv_lambda 0.5, scm_lambda 1.0, tangent_warmup_steps 4000, grad clip 0.1, scm_cfg_scale [4,4.5,5], guidance_embeds_scale 0.1, largest_timestep 1.5708 prob 0.5, head_block_ids [2,8,14,19].
- Native repo: train_scripts/train_scm_ladd.sh -> train_scripts/train_scm_ladd.py, torchrun, config configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml [V].
  - YAML: model SanaMSCM_1600M_P1_D20, load_from hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth (student weights!), teacher_model hf://.../Sana_Sprint_1.6B_1024px_teacher/...pth. So the native repo CAN init from an existing Sprint checkpoint via model.load_from (YAML default already does), distilling against the teacher. [V from YAML; whether training continues cleanly from Sprint weights incl. guidance embed is unverified in practice]
  - Data: data.type SanaWebDatasetMS (WebDataset), data_dir list (default data/toy_data), load_vae_feat true (precomputed VAE latents), load_text_feat false, caption key "prompt", multi_scale true w/ ASPECT_RATIO_1024. Exact shard/file schema NOT documented in sana_sprint.md; see Sana's other data docs / toy_data for format (not read) [U].
  - YAML training: bs 16, lr 2e-6 constant, CAME optimizer, clip 0.1, grad ckpt, LoRA none (full FT), tangent warmup 4000, sCM cfg [4,4.5,5], LADD hinge adv weight 0.5.
- Debug script uses 2 GPUs, --resume_from latest.

## 6. Sprint inference
- Diffusers (https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_sprint) [V]: SanaSprintPipeline.from_pretrained("Efficient-Large-Model/Sana_Sprint_{0.6B,1.6B}_1024px_diffusers", torch_dtype=torch.bfloat16). Transformer bf16, text encoder bf16, VAE bf16 or fp32.
- __call__ defaults: num_inference_steps=2, timesteps=None (list, descending), max_timesteps=1.5708, intermediate_timesteps=1.3 (used only when steps=2), guidance_scale=4.5 (embedded), height=width=1024, use_resolution_binning=True, max_sequence_length=300, generator, latents, prompt_embeds, prompt_attention_mask.
- Paper Tab 7 schedules (same for 0.6B/1.6B) [V]: 1 step [pi/2, 0]; 2 steps [arctan(200/0.5)=1.5683, 1.3, 0]; 4 steps [1.5683, 1.3, 1.1, 0.6, 0]. Docs default max_timesteps 1.5708 (pi/2) differs slightly from paper's 1.5683 for 2/4 step; paper ablations used 1.0 intermediate. Default came from inference-time search (docs).
- Fixed initial noise: pass latents=<Tensor> with shape (B, 32, H/32, W/32) (DC-AE f32c32; 1024px -> (B,32,32,32)), dtype matching transformer (bf16), standard Gaussian; or a torch.Generator(device).manual_seed(s). With use_resolution_binning=True the requested H/W get mapped to a bin, so build latents for the binned size (or disable binning, use_resolution_binning=False). Check how the pipeline scales latents (sigma_data multiply) before feeding; whether pipeline multiplies provided latents by sigma_data is unverified [U] - read pipeline_sana_sprint.py prepare_latents. For multi-step, scheduler (SCMScheduler) adds fresh noise each step from generator; for strict reproducibility pass a generator too. Also pass prompt_embeds to hold text constant across LoRA/no-LoRA comparisons.
- Repo-native: app.sana_sprint_pipeline.SanaSprintPipeline, same YAML; guidance_scale 4.5, steps 2, seeded generator [V].
- Optional faster decode: dc-ae-lite-f32c32-sana-1.1-diffusers [V].

## Sources
- https://github.com/NVlabs/Sana (asset/docs/sana_sprint.md; train_scripts/train_scm_ladd.sh; configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml)
- https://github.com/NVlabs/Sana/issues/256 , /240 , /537
- https://github.com/huggingface/diffusers/pull/11514 ; examples/research_projects/sana/{README.md,train_sana_sprint_diffusers.py,train_sana_sprint_diffusers.sh}
- https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md ; PR #10234 (Sana LoRA)
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_sprint
- https://arxiv.org/html/2503.09641v4 (last 14.8k chars unread) ; https://arxiv.org/abs/2508.09968 (HyperNoise)
- https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher_diffusers

## Not verified / gaps
- Any VRAM measurement; any successful FM-loss LoRA on Sprint; whether the teacher works in train_dreambooth_lora_sana.py; Sana WebDataset shard schema; latents scaling in SanaSprintPipeline; whether the diffusers sprint script's log_validation bug was fixed; Reddit/HF-forum searches returned nothing relevant (only 2 web searches run).
