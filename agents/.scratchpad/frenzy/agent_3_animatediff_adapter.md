# Agent 3: AnimateDiff domain adapter and weight-arithmetic precedents

Method: web search and page fetches only. Each claim carries a source URL. "UNVERIFIED" marks things I could not confirm.

## 1. How AnimateDiff trains the domain adapter

Source: AnimateDiff paper, arXiv 2307.04725, Sec. 4.1 "Alleviate Negative Effects from Training Data with Domain Adapter" (fetched via https://arxiv.org/html/2307.04725v2), plus the official README https://github.com/guoyww/AnimateDiff

- **Motivation.** WebVid is low-quality real-world footage (motion blur, compression artifacts, watermarks). LAION-Aesthetic is cleaner. If the motion module is trained directly on the frames, it also has to absorb this pixel-level quality gap. The adapter is a separate network that "fits the domain information" so the motion module can concentrate on motion.
- **What it is.** LoRA layers inserted into the self-attention and cross-attention layers of the base T2I UNet (paper Fig. 3). Paper Eq. 4, shown for the query projection: `Q = W^Q z + AdapterLayer(z) = W^Q z + alpha * A B^T z`. The paper writes LoRA as `W' = W + A B^T` with `A in R^{m x r}`, `B in R^{n x r}`.
- **Data.** "Static frames randomly sampled from video datasets", i.e. single frames, no temporal modelling. The paper uses WebVid-10M with SD v1.5.
- **Loss.** The ordinary SD noise-prediction MSE (paper Eq. 2): `|| eps - eps_theta(z_t, t, tau(y)) ||^2`. Only the adapter parameters are optimized. The base weights stay frozen.
- **Rank.** NOT FOUND. The main text gives no domain-adapter rank. The only ranks it states are for MotionLoRA (rank 2, about 1M params, and rank 128, about 36M). The v3 adapter checkpoint is 97.4 MB per the README. That is a size I could use to estimate the rank, but I did not compute it, so the rank is UNVERIFIED. Hyperparameters are deferred to the supplementary material, which I could not read: the arXiv PDF exceeded the fetch size limit and ar5iv failed to convert.
- **Pipeline order.** The README lists three stages. (1) Train the domain adapter. (2) Train the motion module with the adapter and the base T2I frozen. (3) Optionally train MotionLoRA.
  - README: https://github.com/guoyww/AnimateDiff
  - Training guide, which only covers v1 configs and has nothing on the adapter: https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md
- **v1/v2 versus v3.** In v1/v2 the paper trains the adapter and then keeps it frozen during motion training. The v3 README says (2023.12): "we use Domain Adapter LoRA for image model finetuning, which provides more flexibility at inference". The v3 checkpoint is `v3_sd15_adapter.ckpt`. The README table writes the name as `v3_adapter_sd_v15.ckpt`.

## 2. Use at inference (scale 0 versus 1)

- **Training scale.** `alpha = 1` during training. At inference it "can be adjusted to other values (set to 0 to remove the effects)" (paper). The default recipe drops the adapter at inference, i.e. alpha = 0, so you keep the base T2I plus the motion module only. The adapter can instead be injected into a personalized T2I, with alpha tuned.
- **README wording.** "By default, the adapter can be removed at inference. It can also be integrated into the model and its effects can be adjusted by a lora scaler." (https://github.com/guoyww/AnimateDiff)
- **Community usage.** In ComfyUI it is an ordinary image-model LoRA, not a motion LoRA. AnimateDiff-Evolved's motion LoRA loader rejects it with "contains no temporal keys; it is not a valid motion LoRA". Workflows use a normal LoRA loader. Source: https://github.com/guoyww/AnimateDiff/issues/257
  - The "0.5 weight with mm_v3" in A1111 workflows came from a search summary of a Civitai page. UNVERIFIED, and it is a community habit, not an official recommendation.
- **SparseCtrl needs the full adapter.** A fork README (https://github.com/WiserZhou/AnimateDiff) says: "To utilize the SparseCtrl encoder, it's necessary to use a full Domain Adapter in the pipeline". I could not load the official v3 YAML configs (the URL I guessed returned 404). My reading is that "full" means scale 1.0 and that the SparseCtrl encoders were trained with the adapter in place. That reading is UNVERIFIED.
  - This is the case closest to your plan. AnimateDiff trains a control encoder on top of the base plus adapter, and the encoder then expects the adapter to be present at inference. In your plan it is the opposite: the control model is trained on top of base plus baked adapter, and you remove the adapter at inference. Expect a distribution shift of the same kind.

## 3. Why it helps, and the evidence

- **Mechanism.** The adapter absorbs the dataset-specific appearance (watermark, blur, colour/compression statistics), so the downstream module is not forced to learn that appearance. Removing it at inference returns the base model's own look.
- **Evidence is qualitative only.** Paper Sec. 5.3 and Fig. 6 vary the scale from 1 down to 0 and show first frames. As the scale drops, overall quality improves and WebVid-specific content fades, e.g. the watermark. There is no quantitative table for this ablation. Table 1 compares methods only. I could not find any independent numerical replication.
- **The paper never tested "bake into weights, then subtract after training a second model".** That precise recipe is not studied there. In AnimateDiff the second model (the motion module) has separate parameters. It is not a delta on the same weights.

## 4. Related work on weight arithmetic

- **Task arithmetic** (Ilharco et al., ICLR 2023, arXiv 2212.04089, https://arxiv.org/abs/2212.04089). Task vector = `W_ft - W_0`. Negating a vector reduces performance on that task with little change on control tasks. Adding vectors gives multi-task ability. The paper also has an analogy operation. The experiments are CLIP and T5. Relevance: it validates the idea of linear delta addition and subtraction in weight space, but only for deltas fine-tuned from the same pre-trained init.
- **Tangent-space paper** (Ortiz-Jimenez et al., NeurIPS 2023, arXiv 2305.12827, https://arxiv.org/abs/2305.12827). Task arithmetic works to the extent that the weights are "disentangled": different directions in weight space change the network on disjoint input regions. Disentanglement is an emergent property of pre-training. It is stronger with linearized fine-tuning, but present in non-linear models too. It explains why subtraction can leave residual interference.
- **LCM-LoRA** (arXiv 2311.05556, https://arxiv.org/abs/2311.05556). The paper calls the LCM LoRA an "acceleration vector". It combines it linearly with a style LoRA vector: `theta' = theta_pre + lambda1 * tau_style + lambda2 * tau_LCM`. The authors claim it plugs into other SD fine-tunes such as Dreamshaper 7 without training. The diffusers model card uses `adapter_weights=[1.0, 0.8]`, https://huggingface.co/latent-consistency/lcm-lora-sdxl. I found no independent study of how well the transfer works on checkpoints far from the base, nor of interference between the two vectors.
- **ControlNet transfer to other checkpoints.** lllyasviel's Discussion #12 (https://github.com/lllyasviel/ControlNet/discussions/12) gives the formula `AnythingV3_control = AnythingV3 + SD15_control - SD15`. This is applied to the control model weights; the same offset on the base UNet is "less important, but better to have". The discussion warns that simply swapping the base "MAY work but is WRONG" because ControlNet training may unlock some SD layers. It also says the CLIP text-encoder difference must be considered. A later edit marks this "(Obsolete) Some experiments show that results are equally good without such offsets." The pretrained SD1.5 ControlNets do not work on SD2.1-based models.
  - ControlNet-for-Diffusers repeats the caveat that the transfer "may not work always": https://github.com/haofanwang/ControlNet-for-Diffusers
  - Diffusers issue #2722 shows that SD ControlNet checkpoints contain different UNet-side weights, which is consistent with the "unlocked layers" warning: https://github.com/huggingface/diffusers/issues/2722
  - ComfyUI's "Add Difference" model merge `(A - B) * 1.0 + C` and its "Load ControlNet Model (diff)" node implement the same idea. Source: https://comfyanonymous.github.io/ComfyUI_examples/model_merging/
- **Direct reports of "base - adapter + control" failing or succeeding.** I found none. The closest cases are the ControlNet delta transfer (mixed evidence, summarized above) and SparseCtrl requiring the adapter (the opposite direction). I did not find a case where someone trained a control model on a baked adapter and then removed the adapter. Absence of reports is not evidence either way.

## 5. Pitfalls relevant to the plan

1. **Algebra is exact only if the control model starts from the baked weights and only adds a delta.**
   - W_control_trained = W_base + D_a + D_c, where D_c is whatever training changed after the bake.
   - W_base - D_a + D_c is therefore exactly the base plus D_c. That removes the baked adapter in a literal weight sense.
   - D_c was learned in the presence of D_a. It may rely on D_a through non-linear interactions (attention softmax, norms, later layers). Subtracting D_a changes the function the control delta was fitted to. Disentanglement helps but is not guaranteed (Ortiz-Jimenez above).
   - AnimateDiff's SparseCtrl requiring the full adapter is a hint that this reliance is real. The reading is unverified.
2. **Low-rank adapter versus full-weight control training.**
   - If the control model is trained with full fine-tuning, D_c can partly "undo" or "reuse" D_a, because nothing separates them. That makes subtraction less clean.
   - If the control model is itself a LoRA on the baked weights, the two deltas live in separate low-rank subspaces. That is easier to reason about, but they can still overlap.
   - Keeping D_a frozen during control training is required for the algebra above.
3. **bf16 merge precision loss.** A bf16 merge rounds `W + alpha * B A` back to bf16.
   - Reported losses: a bf16 merge keeps about 58-97% of the delta, median 70%, with about 87% of the logit-level effect (https://github.com/on22s/alexandria-audiobook2/issues/1710, which cites https://github.com/unslothai/unsloth-zoo/pull/1765 for a Qwen3-0.6B case with about 41% of the delta lost and about 45% of weights unchanged). A blog measured a max per-layer error of 0.03125 for bf16 merge: https://resolve.ai/blog/debugging-performance-gaps-in-scaling-LoRA-training. These are secondary sources and single reports, not controlled studies.
   - Consequence for this plan: if you bake D_a in bf16, the baked weights are not exactly `W_base + D_a`. Later subtracting D_a does not return exactly W_base. The final model also inherits per-weight rounding noise. Deltas smaller than half a bf16 ulp of the weight (relative 2^-8) are lost, and low-rank deltas tend to be small.
   - Mitigations from these sources:
     - Do the arithmetic in fp32: load the base and the adapter in fp32, compute `W_base - D_a + D_c` once, and round to bf16 once at the end. Keep the original base checkpoint and the adapter A and B factors in fp32.
     - Train the control model on the baked weights stored in fp32 (master weights), or keep the adapter as an unmerged LoRA path during control training so no rounding happens at the bake step.
     - Do not unmerge by subtracting from a rounded sum: diffusers `unfuse_lora` is known to be fragile (https://github.com/huggingface/diffusers/issues/5000, with a clean-restore design described in https://github.com/unslothai/unsloth/pull/16675). Restore from a stored copy of the originals instead.
4. **Training the control model on the bf16-baked weights versus applying the delta to the clean fp32 base.** The training-time weights differ from the inference-time weights by the rounding error plus D_a. The error is small, but a worst case is possible for layers where the control delta is tiny. Evaluate by comparing outputs with and without the final subtraction.
5. **Scale mismatch.** AnimateDiff trains with alpha = 1 and the ablation shows a smooth effect as the scale goes toward 0. A fractional subtraction (for example removing 0.7 D_a) is an obvious tuning knob. Treat it as an experiment, since there is no published evidence for control models.
6. **Text encoder and other modules.** If the adapter touches the text encoder or modules outside the main model, ControlNet's note says the text-encoder difference must be included in the transfer formula (Discussion #12 above). Include every adapted module in the arithmetic.
7. **Domain-removal expectations.** AnimateDiff's evidence is that the adapter-free output looks cleaner (the watermark fades). Nothing shows that the control behaviour stays intact after removal. For your gray-background case, check explicitly that the control model does not depend on the gray-background cue.

## 6. Suggested minimal tests (my suggestion, not from sources)

- Compare four outputs on a fixed seed and prompt set: (a) base, (b) base + D_a, (c) base + D_a + D_c (as trained), (d) base - D_a + D_c, where (d) means subtracting D_a from the trained weights.
- Add an intermediate variant: subtract only part of D_a (scale 0.5) to check how smooth the degradation is.
- Do all arithmetic in fp32 and cast once. Also run a bf16-baked variant to measure the rounding effect directly.
- Evaluate whether the gray background disappears in (d) and whether control adherence matches (c).

## 7. Open items and unverified claims

- Domain adapter LoRA rank, learning rate and step count: not found (supplementary or the official training code; the v3 adapter training code does not appear to be public, UNVERIFIED).
- "Full Domain Adapter needed for SparseCtrl" comes from a fork README. The official config values (`adapter_lora_scale`) were not read.
- The "0.5" recommendation in A1111 workflows is community lore.
- No quantitative evidence for the adapter's benefit was found, only the qualitative Fig. 6.
- Search summaries were the source for the bf16 numbers. I read the Resolve.ai, alexandria and unsloth items only through those summaries.
