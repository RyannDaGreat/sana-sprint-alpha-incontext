"""
FIBO2 trimask model (manifest §14) in the browser (manifest §16): ONNX graphs for ONNX Runtime Web (WebGPU EP, Memory64
build of web/scripts/build_ort_memory64.sh, because the int4 transformer alone is ≈ 4.5 GB, over wasm32's 4 GB).

Graphs (all inputs/outputs float32, batch 1, so no attention mask is needed: nothing is padded):
  transformer   (latents (1,32,LH,LW), cond_latents (1,32,LH,LW), sigma (1,), prompt_embeds (1,6,L,7680)
                 [, adapter_scale (1,)]) -> velocity (1,32,LH,LW). Normalized latents (fibo2.normalize_latents).
                 One call is one model pass; the app runs CFG (two passes against the null prompt) and the Euler step
                 x += (σ' − σ)·v in JS, as fibo2.sample does, so steps and CFG stay user controls (m00638).
  vae_encoder   image (1,3,H,W) in [−1,1] -> normalized latents (mode, BatchNorm normalization inside).
  vae_decoder   normalized latents -> image (1,3,H,W) RGB in [0,1].
  text_encoder  Qwen3-VL-4B language model only (text prompts need no vision tower): input_ids, attention_mask
                -> (1,6,L,7680): six bundles of three RMS-normalized hidden layers (BriaFibo2Pipeline._get_qwen_prompt_embeds).
Quantization: web_quant's SVDQuant weight-only int4 on every linear of the 46 blocks (fibo2.LORA_TARGETS) and of the
text encoder. Runtime λ: the adapter LoRA itself is the delta W_adapter − W_base (exactly rank 64), so each LoRA'd linear
becomes web_export.LambdaLinear(base, B·α/r, A): no SVD, no approximation.

    PYTHONPATH=src uv run python -m sana_alpha.fibo2_web export models/fibo2/fibo2_tri/step_002000 --adapter models/fibo2/fibo2_adapter/step_001000
"""

import json
import re
import time
from pathlib import Path

import fire
import torch
from torch import nn

from sana_alpha import fibo2
from sana_alpha import web_export as we
from sana_alpha import web_quant as wq

WEB_DIR = we.WEB_DIR
OPSET, IR_VERSION = we.OPSET, we.IR_VERSION
TEXT_LAYERS = 36  # Qwen3-VL-4B decoder layers; the deepest one read is 35 (TEXT_ENCODER_OUT_LAYERS)
MIN_LATENT, MAX_LATENT = 32, 256  # latent side (H/8): 256 to 2048 px, even (2x2 patches)


def rotate_real(x, freqs):
    """
    Pure function. RoPE without complex numbers (ONNX has none): x's (even, odd) pairs rotated by (cos, sin).

    Args:
        x (torch.Tensor): (B, S, heads, D)
        freqs (torch.Tensor): (B or 1, S, D/2, 2) float32 (cos, sin) = view_as_real of FIBO2's freqs_cis

    Returns:
        torch.Tensor: (B, S, heads, D) in x's dtype, equal to FIBO2's complex apply_rotary_emb

    Examples:
        >>> x = torch.randn(1, 3, 2, 4); f = torch.polar(torch.ones(1, 3, 2), torch.randn(1, 3, 2))
        >>> ref = torch.view_as_real(torch.view_as_complex(x.reshape(1, 3, 2, 2, 2)) * f.unsqueeze(2)).flatten(3)
        >>> bool(torch.allclose(rotate_real(x, torch.view_as_real(f)), ref, atol=1e-6))
        True
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = freqs[..., 0].unsqueeze(2), freqs[..., 1].unsqueeze(2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(3).type_as(x)


class RealRopeProcessor:
    """FIBO2's BriaFibo2AttnProcessor with rotate_real for the RoPE; freqs_cis arrives as real (…, D/2, 2) tensors."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        """Pure function (given weights). (B, S, dim) -> (B, S, dim), as BriaFibo2AttnProcessor."""
        import torch.nn.functional as F
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        heads = lambda t: t.unflatten(-1, (attn.heads, -1))
        q, k, v = attn.norm_q(heads(attn.to_q(hidden_states))), attn.norm_k(heads(attn.to_k(context))), heads(attn.to_v(context))
        if freqs_cis is not None:
            q, k = rotate_real(q, freqs_cis), rotate_real(k, freqs_cis)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attention_mask)
        return attn.to_out[0](out.transpose(1, 2).flatten(2, 3))


def make_exportable(transformer):
    """
    Command (replaces the RoPE table and every attention processor). FIBO2 with real-valued RoPE: the rope embedder's
    complex tables become view_as_real float32 tables (concatenated on the pair axis), every attention uses
    RealRopeProcessor. Same outputs (tested in the CPU prototype).

    Returns:
        the same transformer
    """
    rope = transformer.rope_embedder
    if not hasattr(rope, "precompute_freqs_cis"):  # already real (a copy of a prepared transformer)
        return transformer
    tables =[torch.view_as_real(t) for t in rope.precompute_freqs_cis(rope.axes_dims, rope.axes_lens, theta=rope.theta)]

    def real_rope(ids):
        """Pure function. (S, 3) int position ids -> (S, Σ d/2, 2) float32 (cos, sin)."""
        return torch.cat([tables[i].to(ids.device)[ids[:, i].long()] for i in range(len(tables))], dim=-2)  # int64: ONNX GatherND takes no int32 indices

    transformer.rope_embedder = real_rope
    for module in transformer.modules():
        if hasattr(module, "set_processor") and hasattr(module, "to_q"):
            module.set_processor(RealRopeProcessor())
    return transformer


class VelocityGraph(nn.Module):
    """One FIBO2 pass with the trimask as a clean context image: the predicted velocity."""

    def __init__(self, transformer):
        """Command (holds the module). transformer: BriaFibo2Transformer2DModel (any LoRA merged or attached)."""
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, cond_latents, sigma, prompt_embeds):
        """
        Pure function (given fixed weights). (1,32,LH,LW) x2, (1,), (1,6,L,C) -> (1,32,LH,LW) float32 velocity v = ε − x_0.
        """
        dtype = self.transformer.dtype
        return self.transformer(hidden_states=latents.to(dtype), timestep=sigma, encoder_hidden_states=prompt_embeds.to(dtype),
                                context_latents=[cond_latents.to(dtype)], return_dict=False)[0].float()


class LambdaVelocityGraph(VelocityGraph):
    """VelocityGraph with the adapter fraction λ as one more input (adapter_scale), web_export.LambdaStepTransformer's analogue."""

    def __init__(self, transformer, scale):
        """Command (holds the module). transformer: prepared with attach_lora_lambda(..., scale)."""
        super().__init__(transformer)
        self.scale = scale

    def forward(self, latents, cond_latents, sigma, prompt_embeds, adapter_scale):
        """Pure function (given fixed weights). As VelocityGraph; adapter_scale (1,) float32 λ."""
        self.scale.value = adapter_scale
        return super().forward(latents, cond_latents, sigma, prompt_embeds)


def target_linears(transformer):
    """
    Query. The FIBO2 linears to quantize: every linear of the 46 blocks (fibo2.LORA_TARGETS), the same set the LoRAs train.

    Returns:
        dict: qualified name -> nn.Linear (322 for FIBO2: 46 blocks × 7)
    """
    pattern = re.compile(fibo2.LORA_TARGETS + "$")
    return {n: m for n, m in transformer.named_modules() if isinstance(m, nn.Linear) and pattern.match(n)}


def quantize_linears(root, linears, inputs, bits, rank, device):
    """
    Command. Replaces each linear of root by a web_quant.QuantLinear (SVDQuant weight-only), like
    web_export.quantize_linears but on any device (the CPU prototype runs here too).

    Args:
        root (nn.Module): owner of the dotted names
        linears (dict): name -> nn.Linear
        inputs (dict): name -> (T, K) float32 calibration inputs (consumed)
        bits (int), rank (int): see web_quant.svdquant_weight
        device (str): where the SVDs run

    Returns:
        dict: name -> {alpha, rel_error}
    """
    report = {}
    for name, linear in linears.items():
        with torch.no_grad():
            q = wq.svdquant_weight(linear.weight.float().to(device), inputs.pop(name).to(device), bits, rank)
        wq.set_module(root, name, wq.QuantLinear(q, linear.bias, bits).to(linear.weight.device))
        report[name] = dict(alpha=q["alpha"], rel_error=round(q["rel_error"], 5))
    return report


def attach_lora_lambda(transformer, lora, scale):
    """
    Command (replaces modules). Adds λ·(W_adapter − W_base) as the adapter LoRA itself: each LoRA'd linear (or the
    QuantLinear in its place) becomes web_export.LambdaLinear(base, up = B·α/r, down = A).

    Args:
        transformer (nn.Module): weights W(λ = 0) = FIBO2 + control LoRA, possibly quantized
        lora (dict): the adapter's fibo2 LoRA state ("<m>.lora_A.weight", "<m>.lora_B.weight", "<m>.alpha")
        scale (web_export.AdapterScale)

    Returns:
        int: number of LambdaLinear modules

    Examples:
        >>> lin = nn.Linear(3, 2, bias=False); s = we.AdapterScale("cpu"); root = nn.Sequential(lin)
        >>> lora = {"0.lora_A.weight": torch.ones(1, 3), "0.lora_B.weight": torch.ones(2, 1), "0.alpha": torch.tensor(1.0)}
        >>> attach_lora_lambda(root, lora, s); s.value = torch.ones(1); x = torch.ones(1, 3)
        1
        >>> bool(torch.allclose(root(x) - lin(x), x @ fibo2.lora_delta(lora, "0").T))
        True
    """
    names = sorted({k.rsplit(".lora_A.weight", 1)[0] for k in lora if k.endswith(".lora_A.weight")})
    for name in names:
        a, b = lora[f"{name}.lora_A.weight"].float(), lora[f"{name}.lora_B.weight"].float()
        module = transformer.get_submodule(name)
        dtype = next(module.buffers(), next(module.parameters(), None)).dtype
        dtype = torch.float16 if dtype == torch.uint8 else dtype  # a QuantLinear's first buffer is its packed codes
        up, down = (b * float(lora[f"{name}.alpha"]) / a.shape[0]).to(dtype), a.to(dtype)
        device = next(module.buffers(), next(module.parameters(), None)).device
        wq.set_module(transformer, name, we.LambdaLinear(module, up.to(device), down.to(device), None, scale))
    return len(names)


def example_inputs(config, latent_side, text_len, generator=None):
    """
    Query (reads the RNG). Random graph inputs for tracing and numerics: latents, cond_latents, sigma, prompt_embeds.

    Args:
        config: the transformer's config (cap_feat_dim, injection_layer_ids)
        latent_side (int): LH = LW (even)
        text_len (int): L prompt tokens

    Returns:
        tuple of float32 CPU tensors
    """
    bundles = 1 + len(config.injection_layer_ids)
    latent = lambda: torch.randn(1, 32, latent_side, latent_side, generator=generator)
    return latent(), latent(), torch.tensor([0.7]), torch.randn(1, bundles, text_len, config.cap_feat_dim, generator=generator)


INPUT_NAMES = ["latents", "cond_latents", "sigma", "prompt_embeds"]


def named_inputs(inputs, adapter_scale=None):
    """
    Pure function. Graph feed dict from example_inputs (+ λ).

    Examples:
        >>> sorted(named_inputs((1, 2, 3, 4), 5))
        ['adapter_scale', 'cond_latents', 'latents', 'prompt_embeds', 'sigma']
    """
    named = dict(zip(INPUT_NAMES, inputs))
    return named if adapter_scale is None else {**named, "adapter_scale": adapter_scale}


def gist_counts(config):
    """
    Pure function. Every gist-token count FIBO2 can use (min to max in gist_step), with the longest text length (in
    tokens) that still gets each: FIBO2 maps text lengths min_text_len … max_text_len linearly onto min … max gists,
    rounded up to gist_step (diffusers compute_gist_count). One transformer graph per count (fixed_gist): the count is a
    Python int inside FIBO2's forward, so a traced graph has one.

    Returns:
        list of (gist count, longest text length)

    Examples:
        >>> from types import SimpleNamespace as C
        >>> gist_counts(C(min_num_gist_tokens=256, max_num_gist_tokens=512, gist_step=64, gist_min_text_len=256, gist_max_text_len=2048))
        [(256, 256), (320, 704), (384, 1152), (448, 1600), (512, 2048)]
    """
    from diffusers.models.transformers.transformer_bria_fibo2 import compute_gist_count
    args = dict(min_gist=config.min_num_gist_tokens, max_gist=config.max_num_gist_tokens, gist_step=config.gist_step,
                min_text_len=config.gist_min_text_len, max_text_len=config.gist_max_text_len)
    lengths = range(1, config.gist_max_text_len + 1)
    counts = compute_gist_count(list(lengths), **args)
    return [(g, max(n for n, c in zip(lengths, counts) if c == g)) for g in sorted(set(counts))]


class fixed_gist:
    """Context manager: FIBO2's forward uses `count` gist tokens whatever the text length (for tracing one graph per count)."""

    def __init__(self, count):
        """Command (keeps the count)."""
        self.count = count

    def __enter__(self):
        """Command (patches diffusers' compute_gist_count in the FIBO2 module)."""
        from diffusers.models.transformers import transformer_bria_fibo2 as module
        self.module, self.original = module, module.compute_gist_count
        module.compute_gist_count = lambda text_lengths, **_: [self.count] * len(text_lengths)

    def __exit__(self, *exc):
        """Command (restores compute_gist_count)."""
        self.module.compute_gist_count = self.original


def export_velocity(model, path, inputs, adapter_scale=False, gist=None):
    """
    Command. Traces a VelocityGraph (dynamo exporter, MatMulNBits translation for QuantLinear) and writes path
    (+ path.data). Dynamic: latent side LH, LW (even, 256–2048 px) and text length L; the gist count is fixed.

    Args:
        model (VelocityGraph | LambdaVelocityGraph)
        path (Path): e.g. models/web/<version>/transformer_g256.onnx
        inputs (tuple): example_inputs on the model's device
        adapter_scale (bool): model takes λ
        gist (int | None): gist-token count of this graph (gist_counts); default the config's minimum
    """
    began = time.perf_counter()
    half = lambda n: torch.export.Dim(n, min=MIN_LATENT // 2, max=MAX_LATENT // 2)
    LH, LW, L = 2 * half("LH2"), 2 * half("LW2"), torch.export.Dim("L", min=4, max=4096)
    latent = {2: LH, 3: LW}
    dynamic = [latent, latent, None, {2: L}] + [None] * adapter_scale
    args = (*inputs, torch.tensor([0.5], device=inputs[0].device)) if adapter_scale else inputs
    gist = model.transformer.config.min_num_gist_tokens if gist is None else gist
    with torch.no_grad(), fixed_gist(gist):
        program = torch.onnx.export(model, args, dynamo=True, opset_version=OPSET, optimize=False,
                                    input_names=INPUT_NAMES + ["adapter_scale"] * adapter_scale, output_names=["velocity"],
                                    dynamic_shapes=dynamic,
                                    custom_translation_table={torch.ops.websana.matmul_nbits.default: wq.matmul_nbits_onnx})
    program.model.ir_version = IR_VERSION
    for old in (path, path.with_name(path.name + ".data")):
        old.unlink(missing_ok=True)
    program.save(str(path), external_data=True)
    print(f"export transformer: {path} in {time.perf_counter() - began:.0f} s", flush=True)


def prototype_models(transformer, inputs, rank=8, lora_rank=4):
    """
    Command (copies and quantizes the transformer). The three graph variants of a (tiny) FIBO2 for the numerics check:
    fp32, int4 (SVDQuant, calibrated on the given inputs), int4 + λ branch from a random adapter LoRA.

    Args:
        transformer: BriaFibo2Transformer2DModel float32 on CPU
        inputs (tuple): example_inputs
        rank (int): SVDQuant low-rank rank (web_quant.RANK = 32 at full size; small here for tiny dims)
        lora_rank (int): rank of the random adapter LoRA

    Yields:
        (label, module, adapter_scale tensor or None)
    """
    import copy
    yield "fp32", VelocityGraph(transformer).eval(), None
    q = copy.deepcopy(transformer)
    linears = target_linears(q)
    calib = we.record_inputs(linears, lambda: VelocityGraph(q)(*inputs), rows_per_item=256)
    quantize_linears(q, linears, calib, 4, rank, "cpu")
    yield "int4", VelocityGraph(q).eval(), None
    generator = torch.Generator().manual_seed(1)
    lora = {}
    for name, m in target_linears(transformer).items():
        lora[f"{name}.lora_A.weight"] = torch.randn(lora_rank, m.in_features, generator=generator) * 0.02
        lora[f"{name}.lora_B.weight"] = torch.randn(m.out_features, lora_rank, generator=generator) * 0.02
        lora[f"{name}.alpha"] = torch.tensor(float(lora_rank))
    scale = we.AdapterScale("cpu")
    attach_lora_lambda(q, lora, scale)
    yield "int4_lambda", LambdaVelocityGraph(q, scale).eval(), torch.tensor([0.5])


class VaeEncoder(nn.Module):
    """FLUX.2 VAE encoder for the trimask: image in [−1, 1] -> normalized latents (mode, then fibo2.normalize_latents)."""

    def __init__(self, vae, bn_mean, bn_std):
        """Command (holds the module and the BatchNorm statistics). vae: AutoencoderKLFlux2; bn: fibo2.bn_stats."""
        super().__init__()
        self.vae = vae
        self.register_buffer("bn_mean", bn_mean)
        self.register_buffer("bn_std", bn_std)

    def forward(self, image):
        """Pure function (given fixed weights). (1, 3, H, W) float32 in [−1, 1] -> (1, 32, H/8, W/8) float32 normalized."""
        z = self.vae.encode(image.to(self.vae.dtype)).latent_dist.mode()
        return fibo2.normalize_latents(z, self.bn_mean, self.bn_std)


class VaeDecoder(nn.Module):
    """FLUX.2 VAE decoder: normalized latents -> RGB in [0, 1] (generate_images' postprocess)."""

    def __init__(self, vae, bn_mean, bn_std):
        """Command (holds the module and the BatchNorm statistics)."""
        super().__init__()
        self.vae = vae
        self.register_buffer("bn_mean", bn_mean)
        self.register_buffer("bn_std", bn_std)

    def forward(self, latents):
        """Pure function (given fixed weights). (1, 32, LH, LW) float32 -> (1, 3, 8·LH, 8·LW) float32 RGB in [0, 1]."""
        z = fibo2.denormalize_latents(latents, self.bn_mean, self.bn_std).to(self.vae.dtype)
        return (self.vae.decode(z).sample.float().clamp(-1, 1) + 1) / 2


class TextEncoder(nn.Module):
    """
    Qwen3-VL-4B's language model only (text prompts: no image markers, so no vision tower) up to the deepest layer FIBO2
    reads, then BriaFibo2Pipeline._get_qwen_prompt_embeds' bundles: every read layer scaled to unit RMS per token in
    float32, three layers concatenated per bundle.
    """

    def __init__(self, language_model, layers):
        """
        Command (holds the module; drops the decoder layers after the deepest read one, so the graph has no unused weights).

        Args:
            language_model: Qwen3VLForConditionalGeneration.model.language_model
            layers (tuple): TEXT_ENCODER_OUT_LAYERS of the pipeline, e.g. ((9, 20, 31), (5, 16, 27), …)
        """
        super().__init__()
        self.layers = layers
        language_model.layers = language_model.layers[:max(max(b) for b in layers)]
        language_model.norm = nn.Identity()  # HF puts the final norm on the last hidden state: after the cut that is a read layer
        self.model = language_model

    def forward(self, input_ids, attention_mask):
        """Pure function (given fixed weights). (1, T) int64 x2 -> (1, len(layers), T, 3·hidden) float32."""
        states = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        unit = {i: (s := states[i].float()) * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) for i in sorted({i for b in self.layers for i in b})}
        return torch.stack([torch.cat([unit[i] for i in bundle], -1) for bundle in self.layers], 1)


def calibration_inputs(model, eval_dirs=("resources/showcase_trimask_held",), items_per_dir=4, sigmas=(0.95, 0.6, 0.25)):
    """
    untested. Query (runs the GPU). Graph inputs for SVDQuant calibration: trimask items of the showcase sets, each at a
    few noise levels, encoded as the shipped model encodes them (fibo2.generate_images).

    Args:
        model (dict): fibo2.load_eval_model of the checkpoint being exported
        eval_dirs (tuple): showcase trimask sets
        items_per_dir (int): items per set
        sigmas (tuple): noise levels per item

    Returns:
        list of VelocityGraph input tuples (on the GPU)
    """
    from sana_alpha.eval_set import load_eval_set
    vae, bn, device = model["vae"], model["bn"], model["vae"].device
    out = []
    for d in eval_dirs:
        for method_dir in sorted(p for p in (we.ROOT / d).iterdir() if p.is_dir()):
            data = load_eval_set(method_dir)
            for item, trimask in list(zip(data["items"], data["mattes"]))[:items_per_dir]:
                x = torch.from_numpy(trimask).float().to(device).mul(255).round().div(127.5).sub(1).expand(1, 3, *trimask.shape)
                cond = fibo2.normalize_latents(vae.encode(x.to(vae.dtype)).latent_dist.mode(), *bn)
                embeds, _ = fibo2.encode_prompts(*model["text"], [fibo2.caption_prompt(item["caption"])])
                for s in sigmas:
                    noisy = s * torch.randn_like(cond) + (1 - s) * cond  # x_t with the trimask latent standing in for x_0 (same scale)
                    out.append((noisy, cond, torch.tensor([s], device=device), embeds.float()))
    return out


def export(checkpoint, adapter, version=None, bits=4, rank=wq.RANK, rows_per_item=128):
    """
    untested (full size; the CPU prototype in agents/.scratchpad/scratchpad_agent_fibo2web.py tests the same path on a
    tiny FIBO2). Command (needs one free 80 GB GPU; estimate ≈ 60–90 GPU-min). Writes models/web/<version>/:
    transformer_g<G>.onnx for every gist count G (gist_counts: 256…512, one shared transformer.onnx.data, int4 + λ),
    text_encoder.onnx (+ .data, int4), vae_encoder.onnx / vae_decoder.onnx (fp16), the tokenizer, manifest.json.
    Size estimate: transformer ≈ 4.6 GB int4 + ≈ 0.4 GB λ branch, text encoder ≈ 2.8 GB (int4 + the 0.78 GB fp16
    embedding table), VAE ≈ 0.2 GB: ≈ 8 GB.

    Args:
        checkpoint (str): train_fibo2 control step folder, e.g. models/fibo2/fibo2_tri/step_002000
        adapter (str): the adapter step the control trained on, e.g. models/fibo2/fibo2_adapter/step_001000 (the λ branch)
        version (str | None): default fibo2_<run>_step_<N>-int4-lambda
        bits, rank, rows_per_item: see web_export.quantize
    """
    from safetensors.torch import load_file
    from sana_alpha import models
    checkpoint, device = Path(checkpoint), "cuda"
    version = version or f"fibo2_{checkpoint.parent.name}_{checkpoint.name}-int{bits}-lambda"
    folder = WEB_DIR / version
    folder.mkdir(parents=True, exist_ok=True)
    model = fibo2.load_eval_model(str(checkpoint), device)
    transformer = model["transformer"].half()
    make_exportable(transformer)
    graph = VelocityGraph(transformer).eval()
    calib = calibration_inputs(model)
    linears = target_linears(transformer)
    inputs = we.record_inputs(linears, lambda: [graph(*c) for c in calib], rows_per_item)
    report = quantize_linears(transformer, linears, inputs, bits, rank, device)
    scale = we.AdapterScale(device)
    attach_lora_lambda(transformer, load_file(models.localize(adapter) / "lora.safetensors"), scale)
    graph = LambdaVelocityGraph(transformer, scale).eval()
    first = None
    for gist, longest in gist_counts(transformer.config):
        path = folder / f"transformer_g{gist}.onnx"
        export_velocity(graph, path, calib[0], adapter_scale=True, gist=gist)
        if first is None:
            first = path
        else:
            we.share_external_data(path, first)  # same weights: one transformer data file
    (folder / "quantization.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {folder}: transformer graphs for gist counts {[g for g, _ in gist_counts(transformer.config)]}", flush=True)


if __name__ == "__main__":
    fire.Fire()
