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

import gc
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


MIN_QUANT_SIDE = 256  # all_linears: smaller linears (embedders, the final projection) stay fp16


def target_linears(transformer, all_linears=False):
    """
    Query. The FIBO2 linears to quantize: every linear of the 46 blocks (fibo2.LORA_TARGETS), the same set the LoRAs train;
    with all_linears, also every other linear with both sides ≥ MIN_QUANT_SIDE (the Perceiver, the injection blocks'
    cross-attention, adaLN modulation: 2.1 GB of fp16 in the int4 version of step 2000).

    Returns:
        dict: qualified name -> nn.Linear (322 for FIBO2's blocks)
    """
    pattern = re.compile(fibo2.LORA_TARGETS + "$")
    big = lambda n, m: min(m.in_features, m.out_features) >= MIN_QUANT_SIDE and not n.startswith("t_embedder")  # FIBO2 reads t_embedder's .weight.dtype
    return {n: m for n, m in transformer.named_modules() if isinstance(m, nn.Linear) and (pattern.match(n) or (all_linears and big(n, m)))}


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


MAX_TEXT_TOKENS = 256  # gist_counts(config)[0][1]: prompts up to this many tokens get the minimum gist count, the one the graph is traced with
PART_LIMIT = 1_900_000_000  # bytes per external-data file: the app reads each file into one ArrayBuffer (Chrome: < 2 GB)
DEFAULT_STEPS, DEFAULT_GUIDANCE = 20, 5.0  # the training previews' sampler (train_fibo2: 20 steps, CFG 5)
BASES = dict(dev=dict(dir=fibo2.FIBO2_DIR, steps=DEFAULT_STEPS, guidance=DEFAULT_GUIDANCE, cfg=True, symbol="W_fibo2"),
             turbo=dict(dir=fibo2.TURBO_DIR, steps=4, guidance=1.0, cfg=False, symbol="W_turbo"))  # turbo: Bria's 4-step distilled model, no CFG (the control transfers onto it, manifest §14)
TRAINED_SIZES = [1024, 768, 512, 256]  # §13 multires, as train_fibo2


def split_external_data(path, limit=PART_LIMIT):
    """
    Command (rewrites path, replaces path.data). Moves the external tensors of ONNX graph `path` into files
    <path>.p<k>.data of at most `limit` bytes each, in tensor order (the app reads each file into one ArrayBuffer).

    Returns:
        list[str]: the new file names
    """
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    source, names, used, out = path.with_name(path.name + ".data"), [], limit, None
    with open(source, "rb") as f:
        for t in model.graph.initializer:
            if t.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = {e.key: e.value for e in t.external_data}
            f.seek(int(info.get("offset", 0)))
            data = f.read(int(info["length"]))
            assert len(data) <= limit, f"{t.name}: {len(data)} bytes, over the {limit}-byte file limit"
            if used + len(data) > limit:
                out and out.close()
                names.append(f"{path.name}.p{len(names)}.data")
                out, used = open(path.with_name(names[-1]), "wb"), 0
            del t.external_data[:]
            for key, value in (("location", names[-1]), ("offset", str(used)), ("length", str(len(data)))):
                entry = t.external_data.add()
                entry.key, entry.value = key, value
            out.write(data)
            used += len(data)
    out.close()
    source.unlink()
    onnx.save(model, str(path))
    return names


def eval_items(eval_dir="resources/showcase_trimask", split="train", per_method=1):
    """
    Query. (caption, trimask (S, S) float32 in [0, 1]) pairs from the showcase trimask sets: `per_method` items of the
    given split from every method. Calibration uses "train" (held-out items are never calibrated on); verify uses "held-out".
    """
    from sana_alpha.eval_set import load_eval_set
    out = []
    for method_dir in sorted(p for p in (we.ROOT / eval_dir).iterdir() if p.is_dir()):
        data = load_eval_set(method_dir)
        chosen = [k for k, item in enumerate(data["items"]) if item["split"] == split][:per_method]
        out += [(data["items"][k]["caption"], data["mattes"][k]) for k in chosen]
    return out


def encode_condition(model, trimask):
    """Query (runs the GPU). A trimask (S, S) in [0, 1] -> normalized latents (1, 32, S/8, S/8), as fibo2.generate_images."""
    vae, bn = model["vae"], model["bn"]
    x = torch.from_numpy(trimask).float().to(vae.device).mul(255).round().div(127.5).sub(1).expand(1, 3, *trimask.shape)
    return fibo2.normalize_latents(vae.encode(x.to(vae.dtype)).latent_dist.mode(), *bn)


def calibration_inputs(model, items, sigmas=(0.9, 0.4)):
    """
    Query (runs the GPU). VelocityGraph inputs for SVDQuant calibration: each item at a few noise levels, x_t with the
    trimask latent standing in for x_0 (the same scale; no clean target is needed to record linear inputs).

    Returns:
        list of (latents, cond_latents, sigma, prompt_embeds) on the GPU
    """
    out = []
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for caption, trimask in items:
            cond = encode_condition(model, trimask)
            embeds, _ = fibo2.encode_prompts(*model["text"], [fibo2.caption_prompt(caption)])
            for s in sigmas:
                noise = torch.randn(cond.shape, generator=generator, device="cuda")
                out.append((s * noise + (1 - s) * cond, cond, torch.tensor([s], device="cuda"), embeds.float()))
    return out


def graph_specs():
    """
    Pure function. Input/output names, shapes and dtypes of the FIBO2 graphs (strings are dynamic: LH = H/8, L tokens, T).

    Examples:
        >>> [n for n, _, _ in graph_specs()["transformer"]["inputs"]]
        ['latents', 'cond_latents', 'sigma', 'prompt_embeds', 'adapter_scale']
    """
    latent, image = [1, 32, "LH", "LW"], [1, 3, "H", "W"]
    return dict(
        transformer=dict(inputs=[("latents", latent, "float32"), ("cond_latents", latent, "float32"), ("sigma", [1], "float32"),
                                 ("prompt_embeds", [1, 6, "L", 7680], "float32"), ("adapter_scale", [1], "float32")],
                         outputs=[("velocity", latent, "float32")]),
        vae_encoder=dict(inputs=[("image", image, "float32")], outputs=[("latent", latent, "float32")]),
        vae_decoder=dict(inputs=[("latent", latent, "float32")], outputs=[("image", image, "float32")]),
        text_encoder=dict(inputs=[("input_ids", [1, "T"], "int64"), ("attention_mask", [1, "T"], "int64")],
                          outputs=[("prompt_embeds", [1, 6, "T", 7680], "float32")]))


def export_plain(module, name, path, args, dims):
    """Command. torch.onnx (dynamo) export of a non-quantized wrapper with graph_specs names; writes path (+ .data)."""
    spec = graph_specs()[name]
    began = time.perf_counter()
    with torch.no_grad():
        program = torch.onnx.export(module, args, dynamo=True, opset_version=OPSET, optimize=False,
                                    input_names=[n for n, _, _ in spec["inputs"]], output_names=[n for n, _, _ in spec["outputs"]],
                                    dynamic_shapes=dims,
                                    custom_translation_table={torch.ops.websana.matmul_nbits.default: wq.matmul_nbits_onnx})
    program.model.ir_version = IR_VERSION
    for old in (path, path.with_name(path.name + ".data")):
        old.unlink(missing_ok=True)
    program.save(str(path), external_data=True)
    print(f"export {name}: {path} in {time.perf_counter() - began:.0f} s", flush=True)


def text_encoder_parts(model):
    """
    Query. The pipeline's prompt template and hidden-layer bundles (diffusers BriaFibo2Pipeline module constants).

    Returns:
        tuple: (template str with one {}, layers tuple of 3-tuples)
    """
    from diffusers.pipelines.bria_fibo2 import pipeline_bria_fibo2 as pipe
    return pipe.PROMPT_TEMPLATE, pipe.TEXT_ENCODER_OUT_LAYERS


def export(checkpoint, version=None, bits=4, text_bits=None, all_linears=False, rows_per_item=32, base="dev"):
    """
    Command (one GPU, ≈ 28 GB: the models take turns on it; ≈ 60–90 min). Writes models/web/<version>/, the
    web-model-contract folder of a FIBO2 control checkpoint, with runtime λ (manifest §16):
      transformer.onnx (+ .p<k>.data)   FIBO2 + control LoRA, SVDQuant int4, the adapter LoRA as the exact λ branch, fp16
      text_encoder.onnx (+ .p<k>.data)  Qwen3-VL-4B language model, SVDQuant int4, fp16
      vae_encoder.onnx / vae_decoder.onnx  FLUX.2 VAE, fp32 (small; no fp16 risk in the convolutions)
      tokenizer.json, tokenizer_config.json, manifest.json
    Calibration on the "train" items of resources/showcase_trimask (held-out items are never used).

    Args:
        checkpoint (str): train_fibo2 control step folder, e.g. models/fibo2/fibo2_tri/step_002000 (its meta.json names the adapter)
        version (str | None): default fibo2_<run>_step_<NNNNNN>-int<bits>-lambda (-int<bits>te<text_bits>- when they differ)
        bits (int): transformer weights, 4 (SVDQuant rank 32) or 8 (rank 0, as web_export)
        text_bits (int | None): text encoder weights; default = bits. 8 is needed for images close to PyTorch (manifest §16)
        all_linears (bool): also quantize the transformer's linears outside the blocks (target_linears)
        rows_per_item (int): calibration tokens per linear per forward
        base (str): "dev" (FIBO2 + control) or "turbo" (the control transferred onto Fibo-2.0-edit-turbo, 4 steps, no CFG;
            version prefix fibo2_turbo_); the λ branch is the adapter LoRA (W_adapter − W_dev) either way, as SANA's on Sprint
    """
    import shutil
    from safetensors.torch import load_file
    from sana_alpha import models
    checkpoint = Path(checkpoint)
    meta = json.loads((models.localize(checkpoint) / "meta.json").read_text())
    adapter = meta["adapter"]
    text_bits = bits if text_bits is None else text_bits
    rank_of = lambda b: wq.RANK if b == 4 else 0
    tag = f"int{bits}" if text_bits == bits else f"int{bits}te{text_bits}"
    version = version or f"fibo2_{'turbo_' if base == 'turbo' else ''}{meta['run']}_{checkpoint.name}-{tag}-lambda"
    folder = WEB_DIR / version
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)  # randomized SVD: the same weights give the same files
    model = fibo2.load_eval_model(str(checkpoint), "cuda", BASES[base]["dir"])
    calib = calibration_inputs(model, eval_items())
    print(f"calibration: {len(calib)} transformer inputs", flush=True)
    template, layers = text_encoder_parts(model)
    encoder, processor = model["text"]
    prompts = [template.format(fibo2.caption_prompt(c)) for c, _ in eval_items(per_method=2)] + [template.format(fibo2.NULL_PROMPT)]
    tokens = [processor(text=[p], return_tensors="pt").to("cuda") for p in prompts]
    encoder.cpu()  # the models take turns on the GPU
    torch.cuda.empty_cache()
    quantization = {}

    lora = load_file(models.localize(adapter) / "lora.safetensors")
    lora_rank = lora[next(k for k in lora if k.endswith(".lora_A.weight"))].shape[0]
    record = folder / "quantization_transformer.json"  # written when the transformer graph is done: a rerun skips it
    if record.exists():
        quantization["transformer"] = json.loads(record.read_text())
        print(f"transformer: already exported ({record})", flush=True)
    else:
        transformer = model["transformer"].half()
        make_exportable(transformer)
        graph = VelocityGraph(transformer).eval()
        linears = target_linears(transformer, all_linears)
        report = quantize_linears(transformer, linears, we.record_inputs(linears, lambda: [graph(*c) for c in calib], rows_per_item),
                                  bits, rank_of(bits), "cuda")
        del linears, graph  # the dict still holds the replaced fp16 linears
        scale = we.AdapterScale("cuda")
        attach_lora_lambda(transformer, lora, scale)
        export_velocity(LambdaVelocityGraph(transformer, scale).eval(), folder / "transformer.onnx", calib[0], adapter_scale=True,
                        gist=transformer.config.min_num_gist_tokens)
        split_external_data(folder / "transformer.onnx")
        quantization["transformer"] = quant_record(report, bits, rank_of(bits), rows_per_item)
        record.write_text(json.dumps(quantization["transformer"]))
        del transformer, scale
    del calib
    model["transformer"] = None
    gc.collect()
    torch.cuda.empty_cache()

    text = TextEncoder(encoder.model.language_model.half().cuda(), layers).eval()
    linears = {n: m for n, m in text.model.layers.named_modules(prefix="model.layers") if isinstance(m, nn.Linear)}
    report = quantize_linears(text, linears, we.record_inputs(linears, lambda: [text(t.input_ids, t.attention_mask) for t in tokens], rows_per_item),
                              text_bits, rank_of(text_bits), "cuda")
    T = torch.export.Dim("T", min=2, max=4096)
    export_plain(text, "text_encoder", folder / "text_encoder.onnx", (tokens[0].input_ids, tokens[0].attention_mask), [{1: T}, {1: T}])
    split_external_data(folder / "text_encoder.onnx")
    quantization["text_encoder"] = quant_record(report, text_bits, rank_of(text_bits), rows_per_item)
    del text
    torch.cuda.empty_cache()

    vae = model["vae"].float()
    bn = [b.cuda() for b in model["bn"]]
    H, W = torch.export.Dim("H8", min=MIN_LATENT, max=MAX_LATENT), torch.export.Dim("W8", min=MIN_LATENT, max=MAX_LATENT)
    export_plain(VaeEncoder(vae, *bn).eval(), "vae_encoder", folder / "vae_encoder.onnx", (torch.zeros(1, 3, 512, 512, device="cuda"),), [{2: 8 * H, 3: 8 * W}])
    export_plain(VaeDecoder(vae, *bn).eval(), "vae_decoder", folder / "vae_decoder.onnx", (torch.zeros(1, 32, 64, 64, device="cuda"),), [{2: H, 3: W}])
    quantization.update(vae_encoder={"fp32": "vae_encoder.onnx"}, vae_decoder={"fp32": "vae_decoder.onnx"})

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(models.localize(f"{fibo2.FIBO2_DIR}/processor") / name, folder / name)
    write_manifest(folder, version, str(checkpoint), meta, adapter, lora_rank, template, layers, quantization, base)


def quant_record(report, bits, rank, rows_per_item):
    """Pure function. The manifest's quantization entry of one graph from quantize_linears' report."""
    import numpy as np
    errors = [r["rel_error"] for r in report.values()]
    return dict(method="SVDQuant weight-only (arXiv 2411.05007), src/sana_alpha/web_quant.py", bits=bits, group_size=wq.BLOCK,
                symmetric=True, scales="fp16", lowrank_rank=rank, activations="fp16",
                calibration=f"resources/showcase_trimask train items, {rows_per_item} tokens/linear/forward",
                quantized_linears=len(report), mean_output_rel_error=float(np.mean(errors)), max_output_rel_error=float(np.max(errors)))


def write_manifest(folder, version, checkpoint, meta, adapter, lora_rank, template, layers, quantization, base="dev"):
    """Command. Writes folder/manifest.json in the exporter format of the web-model-contract (manifest §12, §16), family "fibo2"."""
    files = sorted(p.name for p in folder.iterdir() if p.suffix in (".onnx", ".data") or p.name.startswith("tokenizer"))
    manifest = dict(
        version=version, family="fibo2", base=base, source=checkpoint, cond="trimask", trained_sizes=TRAINED_SIZES,
        weights=f"{BASES[base]['symbol']} + λ·(W_adapter − W_fibo2) + (W_control − W_adapter) · {meta['run']} step {meta['step']}",
        adapter=dict(source=adapter, rank=lora_rank, delta="exact: the adapter LoRA itself (up = B·α/r, down = A)"),
        files={f: (folder / f).stat().st_size for f in files}, graphs=graph_specs(), quantization=quantization,
        text_encoder=dict(model=f"{fibo2.FIBO2_DIR}/text_encoder (Qwen3-VL-4B language model, cut after layer {max(max(b) for b in layers)})",
                          template=template, caption_key=fibo2.CAPTION_KEY, null_prompt=fibo2.NULL_PROMPT, max_tokens=MAX_TEXT_TOKENS,
                          layers=layers, output="(1, 6, T, 7680): every token, no padding, no mask"),
        sampler=dict(kind="flow_euler_cfg", steps=BASES[base]["steps"], guidance=BASES[base]["guidance"], cfg=BASES[base]["cfg"], shift_seq=list(fibo2.SHIFT_SEQ),
                     shift_mu=list(fibo2.SHIFT_MU),
                     schedule="u_k = linspace(1, 1/steps, steps), σ_k = e^mu / (e^mu + 1/u_k − 1), σ_steps = 0; mu linear in image tokens (LH·LW/4)",
                     step="v = transformer(x, σ_k, prompt); CFG: v = v_null + guidance·(v − v_null) (null_prompt); x += (σ_{k+1} − σ_k)·v",
                     latents="standard-normal noise (1, 32, H/8, W/8), normalized latent space"),
        vae=dict(latent_factor=8, matte_input="trimask 8-bit v, 2·v/255 − 1, copied to 3 channels",
                 note="vae_encoder normalizes and vae_decoder denormalizes inside the graph", decoder_output="RGB in [0, 1]"))
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {folder / 'manifest.json'}: {sum(manifest['files'].values()) / 1e9:.2f} GB", flush=True)


def ort_sessions(folder, providers=("CUDAExecutionProvider",), chunks=False):
    """
    Query (reads the graphs). One onnxruntime session per graph of an exported FIBO2 version; with chunks, the
    transformer is its chained transformer.chunk*.onnx graphs (web_chunks.split).
    """
    import onnxruntime as ort
    from sana_alpha import web_chunks
    options = ort.SessionOptions()
    make = lambda path: ort.InferenceSession(str(path), options, providers=list(providers))
    return {name: web_chunks.ChainedSession(folder, make) if chunks and name == "transformer" else make(folder / f"{name}.onnx")
            for name in graph_specs()}


def onnx_generate(sessions, tokenizer_ids, null_ids, trimask, noise, steps, guidance, adapter_scale=0.0):
    """
    Query (runs the sessions). The app's FIBO2 pipeline in numpy (web/src/pipeline.js mirrors it): text encoder for the
    prompt and the null prompt, vae_encoder on the trimask, the flow Euler sampler with CFG, vae_decoder.

    Args:
        sessions (dict): ort_sessions
        tokenizer_ids, null_ids (np.ndarray): (1, T) int64 token ids of the templated prompt and null prompt
        trimask (np.ndarray): (S, S) float32 in [0, 1]
        noise (np.ndarray): (1, 32, S/8, S/8) float32 standard normal
        steps (int), guidance (float), adapter_scale (float): sampler settings and λ

    Returns:
        np.ndarray: (S, S, 3) float32 RGB in [0, 1]
    """
    import numpy as np
    embed = lambda ids: sessions["text_encoder"].run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
    prompt, null = embed(tokenizer_ids), embed(null_ids)
    image = np.repeat((np.round(trimask * 255) / 127.5 - 1)[None, None].astype(np.float32), 3, axis=1)
    cond = sessions["vae_encoder"].run(None, {"image": image})[0]
    h, w = noise.shape[2:]
    mu = fibo2.shift_mu(h * w // 4)
    sigmas = [float(fibo2.shift_sigmas(torch.tensor(u), mu)) for u in np.linspace(1.0, 1 / steps, steps)] + [0.0]
    x = noise.astype(np.float32)
    run = lambda e, s: sessions["transformer"].run(None, {"latents": x, "cond_latents": cond, "sigma": np.array([s], np.float32),
                                                          "prompt_embeds": e, "adapter_scale": np.array([adapter_scale], np.float32)})[0]
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        v = run(prompt, s)
        if guidance > 1:
            v_null = run(null, s)
            v = v_null + guidance * (v - v_null)
        x = x + (s_next - s) * v
    return sessions["vae_decoder"].run(None, {"latent": x})[0][0].transpose(1, 2, 0)


def verify(version, items=4, steps=4, guidance=None, size=512, chunks=False):
    """
    Command (one GPU). Exported graphs (ORT CUDA EP) vs the shipped PyTorch model (fibo2.load_eval_model, bf16) on held-out
    eval items: text embeddings, condition latents and one velocity pass (relative error), then whole generations with the
    same noise (PSNR between the two images). Writes PNG pairs (lossless, full size) and verify.json to
    agents/.scratchpad/web_verify/<version>/.

    Args:
        version (str): folder in models/web/
        items (int): held-out items (one per method, in method order)
        steps, guidance (int, float | None): sampler settings of the generations; guidance default = the manifest's
        size (int): image side of the generations (a trained size)
        chunks (bool): run the transformer as its chunk chain (web_chunks.split); results go to verify_chunks.json
    """
    import numpy as np
    from PIL import Image
    folder = WEB_DIR / version
    manifest = json.loads((folder / "manifest.json").read_text())
    guidance = manifest["sampler"]["guidance"] if guidance is None else guidance
    out = we.VERIFY_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    model = fibo2.load_eval_model(manifest["source"], "cuda", BASES[manifest.get("base", "dev")]["dir"])
    encoder, processor = model["text"]
    template = manifest["text_encoder"]["template"]
    ids = lambda p: processor(text=[template.format(p)], return_tensors="np")["input_ids"].astype(np.int64)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    psnr = lambda a, b: float(10 * np.log10(1 / np.mean((a - b) ** 2)))
    cases = []  # PyTorch first, then free its GPU memory for onnxruntime
    for k, (caption, trimask) in enumerate(eval_items(split="held-out", per_method=1)[:items]):
        trimask = np.asarray(Image.fromarray((trimask * 255).round().astype(np.uint8)).resize((size, size), Image.NEAREST), np.float32) / 255
        prompt = fibo2.caption_prompt(caption)
        with torch.no_grad():
            emb_t, _ = fibo2.encode_prompts(encoder, processor, [prompt])
            cond_t = encode_condition(model, trimask)
            noise = torch.randn(cond_t.shape, generator=torch.Generator().manual_seed(k)).numpy()
            v_t = VelocityGraph(model["transformer"])(torch.from_numpy(noise).cuda() * 0.8 + 0.2 * cond_t, cond_t,
                                                       torch.tensor([0.8], device="cuda"), emb_t.float()).cpu().numpy()
            image_t = fibo2.generate_images(model, [caption], trimask[None], [k], steps, guidance)[0][0]
        cases.append(dict(k=k, caption=caption, trimask=trimask, prompt=prompt, ids=ids(prompt), noise=noise, emb_t=emb_t.float().cpu().numpy(),
                          cond_t=cond_t.cpu().numpy(), v_t=v_t, image_t=image_t))
    null_ids = ids(fibo2.NULL_PROMPT)
    del model, encoder
    torch.cuda.empty_cache()
    sessions = ort_sessions(folder, chunks=chunks)
    results = []
    for c in cases:
        emb_o = sessions["text_encoder"].run(None, {"input_ids": c["ids"], "attention_mask": np.ones_like(c["ids"])})[0]
        image_in = np.repeat((np.round(c["trimask"] * 255) / 127.5 - 1)[None, None].astype(np.float32), 3, axis=1)
        cond_o = sessions["vae_encoder"].run(None, {"image": image_in})[0]
        v_o = sessions["transformer"].run(None, {"latents": c["noise"] * 0.8 + 0.2 * cond_o, "cond_latents": cond_o, "sigma": np.array([0.8], np.float32),
                                                 "prompt_embeds": emb_o, "adapter_scale": np.array([0.0], np.float32)})[0]
        image_o = onnx_generate(sessions, c["ids"], null_ids, c["trimask"], c["noise"], steps, guidance)
        r = dict(caption=c["caption"], text_rel=rel(emb_o, c["emb_t"]), cond_rel=rel(cond_o, c["cond_t"]), velocity_rel=rel(v_o, c["v_t"]),
                 image_psnr=psnr(image_o, c["image_t"]), nonfinite=int((~np.isfinite(image_o)).sum()))
        results.append(r)
        print(json.dumps(r), flush=True)
        for tag, img in (("onnx_chunks" if chunks else "onnx", image_o), ("torch", c["image_t"])):
            Image.fromarray((img.clip(0, 1) * 255).round().astype(np.uint8)).save(out / f"item{c['k']}_{tag}.png")
    (out / ("verify_chunks.json" if chunks else "verify.json")).write_text(json.dumps(dict(version=version, steps=steps, guidance=guidance, size=size, chunks=chunks, items=results), indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    fire.Fire()
