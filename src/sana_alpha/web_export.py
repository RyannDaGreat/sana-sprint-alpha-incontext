"""
ONNX export of the one-step matte model for ONNX Runtime Web (WebGPU EP). The files and their inputs follow the model
contract in agents/manifest.md §12 (SEMANTIC BINDING: web-model-contract); `write_manifest` records the exact values.

    PYTHONPATH=src uv run python -m sana_alpha.web_export export models/control/control_v1/sprint_step_001500
    PYTHONPATH=src uv run python -m sana_alpha.web_export verify control_v1_sprint_step_001500
    PYTHONPATH=src uv run python -m sana_alpha.web_export quantize control_v1_sprint_step_001500 --bits 4

The graphs are traced from the fp16 PyTorch modules (WebGPU has no bf16), so the ops gatecontrol.py keeps in float32
(linear attention, gates) stay float32 in ONNX. All graph inputs and outputs are float32 so the browser needs no fp16
arrays. The Sprint TrigFlow wrapper (SanaSprintPipeline + SCMScheduler) is inside the transformer graph: it takes
unit-scale latents and returns the denoised image latent, so one call is one sampler step.
"""

import inspect
import json
import shutil
import time
from pathlib import Path

import einx
import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import SanaSprintPipeline
from transformers import AutoModel

from sana_alpha import gatecontrol as gc
from sana_alpha import models

ROOT = models.ROOT
WEB_DIR = ROOT / "models/web"
VERIFY_DIR = ROOT / "agents/.scratchpad/web_verify"
SIGMA_DATA = 0.5  # SCMScheduler sigma_data (Sprint scheduler config)
ONE_STEP_TIMESTEP = 1.5708  # TrigFlow angle s of the single step: SanaSprintPipeline max_timesteps (≈ π/2)
GUIDANCE = 4.5  # Sprint's embedded guidance scale (models.SPRINT_DEFAULTS)
TEXT_TOKENS = 300  # SanaSprintPipeline max_sequence_length: embeddings keep token 0 + the last 299
OPSET = 21
# SanaSprintPipeline.__call__'s prompt template; encode_prompt's own default is None (no template), so pass it
INSTRUCTION = inspect.signature(SanaSprintPipeline.__call__).parameters["complex_human_instruction"].default
IR_VERSION = 13  # onnxruntime-web 1.31 reads ONNX IR ≤ 13; onnx 1.23 writes 14 by default


class Conv1x1AsLinear(nn.Module):
    """A 1×1 Conv2d run as a MatMul over channels, so ORT's MatMulNBits weight quantizer can reach the Mix-FFN."""

    def __init__(self, conv):
        """
        Command (copies the weights).

        Args:
            conv (nn.Conv2d): kernel 1×1, stride 1, no groups
        """
        super().__init__()
        assert conv.kernel_size == (1, 1) and conv.groups == 1 and conv.stride == (1, 1), f"not a plain 1x1 conv: {conv}"
        self.linear = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        self.linear.to(conv.weight)
        self.linear.weight.data = conv.weight.data.flatten(1)
        if conv.bias is not None:
            self.linear.bias.data = conv.bias.data

    def forward(self, x):
        """Pure function (given fixed weights). (B, C, H, W) -> (B, O, H, W), equal to the 1×1 conv."""
        H, W = x.shape[2:]
        y = self.linear(einx.rearrange("B C H W -> B (H W) C", x))
        return einx.rearrange("B (H W) O -> B O H W", y, H=H, W=W)


def linearize_mix_ffn(transformer):
    """
    Command. Replaces conv_inverted / conv_point of every block's Mix-FFN with Conv1x1AsLinear (same math).

    Args:
        transformer (SanaTransformer2DModel)

    Returns:
        int: number of convs replaced (40 for SANA 1.6B)
    """
    for block in transformer.transformer_blocks:
        block.ff.conv_inverted = Conv1x1AsLinear(block.ff.conv_inverted)
        block.ff.conv_point = Conv1x1AsLinear(block.ff.conv_point)
    return 2 * len(transformer.transformer_blocks)


class Fp32Linear(nn.Module):
    """A Linear whose fp16 weights are applied in float32 (the ONNX graph stores fp16 and casts), output float32."""

    def __init__(self, linear):
        """Command (holds the module). linear: nn.Linear."""
        super().__init__()
        self.linear = linear

    def forward(self, x):
        """Pure function (given fixed weights). (..., in) any float dtype -> (..., out) float32."""
        bias = None if self.linear.bias is None else self.linear.bias.float()
        return F.linear(x.float(), self.linear.weight.float(), bias)


class CastOutput(nn.Module):
    """Runs a module and casts its output to a fixed dtype."""

    def __init__(self, module, dtype):
        """Command (holds the module). module: nn.Module; dtype: torch.dtype of the output."""
        super().__init__()
        self.module, self.dtype = module, dtype

    def forward(self, x):
        """Pure function (given fixed weights). Same shapes as module(x), dtype self.dtype."""
        return self.module(x).to(self.dtype)


def fix_fp16_overflow(transformer):
    """
    Command. Makes the two fp16 overflow spots of SANA 1.6B run in float32 (measured on the eval set, bf16 reference):
    cross-attention `attn2.to_q` reaches |q| ≈ 1.6e5 in blocks 14 and 17 (fp16 max 65504; RMSNorm `norm_q` right after
    rescales it, so it is computed in float32 and cast to fp16 after the norm), and `caption_projection.linear_2`
    reaches 2.3e4 (kept in float32 for margin; caption_norm casts back to fp16). Weights stay fp16.

    Args:
        transformer (SanaTransformer2DModel): in fp16

    Returns:
        the same transformer
    """
    for block in transformer.transformer_blocks:
        assert block.attn2.norm_q is not None, "the fp32 to_q relies on norm_q to bring q back into fp16 range"
        block.attn2.to_q = Fp32Linear(block.attn2.to_q)
        block.attn2.norm_q = CastOutput(block.attn2.norm_q, transformer.dtype)
    transformer.caption_projection.linear_2 = Fp32Linear(transformer.caption_projection.linear_2)
    return transformer


def trigflow_step(latents, timestep, velocity_fn):
    """
    Pure function (if velocity_fn is). One SanaSprintPipeline + SCMScheduler step in unit-scale latents (x / sigma_data).

    With t = sin s / (cos s + sin s) and n = √(t² + (1−t)²):
    x_in = n·latents, v = velocity_fn(x_in, t), F = ((1−2t)·x_in + (1−2t+2t²)·v) / n, denoised = cos s·latents − sin s·F.
    For more steps, the next input is cos(s')·denoised + sin(s')·ε (ε standard normal); the pipeline does the same
    with sigma_data factored out.

    Args:
        latents (torch.Tensor): (B, LC, LH, LW) float32; at s = π/2 the standard-normal noise itself
        timestep (torch.Tensor): (B,) float32 TrigFlow angle s
        velocity_fn: (x_in (B, LC, LH, LW) float32, t (B,) float32) -> flow velocity (B, LC, LH, LW)

    Returns:
        torch.Tensor: (B, LC, LH, LW) float32 denoised latent (the VAE latent times scaling_factor)

    Examples:
        >>> x = torch.ones(1, 1, 1, 1)
        >>> trigflow_step(x, torch.tensor([1.5708]), lambda x_in, t: x_in - 3).flatten().round(decimals=3).tolist()  # v = noise − x0, x0 = 3
        [3.0]
    """
    s = timestep.float()
    t = torch.sin(s) / (torch.cos(s) + torch.sin(s))
    t4, s4 = t.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)
    norm = torch.sqrt(t4 ** 2 + (1 - t4) ** 2)
    x_in = latents.float() * norm
    v = velocity_fn(x_in, t).float()
    f = ((1 - 2 * t4) * x_in + (1 - 2 * t4 + 2 * t4 ** 2) * v) / norm
    return torch.cos(s4) * latents.float() - torch.sin(s4) * f


class OneStepTransformer(nn.Module):
    """The gated two-stream Sprint transformer wrapped in trigflow_step; float32 I/O, compute in the model's dtype."""

    def __init__(self, transformer):
        """
        Command (holds the module).

        Args:
            transformer (SanaTransformer2DModel): guidance-embedded student with gatecontrol_gates attached
        """
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, cond_latents, timestep, guidance, prompt_embeds, prompt_mask):
        """
        Pure function (given fixed weights).

        Args:
            latents: (1, 32, 32, 32) float32 unit-scale sample (step 1: standard-normal noise)
            cond_latents: (1, 32, 32, 32) float32 matte latent (vae_encoder output, already × scaling_factor)
            timestep: (1,) float32 TrigFlow angle s (ONE_STEP_TIMESTEP for one step)
            guidance: (1,) float32 raw guidance scale (GUIDANCE); × guidance_embeds_scale inside
            prompt_embeds: (1, 300, 2304) float32 Gemma-2 embeddings
            prompt_mask: (1, 300) float32, 1 = token, 0 = padding

        Returns:
            torch.Tensor: (1, 32, 32, 32) float32 denoised latent (vae_decoder input)
        """
        tr, dtype = self.transformer, self.transformer.dtype
        scaled_guidance = (guidance * tr.config.guidance_embeds_scale).to(dtype)
        velocity = lambda x_in, t: gc.gatecontrol_forward(tr, x_in.to(dtype), cond_latents.to(dtype),
                                                          prompt_embeds.to(dtype), prompt_mask, t,
                                                          guidance=scaled_guidance)
        return trigflow_step(latents, timestep, velocity)


class VaeEncoder(nn.Module):
    """DC-AE encoder: RGB in [−1, 1] -> latent × scaling_factor; float32 I/O."""

    def __init__(self, vae):
        """Command (holds the module). vae: AutoencoderDC."""
        super().__init__()
        self.vae = vae

    def forward(self, image):
        """Pure function (given fixed weights). (1, 3, 1024, 1024) float32 in [−1, 1] -> (1, 32, 32, 32) float32."""
        return self.vae.encode(image.to(self.vae.dtype)).latent.float() * self.vae.config.scaling_factor


class VaeDecoder(nn.Module):
    """DC-AE decoder: latent × scaling_factor -> RGB in [0, 1] (the pipeline's postprocess); float32 I/O."""

    def __init__(self, vae):
        """Command (holds the module). vae: AutoencoderDC."""
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Pure function (given fixed weights). (1, 32, 32, 32) float32 -> (1, 3, 1024, 1024) float32 RGB in [0, 1]."""
        image = self.vae.decode((latents / self.vae.config.scaling_factor).to(self.vae.dtype)).sample.float()
        return (image / 2 + 0.5).clamp(0, 1)


class TextEncoder(nn.Module):
    """Gemma-2 2B (Gemma2Model) last_hidden_state over the whole templated prompt; the app picks the 300 tokens."""

    def __init__(self, model):
        """Command (holds the module). model: transformers Gemma2Model."""
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        """Pure function (given fixed weights). (1, T) int64 ids and mask -> (1, T, 2304) float32."""
        return self.model(input_ids, attention_mask=attention_mask)[0].float()


def graph_specs():
    """
    Pure function. Input and output names, shapes and dtypes of every graph (the model contract).

    Returns:
        dict: graph name -> dict(inputs=[(name, shape, dtype)], outputs=[...])

    Examples:
        >>> [n for n, _, _ in graph_specs()["transformer"]["inputs"]]
        ['latents', 'cond_latents', 'timestep', 'guidance', 'prompt_embeds', 'prompt_mask']
    """
    latent, image = [1, 32, 32, 32], [1, 3, 1024, 1024]
    return dict(
        transformer=dict(inputs=[("latents", latent, "float32"), ("cond_latents", latent, "float32"),
                                 ("timestep", [1], "float32"), ("guidance", [1], "float32"),
                                 ("prompt_embeds", [1, TEXT_TOKENS, 2304], "float32"),
                                 ("prompt_mask", [1, TEXT_TOKENS], "float32")],
                         outputs=[("denoised", latent, "float32")]),
        vae_encoder=dict(inputs=[("image", image, "float32")], outputs=[("latent", latent, "float32")]),
        vae_decoder=dict(inputs=[("latent", latent, "float32")], outputs=[("image", image, "float32")]),
        text_encoder=dict(inputs=[("input_ids", [1, "T"], "int64"), ("attention_mask", [1, "T"], "int64")],
                          outputs=[("last_hidden_state", [1, "T", 2304], "float32")]),
    )


def example_inputs(name, device):
    """
    Query (reads the RNG). Random float32 tensors shaped like a graph's inputs, for tracing.

    Args:
        name (str): graph name in graph_specs()
        device (str): torch device

    Returns:
        tuple of torch.Tensor
    """
    tokens = 506  # length of SANA's templated prompt (instruction + 298 tokens)
    special = dict(timestep=torch.tensor([ONE_STEP_TIMESTEP]), guidance=torch.tensor([GUIDANCE]),
                   prompt_mask=torch.ones(1, TEXT_TOKENS), input_ids=torch.randint(3, 1000, (1, tokens)),
                   attention_mask=torch.ones(1, tokens, dtype=torch.long))
    return tuple(special[n].to(device) if n in special else torch.randn(shape, device=device)
                 for n, shape, _ in graph_specs()[name]["inputs"])


def export_graph(module, name, path, custom_translation_table=None):
    """
    Command. Traces module with torch.onnx (dynamo exporter) and writes path (+ path.data when weights exceed 2 GB).

    Args:
        module (nn.Module): one of the wrappers above, on the GPU
        name (str): graph name in graph_specs()
        path (Path): e.g. models/web/<version>/transformer.onnx
        custom_translation_table (dict | None): torch op -> onnxscript function (web_quant's MatMulNBits)
    """
    spec = graph_specs()[name]
    began = time.perf_counter()
    T = torch.export.Dim("T", min=2, max=2048)
    dynamic = tuple({1: T} if "T" in shape else None for _, shape, _ in spec["inputs"])
    with torch.no_grad():  # optimize=False: onnxscript's optimizer drops the "+ 1e-15" of linear attention -> 0/0 = NaN
        program = torch.onnx.export(module, example_inputs(name, "cuda"), dynamo=True, opset_version=OPSET, optimize=False,
                                    input_names=[n for n, _, _ in spec["inputs"]],
                                    output_names=[n for n, _, _ in spec["outputs"]],
                                    dynamic_shapes=dynamic if any(dynamic) else None,
                                    custom_translation_table=custom_translation_table)
    program.model.ir_version = IR_VERSION
    program.save(str(path), external_data=True)
    print(f"export {name}: {path} in {time.perf_counter() - began:.0f} s", flush=True)


def load_source(source, dtype):
    """
    Query (reads weights, allocates GPU memory). The Sprint pipeline of a merge folder (student + deltas + gates).

    Args:
        source (str): repo-relative merge_final --student folder, e.g. models/control/control_v1/sprint_step_001500
        dtype (torch.dtype): torch.float16 for export, torch.bfloat16 for the reference

    Returns:
        SanaSprintPipeline with the gated transformer
    """
    pipe, _, conditioned = models.load_eval_pipeline("sprint", source)
    assert conditioned, f"{source} has no gates.safetensors"
    return pipe.to(dtype=dtype)


def write_manifest(folder, version, source, quantization):
    """
    Command. Writes folder/manifest.json: files, I/O specs, quantization, and the Sprint math (web-model-contract).

    Args:
        folder (Path): models/web/<version>
        version (str): e.g. control_v1_sprint_step_001500
        source (str): the merge folder it came from
        quantization (dict): graph name -> list of variants, e.g. {"transformer": {"fp16": "transformer.onnx"}}
    """
    files = sorted(p.name for p in folder.iterdir() if p.suffix in (".onnx", ".data") or p.name.startswith("tokenizer"))
    manifest = dict(
        version=version, source=source, files={f: (folder / f).stat().st_size for f in files},
        graphs=graph_specs(), quantization=quantization,
        text_encoder=dict(
            model="models/Sana_Sprint_1.6B_1024px_diffusers/text_encoder (Gemma2Model, gemma-2-2b-it), last_hidden_state",
            prompt="complex_human_instruction lines joined by '\\n' + caption.lower().strip()",
            tokenize="right padding to len(tokenize(instruction)) + 298 tokens, add_special_tokens, truncation",
            select="keep token 0 and the last 299 tokens of embeddings and mask -> (1, 300, 2304), (1, 300)"),
        sampler=dict(
            steps=1, timestep=ONE_STEP_TIMESTEP, guidance=GUIDANCE, sigma_data=SIGMA_DATA,
            latents="standard-normal noise (1, 32, 32, 32); the graph handles sigma_data and the TrigFlow scaling",
            output="denoised = image latent × scaling_factor: feed it to vae_decoder as is",
            multi_step="next latents = cos(s')·denoised + sin(s')·ε, timestep s' (2 steps: s = 1.5708 then 1.3)"),
        vae=dict(scaling_factor=0.41407, note="vae_encoder multiplies and vae_decoder divides inside the graph",
                 matte_input="matte copied to 3 channels, 2·m − 1", decoder_output="RGB in [0, 1]"),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {folder / 'manifest.json'}")


def export(source, version=None, parts="transformer,vae_encoder,vae_decoder,text_encoder"):
    """
    Command. Exports fp16 ONNX graphs of a Sprint merge folder to models/web/<version>/ and writes manifest.json.

    Args:
        source (str): e.g. models/control/control_v1/sprint_step_001500
        version (str | None): default "<run>_<folder>", e.g. control_v1_sprint_step_001500
        parts (str | tuple): comma-separated graph names to (re)export
    """
    # SEMANTIC BINDING: web-model-contract
    source_path = Path(source)
    version = version or f"{source_path.parent.name}_{source_path.name}"
    folder = WEB_DIR / version
    folder.mkdir(parents=True, exist_ok=True)
    parts = parts.split(",") if isinstance(parts, str) else list(parts)
    pipe = load_source(source, torch.float16)
    pipe.text_encoder.to("cpu")
    print(f"linearized {linearize_mix_ffn(pipe.transformer)} Mix-FFN 1x1 convs", flush=True)
    fix_fp16_overflow(pipe.transformer)
    wrappers = dict(transformer=lambda: OneStepTransformer(pipe.transformer), vae_encoder=lambda: VaeEncoder(pipe.vae),
                    vae_decoder=lambda: VaeDecoder(pipe.vae), text_encoder=lambda: TextEncoder(text_encoder()))
    text_encoder = lambda: AutoModel.from_pretrained(models.localize(models.SPRINT_DIR) / "text_encoder", dtype=torch.float16,
                                                     attn_implementation="eager").cuda()
    for name in parts:
        if name == "text_encoder":  # Gemma needs the GPU memory the transformer holds
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()
        export_graph(wrappers[name]().eval(), name, folder / f"{name}.onnx")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(models.localize(models.SPRINT_DIR) / "tokenizer" / name, folder / name)
    quantization = {n: {"fp16": f"{n}.onnx"} for n in graph_specs()}
    write_manifest(folder, version, source, quantization)


def record_inputs(linears, run, rows_per_item):
    """
    Command (hooks modules temporarily, runs `run`). A random token subsample of each Linear's input per forward.

    Args:
        linears (dict): name -> nn.Linear
        run: callable that runs the calibration forwards
        rows_per_item (int): tokens kept per Linear per forward

    Returns:
        dict: name -> (T, K) float32 CPU tensor
    """
    inputs = {n: [] for n in linears}
    generator = torch.Generator().manual_seed(0)
    def record(name, x):
        x = x.reshape(-1, x.shape[-1])
        inputs[name].append(x[torch.randperm(len(x), generator=generator)[:rows_per_item].to(x.device)].float().cpu())
    hooks = [m.register_forward_pre_hook(lambda mod, args, n=n: record(n, args[0])) for n, m in linears.items()]
    with torch.no_grad():
        run()
    for h in hooks:
        h.remove()
    return {n: torch.cat(v) for n, v in inputs.items()}


def calibration_runs(pipe, calib_dir):
    """
    Query (reads the calibration set; returns closures that run the GPU). One-step transformer forwards and Gemma
    forwards (SANA template) on the "train" items of calib_dir; the held-out eval items are never used.

    Args:
        pipe: fp16 Sprint pipeline from load_source (transformer linearized and fp16-fixed)
        calib_dir (str): e.g. resources/showcase

    Returns:
        dict: "transformer" / "text_encoder" -> callable(model) running all calibration items through that model
    """
    from sana_alpha.eval_set import load_eval_set
    from sana_alpha.generate import encode_mattes
    data = load_eval_set(ROOT / calib_dir)
    train = [i for i, item in enumerate(data["items"]) if item["split"] == "train"]
    captions = [data["items"][i]["caption"] for i in train]
    cond = encode_mattes(pipe.vae, data["mattes"][train]).cuda()
    with torch.no_grad():
        prompts = [pipe.encode_prompt(c, device="cuda", complex_human_instruction=INSTRUCTION)[:2] for c in captions]
    tokens = [sana_tokens(pipe.tokenizer, c)[:2] for c in captions]
    print(f"calibration: {len(train)} train items of {calib_dir}", flush=True)
    step = lambda model, k: model(data["noise"][train[k]:train[k] + 1].cuda(), cond[k:k + 1],
                                  torch.tensor([ONE_STEP_TIMESTEP]).cuda(), torch.tensor([GUIDANCE]).cuda(),
                                  prompts[k][0].float(), prompts[k][1].float())
    return dict(transformer=lambda model: [step(model, k) for k in range(len(train))],
                text_encoder=lambda model: [model(ids.cuda(), m.cuda()) for ids, m in tokens])


def quantize_linears(root, linears, inputs, bits, rank):
    """
    Command. Replaces each Linear of root with an SVDQuant QuantLinear (web_quant.svdquant_weight).

    Args:
        root (nn.Module): owner of the dotted names in linears
        linears (dict): name -> nn.Linear
        inputs (dict): name -> (T, K) calibration inputs (consumed)
        bits (int): 4 or 8
        rank (int): low-rank branch rank

    Returns:
        dict: name -> {alpha, rel_error}
    """
    from sana_alpha import web_quant as wq
    report = {}
    for name, linear in linears.items():
        with torch.no_grad():  # the weights require grad: without this every intermediate stays alive (OOM)
            q = wq.svdquant_weight(linear.weight.float(), inputs.pop(name).cuda(), bits, rank)
        wq.set_module(root, name, wq.QuantLinear(q, linear.bias, bits).cuda())
        report[name] = dict(alpha=q["alpha"], rel_error=round(q["rel_error"], 5))
        print(f"{name}: alpha {q['alpha']}, output rel error {q['rel_error']:.4f}", flush=True)
    torch.cuda.empty_cache()
    return report


def quantize(version, bits=4, rank=None, parts="transformer,text_encoder", reuse=None, calib_dir="resources/showcase",
             rows_per_item=128):
    """
    Command. SVDQuant-style weight-only quantization (web_quant.py) of the transformer and the Gemma text encoder,
    exported to models/web/<version>-int<bits>/; the files not quantized are hard links to the fp16 version's.

    Transformer: every block Linear except attn2.to_q. Gemma: every Linear of its 26 decoder layers (the
    embedding table and norms stay fp16).

    Args:
        version (str): fp16 version folder in models/web/
        bits (int): 4 or 8
        rank (int | None): low-rank branch rank; default web_quant.RANK for 4 bits, 0 for 8 bits
        parts (str | tuple): graphs to (re)quantize
        reuse (str | None): a quantized version folder whose other graphs are hard-linked (with their manifest
            entries) instead of the fp16 ones, e.g. the same Gemma int4 for a new transformer
        calib_dir (str): calibration set (its "train" items)
        rows_per_item (int): calibration tokens per Linear per item
    """
    from sana_alpha import web_quant as wq
    rank = (wq.RANK if bits == 4 else 0) if rank is None else rank
    parts = parts.split(",") if isinstance(parts, str) else list(parts)
    base, folder = WEB_DIR / version, WEB_DIR / f"{version}-int{bits}"
    folder.mkdir(exist_ok=True)
    source = json.loads((base / "manifest.json").read_text())["source"]
    donor = WEB_DIR / reuse if reuse else base
    old = folder / "manifest.json" if (folder / "manifest.json").exists() else donor / "manifest.json"
    quantization = json.loads(old.read_text())["quantization"]
    torch.manual_seed(0)  # randomized SVD: the same weights give the same files
    pipe = load_source(source, torch.float16)
    linearize_mix_ffn(pipe.transformer)
    fix_fp16_overflow(pipe.transformer)
    runs = calibration_runs(pipe, calib_dir)
    pipe.text_encoder.to("cpu")
    table = {torch.ops.websana.matmul_nbits.default: wq.matmul_nbits_onnx}
    for name in parts:
        if name == "transformer":
            model = OneStepTransformer(pipe.transformer)
            linears = wq.target_linears(pipe.transformer)
        else:
            pipe.transformer.to("cpu")
            model = TextEncoder(AutoModel.from_pretrained(models.localize(models.SPRINT_DIR) / "text_encoder",
                                                          dtype=torch.float16, attn_implementation="eager").cuda())
            linears = {f"model.{n}": m for n, m in model.model.layers.named_modules(prefix="layers")
                       if isinstance(m, nn.Linear)}
        report = quantize_linears(model, linears, record_inputs(linears, lambda: runs[name](model), rows_per_item),
                                  bits, rank)
        for f in folder.glob(f"{name}.onnx*"):
            f.unlink()  # may be a hard link to the fp16 file: writing through it would overwrite fp16
        export_graph(model.eval(), name, folder / f"{name}.onnx", table)
        del model
        torch.cuda.empty_cache()
        errors = [r["rel_error"] for r in report.values()]
        quantization[name] = dict(
            method="SVDQuant weight-only (arXiv 2411.05007; deepcompressor/nunchaku), see src/sana_alpha/web_quant.py",
            bits=bits, group_size=wq.BLOCK, symmetric=True, scales="fp16", lowrank_rank=rank, lowrank_dtype="fp16",
            smoothing_alphas=list(wq.ALPHAS), activations="fp16",
            calibration=f"{calib_dir} train split, {rows_per_item} tokens/Linear/item",
            quantized_linears=len(report), mean_output_rel_error=float(np.mean(errors)),
            max_output_rel_error=float(np.max(errors)), per_linear=report)
    for f in [*donor.iterdir(), *base.iterdir()]:
        if f.name != "manifest.json" and not (folder / f.name).exists():
            (folder / f.name).hardlink_to(f)
    write_manifest(folder, folder.name, source, quantization)


def ort_session(path):
    """
    Query (reads the file, allocates GPU memory). An ONNX Runtime CUDA session (the A100 stand-in for WebGPU).

    Args:
        path (Path): .onnx file

    Returns:
        onnxruntime.InferenceSession
    """
    import onnxruntime as ort
    ort.preload_dlls()  # CUDA/cuDNN from the torch wheels
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = [("CUDAExecutionProvider", dict(arena_extend_strategy="kSameAsRequested"))]
    session = ort.InferenceSession(str(path), options, providers=providers)
    assert session.get_providers()[0] == "CUDAExecutionProvider", f"no CUDA EP: {session.get_providers()}"
    return session


def run_session(session, *inputs):
    """
    Command (runs the GPU). Runs a session on numpy copies of the inputs, cast to each graph input's type (float32 or
    int64), in graph input order.

    Args:
        session (onnxruntime.InferenceSession)
        *inputs (torch.Tensor | np.ndarray): one per graph input

    Returns:
        np.ndarray: the first output
    """
    as_numpy = lambda x: np.asarray(x.float().cpu().numpy() if torch.is_tensor(x) else x)
    types = {"tensor(float)": np.float32, "tensor(int64)": np.int64}
    feeds = {i.name: as_numpy(x).astype(types[i.type]) for i, x in zip(session.get_inputs(), inputs)}
    return session.run(None, feeds)[0]


def sana_tokens(tokenizer, caption):
    """
    Pure function (given the tokenizer). SanaSprintPipeline's Gemma input: complex human instruction + caption,
    right-padded to len(instruction tokens) + TEXT_TOKENS − 2, and the indices of the TEXT_TOKENS kept embeddings.

    Args:
        tokenizer: the Sprint pipeline's Gemma tokenizer
        caption (str): user prompt

    Returns:
        tuple: input_ids (1, T) int64, attention_mask (1, T) int64, select (TEXT_TOKENS,) list of ints

    Examples:
        >>> # ids, mask, select = sana_tokens(pipe.tokenizer, "two dice"); ids.shape, len(select)
        >>> # (torch.Size([1, 506]), 300)
    """
    instruction = "\n".join(INSTRUCTION)
    total = len(tokenizer.encode(instruction)) + TEXT_TOKENS - 2
    tokenizer.padding_side = "right"
    t = tokenizer(instruction + caption.lower().strip(), padding="max_length", max_length=total, truncation=True,
                  add_special_tokens=True, return_tensors="pt")
    return t.input_ids, t.attention_mask, [0] + list(range(total - TEXT_TOKENS + 1, total))


def image_errors(images, reference, mattes):
    """
    Pure function. Mean abs difference to a reference image set, plus the eval metrics (metrics.image_metrics).

    Args:
        images, reference (np.ndarray): (N, H, W, 3) float RGB in [0, 1]
        mattes (np.ndarray): (N, H, W) float in [0, 1]

    Returns:
        dict: mean_abs_diff, max_abs_diff, nonfinite (count of inf/NaN pixels), iou, exterior_gray_error

    Examples:
        >>> m = np.zeros((1, 4, 4)); a = np.full((1, 4, 4, 3), 0.5); b = a + 0.1
        >>> {k: round(v, 3) for k, v in image_errors(b, a, m).items()}
        {'mean_abs_diff': 0.1, 'max_abs_diff': 0.1, 'nonfinite': 0, 'iou': 0.0, 'exterior_gray_error': 0.1}
    """
    from sana_alpha.metrics import image_metrics, mean_metrics
    diff = np.abs(np.nan_to_num(images, nan=1.0) - reference)
    scores = mean_metrics([image_metrics(np.nan_to_num(i), m) for i, m in zip(images, mattes)])
    return dict(mean_abs_diff=float(diff.mean()), max_abs_diff=float(diff.max()),
                nonfinite=int((~np.isfinite(images)).sum()), iou=scores["iou"],
                exterior_gray_error=scores["exterior_gray_error"])


def verify(version, transformers="transformer.onnx", eval_dir="resources/eval"):
    """
    Command. Runs the ONNX graphs (ORT CUDA EP) on the eval set and compares them with the bf16 PyTorch pipeline
    (SanaSprintPipeline, 1 step). Prints per-graph and end-to-end errors, writes PNGs and a montage to
    agents/.scratchpad/web_verify/<version>/, appends the numbers to verify.json there.

    Args:
        version (str): folder in models/web/
        transformers (str | tuple): comma-separated transformer .onnx files to test (fp16, int8, int4 variants)
        eval_dir (str): eval set folder (29 held-out items)
    """
    from PIL import Image
    from sana_alpha.eval_set import load_eval_set
    from sana_alpha.generate import encode_mattes
    from sana_alpha.metrics import montage
    folder, out = WEB_DIR / version, VERIFY_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((folder / "manifest.json").read_text())["source"]
    data = load_eval_set(ROOT / eval_dir)
    items, mattes, noise = data["items"], data["mattes"], data["noise"]
    captions = [i["caption"] for i in items]

    pipe = load_source(source, torch.bfloat16)
    cond = encode_mattes(pipe.vae, mattes)  # the training path: bf16 DC-AE, latents rounded to bf16
    with torch.no_grad():
        embeds, mask = zip(*(pipe.encode_prompt(c, device="cuda", complex_human_instruction=INSTRUCTION)[:2]
                             for c in captions))
    kwargs = dict(num_inference_steps=1, max_timesteps=ONE_STEP_TIMESTEP, intermediate_timesteps=None,
                  guidance_scale=GUIDANCE, output_type="latent", use_resolution_binning=False)
    ref_latents = np.concatenate([pipe(prompt_embeds=e, prompt_attention_mask=m, latents=noise[i:i + 1], **kwargs,
                                       attention_kwargs={gc.COND_KWARG: cond[i:i + 1]}).images.float().cpu().numpy()
                                  for i, (e, m) in enumerate(zip(embeds, mask))])
    pipe.transformer.to("cpu"), pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    to01 = lambda t: einx.rearrange("N C H W -> N H W C", (torch.cat(t) / 2 + 0.5).clamp(0, 1).cpu().numpy())
    with torch.no_grad():  # the pipeline's own decode: bf16 latents / scaling_factor
        reference = to01([pipe.vae.decode(torch.from_numpy(l[None]).cuda().bfloat16() / pipe.vae.config.scaling_factor)
                          .sample.float() for l in ref_latents])
    matte_images = einx.rearrange("N H W -> N C H W", torch.from_numpy(mattes) * 2 - 1, C=3)

    results = {}
    enc = ort_session(folder / "vae_encoder.onnx")
    onnx_cond = np.concatenate([run_session(enc, m[None]) for m in matte_images])
    results["vae_encoder"] = dict(latent_mean_abs_diff=float(np.abs(onnx_cond - cond.numpy()).mean()),
                                  latent_mean_abs=float(np.abs(cond.numpy()).mean()),
                                  nonfinite=int((~np.isfinite(onnx_cond)).sum()))
    del enc
    text = ort_session(folder / "text_encoder.onnx")  # the app's path: template + tokens -> Gemma -> 300 selected
    tokens = [sana_tokens(pipe.tokenizer, c) for c in captions]
    onnx_embeds = [torch.from_numpy(run_session(text, ids, m)[:, select]) for ids, m, select in tokens]
    del text
    results["text_encoder"] = dict(
        embeds_mean_abs_diff=float(np.mean([(o - e.float().cpu()).abs().mean() for o, e in zip(onnx_embeds, embeds)])),
        embeds_mean_abs=float(np.mean([e.float().abs().mean().item() for e in embeds])),
        embeds_cosine_valid_tokens=float(np.mean([F.cosine_similarity(o[0], e[0].float().cpu(), dim=-1)[pm[0].bool().cpu()].mean()
                                                  for o, e, pm in zip(onnx_embeds, embeds, mask)])),
        mask_equal=all(bool((m[:, s] == pm.cpu()).all()) for (_, m, s), pm in zip(tokens, mask)))
    embeds, mask = onnx_embeds, [m[:, s] for _, m, s in tokens]
    dec = ort_session(folder / "vae_decoder.onnx")
    decode = lambda latents: einx.rearrange("N C H W -> N H W C", np.concatenate([run_session(dec, l[None]) for l in latents]))
    results["vae_decoder"] = image_errors(decode(ref_latents), reference, mattes)
    timestep, guidance = np.array([ONE_STEP_TIMESTEP], np.float32), np.array([GUIDANCE], np.float32)
    rows = [[(r, c) for r, c in zip(reference, captions)]]
    for name in transformers.split(",") if isinstance(transformers, str) else transformers:
        began = time.perf_counter()
        session = ort_session(folder / name)
        load_s = time.perf_counter() - began
        latents, seconds = [], []
        for i in range(len(items)):
            began = time.perf_counter()
            latents.append(run_session(session, noise[i:i + 1], onnx_cond[i:i + 1], timestep, guidance, embeds[i], mask[i]))
            seconds.append(time.perf_counter() - began)
        latents = np.concatenate(latents)
        del session
        images = decode(latents)
        results[name] = dict(image_errors(images, reference, mattes), load_seconds=load_s,
                             seconds_per_step=float(np.median(seconds[1:])),
                             latent_mean_abs_diff=float(np.abs(np.nan_to_num(latents) - ref_latents).mean()),
                             nonfinite_latents=int((~np.isfinite(latents)).sum()))
        (out / name).mkdir(exist_ok=True)
        for item, image in zip(items, images):
            Image.fromarray(np.round(np.nan_to_num(image) * 255).astype(np.uint8)).save(out / name / f"{item['page_id']}.png")
        rows.append([(i, f"{name}: {c}") for i, c in zip(images, captions)])
        print(name, json.dumps(results[name]), flush=True)
    results["reference"] = dict(image_errors(reference, reference, mattes), what="bf16 SanaSprintPipeline, 1 step")
    Image.fromarray(montage([list(col) for col in zip(*rows)][:12], size=256)).save(out / "montage.jpg", quality=90)
    print(json.dumps(results, indent=1))
    log = out / "verify.json"
    history = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(history + [dict(time=time.strftime("%Y-%m-%d %H:%M"), results=results)], indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(dict(export=export, verify=verify, quantize=quantize))
