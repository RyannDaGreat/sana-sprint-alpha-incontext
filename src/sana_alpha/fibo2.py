"""
Bria Fibo 2.0 (FIBO2) helpers for the trimask model of manifest §14: latent normalization, the resolution shift of the
noise levels, prompts, text encoding, LoRA, and a sampler with the trimask as a clean context image.

Latents. encode_fibo2 stores the raw FLUX.2 VAE mode z: (32, S/8, S/8). The transformer reads normalized latents:
2x2 patchify to (128, S/16, S/16), then (z − bn_mean) / sqrt(bn_var + eps) per channel, then unpatchify back.
Flow. x_t = (1 − σ)·x_0 + σ·ε, timestep = σ, target v = ε − x_0 (train_objective); σ comes from a logit-normal u shifted
by FIBO2's own resolution shift: σ = e^mu / (e^mu + 1/u − 1), mu linear in the image token count (the pipeline's).
"""

import json
import math

import numpy as np
import torch
from safetensors.torch import load_file

from sana_alpha import models

FIBO2_DIR = "models/Fibo-2.0-dev"  # repo-relative; each subfolder is localized on its own (models.localize treats an existing folder as complete)
TURBO_DIR = "models/Fibo-2.0-edit-turbo"  # Bria's 4-step distilled model (no CFG): same architecture, text encoder and VAE as dev
CAPTION_KEY = "short_description"  # A caption is a JSON object with only this key: one of the forms FIBO2 was trained on
NULL_PROMPT = ('{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},'
               '"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}')  # FIBO2's trained null caption
SHIFT_SEQ = (256, 4096)  # scheduler base_image_seq_len, max_image_seq_len
SHIFT_MU = (0.5, 1.15)  # scheduler base_shift, max_shift
GRAY = 128  # 8-bit value of the trimask's gray ("unknown")
LORA_TARGETS = r"(layers|noise_refiner)\.\d+\.(attention\.(to_q|to_k|to_v|to_out\.0)|feed_forward\.(w1|w2|w3))"  # every linear of the 46 blocks


def patchify(z):
    """
    Pure function. 2x2 patches into channels, as the FIBO2 pipeline's _patchify_latents.

    Args:
        z (torch.Tensor): (B, LC, LH, LW), e.g. (2, 32, 128, 128)

    Returns:
        torch.Tensor: (B, 4·LC, LH/2, LW/2), e.g. (2, 128, 64, 64)

    Examples:
        >>> patchify(torch.arange(16.).view(1, 1, 4, 4))[0, :, 0, 0].tolist()
        [0.0, 1.0, 4.0, 5.0]
    """
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(z):
    """
    Pure function. Inverse of patchify.

    Examples:
        >>> x = torch.randn(2, 32, 8, 6)
        >>> bool((unpatchify(patchify(x)) == x).all())
        True
    """
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def normalize_latents(z, bn_mean, bn_std):
    """
    Pure function. Raw FLUX.2 VAE latents -> the normalized latents FIBO2 reads.

    Args:
        z (torch.Tensor): (B, 32, LH, LW) raw VAE mode
        bn_mean, bn_std (torch.Tensor): (128,) the VAE BatchNorm's running mean and sqrt(running_var + eps)

    Returns:
        torch.Tensor: (B, 32, LH, LW) float32

    Examples:
        >>> z = torch.ones(1, 32, 2, 2)
        >>> normalize_latents(z, torch.full((128,), 0.5), torch.full((128,), 0.25)).unique().tolist()
        [2.0]
    """
    mean, std = bn_mean.view(1, -1, 1, 1).to(z.device), bn_std.view(1, -1, 1, 1).to(z.device)
    return unpatchify((patchify(z.float()) - mean) / std)


def denormalize_latents(x, bn_mean, bn_std):
    """
    Pure function. Inverse of normalize_latents (before vae.decode).

    Examples:
        >>> z, m, s = torch.randn(1, 32, 4, 4), torch.randn(128), torch.rand(128) + 0.5
        >>> bool(torch.allclose(denormalize_latents(normalize_latents(z, m, s), m, s), z, atol=1e-5))
        True
    """
    mean, std = bn_mean.view(1, -1, 1, 1).to(x.device), bn_std.view(1, -1, 1, 1).to(x.device)
    return unpatchify(patchify(x.float()) * std + mean)


def bn_stats(vae_dir):
    """
    Query. The VAE BatchNorm statistics, read from its weights file without loading the VAE.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: bn_mean (128,), bn_std = sqrt(running_var + eps) (128,), float32
    """
    path = models.localize(vae_dir)
    weights = load_file(path / "diffusion_pytorch_model.safetensors")
    eps = json.loads((path / "config.json").read_text())["batch_norm_eps"]
    return weights["bn.running_mean"].float(), (weights["bn.running_var"].float() + eps).sqrt()


def shift_mu(image_tokens):
    """
    Pure function. FIBO2's resolution shift mu for an image of this many tokens (the pipeline's calculate_shift).

    Examples:
        >>> [round(shift_mu(t), 3) for t in (256, 1024, 4096)]  # 256², 512², 1024²
        [0.5, 0.63, 1.15]
    """
    slope = (SHIFT_MU[1] - SHIFT_MU[0]) / (SHIFT_SEQ[1] - SHIFT_SEQ[0])
    return SHIFT_MU[0] + slope * (image_tokens - SHIFT_SEQ[0])


def shift_sigmas(u, mu):
    """
    Pure function. FIBO2's exponential time shift: σ = e^mu / (e^mu + (1/u − 1)).

    Args:
        u (torch.Tensor): (B,) noise levels in (0, 1]
        mu (float): from shift_mu

    Returns:
        torch.Tensor: (B,) shifted noise levels

    Examples:
        >>> [round(s, 3) for s in shift_sigmas(torch.tensor([0.25, 0.5, 1.0]), 1.15).tolist()]
        [0.513, 0.76, 1.0]
    """
    return math.exp(mu) / (math.exp(mu) + (1 / u - 1))


def sample_sigmas(batch_size, image_tokens, generator=None):
    """
    Near-pure function (advances the RNG state). Training noise levels: u = sigmoid(N(0, 1)), then FIBO2's shift.

    Returns:
        torch.Tensor: (B,) float32 in (0, 1)

    Examples:
        >>> s = sample_sigmas(100000, 4096, torch.Generator().manual_seed(0))
        >>> round(s.median().item(), 2)  # median u = 0.5, shifted at 1024²
        0.76
    """
    return shift_sigmas(torch.sigmoid(torch.randn(batch_size, generator=generator)), shift_mu(image_tokens))


def caption_prompt(caption):
    """
    Pure function. A MAGICK caption as a FIBO2 prompt: a JSON caption that holds only the short description.

    Examples:
        >>> caption_prompt("red apple on a plate .")
        '{"short_description":"red apple on a plate ."}'
    """
    return json.dumps({CAPTION_KEY: caption}, separators=(",", ":"), ensure_ascii=False)


def load_text_encoder(device):
    """
    Query (reads weights). FIBO2's Qwen3-VL-4B text encoder and its processor, bf16, frozen.

    Returns:
        tuple: (Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
    """
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(models.localize(f"{FIBO2_DIR}/text_encoder"), dtype=torch.bfloat16)
    return encoder.to(device).requires_grad_(False).eval(), Qwen3VLProcessor.from_pretrained(models.localize(f"{FIBO2_DIR}/processor"))


@torch.no_grad()
def encode_prompts(text_encoder, processor, prompts):
    """
    Query (runs the GPU). Prompt embeddings exactly as BriaFibo2Pipeline encodes a generation prompt.

    Args:
        text_encoder, processor: from load_text_encoder
        prompts (list[str]): JSON prompts (caption_prompt or NULL_PROMPT)

    Returns:
        tuple: embeds (B, 6, L, 7680) bf16, mask (B, L) bool (padding to the longest prompt)
    """
    from diffusers import BriaFibo2Pipeline

    return BriaFibo2Pipeline._get_qwen_prompt_embeds(text_encoder, processor, prompts, device=text_encoder.device)


def load_transformer(device, adapter=None, base=FIBO2_DIR):
    """
    Query (reads weights). The frozen FIBO2 transformer in bf16; with adapter, its LoRA delta is merged in (W_adapter).

    Args:
        device (torch.device)
        adapter (str | None): a train_fibo2 checkpoint folder of the adapter (lora.safetensors)
        base (str): FIBO2_DIR (dev) or TURBO_DIR (the 4-step model the control is transferred onto)

    Returns:
        BriaFibo2Transformer2DModel: frozen, eval mode
    """
    from diffusers import BriaFibo2Transformer2DModel

    transformer = BriaFibo2Transformer2DModel.from_pretrained(models.localize(f"{base}/transformer"), torch_dtype=torch.bfloat16)
    transformer = transformer.to(device).requires_grad_(False).eval()
    if adapter:
        merge_lora(transformer, load_file(models.localize(adapter) / "lora.safetensors"), scale=1.0)
    return transformer


def lora_delta(lora, module_name):
    """
    Pure function. The weight delta B·A·(alpha / rank) of one module from a saved LoRA state dict.

    Args:
        lora (dict): {"<module>.lora_A.weight": (r, in), "<module>.lora_B.weight": (out, r), "<module>.alpha": ()}
        module_name (str): e.g. "layers.0.attention.to_q"

    Returns:
        torch.Tensor: (out, in) float32

    Examples:
        >>> lora = {"m.lora_A.weight": torch.ones(2, 3), "m.lora_B.weight": torch.ones(4, 2), "m.alpha": torch.tensor(2.0)}
        >>> lora_delta(lora, "m").shape, lora_delta(lora, "m")[0, 0].item()
        (torch.Size([4, 3]), 2.0)
    """
    a, b = lora[f"{module_name}.lora_A.weight"].float(), lora[f"{module_name}.lora_B.weight"].float()
    return b @ a * (float(lora[f"{module_name}.alpha"]) / a.shape[0])


@torch.no_grad()
def merge_lora(transformer, lora, scale):
    """
    Command (changes transformer weights in place). W += scale · lora_delta for every module in the LoRA.

    Args:
        transformer (torch.nn.Module)
        lora (dict): saved by save_lora
        scale (float): 1 adds the delta, −1 removes it
    """
    names = sorted({k.rsplit(".lora_A.weight", 1)[0] for k in lora if k.endswith(".lora_A.weight")})
    assert names, "no LoRA modules in the state dict"
    for name in names:
        weight = transformer.get_submodule(name).weight
        weight.copy_((weight.float() + scale * lora_delta(lora, name).to(weight.device)).to(weight.dtype))


def add_lora(transformer, rank):
    """
    Command (adds trainable float32 LoRA modules to transformer). Rank-r LoRA, alpha = r, on LORA_TARGETS.

    Returns:
        peft.LoraConfig
    """
    from peft import LoraConfig

    config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights=True, target_modules=LORA_TARGETS)
    transformer.add_adapter(config)
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return config


def lora_state(transformer, rank):
    """
    Query. The trainable LoRA as a plain state dict: "<module>.lora_A.weight", "<module>.lora_B.weight", "<module>.alpha".

    Returns:
        dict[str, torch.Tensor]: float32 on CPU
    """
    from peft.utils import get_peft_model_state_dict

    state = {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(transformer).items()}
    for name in {k.rsplit(".lora_A.weight", 1)[0] for k in state if k.endswith(".lora_A.weight")}:
        state[f"{name}.alpha"] = torch.tensor(float(rank))
    return state


class GatedContextProcessor:
    """
    FIBO2 self-attention with a GateControl-style gate on the context tokens (the trimask), for --gates:
    out = attn(q, K_rest, V_rest) + tanh(g) · attn(q, K_ctx, V_ctx), g one zero-initialized scalar per block (refiner and main), so the
    model starts as FIBO2 without the condition. The context tokens are the last `state["context_tokens"]` tokens of the
    stream (FIBO2's order when no gist padding: gist, image, context).
    """

    def __init__(self, gate, state):
        """Command (keeps a reference to the gate parameter and to the shared state dict)."""
        self.gate, self.state = gate, state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        """Pure function (given weights). Gated self-attention; see the class docstring. (B, S, D) -> (B, S, D)."""
        import torch.nn.functional as F

        n = self.state["context_tokens"]
        assert encoder_hidden_states is None and attention_mask is None and n > 0, "gates need context tokens and no gist padding"
        heads = lambda x: x.unflatten(-1, (attn.heads, -1))
        q, k, v = heads(attn.to_q(hidden_states)), heads(attn.to_k(hidden_states)), heads(attn.to_v(hidden_states))
        q, k = attn.norm_q(q), attn.norm_k(k)
        rotate = lambda x: torch.view_as_real(torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)
        q, k = rotate(q), rotate(k)
        sdpa = lambda kk, vv: F.scaled_dot_product_attention(q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)).transpose(1, 2)
        out = sdpa(k[:, :-n], v[:, :-n]) + self.gate.tanh().to(q.dtype) * sdpa(k[:, -n:], v[:, -n:])
        return attn.to_out[0](out.flatten(2, 3))


def attach_context_gates(transformer):
    """
    Command (adds transformer.context_gates, 46 zero-initialized float32 scalars (2 refiner + 44 main blocks), replaces their
    self-attention processors, and sets transformer.context_gate_state = {"context_tokens": 0}, which
    set_context_tokens updates before each forward).
    """
    state = transformer.context_gate_state = {"context_tokens": 0}
    device = transformer.layers[0].attention.to_q.weight.device
    blocks = list(transformer.noise_refiner) + list(transformer.layers)  # the image tokens meet the context in both
    transformer.context_gates = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1, device=device)) for _ in blocks])
    for gate, block in zip(transformer.context_gates, blocks):
        block.attention.set_processor(GatedContextProcessor(gate, state))


def set_context_tokens(transformer, cond):
    """Command (updates transformer.context_gate_state when gates are attached). cond: (B, 32, LH, LW) or None."""
    if hasattr(transformer, "context_gate_state"):
        transformer.context_gate_state["context_tokens"] = 0 if cond is None else cond.shape[2] * cond.shape[3] // 4


SPLIT_BACKGROUNDS = {"gray": 0.5, "white": 1.0, "black": 0.0}  # splitter heads in output order (§17), the background each target is composited on (split_alpha.composite_on_gray's gray_level); head 0 is the model's original output
SPLITTER_HEAD_FILE = "splitter_head.safetensors"  # In a splitter step folder: SplitterHead.delta (weight, bias), float32
MATTE_HEAD_FILE = "matte_head.safetensors"  # In a matte step folder (§21): SplitterHead.matte (weight, bias), float32


class SplitterHead(torch.nn.Module):
    """
    FIBO2's final linear (dim -> 128 = 2·2 patch × 32 latent channels) made K heads for the splitter (§17), one per
    SPLIT_BACKGROUNDS: head k's weights are W_base + γ·Δ_k, Δ zero-initialized, so every head starts as the base layer
    (the user's "triplicate it": rows 128–383 begin as copies of 0–127) and γ = 0 is the original model.
    forward returns head 0, the shape the transformer's own unpatchify expects, and keeps every head's output in
    self.last (B, N, K·128) for split_velocities.
    """

    def __init__(self, base, heads=len(SPLIT_BACKGROUNDS)):
        """Command (keeps base frozen, adds the trainable float32 delta Linear(dim, K·128), zeros)."""
        super().__init__()
        self.base, self.heads, self.gamma, self.last = base.requires_grad_(False), heads, 1.0, None
        self.delta = torch.nn.Linear(base.in_features, heads * base.out_features, device=base.weight.device, dtype=torch.float32)
        torch.nn.init.zeros_(self.delta.weight)
        torch.nn.init.zeros_(self.delta.bias)
        self.matte, self.mu = None, 1.0  # the matte adapter (§21): add_matte

    def add_matte(self):
        """
        Command (adds the trainable float32 Linear(dim, 128) self.matte, zeros). A 4th head, the alpha matte (§21):
        the gray head + μ·Δ_matte, so it starts as a copy of the gray head and μ = 0 leaves the splitter's 3 heads and
        outputs unchanged (the matte head then equals the gray head). Returns self.
        """
        self.matte = torch.nn.Linear(self.base.in_features, self.base.out_features, device=self.base.weight.device, dtype=torch.float32)
        torch.nn.init.zeros_(self.matte.weight)
        torch.nn.init.zeros_(self.matte.bias)
        return self

    def forward(self, x):
        """Command (sets self.last). (B, N, dim) -> head 0 (B, N, 128); self.last = all heads (B, N, K·128)."""
        import torch.nn.functional as F

        base = self.base(x)
        delta = F.linear(x, self.delta.weight.to(x.dtype), self.delta.bias.to(x.dtype))
        # "B N C, B N (K C) -> B N (K C)": base tiled K times. Not einx: the web export traces N symbolically, and einx
        # bakes the traced N into a reshape (fibo2_web.SplitterVelocityGraph failed at a second latent size)
        self.last = base.repeat(1, 1, self.heads) + self.gamma * delta
        if self.matte is not None:  # head 4 = the gray head + μ·Δ_matte (§21)
            matte = self.last[..., :base.shape[-1]] + self.mu * F.linear(x, self.matte.weight.to(x.dtype), self.matte.bias.to(x.dtype))
            self.last = torch.cat([self.last, matte], dim=-1)
        return self.last[..., :base.shape[-1]]


def attach_splitter(transformer):
    """
    Command (replaces transformer.final_layer.linear with a SplitterHead around it). Returns the SplitterHead.
    """
    head = transformer.final_layer.linear = SplitterHead(transformer.final_layer.linear)
    return head


def splitter_of(transformer):
    """Query. The transformer's SplitterHead, or None for a model without the splitter."""
    head = transformer.final_layer.linear
    return head if isinstance(head, SplitterHead) else None


def split_velocities(head, height, width):
    """
    Pure function (of head.last, the last forward's output). Every head's velocity, unpatchified exactly as FIBO2's
    forward unpatchifies its output (token (h, w) row-major, features (C, ph, pw)).

    Args:
        head (SplitterHead): after a transformer forward
        height, width (int): LH, LW of the latents

    Returns:
        torch.Tensor: (B, K, 32, LH, LW), head order SPLIT_BACKGROUNDS (+ the matte head last when added, K = 4)

    Examples:
        >>> head = SplitterHead(torch.nn.Linear(4, 8))  # 8 = 2·2 patch × 2 channels, K = 3
        >>> _ = head(torch.randn(1, 6, 4))  # 6 tokens = a (4, 6) latent
        >>> split_velocities(head, 4, 6).shape
        torch.Size([1, 3, 2, 4, 6])
        >>> v = split_velocities(head, 4, 6); bool((v[:, 0] == v[:, 2]).all())  # Δ = 0: the heads are copies
        True
    """
    import einx

    return einx.rearrange("B (H W) (K C PH PW) -> B K C (H PH) (W PW)", head.last, H=height // 2,
                          K=head.heads + (head.matte is not None), PH=2, PW=2)


ALPHA_ILS_FLOOR = 1e-9  # rgba_from_composites' F step divides by max(α, this): at α = 0 the objective ignores F, but its direction lets α leave 0
ALPHA_ILS_ITERATIONS = 32  # rgba_from_composites' alternating steps: 16 reach the multi-start L-BFGS-B optimum (gap ≤ 2e-14) on every validated pixel (K = 2, 3, 9, noise ≤ 0.1, manifest §19); 32 for margin


def rgba_from_composites(composites, backgrounds, iterations=ALPHA_ILS_ITERATIONS):
    """
    Pure function. RGBA with decontaminated color (m00824, m00826) from one image composited on K known solid
    backgrounds (§17): per pixel, the box-constrained least squares
        min over α ∈ [0, 1], F ∈ [0, 1]³ of  Σ_k,c (C_kc − α·F_c − (1 − α)·b_kc)²
    solved by iterative least squares (alternating exact 1-D solves, each the global optimum of its block):
        F_c ← clip(mean_k (C_kc − (1 − α)·b_kc) / α, 0, 1)                      (α fixed; α floored at ALPHA_ILS_FLOOR here)
        α  ← clip(Σ_k,c (C_kc − b_kc)·(F_c − b_kc) / Σ_k,c (F_c − b_kc)², 0, 1)  (F fixed)
    started from the unconstrained linear fit 1 − α = Σ_k,c (C_kc − C̄_c)·(b_kc − b̄_c) / Σ_k,c (b_kc − b̄_c)², clipped.
    Unlike that one-shot fit + clipping, it reaches the constrained optimum (validated against projected Adam and
    multi-start L-BFGS-B, manifest §19). F is decontaminated: the known backgrounds' share is removed exactly. The
    uncertainty is the RMS residual: 0 when the K images agree with one RGBA. F is reported as 0 where α = 0.

    Args:
        composites (np.ndarray): (K, H, W, 3) float RGB in [0, 1], e.g. the splitter's decoded heads
        backgrounds (Sequence): (K,) gray levels, e.g. SPLIT_BACKGROUNDS.values(), or (K, 3) RGB background colors
        iterations (int): alternating steps

    Returns:
        tuple: rgba (H, W, 4) float32 straight (not premultiplied) alpha, uncertainty (H, W) float32 RMS residual

    Examples:
        >>> F, a = np.array([[[0.2, 0.6, 1.0], [0.9, 0.1, 0.4]]]), np.array([[1.0, 0.25]])  # (1, 2, 3), (1, 2)
        >>> b = list(SPLIT_BACKGROUNDS.values())
        >>> C = np.stack([a[..., None] * F + (1 - a[..., None]) * k for k in b])
        >>> rgba, err = rgba_from_composites(C, b)
        >>> np.round(rgba.astype(float), 6).tolist(), float(err.max()) < 1e-6
        ([[[0.2, 0.6, 1.0, 1.0], [0.9, 0.1, 0.4, 0.25]]], True)
        >>> C[0, 0, 0] += 0.3  # the gray image disagrees at one pixel: its uncertainty rises (residual of the feasible fit)
        >>> np.round(rgba_from_composites(C, b)[1].astype(float), 3).tolist()
        [[0.153, 0.0]]
        >>> rgb = np.array([[0.5] * 3, [1.0] * 3, [0.0] * 3, [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> C9 = np.stack([0.4 * np.array([0.9, 0.3, 0.1]) + 0.6 * k for k in rgb])[:, None, None]  # 9 RGB backgrounds
        >>> np.round(rgba_from_composites(C9, rgb)[0].astype(float), 6).tolist()  # exact with RGB background colors
        [[[0.9, 0.3, 0.1, 0.4]]]
        >>> Cs = np.stack([np.full(3, 0.02 * 1.0 + 0.98 * k) for k in b])[:, None, None]  # α = 0.02, F = white: faint edge
        >>> np.round(rgba_from_composites(Cs, b)[0].astype(float), 6).tolist()
        [[[1.0, 1.0, 1.0, 0.02]]]
    """
    import einx

    c = np.asarray(composites, dtype=np.float64)
    b = np.asarray(list(backgrounds), dtype=np.float64)
    b = einx.rearrange("K -> K C", b, C=3) if b.ndim == 1 else b  # (K, 3)  # (r, g, b)
    b_dev = b - b.mean(0)
    if (b_dev ** 2).sum() == 0:
        raise ValueError(f"rgba_from_composites needs at least two different backgrounds, got {b.tolist()}")
    k = len(b)
    # Every step only needs these per-pixel sums over k, so an iteration costs a few (H, W, 3) operations
    c_mean = c.mean(0)  # (H, W, 3)
    offset = einx.subtract("K H W C, K C -> K H W C", c, b)  # C_k − b_k
    c_minus_b = offset.sum(0)  # (H, W, 3): Σ_k (C_k − b_k)
    cross = einx.dot("K H W C, K C -> H W", offset, b)  # Σ_k,c (C_kc − b_kc)·b_kc
    alpha = np.clip(1 - einx.dot("K H W C, K C -> H W", c - c_mean, b_dev) / (b_dev ** 2).sum(), 0, 1)
    for _ in range(iterations):
        # α floored only inside this step: at α = 0 the best F is still a direction, so α can leave 0 next step
        color = np.clip((c_mean - (1 - alpha)[..., None] * b.mean(0)) / np.maximum(alpha, ALPHA_ILS_FLOOR)[..., None], 0, 1)
        numerator = (color * c_minus_b).sum(-1) - cross  # Σ_k,c (C_kc − b_kc)·(F_c − b_kc)
        denominator = k * (color ** 2).sum(-1) - 2 * color @ b.sum(0) + (b ** 2).sum()  # Σ_k,c (F_c − b_kc)²
        alpha = np.clip(np.divide(numerator, denominator, out=alpha.copy(), where=denominator > 0), 0, 1)
    color = np.clip((c_mean - (1 - alpha)[..., None] * b.mean(0)) / np.maximum(alpha, ALPHA_ILS_FLOOR)[..., None], 0, 1)
    color = np.where(alpha[..., None] > 0, color, 0)
    fit = einx.multiply("H W, H W C -> 1 H W C", alpha, color) + einx.multiply("H W, K C -> K H W C", 1 - alpha, b)
    uncertainty = np.sqrt(((c - fit) ** 2).mean(axis=(0, 3)))
    return np.concatenate([color, alpha[..., None]], -1).astype(np.float32), uncertainty.astype(np.float32)


def gray_image(size):
    """
    Pure function. The all-gray trimask ("unknown everywhere") as the VAE input: (1, 3, S, S) in [-1, 1].

    Examples:
        >>> round(gray_image(8)[0, 0, 0, 0].item(), 4)
        0.0039
    """
    return torch.full((1, 3, size, size), GRAY / 127.5 - 1)


def load_splitter(transformer, folder, gamma):
    """
    Command (changes transformer weights in place, replaces its final linear). A splitter step folder (§17) at strength γ:
    its LoRA merged with scale γ, and attach_splitter with the saved head delta and head.gamma = γ. γ = 0 leaves the
    model's outputs unchanged (three copies of head 0).

    Returns:
        SplitterHead
    """
    merge_lora(transformer, load_file(folder / "lora.safetensors"), scale=gamma)
    head = attach_splitter(transformer)
    head.delta.load_state_dict(load_file(folder / SPLITTER_HEAD_FILE))
    head.gamma = gamma
    return head


def load_matte(transformer, folder, mu):
    """
    Command (changes transformer weights in place). A matte step folder (§21) at strength μ, on a model that already has
    its splitter (load_splitter): the matte LoRA merged with scale μ, and the head's matte delta with head.mu = μ.
    μ = 0 leaves the splitter model's outputs unchanged (its 4th head is then a copy of the gray head). Any real μ works
    (m00893: negative and > 1 too).

    Returns:
        SplitterHead
    """
    merge_lora(transformer, load_file(folder / "lora.safetensors"), scale=mu)
    head = splitter_of(transformer).add_matte()
    head.matte.load_state_dict(load_file(folder / MATTE_HEAD_FILE))
    head.mu = mu
    return head


def load_eval_model(checkpoint, device, base=FIBO2_DIR, gamma=1.0, mu=1.0):
    """
    Query (reads weights, allocates GPU memory). The shipped model of one control checkpoint (§14 step 3: FIBO2 + the
    control LoRA alone, the adapter removed) with the text encoder, the VAE, its BatchNorm stats and the null prompt.
    base = TURBO_DIR transfers the control onto the 4-step model (W_turbo + control delta), as §13 moved SANA's onto Sprint.
    A checkpoint folder made by distill_fibo2.transfer also holds student.safetensors, our distilled 4-step student LoRA:
    W_dev + student + control.

    Args:
        checkpoint (str): a train_fibo2 control step folder, e.g. models/fibo2/fibo2_tri/step_002000
        device (torch.device)
        base (str): FIBO2_DIR or TURBO_DIR
        gamma (float): splitter and matte checkpoints: the splitter strength γ (load_splitter)
        mu (float): matte checkpoints only: the matte strength μ (load_matte)

    Returns:
        dict: transformer, text (encoder, processor), vae, bn, null (embeds, mask)
    """
    from diffusers import AutoencoderKLFlux2

    meta = json.loads((models.localize(checkpoint) / "meta.json").read_text())
    assert meta["mode"] in ("control", "splitter", "matte") and not meta["gates"], f"{checkpoint}: only control, splitter or matte checkpoints without gates are evaluated"
    transformer = load_transformer(device, base=base)
    if meta["mode"] in ("splitter", "matte"):  # W_base + control + γ·splitter (§17) [+ μ·matte (§21)]: each names the checkpoint it was trained on
        merge_lora(transformer, load_file(models.localize(meta["control"]) / "lora.safetensors"), scale=1.0)
        load_splitter(transformer, models.localize(meta["splitter"] if meta["mode"] == "matte" else checkpoint), gamma)
        if meta["mode"] == "matte":
            load_matte(transformer, models.localize(checkpoint), mu)
    else:
        merge_lora(transformer, load_file(models.localize(checkpoint) / "lora.safetensors"), scale=1.0)
    if (models.localize(checkpoint) / "student.safetensors").exists():  # a distill_fibo2.transfer folder: + the 4-step student LoRA
        merge_lora(transformer, load_file(models.localize(checkpoint) / "student.safetensors"), scale=1.0)
    text = load_text_encoder(device)
    vae = AutoencoderKLFlux2.from_pretrained(models.localize(f"{FIBO2_DIR}/vae"), torch_dtype=torch.bfloat16).to(device)
    return dict(transformer=transformer, text=text, vae=vae, bn=bn_stats(f"{FIBO2_DIR}/vae"), null=encode_prompts(*text, [NULL_PROMPT]))


@torch.no_grad()
def generate_images(model, captions, trimasks, seeds, steps, guidance):
    """
    Query (runs the GPU). One image per caption from its trimask, as the shipped model runs: caption_prompt, the trimask
    encoded as training encodes it (8-bit, 2·v/255 − 1, copied to 3 channels, VAE mode, normalized), sample, decode.

    Args:
        model (dict): load_eval_model
        captions (list[str]): N MAGICK captions
        trimasks (np.ndarray): (N, S, S) float32 in [0, 1] (8-bit PNG / 255: unknown = 128 / 255)
        seeds (list[int]): N noise seeds (the item's noise, fixed across checkpoints)
        steps (int), guidance (float): sampler steps and CFG

    Returns:
        tuple: images (N, S, S, 3) float32 RGB in [0, 1] — with a splitter (N, K, S, S, 3), one per SPLIT_BACKGROUNDS
               head (rgba_from_composites makes RGBA of them) —, seconds per image (list of N floats)
    """
    import time

    vae, bn, device = model["vae"], model["bn"], model["vae"].device
    images, seconds = [], []
    for caption, trimask, seed in zip(captions, trimasks, seeds):
        torch.cuda.synchronize()
        began = time.perf_counter()
        x = torch.from_numpy(np.round(trimask * 255) / 127.5 - 1).float().to(device).expand(1, 3, *trimask.shape)
        cond = normalize_latents(vae.encode(x.to(vae.dtype)).latent_dist.mode(), *bn)
        embeds, mask = encode_prompts(*model["text"], [caption_prompt(caption)])
        out = sample(model["transformer"], embeds, mask, *model["null"], cond, steps, guidance, seed, cond.shape)
        heads = out[0] if out.dim() == 5 else out  # (K, 32, LH, LW) with a splitter: the VAE decodes each head
        image = (vae.decode(denormalize_latents(heads, *bn).to(vae.dtype)).sample.float().clamp(-1, 1) + 1) / 2
        images.append(image.permute(0, 2, 3, 1).cpu().numpy()[0 if out.dim() == 4 else slice(None)])
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - began)
    return np.stack(images), seconds


@torch.no_grad()
def sample(transformer, embeds, mask, null_embeds, null_mask, cond, steps, guidance, seed, shape):
    """
    Query (runs the GPU). FIBO2's sampler (FlowMatchEulerDiscrete, sigmas 1 … 1/steps, the resolution shift, CFG
    against the null caption), with the trimask, if any, as one clean context image in both passes.

    Args:
        transformer: FIBO2 transformer (with any LoRA)
        embeds, mask: (B, 6, L, 7680), (B, L) from encode_prompts
        null_embeds, null_mask: (1, 6, L0, 7680), (1, L0) the null prompt
        cond (torch.Tensor | None): (B, 32, LH, LW) normalized trimask latents; None = text only
        steps (int), guidance (float), seed (int)
        shape (tuple): (B, 32, LH, LW) of the output

    Returns:
        torch.Tensor: (B, 32, LH, LW) normalized latents; with a splitter (attach_splitter), every head's clean estimate
        (B, K, 32, LH, LW) from the last step: the gray ODE is integrated as without it, and with the shared noise
        estimate ε̂ = x_t + (1 − σ)·v_gray, x0_k = ε̂ − v_k (§17); head 0 equals the plain sampler's output
    """
    b, c, h, w = shape
    head = splitter_of(transformer)
    device = embeds.device
    mu = shift_mu(h * w // 4)
    sigmas = [float(shift_sigmas(torch.tensor(s), mu)) for s in np.linspace(1.0, 1 / steps, steps)] + [0.0]
    x = torch.randn(b, c, h, w, generator=torch.Generator().manual_seed(seed)).to(device)
    context = None if cond is None else [cond.to(transformer.dtype)]
    set_context_tokens(transformer, cond)
    run = lambda e, m: transformer(hidden_states=x.to(transformer.dtype), timestep=torch.full((b,), s, device=device),
                                   encoder_hidden_states=e, encoder_attention_mask=m, context_latents=context,
                                   return_dict=False)[0].float()
    heads = lambda v: v[:, None] if head is None else split_velocities(head, h, w).float()  # (B, K, 32, LH, LW)
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        v = heads(run(embeds, mask))
        if guidance > 1:
            uncond = heads(run(null_embeds.expand(b, -1, -1, -1), None if null_mask is None else null_mask.expand(b, -1)))
            v = uncond + guidance * (v - uncond)
        x0 = x[:, None] + (1 - s) * v[:, :1] - v
        x = x + (s_next - s) * v[:, 0]
    return x if head is None else x0
