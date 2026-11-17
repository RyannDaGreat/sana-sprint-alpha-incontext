"""
GateControl (arXiv 2603.27666, https://github.com/Carol-lyh/GateControl) ported to diffusers' SanaTransformer2DModel.

The condition (a clean matte latent) is a second token stream. It runs through the same patch_embed and the same 20
blocks with the same weights, the same adaLN modulation and the same timestep as the noisy target stream x. Text
cross-attention and the Mix-FFN run per stream. Only the linear self-attention mixes the streams: its K/V cover
[x; c] (bidirectional). Per block, two gates read the modulated pre-attention input:

    g_x = sigmoid(W_x(LN(x_mod)))    g_c = sigmoid(W_c(LN(c_mod)))    (LN without affine, W: Linear(2240, 1))
    x  += gate_msa · (g_x · attn_x + g_c · attn_c · keep)             (attn_c added token by token)
    c  += gate_msa · attn_c

`keep` is 0 for condition-dropped samples; their condition keys are also zeroed, so x attends to x alone. Gates are
zero-initialised, so g = 0.5 at init (not identity to the base: see agents/concerns.md for the measured deviation).

Differences from the official code (stock SANA): the Sprint teacher has QK RMSNorm (per token, so it works on the
concatenated sequence unchanged), timestep_scale 0.001 (the transformer takes sigma in [0, 1], not 1000·sigma; see
`train_control.py`) and no guidance embedding. Here the two streams are stacked on the batch axis for everything
except self-attention, instead of duplicating every call.

The base model is not modified: gates live in `transformer.gatecontrol_gates` and the control LoRA is a PEFT adapter
named "control". `enable_gatecontrol` replaces the instance's forward so a stock SanaPipeline can drive it.
"""

import types

import einx
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

ADAPTER_NAME = "control"
# The official GateControl LoRA targets (train_sana_gatecontrol.py). "proj" is patch_embed.proj, "linear" is
# time_embed.linear, linear_1/linear_2 are the caption projection and timestep embedder MLPs.
LORA_TARGETS = ["proj", "linear_1", "linear_2", "linear", "to_q", "to_k", "to_v", "to_out.0",
                "conv_point", "conv_inverted", "proj_out"]
COND_KWARG = "gatecontrol_cond"  # key in the pipeline's attention_kwargs that carries the condition latents
LINEAR_ATTN_EPS = 1e-15  # same epsilon as diffusers' SanaLinearAttnProcessor2_0


class GateControlGates(nn.Module):
    """Two Linear(dim, 1) gates per block, zero-initialised (so each gate starts at sigmoid(0) = 0.5)."""

    def __init__(self, num_blocks, dim):
        """
        Command (allocates parameters).

        Args:
            num_blocks (int): transformer blocks, 20 for SANA 1.6B
            dim (int): hidden size, 2240 for SANA 1.6B
        """
        super().__init__()
        self.x_gates = nn.ModuleList(nn.Linear(dim, 1) for _ in range(num_blocks))
        self.c_gates = nn.ModuleList(nn.Linear(dim, 1) for _ in range(num_blocks))
        for gate in [*self.x_gates, *self.c_gates]:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)


def tokens_to_grid(x, height, width):
    """
    Pure function. (B, H·W, C) tokens -> (B, C, H, W) grid. torch ops, not einx: einx cannot take symbolic sizes,
    which the dynamic-size ONNX export (web_export) traces with.

    Examples:
        >>> tokens_to_grid(torch.arange(6.).reshape(1, 3, 2), 1, 3).tolist()
        [[[[0.0, 2.0, 4.0]], [[1.0, 3.0, 5.0]]]]
    """
    return x.transpose(1, 2).unflatten(2, (height, width))


def grid_to_tokens(x):
    """
    Pure function. (B, C, H, W) grid -> (B, H·W, C) tokens; inverse of tokens_to_grid.

    Examples:
        >>> grid_to_tokens(tokens_to_grid(torch.arange(6.).reshape(1, 3, 2), 1, 3)).flatten().tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    """
    return x.flatten(2).transpose(1, 2)


def streams_to_sequence(t, heads):
    """
    Pure function. einx "(s B) N (H d) -> B (s N) H d" with s = 2, in torch ops: einx writes the traced N into its
    Reshape, which breaks the dynamic-size ONNX export.

    Examples:
        >>> t = torch.randn(4, 3, 6)
        >>> bool((streams_to_sequence(t, 2) == einx.rearrange("(s B) N (H d) -> B (s N) H d", t, s=2, H=2)).all())
        True
    """
    return t.unflatten(0, (2, -1)).transpose(0, 1).flatten(1, 2).unflatten(-1, (heads, -1))


def sequence_to_streams(t):
    """
    Pure function. einx "B (s N) H d -> (s B) N (H d)" with s = 2; inverse of streams_to_sequence.

    Examples:
        >>> t = torch.randn(2, 6, 2, 3)
        >>> bool((sequence_to_streams(t) == einx.rearrange("B (s N) H d -> (s B) N (H d)", t, s=2)).all())
        True
    """
    return t.flatten(-2).unflatten(1, (2, -1)).transpose(0, 1).flatten(0, 1)


def gate_value(gate, h):
    """
    Pure function. GateControl gate: sigmoid(Linear(LayerNorm_no_affine(h))), computed in float32.

    Args:
        gate (nn.Linear): Linear(D, 1)
        h (torch.Tensor): (B, N, D) modulated pre-attention hidden states

    Returns:
        torch.Tensor: (B, N, 1) float32 in (0, 1)

    Examples:
        >>> g = nn.Linear(4, 1); _ = nn.init.zeros_(g.weight); _ = nn.init.zeros_(g.bias)
        >>> gate_value(g, torch.randn(2, 3, 4)).flatten().tolist()
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    """
    with torch.autocast(h.device.type, enabled=False):
        h = F.layer_norm(h.float(), (h.shape[-1],))
        return torch.sigmoid(F.linear(h, gate.weight.float(), gate.bias.float()))


def relu_linear_attention(q, k, v):
    """
    Pure function. SANA's ReLU linear attention, out_i = Σ_j (q_i·k_j) v_j / Σ_j (q_i·k_j), in float32.

    Same math as diffusers' SanaLinearAttnProcessor2_0 (pad V with a ones channel to get the denominator).

    Args:
        q (torch.Tensor): (B, N, H, d) queries, already ReLU'd
        k (torch.Tensor): (B, M, H, d) keys, already ReLU'd (zero rows are ignored exactly)
        v (torch.Tensor): (B, M, H, d) values

    Returns:
        torch.Tensor: (B, N, H, d) float32

    Examples:
        >>> q = torch.ones(1, 1, 1, 2); k = torch.tensor([[[[1., 0.]], [[0., 0.]]]]); v = torch.tensor([[[[2., 4.]], [[9., 9.]]]])
        >>> relu_linear_attention(q, k, v).flatten().tolist()  # the zero key contributes nothing
        [2.0, 4.0]
    """
    q, k, v = q.float(), k.float(), v.float()
    v = F.pad(v, (0, 1), value=1.0)
    with torch.autocast(q.device.type, enabled=False):  # keep float32 even under autocast, like the base runs it
        # torch transposes, not einx: einx writes traced sizes into Reshape nodes (breaks dynamic-size ONNX export)
        kv = k.permute(0, 2, 3, 1) @ v.transpose(1, 2)  # (B, H, d, M) @ (B, H, M, e) -> (B, H, d, e)
        out = (q.transpose(1, 2) @ kv).transpose(1, 2)  # (B, H, N, e) -> (B, N, H, e)
    return out[..., :-1] / (out[..., -1:] + LINEAR_ATTN_EPS)


def two_stream_self_attention(attn, h, keep):
    """
    Pure function (given fixed module weights). Bidirectional linear self-attention over [x; c] with one SANA attn1.

    Args:
        attn (diffusers Attention): block.attn1 (to_q/k/v, optional norm_q/norm_k RMSNorm, to_out)
        h (torch.Tensor): (2B, N, D) modulated inputs, streams stacked on the batch axis as [x; c]
        keep (torch.Tensor): (B,) bool, False = condition dropped (its keys are zeroed, so x attends to x only)

    Returns:
        torch.Tensor: (2B, N, D) attention outputs [attn_x; attn_c] after to_out, in h's dtype
    """
    q, k, v = attn.to_q(h), attn.to_k(h), attn.to_v(h)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    q, k = F.relu(q), F.relu(k)
    k_x, k_c = k.chunk(2)
    k_c = k_c * keep.to(k_c.dtype)[:, None, None]  # (B, N, D) · (B,); not einx: it bakes traced N into a Reshape
    k = torch.cat([k_x, k_c])
    # [x; c] on the batch axis -> one sequence of 2N tokens per sample
    q, k, v = (streams_to_sequence(t, attn.heads) for t in (q, k, v))
    out = relu_linear_attention(q, k, v).to(h.dtype)
    out = sequence_to_streams(out)
    return attn.to_out[1](attn.to_out[0](out))


def gated_block_forward(block, x_gate, c_gate, h, temb, text, text_mask, height, width, keep):
    """
    Pure function (given fixed module weights). One SanaTransformerBlock on both streams with GateControl gating.

    Args:
        block (SanaTransformerBlock): the diffusers block (weights shared by both streams)
        x_gate, c_gate (nn.Linear): this block's gates, Linear(D, 1)
        h (torch.Tensor): (2B, N, D) hidden states, [x; c] stacked on the batch axis, N = height·width
        temb (torch.Tensor): (2B, 6·D) adaLN timestep embedding (same for both streams)
        text (torch.Tensor): (2B, L, D) projected caption tokens
        text_mask (torch.Tensor | None): (2B, 1, L) additive bias
        height, width (int): latent grid, e.g. 32, 32 at 1024²
        keep (torch.Tensor): (B,) bool, False = condition dropped

    Returns:
        torch.Tensor: (2B, N, D)
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.scale_shift_table[None] + einx.rearrange("B (s D) -> B s D", temb, s=6)
    ).chunk(6, dim=1)

    hn = (block.norm1(h) * (1 + scale_msa) + shift_msa).to(h.dtype)
    attn = two_stream_self_attention(block.attn1, hn, keep)
    hn_x, hn_c = hn.chunk(2)
    attn_x, attn_c = attn.chunk(2)
    cond_in = attn_c * keep.to(attn_c.dtype)[:, None, None]  # (B, N, D) · (B,), torch broadcast as above
    mixed_x = gate_value(x_gate, hn_x) * attn_x + gate_value(c_gate, hn_c) * cond_in
    h = h + gate_msa * torch.cat([mixed_x.to(h.dtype), attn_c])

    h = h + block.attn2(h, encoder_hidden_states=text, attention_mask=text_mask)

    hn = block.norm2(h) * (1 + scale_mlp) + shift_mlp
    ff = grid_to_tokens(block.ff(tokens_to_grid(hn, height, width)))
    return h + gate_mlp * ff


def gatecontrol_forward(transformer, latents, cond_latents, text_embeds, text_mask, timestep, keep=None,
                        guidance=None):
    """
    Pure function (given fixed weights). Two-stream GateControl forward of a SanaTransformer2DModel.

    Args:
        transformer (SanaTransformer2DModel): with `gatecontrol_gates` attached (see attach_gates)
        latents (torch.Tensor): (B, LC, LH, LW) noisy target latents, e.g. (B, 32, 32, 32) at 1024²
        cond_latents (torch.Tensor): (B, LC, LH, LW) clean condition latents (scaled like latents)
        text_embeds (torch.Tensor): (B, L, 2304) Gemma-2 caption embeddings
        text_mask (torch.Tensor): (B, L) 1 = token, 0 = padding
        timestep (torch.Tensor): (B,) what the transformer expects (teacher: sigma in [0, 1])
        keep (torch.Tensor | None): (B,) bool, False = drop the condition for that sample; None = keep all
        guidance (torch.Tensor | None): (B,) for guidance-embedded models (Sprint student); None for the teacher

    Returns:
        torch.Tensor: (B, LC, LH, LW) velocity prediction for the target stream only
    """
    B, _, height, width = latents.shape
    assert transformer.config.patch_size == 1, "GateControl port assumes patch_size 1 (SANA 1.6B)"
    assert cond_latents.shape == latents.shape, f"cond {tuple(cond_latents.shape)} != latents {tuple(latents.shape)}"
    keep = torch.ones(B, dtype=torch.bool, device=latents.device) if keep is None else keep.to(latents.device)

    h = transformer.patch_embed(torch.cat([latents, cond_latents]))
    if guidance is not None:
        temb, embedded = transformer.time_embed(timestep, guidance=guidance, hidden_dtype=h.dtype)
    else:
        temb, embedded = transformer.time_embed(timestep, batch_size=B, hidden_dtype=h.dtype)
    text = transformer.caption_norm(transformer.caption_projection(text_embeds).view(B, -1, h.shape[-1]))
    bias = einx.rearrange("B L -> B 1 L", (1 - text_mask.to(h.dtype)) * -10000.0)

    temb2, text2, bias2 = (torch.cat([t, t]) for t in (temb, text, bias))
    gates = transformer.gatecontrol_gates
    use_ckpt = torch.is_grad_enabled() and getattr(transformer, "gatecontrol_checkpointing", False)
    for i, block in enumerate(transformer.transformer_blocks):
        args = (block, gates.x_gates[i], gates.c_gates[i], h, temb2, text2, bias2, height, width, keep)
        h = checkpoint(gated_block_forward, *args, use_reentrant=False) if use_ckpt else gated_block_forward(*args)

    x = transformer.norm_out(h[:B], embedded, transformer.scale_shift_table)
    x = transformer.proj_out(x)
    return tokens_to_grid(x, height, width)


def compile_blocks():
    """
    Command (rebinds this module's gated_block_forward). torch.compile of the block function, static shapes.

    Measured on an A100 (shared): 1.66× faster, 30% less activation memory than eager. One graph per block
    (modules are guarded by identity), hence the larger dynamo cache.
    """
    global gated_block_forward
    torch._dynamo.config.cache_size_limit = 64
    gated_block_forward = torch.compile(gated_block_forward, dynamic=False)


def attach_gates(transformer):
    """
    Command. Adds zero-initialised GateControl gates as `transformer.gatecontrol_gates` (on the transformer's device).

    Args:
        transformer (SanaTransformer2DModel)

    Returns:
        GateControlGates: the new module (float32)
    """
    assert not hasattr(transformer, "gatecontrol_gates"), "gates already attached"
    cfg = transformer.config
    gates = GateControlGates(cfg.num_layers, cfg.num_attention_heads * cfg.attention_head_dim)
    transformer.gatecontrol_gates = gates.to(transformer.device)
    return transformer.gatecontrol_gates


def add_control_lora(transformer, rank, alpha=None):
    """
    Command. Adds the PEFT LoRA adapter "control" on LORA_TARGETS (gaussian init for A, zeros for B).

    Args:
        transformer (SanaTransformer2DModel): base weights (teacher with the adapter baked in)
        rank (int): LoRA rank r
        alpha (int | None): LoRA alpha; None = rank (scale 1, as in GateControl)

    Returns:
        LoraConfig: the config used
    """
    config = LoraConfig(r=rank, lora_alpha=rank if alpha is None else alpha, target_modules=LORA_TARGETS,
                        init_lora_weights="gaussian")
    transformer.add_adapter(config, adapter_name=ADAPTER_NAME)
    return config


def _pipeline_forward(self, hidden_states, encoder_hidden_states, timestep, guidance=None,
                      encoder_attention_mask=None, attention_kwargs=None, return_dict=True, **unused):
    """
    Pure function (given fixed weights). SanaTransformer2DModel.forward replacement used by SanaPipeline.

    Reads the condition from attention_kwargs[COND_KWARG]: (b, LC, LH, LW) where b divides the latent batch
    (the pipeline stacks [negative; positive] for CFG; the condition is repeated to match, as GateControl does).
    Without COND_KWARG it runs the base forward unchanged.
    """
    kwargs = dict(attention_kwargs or {})
    cond = kwargs.pop(COND_KWARG, None)
    assert kwargs.get("scale", 1.0) == 1.0 and set(kwargs) <= {"scale"}, f"unsupported attention_kwargs {kwargs}"
    assert not unused.get("controlnet_block_samples") and unused.get("attention_mask") is None, f"unsupported {unused}"
    if cond is None:
        return type(self).forward(self, hidden_states, encoder_hidden_states, timestep, guidance=guidance,
                                  encoder_attention_mask=encoder_attention_mask, return_dict=return_dict)
    B = hidden_states.shape[0]
    assert B % cond.shape[0] == 0, f"latent batch {B} is not a multiple of condition batch {cond.shape[0]}"
    cond = cond.to(hidden_states).repeat(B // cond.shape[0], 1, 1, 1)
    out = gatecontrol_forward(self, hidden_states, cond, encoder_hidden_states, encoder_attention_mask,
                              timestep, guidance=guidance)
    return (out,) if not return_dict else types.SimpleNamespace(sample=out)


def enable_gatecontrol(transformer):
    """
    Command. Makes a stock SanaPipeline drive the gated two-stream model: replaces this instance's forward.

    Pass the condition per call: pipe(..., attention_kwargs={COND_KWARG: cond_latents}).

    Args:
        transformer (SanaTransformer2DModel): with gates attached (and the control LoRA, unless merged)

    Returns:
        the same transformer
    """
    assert hasattr(transformer, "gatecontrol_gates"), "attach_gates / load_control first"
    transformer.forward = types.MethodType(_pipeline_forward, transformer)
    return transformer


def control_state_dicts(transformer):
    """
    Query. The trainable control weights as float32 CPU tensors.

    Args:
        transformer (SanaTransformer2DModel): with the "control" LoRA adapter and gates

    Returns:
        tuple: (lora dict with PEFT keys like "transformer_blocks.0.attn1.to_q.lora_A.weight", gates dict)
    """
    lora = get_peft_model_state_dict(transformer, adapter_name=ADAPTER_NAME)
    gates = transformer.gatecontrol_gates.state_dict()
    to32 = lambda d: {k: v.detach().float().cpu().contiguous() for k, v in d.items()}
    return to32(lora), to32(gates)


def save_control(transformer, folder, lora_config, metadata=None):
    """
    Command. Writes folder/control_lora.safetensors and folder/gates.safetensors (float32).

    Args:
        transformer (SanaTransformer2DModel): with the control LoRA and gates
        folder (pathlib.Path): e.g. models/control/<run>/step_002000
        lora_config (LoraConfig): its r and lora_alpha are stored as metadata (needed to merge)
        metadata (dict | None): extra string metadata (run name, step, adapter path, ...)
    """
    folder.mkdir(parents=True, exist_ok=True)
    lora, gates = control_state_dicts(transformer)
    meta = dict(rank=str(lora_config.r), alpha=str(lora_config.lora_alpha), targets=",".join(LORA_TARGETS),
                **{k: str(v) for k, v in (metadata or {}).items()})
    save_file(lora, folder / "control_lora.safetensors", metadata=meta)
    save_file(gates, folder / "gates.safetensors", metadata=meta)


def load_control_files(folder):
    """
    Query. Reads a saved control folder.

    Args:
        folder (pathlib.Path): written by save_control

    Returns:
        tuple: (lora dict float32, gates dict float32, metadata dict)
    """
    from safetensors import safe_open
    with safe_open(folder / "control_lora.safetensors", "pt") as f:
        meta = f.metadata()
    return load_file(folder / "control_lora.safetensors"), load_file(folder / "gates.safetensors"), meta


def load_control(transformer, folder, merge=False):
    """
    Command. Attaches gates and the control LoRA from a saved folder, then enables the pipeline forward.

    Args:
        transformer (SanaTransformer2DModel): base weights (adapter baked in for the control model, or the clean
            base for the final model W_base + (W_control − W_adapter))
        folder (pathlib.Path): written by save_control
        merge (bool): True = add the LoRA delta into the weights in float32 (see merge_lora_into_state_dict) and
            round once to the transformer's dtype; False = keep a PEFT adapter. A Sprint student (guidance_embeds)
            needs merge=True and gets the keys of student_lora_keys

    Returns:
        the same transformer
    """
    lora, gates, meta = load_control_files(folder)
    if transformer.config.guidance_embeds:  # control transfer onto the Sprint student (merge only: see student_lora_keys)
        assert merge, "a PEFT adapter would also hit the student's guidance_embedder.linear_1/2: use merge=True"
        lora = student_lora_keys(lora)
    if merge:
        base = {k: v.float().cpu() for k, v in transformer.state_dict().items()}
        merged = merge_lora_into_state_dict(base, lora, int(meta["rank"]), float(meta["alpha"]))
        transformer.load_state_dict({k: v.to(transformer.dtype) for k, v in merged.items()}, strict=True)
    else:
        config = add_control_lora(transformer, int(meta["rank"]), float(meta["alpha"]))
        result = set_peft_model_state_dict(transformer, lora, adapter_name=ADAPTER_NAME)
        assert not result.unexpected_keys, f"unexpected LoRA keys: {result.unexpected_keys[:5]}"
        assert config.r == int(meta["rank"])
    attach_gates(transformer).load_state_dict(gates, strict=True)
    return enable_gatecontrol(transformer)


def student_lora_keys(lora):
    """
    Pure function. Renames teacher LoRA keys to the Sprint student's: its time_embed holds the teacher's timestep MLP
    `time_embed.emb.timestep_embedder.*` as `time_embed.timestep_embedder.*` (plus a new guidance_embedder).

    Args:
        lora (dict): PEFT keys of a teacher-trained control LoRA

    Returns:
        dict: same tensors, student keys

    Examples:
        >>> list(student_lora_keys({"time_embed.emb.timestep_embedder.linear_1.lora_A.weight": 0, "proj_out.lora_A.weight": 1}))
        ['time_embed.timestep_embedder.linear_1.lora_A.weight', 'proj_out.lora_A.weight']
    """
    return {k.replace("time_embed.emb.timestep_embedder.", "time_embed.timestep_embedder."): v for k, v in lora.items()}


def lora_delta(lora_a, lora_b, scale):
    """
    Pure function. Dense weight delta of one LoRA pair in float32: scale · B @ A (1×1 convs are flattened and restored).

    Args:
        lora_a (torch.Tensor): (r, in) Linear or (r, in, 1, 1) Conv2d
        lora_b (torch.Tensor): (out, r) Linear or (out, r, 1, 1) Conv2d
        scale (float): alpha / r

    Returns:
        torch.Tensor: (out, in) or (out, in, 1, 1) float32

    Examples:
        >>> lora_delta(torch.ones(2, 3), torch.ones(4, 2), 0.5).tolist()[0]
        [1.0, 1.0, 1.0]
        >>> lora_delta(torch.ones(2, 3, 1, 1), torch.ones(4, 2, 1, 1), 1.0).shape
        torch.Size([4, 3, 1, 1])
    """
    assert lora_a.shape[2:] == lora_b.shape[2:] and all(s == 1 for s in lora_a.shape[2:]), "only Linear / 1x1 conv"
    delta = scale * (lora_b.flatten(1).double() @ lora_a.flatten(1).double())
    return delta.float().reshape(lora_b.shape[0], lora_a.shape[1], *lora_a.shape[2:])


def merge_lora_into_state_dict(base, lora, rank, alpha):
    """
    Pure function. W + (alpha/r)·B@A for every LoRA pair, in float32. Exact path for W_base + (W_control − W_adapter).

    Args:
        base (dict): float32 SanaTransformer2DModel state dict (e.g. the clean teacher), keys like "proj_out.weight"
        lora (dict): PEFT keys "<module>.lora_A.weight" / "<module>.lora_B.weight"
        rank (int): r
        alpha (float): lora_alpha

    Returns:
        dict: new float32 state dict (base is not modified)

    Examples:
        >>> base = {"proj_out.weight": torch.zeros(2, 2), "proj_out.bias": torch.zeros(2)}
        >>> lora = {"proj_out.lora_A.weight": torch.ones(1, 2), "proj_out.lora_B.weight": torch.ones(2, 1)}
        >>> merge_lora_into_state_dict(base, lora, 1, 1.0)["proj_out.weight"].tolist()
        [[1.0, 1.0], [1.0, 1.0]]
    """
    out = dict(base)
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
    assert len(lora) == 2 * len(modules), "every LoRA module needs exactly lora_A and lora_B"
    for m in modules:
        key = m + ".weight"
        assert key in base, f"LoRA module {m} has no base weight {key}"
        out[key] = base[key].float() + lora_delta(lora[m + ".lora_A.weight"], lora[m + ".lora_B.weight"], alpha / rank)
    return out
