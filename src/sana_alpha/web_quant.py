"""
Weight-only 4/8-bit quantization of the SANA transformer for ONNX Runtime Web, after SVDQuant (Li et al., ICLR 2025,
arXiv 2411.05007; MIT Han Lab, the SANA authors' own scheme: github.com/mit-han-lab/deepcompressor, nunchaku).

Per Linear W (N out, K in), with calibration inputs X (tokens, K) from the model's own one-step forwards:
  1. Smoothing: W' = W·diag(λ), x' = x / λ, λ_j = amax(X_j)^α / amax(W_:,j)^(1−α), α grid-searched on the output
     error ‖X Wᵀ − X Ŵᵀ‖ (deepcompressor svdquant/__default__.yaml: GridSearch, OutputsError); λ = 1 is a candidate.
  2. Low-rank branch: W' ≈ L + R, L = rank-32 SVD kept in fp16 (deepcompressor `rank: 32`), refit as
     L = SVD_r(W' − Q(W' − L)) for a few iterations (their `num_iters` loop).
  3. Residual R: symmetric signed integers per group of BLOCK input channels, fp16 scales (`sint4`, no zero point).
Deviations, all for the target runtime: activations stay fp16 (W4A16: WebGPU has no int4 MMA, so SVDQuant's
activation quantization does not apply), group size 32 instead of 64 (ORT WebGPU MatMulNBits fast path needs 32),
randomized SVD (torch.svd_lowrank) for speed. Layers kept fp16 as in nunchaku's SANA checkpoint: patch_embed,
caption_projection, time embedding, proj_out, depthwise convs, norms; also attn2.to_q (runs in fp32, see
web_export.fix_fp16_overflow) and the GateControl gates.

At runtime y = MatMulNBits(x/λ, R) + (x/λ)·Bᵀ·Aᵀ + bias (com.microsoft MatMulNBits, zero point = 2^(bits−1)).
"""

import torch
import torch.nn.functional as F
from torch import nn

RANK = 32  # SVDQuant low-rank branch rank (deepcompressor svdquant config)
BLOCK = 32  # quantization group size along K; ORT WebGPU MatMulNBits' fast path requires 32
ALPHAS = (None, 0.25, 0.5, 0.75)  # smoothing migration strengths tried per layer; None = no smoothing (λ = 1)
LOWRANK_ITERS = 4  # L = SVD_r(W' − Q(W' − L)) refits


def group_quantize(w, bits, block=BLOCK):
    """
    Pure function. Symmetric per-group integer quantization along K with fp16 scales (scale = amax / (2^(bits−1) − 1)).

    Args:
        w (torch.Tensor): (N, K) float32, K divisible by block
        bits (int): 4 or 8
        block (int): group size along K

    Returns:
        tuple: codes (N, K) int8 in [−2^(bits−1), 2^(bits−1) − 1], scales (N, K/block) float16, dequantized (N, K) float32

    Examples:
        >>> codes, scales, deq = group_quantize(torch.tensor([[0.7, -0.35, 0.1, 0.0]]), 4, block=2)
        >>> codes.tolist(), scales.double().round(decimals=3).tolist(), deq.double().round(decimals=2).tolist()
        ([[7, -4, 7, 0]], [[0.1, 0.014]], [[0.7, -0.4, 0.1, 0.0]])
    """
    N, K = w.shape
    g = w.view(N, K // block, block)
    qmax = 2 ** (bits - 1) - 1
    scales = (g.abs().amax(-1) / qmax).clamp(min=1e-8).half()
    codes = torch.round(g / scales.float()[..., None]).clamp(-qmax - 1, qmax)
    return codes.view(N, K).to(torch.int8), scales, (codes * scales.float()[..., None]).view(N, K)


def lowrank(w, rank):
    """
    Near-pure function (randomized SVD: uses torch's global RNG). Rank-r factors of w.

    Args:
        w (torch.Tensor): (N, K) float32
        rank (int): r

    Returns:
        tuple: A (N, r), B (r, K) float32 with w ≈ A @ B

    Examples:
        >>> w = torch.outer(torch.arange(4.), torch.ones(3)); A, B = lowrank(w, 1)
        >>> bool(torch.allclose(A @ B, w, atol=1e-4))
        True
    """
    U, S, V = torch.svd_lowrank(w, q=min(rank + 8, *w.shape), niter=4)
    return U[:, :rank] * S[:rank], V[:, :rank].T


def svdquant_weight(w, x, bits, rank=RANK, alphas=ALPHAS, iters=LOWRANK_ITERS):
    """
    Near-pure function (randomized SVD). SVDQuant weight-only decomposition of one Linear, smoothing α chosen by
    output error on calibration inputs.

    Args:
        w (torch.Tensor): (N, K) float32 weight
        x (torch.Tensor): (T, K) float32 calibration inputs (a token subsample)
        bits (int): 4 or 8
        rank (int): low-rank branch rank (0 = none)
        alphas (tuple): smoothing strengths to try; None = λ = 1
        iters (int): low-rank refits

    Returns:
        dict: inv_smooth (K,) = 1/λ, A (N, r), B (r, K), codes (N, K) int8, scales (N, K/BLOCK) fp16, alpha,
              rel_error = ‖X Wᵀ − X Ŵᵀ‖ / ‖X Wᵀ‖ (Ŵ = (A B + deq(codes)) / λ)

    Examples:
        >>> _ = torch.manual_seed(0); w = torch.randn(64, 64); x = torch.randn(256, 64)
        >>> q4, q8 = svdquant_weight(w, x, 4, rank=8), svdquant_weight(w, x, 8, rank=0)
        >>> q8["rel_error"] < q4["rel_error"] < 0.15
        True
    """
    reference = x @ w.T
    x_amax, w_amax = x.abs().amax(0).clamp(min=1e-5), w.abs().amax(0).clamp(min=1e-5)
    best = None
    for alpha in alphas:
        lam = torch.ones_like(x_amax) if alpha is None else x_amax ** alpha / w_amax ** (1 - alpha)
        lam = lam / lam.log().mean().exp()  # geometric mean 1, keeps fp16 ranges centred
        ws = w * lam
        A, B = lowrank(ws, rank) if rank else (ws.new_zeros(ws.shape[0], 0), ws.new_zeros(0, ws.shape[1]))
        for _ in range(iters if rank else 1):
            codes, scales, deq = group_quantize(ws - A @ B, bits)
            if rank:
                A, B = lowrank(ws - deq, rank)
        codes, scales, deq = group_quantize(ws - A @ B, bits)
        w_hat = (A.half().float() @ B.half().float() + deq) / lam
        error = float((x @ w_hat.T - reference).norm() / reference.norm())
        if best is None or error < best["rel_error"]:
            best = dict(inv_smooth=(1 / lam).half(), A=A.half(), B=B.half(), codes=codes, scales=scales,
                        alpha=alpha, rel_error=error)
    return best


def pack_nbits(codes, bits, block=BLOCK):
    """
    Pure function. ORT MatMulNBits weight layout: unsigned codes (code + 2^(bits−1)), shape (N, K/block, block·bits/8)
    uint8; for 4 bits the even k is the low nibble.

    Args:
        codes (torch.Tensor): (N, K) int8 signed codes
        bits (int): 4 or 8
        block (int): group size

    Returns:
        torch.Tensor: (N, K/block, block·bits/8) uint8

    Examples:
        >>> pack_nbits(torch.tensor([[-8, 7, 0, 1]], dtype=torch.int8), 4, block=4).tolist()
        [[[240, 152]]]
    """
    N, K = codes.shape
    u = (codes.to(torch.int16) + 2 ** (bits - 1)).to(torch.uint8)
    if bits == 4:
        u = u[:, 0::2] | (u[:, 1::2] << 4)
    return u.view(N, K // block, block * bits // 8)


def unpack_nbits(packed, scales, bits, block=BLOCK):
    """
    Pure function. Inverse of pack_nbits times the scales: the dequantized weight.

    Args:
        packed (torch.Tensor): (N, K/block, block·bits/8) uint8
        scales (torch.Tensor): (N, K/block) float16
        bits (int): 4 or 8
        block (int): group size

    Returns:
        torch.Tensor: (N, K) float32

    Examples:
        >>> codes = torch.tensor([[-8, 7, 0, 1]], dtype=torch.int8)
        >>> unpack_nbits(pack_nbits(codes, 4, 4), torch.tensor([[0.5]]).half(), 4, 4).tolist()
        [[-4.0, 3.5, 0.0, 0.5]]
    """
    N = packed.shape[0]
    u = packed.view(N, -1)
    if bits == 4:
        u = torch.stack([u & 15, u >> 4], -1).view(N, -1)
    codes = u.float() - 2 ** (bits - 1)
    return (codes.view(N, -1, block) * scales.float()[..., None]).view(N, -1)


@torch.library.custom_op("websana::matmul_nbits", mutates_args=())
def matmul_nbits(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    """Pure function. x (..., K) @ dequantized weightᵀ -> (..., N), in x's dtype (exported as com.microsoft MatMulNBits)."""
    return (x.float() @ unpack_nbits(packed, scales, bits, block).T).to(x.dtype)


@matmul_nbits.register_fake
def _(x, packed, scales, bits, block):
    """Pure function. Output shape (..., N) for tracing."""
    return x.new_empty(*x.shape[:-1], packed.shape[0])


def matmul_nbits_onnx(x, packed, scales, bits: int, block: int):
    """Pure function. onnxscript translation of websana::matmul_nbits to com.microsoft MatMulNBits."""
    import onnx
    from onnxscript.values import Opset
    if not onnx.defs.has("MatMulNBits", "com.microsoft"):  # contrib op: onnx ships no schema, the tracer needs one
        D, P = onnx.defs.OpSchema, onnx.defs.OpSchema.FormalParameter
        onnx.defs.register_schema(D(
            "MatMulNBits", "com.microsoft", 1, inputs=[P("A", "T1"), P("B", "T2"), P("scales", "T1")],
            outputs=[P("Y", "T1")], type_constraints=[("T1", ["tensor(float16)", "tensor(float)"], ""),
                                                      ("T2", ["tensor(uint8)"], "")],
            attributes=[D.Attribute(a, D.AttrType.INT, "") for a in ("K", "N", "bits", "block_size")]))
    N, n_blocks = packed.shape[0], packed.shape[1]
    return Opset("com.microsoft", 1).MatMulNBits(x, packed, scales, K=n_blocks * block, N=N, bits=bits, block_size=block)


class QuantLinear(nn.Module):
    """y = MatMulNBits(x/λ) + (x/λ)·Bᵀ·Aᵀ + bias: an SVDQuant weight-only Linear in fp16."""

    def __init__(self, q, bias, bits):
        """
        Command (registers buffers).

        Args:
            q (dict): svdquant_weight output
            bias (torch.Tensor | None): (N,) Linear bias
            bits (int): 4 or 8
        """
        super().__init__()
        self.bits = bits
        self.register_buffer("packed", pack_nbits(q["codes"], bits))
        self.register_buffer("scales", q["scales"])
        self.register_buffer("inv_smooth", q["inv_smooth"])
        self.register_buffer("A", q["A"].T.contiguous())  # (r, N)
        self.register_buffer("B", q["B"].T.contiguous())  # (K, r)
        self.register_buffer("bias", None if bias is None else bias.half())

    def forward(self, x):
        """Pure function (given fixed buffers). (..., K) -> (..., N) in x's dtype (fp16 for SANA; the casts are then no-ops)."""
        xs = x * self.inv_smooth.to(x.dtype)
        y = matmul_nbits(xs, self.packed, self.scales.to(x.dtype), self.bits, BLOCK)  # MatMulNBits: scales in A's dtype
        if self.A.shape[0]:
            y = y + (xs @ self.B.to(x.dtype)) @ self.A.to(x.dtype)
        return y if self.bias is None else y + self.bias


def target_linears(transformer):
    """
    Query. The Linears to quantize: every nn.Linear in the transformer blocks except attn2.to_q (fp32 path) and the
    gates (outside the blocks). Mix-FFN 1×1 convs must be linearized first (web_export.linearize_mix_ffn).

    Args:
        transformer (SanaTransformer2DModel)

    Returns:
        dict: qualified name -> nn.Linear (160 for SANA 1.6B)
    """
    return {f"transformer_blocks.{n}": m for n, m in transformer.transformer_blocks.named_modules()
            if isinstance(m, nn.Linear) and "attn2.to_q" not in n}


def set_module(root, name, module):
    """Command. Replaces root's submodule at dotted path name."""
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)
