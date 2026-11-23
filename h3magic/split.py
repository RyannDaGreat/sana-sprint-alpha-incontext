"""
The split adapter of the H3 Magic project (manifest §22, m00972): H3's output projection with K heads, gray / white /
black, the FIBO2 splitter design of manifest §17 on MiniMax H3.

H3's `proj_out` is Linear(hidden 5120, 96): 96 = 24 latent channels x the 2 x 2 patch of one video row. SplitterHead
keeps it frozen and adds a zero-initialized float32 delta Linear(5120, K·96):
    heads = tile(proj_out(x), K) + γ·Δ(x)          (B, N, K, 96), head order SPLIT_BACKGROUNDS
It returns head 0 (gray), the shape H3's forward expects, and keeps every head in `.last`. γ = 0, or Δ = 0, gives the
base model exactly. The split adapter = this Δ + a fresh LoRA trained with it, both on top of baked adapters.

Shared-noise targets (H3 sign, t = 1 clean): the state x_t = t·x0_gray + (1 − t)·ε is the gray trajectory only, and
head k predicts v_k = x0_k − ε with the same ε. At sampling the gray ODE runs as usual; from the last step,
    x0_k = x_t − t·v_gray + v_k          (ε̂ = x_t − t·v_gray; k = gray gives the usual x0 = x_t + (1 − t)·v_gray)
"""
import einx
import torch
from torch import nn

SPLIT_BACKGROUNDS = {"gray": 0.5, "white": 1.0, "black": 0.0}  # heads in output order: the background each target is composited on
HEAD_FILE = "splitter_head.safetensors"  # Δ weight (K·96, 5120) and bias (K·96,), float32, in each split step folder


class SplitterHead(nn.Module):
    """Frozen base Linear + trainable zero-init float32 delta Linear with K times its outputs; returns head 0."""

    def __init__(self, base, heads=len(SPLIT_BACKGROUNDS)):
        """
        Command (allocates the delta on base's device).

        Args:
            base (nn.Linear): H3's proj_out, (in 5120 -> out 96)
            heads (int): K
        """
        super().__init__()
        self.base, self.heads, self.gamma, self.last = base, heads, 1.0, None
        self.delta = nn.Linear(base.in_features, heads * base.out_features, bias=True, device=base.weight.device, dtype=torch.float32)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        """
        Near-pure function (stores every head in self.last). x (B, N, 5120) -> head 0 (B, N, 96) in base's dtype; self.last
        = (B, N, K, 96) float32. With γ = 0 it returns base(x) itself.
        """
        out = self.base(x)
        delta = einx.rearrange("... (K C) -> ... K C", self.delta(x.float()), K=self.heads)
        self.last = einx.add("... C, ... K C -> ... K C", out.float(), self.gamma * delta)
        return out if self.gamma == 0 else self.last[..., 0, :].to(out.dtype)


def attach_splitter(transformer, heads=len(SPLIT_BACKGROUNDS)):
    """Command. Replaces transformer.proj_out by a SplitterHead around it (only Δ trains). Returns the head."""
    assert not isinstance(transformer.proj_out, SplitterHead), "the transformer already has a splitter head"
    head = SplitterHead(transformer.proj_out, heads)
    head.delta.requires_grad_(True)
    transformer.proj_out = head
    return head


def detach_splitter(transformer):
    """Command. Puts the base proj_out back (a no-op without a splitter head)."""
    if isinstance(transformer.proj_out, SplitterHead):
        transformer.proj_out = transformer.proj_out.base


def splitter_of(transformer):
    """Query. The transformer's SplitterHead, or None."""
    return transformer.proj_out if isinstance(transformer.proj_out, SplitterHead) else None


def video_heads(head, video_indices):
    """
    Query (reads head.last). Every head's prediction at the video rows of the last forward.

    Args:
        head (SplitterHead)
        video_indices (Tensor): (rows,) positions of the video rows in the packed sequence

    Returns:
        Tensor: (B, rows, K, 96) float32
    """
    return head.last.index_select(1, video_indices)


def x0_heads(x_t, t, velocities):
    """
    Pure function. Every head's clean estimate from one step of the gray ODE: x0_k = x_t − t·v_0 + v_k (H3: t = 1 clean,
    v = x0 − ε, shared ε).

    Args:
        x_t (Tensor): (..., C) the gray state, e.g. (B, rows, 96)
        t (float): H3's clean level of x_t (1 − σ)
        velocities (Tensor): (..., K, C) head velocities, head 0 = gray

    Returns:
        Tensor: (..., K, C)

    Examples:
        >>> x0 = torch.tensor([[0.2, 0.4], [1.0, 1.0], [-1.0, 0.0]])  # gray, white, black targets (K = 3, C = 2)
        >>> eps, t = torch.tensor([0.5, -0.3]), 0.7
        >>> x_t = t * x0[0] + (1 - t) * eps                            # the gray trajectory
        >>> torch.allclose(x0_heads(x_t, t, x0 - eps), x0, atol=1e-6)  # exact velocities give back every target
        True
    """
    return einx.add("... C, ... K C -> ... K C", x_t - t * velocities[..., 0, :], velocities)


def save_head(head, folder):
    """Command. Writes HEAD_FILE (Δ weight and bias, float32) into folder."""
    from safetensors.torch import save_file

    save_file({"weight": head.delta.weight.detach().float().cpu().contiguous(),
               "bias": head.delta.bias.detach().float().cpu().contiguous()}, str(folder / HEAD_FILE), {"heads": ",".join(SPLIT_BACKGROUNDS)})


def load_head(head, folder):
    """Command. Loads HEAD_FILE of a split step folder into head.delta (shapes must match). Returns head."""
    from safetensors.torch import load_file

    state = load_file(str(folder / HEAD_FILE))
    assert state["weight"].shape == head.delta.weight.shape, f"{folder}: Δ {tuple(state['weight'].shape)} vs head {tuple(head.delta.weight.shape)}"
    with torch.no_grad():
        head.delta.weight.copy_(state["weight"])
        head.delta.bias.copy_(state["bias"])
    return head
