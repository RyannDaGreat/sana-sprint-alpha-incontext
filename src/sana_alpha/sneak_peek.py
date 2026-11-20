"""
Sneak peek (manifest §18, m00755, m00757): live previews during a FIBO2 generation.

- The **sneak peek highway**: one small head per main transformer block. Head i reads block i's image tokens
  (stop-gradient) and predicts what the final layer will output (the velocity tokens), so a preview needs only the
  blocks run so far. Each head is the final layer's frozen norm and timestep modulation followed by its own
  Linear(dim, 128), initialized as a copy of the final linear (the "logit lens" start: head 43 starts exact, and a
  tuned-lens style fit corrects the earlier ones). Trained with MSE to the final layer's output, detached, so the
  model itself never changes (the highway has its own optimizer and its inputs are detached).
- Tweedie in latent space: x̂0 = x_t − σ·v̂ (rectified flow, train_objective), then the fast decoder makes pixels.
"""

import torch
from safetensors.torch import load_file, save_file

from sana_alpha import fibo2

HIGHWAY_FILE = "sneak_peek_highway.safetensors"  # In a step folder: the highway heads (float32)
LOSS_REPORT_BLOCKS = (0, 11, 22, 33, 43)  # blocks whose relative error the training log prints (FIBO2 has 44 main blocks)


def final_linear(transformer):
    """Query. The final layer's base nn.Linear(dim, 128), inside a SplitterHead when one is attached."""
    head = fibo2.splitter_of(transformer)
    return head.base if head is not None else transformer.final_layer.linear


class SneakPeekHighway(torch.nn.Module):
    """One Linear(dim, 128) per main block (float32), each a copy of the final linear at the start."""

    def __init__(self, transformer):
        """Command (allocates the heads on the final linear's device)."""
        super().__init__()
        base = final_linear(transformer)
        self.heads = torch.nn.ModuleList(torch.nn.Linear(base.in_features, base.out_features, device=base.weight.device,
                                                         dtype=torch.float32) for _ in transformer.layers)
        with torch.no_grad():
            for h in self.heads:
                h.weight.copy_(base.weight.float())
                h.bias.copy_(base.bias.float())
        self.final = [transformer.final_layer]  # a list: the frozen final layer must not become a submodule (its weights are the model's)

    def modulated(self, tokens, c):
        """
        Pure function (of the frozen final layer). The final layer's input transform: LayerNorm(tokens)·(1 + adaLN(c)).

        Args:
            tokens (torch.Tensor): (B, N, dim) image tokens of one block
            c (torch.Tensor): (B, D) the timestep conditioning the final layer reads (adaln_noisy)

        Returns:
            torch.Tensor: (B, N, dim) float32, no grad
        """
        f = self.final[0]
        with torch.no_grad():
            return (f.norm_final(tokens) * (1.0 + f.adaLN_modulation(c)).unsqueeze(1)).float()

    def predict(self, block, tokens, c):
        """Near-pure function (builds an autograd graph into head `block` only). (B, N, dim) -> (B, N, 128) float32 velocity tokens."""
        return self.heads[block](self.modulated(tokens, c))


class HighwayTap:
    """
    Command (registers forward hooks on the transformer). While `capturing`, keeps every main block's output, the final
    layer's input and conditioning, and its output, all detached: the stop-gradient. With gradient checkpointing the
    block outputs are the checkpoint boundaries, which the model keeps anyway, so this costs no extra memory; the hooks
    ignore the recomputation in backward because capturing is off by then.
    """

    def __init__(self, transformer):
        self.capturing, self.blocks, self.final_in, self.c, self.final_out = False, {}, None, None, None
        self.handles = [layer.register_forward_hook(self._block_hook(i)) for i, layer in enumerate(transformer.layers)]
        self.handles.append(transformer.final_layer.register_forward_pre_hook(self._final_pre, with_kwargs=True))
        self.handles.append(transformer.final_layer.register_forward_hook(self._final_post))

    def _block_hook(self, i):
        def hook(module, args, output):
            if self.capturing:
                self.blocks[i] = output.detach()
        return hook

    def _final_pre(self, module, args, kwargs):
        if self.capturing:
            self.final_in, self.c = args[0].detach(), kwargs["c"].detach()

    def _final_post(self, module, args, output):
        if self.capturing:
            self.final_out = output.detach()

    def image_tokens(self, block):
        """
        Query. Block `block`'s image tokens: the same slice of its output the final layer reads of the last block's.
        The transformer slices hidden_states[:, image_start : image_start + N] as a view, so the slice's offset in the
        last block's storage gives image_start.

        Returns:
            torch.Tensor: (B, N, dim)
        """
        last = self.blocks[len(self.blocks) - 1]
        assert self.final_in.untyped_storage().data_ptr() == last.untyped_storage().data_ptr(), \
            "the final layer's input is not a view of the last block's output: cannot locate the image tokens"
        start = (self.final_in.storage_offset() - last.storage_offset()) // last.stride(1)
        return self.blocks[block][:, start:start + self.final_in.shape[1]]

    def clear(self):
        """Command. Drops the captured tensors."""
        self.blocks, self.final_in, self.c, self.final_out = {}, None, None, None


def highway_backward(highway, tap):
    """
    Command (accumulates .grad of the highway heads). The MSE of every head against the final layer's output, one
    block at a time so only one head's graph is alive.

    Returns:
        list[float]: per block, the error relative to the target's power, mean((pred − target)²) / mean(target²)
    """
    target = tap.final_out.float()
    power = (target ** 2).mean()
    losses = []
    for i in range(len(highway.heads)):
        loss = ((highway.predict(i, tap.image_tokens(i), tap.c) - target) ** 2).mean()
        loss.backward()
        losses.append(loss.detach() / power)
    return torch.stack(losses).tolist()


def all_reduce_grads(module, world):
    """Command (averages module's grads over the DDP world; the highway is outside DDP)."""
    if world > 1:
        import torch.distributed as dist
        grads = [p.grad for p in module.parameters()]
        flat = torch.cat([g.flatten() for g in grads])
        dist.all_reduce(flat)
        flat /= world
        offset = 0
        for g in grads:
            g.copy_(flat[offset:offset + g.numel()].view_as(g))
            offset += g.numel()


def save_highway(highway, folder):
    """Command. Writes the heads to folder/HIGHWAY_FILE."""
    save_file({k: v.detach().float().cpu().contiguous() for k, v in highway.state_dict().items()}, folder / HIGHWAY_FILE)


def load_highway(highway, folder):
    """Command (loads folder/HIGHWAY_FILE into highway). Returns True when the file existed."""
    path = folder / HIGHWAY_FILE
    if not path.exists():
        return False
    highway.load_state_dict(load_file(path))
    return True


def tweedie_x0(xt, velocity, sigma):
    """
    Pure function. The clean-latent estimate of rectified flow, x̂0 = x_t − σ·v (x_t = (1 − σ)·x0 + σ·ε, v = ε − x0).

    Args:
        xt, velocity (torch.Tensor): (B, 32, LH, LW)
        sigma (torch.Tensor): (B,)

    Examples:
        >>> x0, eps, s = torch.randn(1, 32, 4, 4), torch.randn(1, 32, 4, 4), torch.tensor([0.7])
        >>> xt = (1 - s) * x0 + s * eps
        >>> bool(torch.allclose(tweedie_x0(xt, eps - x0, s), x0, atol=1e-6))
        True
    """
    return xt - sigma.view(-1, 1, 1, 1) * velocity
