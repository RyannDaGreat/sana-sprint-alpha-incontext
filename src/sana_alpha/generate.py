"""Eval generation with fixed noise. Every step (baseline, adapter, control, final) calls `generate`."""

import time

import einx
import numpy as np
import torch

from sana_alpha.gatecontrol import COND_KWARG

STEP_NOISE_SEED = 777  # seed of the per-item CPU generator that the Sprint sampler uses at its intermediate step


@torch.no_grad()
def encode_mattes(vae, mattes, batch_size=8):
    """
    Query (runs the GPU). GateControl condition latents of mattes, made as in training: the 8-bit matte copied to 3
    channels in [-1, 1], DC-AE encoded in the VAE's dtype, rounded to bf16 (encode_latents.py stores bf16), then
    multiplied by vae.config.scaling_factor (0.41407) in float32, as train_control does.

    Args:
        vae (AutoencoderDC): the teacher's DC-AE (bf16, on the GPU)
        mattes (np.ndarray): (N, 1024, 1024) float32 in [0,1] (8-bit / 255), e.g. load_eval_set()["mattes"]
        batch_size (int): images per encoder call

    Returns:
        torch.Tensor: (N, 32, 32, 32) float32 on the CPU
    """
    images = einx.rearrange("N H W -> N C H W", torch.from_numpy(mattes) * 2 - 1, C=3)
    latents = [vae.encode(images[i:i + batch_size].to(vae.device, vae.dtype)).latent.to(torch.bfloat16).cpu()
               for i in range(0, len(images), batch_size)]
    return torch.cat(latents).float() * vae.config.scaling_factor


def generate(pipe, prompts, noise, *, seeds=None, batch_size=1, cond=None, **call_kwargs):
    """
    Command (runs the GPU; advances only its own local generators). Generates one image per prompt from fixed latents.

    Item i always gets latents noise[i] and a CPU generator seeded seeds[i], so results do not depend
    on batch_size or on which other items are in the batch (up to GPU nondeterminism).

    Args:
        pipe: SanaPipeline or SanaSprintPipeline (see models.load_pipeline); its weights may be swapped first
        prompts (list[str]): N prompts
        noise (torch.Tensor): (N, 32, 32, 32) float32 standard-normal latents, e.g. eval_set.load_eval_set()["noise"]
        seeds (list[int] | None): per-item generator seeds; default STEP_NOISE_SEED + eval index (0..N-1)
        batch_size (int): images per pipeline call
        cond (torch.Tensor | None): (N, 32, 32, 32) GateControl matte latents (encode_mattes); needs a gated transformer
        **call_kwargs: passed to the pipeline, e.g. models.SPRINT_DEFAULTS

    Returns:
        tuple: images (N, 1024, 1024, 3) float32 RGB in [0,1], seconds per image (list of N floats; batch time / batch size)
    """
    assert len(prompts) == len(noise), f"{len(prompts)} prompts but {len(noise)} noise tensors"
    seeds = [STEP_NOISE_SEED + i for i in range(len(prompts))] if seeds is None else seeds
    images, seconds = [], []
    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds[start:stop]]
        torch.cuda.synchronize()
        began = time.perf_counter()
        if cond is not None:
            call_kwargs["attention_kwargs"] = {COND_KWARG: cond[start:stop]}
        out = pipe(prompt=list(prompts[start:stop]), latents=noise[start:stop], generator=generators,
                   output_type="np", use_resolution_binning=False, **call_kwargs).images
        torch.cuda.synchronize()
        seconds += [(time.perf_counter() - began) / (stop - start)] * (stop - start)
        images.append(np.asarray(out, np.float32))
    return np.concatenate(images), seconds
