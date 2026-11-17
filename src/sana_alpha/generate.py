"""Eval generation with fixed noise. Every step (baseline, adapter, control, final) calls `generate`."""

import time

import numpy as np
import torch

STEP_NOISE_SEED = 777  # seed of the per-item CPU generator that the Sprint sampler uses at its intermediate step


def generate(pipe, prompts, noise, *, seeds=None, batch_size=1, **call_kwargs):
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
        out = pipe(prompt=list(prompts[start:stop]), latents=noise[start:stop], generator=generators,
                   output_type="np", use_resolution_binning=False, **call_kwargs).images
        torch.cuda.synchronize()
        seconds += [(time.perf_counter() - began) / (stop - start)] * (stop - start)
        images.append(np.asarray(out, np.float32))
    return np.concatenate(images), seconds
