"""
The SANA-Sprint teacher's training objective, copied from its own fine-tune recipe.

Source: NVlabs/Sana `configs/sana_sprint_config/1024ms/SanaSprint_1600M_img1024_bf16_normT_allqknorm_teacher_ft.yaml`
with `train_scripts/train.py` and `diffusion/model/{gaussian_diffusion,respace}.py`:
    scheduler: predict_v, linear_flow, flow_shift 3.0, logit_normal(0, 1), timestep_norm_scale_factor 1000
    vae: DC-AE f32c32, scale_factor 0.41407;  model: class_dropout_prob 0.1;  train: gradient_clip 0.1

So the teacher is a plain rectified-flow model (TrigFlow is only a re-parameterization used for distillation):
    u ~ sigmoid(N(0, 1)),   sigma = shift·u / (1 + (shift − 1)·u)   (shift = 3)
    x_t = (1 − sigma)·x_0 + sigma·eps,   model(x_t, timestep=sigma) ≈ v = eps − x_0,   loss = mean (v̂ − v)²
In diffusers the transformer takes timestep = sigma in [0, 1] directly ("dense" time embedding: the
pipeline multiplies its 0..1000 scheduler timesteps by `timestep_scale` = 0.001).
"""

import torch

FLOW_SHIFT = 3.0  # Sana 1024px flow_shift (also the teacher scheduler's flow_shift)
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0


def shift_sigmas(u, shift=FLOW_SHIFT):
    """
    Pure function. Resolution shift of flow-matching noise levels: sigma = shift·u / (1 + (shift − 1)·u).

    Args:
        u (torch.Tensor): (B,) unshifted noise levels in [0, 1]
        shift (float): flow shift; 3.0 for Sana at 1024²

    Returns:
        torch.Tensor: (B,) shifted noise levels in [0, 1]; same dtype as u

    Examples:
        >>> shift_sigmas(torch.tensor([0.0, 0.25, 0.5, 1.0])).tolist()
        [0.0, 0.5, 0.75, 1.0]
    """
    return shift * u / (1 + (shift - 1) * u)


def sample_sigmas(batch_size, generator=None, logit_mean=LOGIT_MEAN, logit_std=LOGIT_STD, shift=FLOW_SHIFT):
    """
    Near-pure function (advances the RNG state). Draws training noise levels: logit-normal, then flow-shifted.

    Args:
        batch_size (int): B
        generator (torch.Generator | None): CPU generator
        logit_mean (float): mean of the normal before the sigmoid
        logit_std (float): std of the normal before the sigmoid
        shift (float): flow shift

    Returns:
        torch.Tensor: (B,) float32 on CPU, in (0, 1)

    Examples:
        >>> s = sample_sigmas(100000, torch.Generator().manual_seed(0))
        >>> round(s.median().item(), 2)  # median u is 0.5, shifted to 0.75
        0.75
    """
    u = torch.sigmoid(torch.randn(batch_size, generator=generator) * logit_std + logit_mean)
    return shift_sigmas(u, shift)


def add_flow_noise(x0, noise, sigmas):
    """
    Pure function. Rectified-flow forward process x_t = (1 − sigma)·x_0 + sigma·eps.

    Args:
        x0 (torch.Tensor): (B, LC, LH, LW) clean scaled latents, e.g. (16, 32, 32, 32)
        noise (torch.Tensor): (B, LC, LH, LW) standard normal
        sigmas (torch.Tensor): (B,) noise levels in [0, 1]

    Returns:
        torch.Tensor: (B, LC, LH, LW) noisy latents

    Examples:
        >>> add_flow_noise(torch.ones(2, 1, 1, 1), torch.zeros(2, 1, 1, 1), torch.tensor([0.25, 1.0])).flatten().tolist()
        [0.75, 0.0]
    """
    s = sigmas.to(x0.dtype).view(-1, *[1] * (x0.ndim - 1))
    return (1 - s) * x0 + s * noise


def flow_velocity_target(x0, noise):
    """
    Pure function. The teacher's prediction target (Sana `predict_v`, diffusers `flow_prediction`): v = eps − x_0.

    Args:
        x0 (torch.Tensor): (B, LC, LH, LW) clean scaled latents
        noise (torch.Tensor): (B, LC, LH, LW) the noise used in add_flow_noise

    Returns:
        torch.Tensor: (B, LC, LH, LW)

    Examples:
        >>> flow_velocity_target(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5])).tolist()
        [-0.5, -1.5]
    """
    return noise - x0


def denoised_from_velocity(xt, velocity, sigmas):
    """
    Pure function. Recovers the x_0 estimate from a velocity prediction: x_0 = x_t − sigma·v.

    Args:
        xt (torch.Tensor): (B, LC, LH, LW) noisy latents
        velocity (torch.Tensor): (B, LC, LH, LW) predicted v
        sigmas (torch.Tensor): (B,) noise levels

    Returns:
        torch.Tensor: (B, LC, LH, LW)

    Examples:
        >>> x0, eps, s = torch.tensor([[2.0]]), torch.tensor([[-1.0]]), torch.tensor([0.4])
        >>> denoised_from_velocity(add_flow_noise(x0, eps, s), flow_velocity_target(x0, eps), s).tolist()
        [[2.0]]
    """
    return xt - sigmas.to(xt.dtype).view(-1, *[1] * (xt.ndim - 1)) * velocity
