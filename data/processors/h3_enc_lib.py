"""
Shared library of the encoder-bottleneck study (manifest §25, m00992): is H3's encoder the bottleneck of its
reconstructions? The H3 VAE (encoder + frozen decoder), a differentiable decode, the losses a latent is optimized
against (pixel L1/L2, LPIPS vgg/alex, FIBO2's FLUX.2-VAE latent distance, mixes), and the judges.

The metric functions (psnr ... to_frames) are copied verbatim from the decoder frenzy's scorer
(agents/.scratchpad/fastdec_ga/score.py), so the numbers compare with its BASELINE.json; DISTS and the FLUX.2 latent
distance are added as held-out judges.

Latents are normalized like h3magic/h3.py: (z - latents_mean) / latents_std, shape (LC, LT, LH, LW), e.g. (24, 1, 32, 32)
for one 512² image and (24, 42, 32, 32) for a 141-frame clip. Pixels are float RGB in [0, 1], (T, 3, H, W).
"""
import os
import sys

import einx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from h3magic import h3  # noqa: E402

FLUX2_VAE_DIR = "models/Fibo-2.0-dev/vae"  # FIBO2's image encoder (FLUX.2 VAE), repo-relative
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HF_RADIUS = 0.125  # cycles/pixel = 1/4 of Nyquist (0.5): the sharpness band of fastdec_ga/score.py
BATCH = 16  # frames per chunk of the ΔE computation
D65 = (0.95047, 1.0, 1.08883)
SOBEL = (-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------- metrics (verbatim from fastdec_ga/score.py)

def psnr(a, b):
    """
    Pure function. (F, 3, H, W) x2 float [0, 1] -> float: mean over frames of PSNR on uint8-rounded values.

    >>> x = torch.zeros(2, 3, 4, 4); round(psnr(x, x + 1 / 255), 2)
    48.13
    """
    mse = ((a * 255).round() - (b * 255).round()).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return float((10 * torch.log10(255 ** 2 / mse)).mean())


def rgb_to_lab(x):
    """
    Pure function. (F, 3, H, W) sRGB [0, 1] -> (F, 3, H, W) CIE Lab (L, a, b), D65.

    >>> lab = rgb_to_lab(torch.ones(1, 3, 1, 1)); [round(float(v), 1) for v in lab.flatten()]
    [100.0, 0.0, 0.0]
    """
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4, x / 12.92)
    m = x.new_tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = einx.dot("o i, f i h w -> f o h w", m, lin) / x.new_tensor(D65)[None, :, None, None]
    f = torch.where(xyz > 216 / 24389, xyz.clamp_min(1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], dim=1)


def ciede2000(lab1, lab2):
    """
    Pure function. Two (F, 3, H, W) Lab tensors -> (F, H, W) CIEDE2000 (Sharma et al. 2005, kL=kC=kH=1).

    >>> a = torch.tensor([50.0, 2.6772, -79.7751]).view(1, 3, 1, 1); b = torch.tensor([50.0, 0.0, -82.7485]).view(1, 3, 1, 1)
    >>> round(float(ciede2000(a.double(), b.double())), 4)  # Sharma test pair 1
    2.0425
    """
    L1, a1, b1 = lab1.unbind(1)
    L2, a2, b2 = lab2.unbind(1)
    C1, C2 = a1.hypot(b1), a2.hypot(b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - torch.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = a1p.hypot(b1), a2p.hypot(b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > 180, dh - 360, torch.where(dh < -180, dh + 360, dh))
    dh = torch.where(C1p * C2p == 0, torch.zeros_like(dh), dh)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dh) / 2)
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum, hdiff = h1p + h2p, (h1p - h2p).abs()
    hm = torch.where(C1p * C2p == 0, hsum, torch.where(hdiff <= 180, hsum / 2, torch.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = 1 - 0.17 * torch.cos(torch.deg2rad(hm - 30)) + 0.24 * torch.cos(torch.deg2rad(2 * hm)) \
        + 0.32 * torch.cos(torch.deg2rad(3 * hm + 6)) - 0.20 * torch.cos(torch.deg2rad(4 * hm - 63))
    dTheta = 30 * torch.exp(-(((hm - 275) / 25) ** 2))
    Rc = 2 * torch.sqrt(Cmp ** 7 / (Cmp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / torch.sqrt(20 + (Lm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cmp, 1 + 0.015 * Cmp * T
    Rt = -torch.sin(torch.deg2rad(2 * dTheta)) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def de2000(a, b):
    """Pure function. (F, 3, H, W) sRGB [0, 1] x2 -> float mean CIEDE2000 over frames and pixels (computed in chunks, float32)."""
    return float(torch.cat([ciede2000(rgb_to_lab(a[i:i + BATCH]), rgb_to_lab(b[i:i + BATCH])).flatten() for i in range(0, len(a), BATCH)]).mean())


def temporal_diff(x):
    """
    Pure function. (F, 3, H, W) float [0, 1] -> float: mean |x[t+1] - x[t]| on the 0-255 scale.

    >>> temporal_diff(torch.stack([torch.zeros(3, 2, 2), torch.ones(3, 2, 2)]))
    255.0
    """
    return float((x[1:] - x[:-1]).abs().mean() * 255)


def gray(x):
    """Pure function. (F, 3, H, W) RGB -> (F, 1, H, W) Rec.601 luma."""
    return einx.dot("c, f c h w -> f 1 h w", x.new_tensor([0.299, 0.587, 0.114]), x)


def grad_mag_mean(x):
    """
    Pure function. (F, 3, H, W) RGB -> float: mean Sobel gradient magnitude of the luma (0-255 scale, unnormalized kernel, valid region; only ratios are used).

    >>> ramp = torch.linspace(0, 1, 8).repeat(8, 1)[None, None].expand(1, 3, 8, 8); round(grad_mag_mean(ramp), 1)  # Sobel is unnormalized: 8 * (255 / 7) per pixel step
    291.4
    """
    g = gray(x) * 255
    kx = g.new_tensor(SOBEL)[None, None]
    gx, gy = F.conv2d(g, kx), F.conv2d(g, kx.transpose(2, 3))
    return float(gx.hypot(gy).mean())


def hf_energy(x):
    """
    Pure function. (F, 3, H, W) RGB -> float: mean over frames of the luma FFT power (mean-removed, 0-255 scale) at radius > 0.125 cycles/px.

    >>> flat = torch.full((1, 3, 16, 16), 0.5); hf_energy(flat)
    0.0
    """
    g = gray(x)[:, 0] * 255
    g = g - g.mean(dim=(1, 2), keepdim=True)
    power = torch.fft.fft2(g).abs() ** 2 / g[0].numel()
    fy, fx = torch.fft.fftfreq(g.shape[1]), torch.fft.fftfreq(g.shape[2])
    radius = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt().to(power.device)
    return float(power[:, radius > HF_RADIUS].sum(dim=1).mean())


def sharpness(cand, src):
    """Pure function. Candidate / source ratios: {"sharp_grad", "sharp_fft"} (1.0 = as sharp as the source)."""
    return {"sharp_grad": grad_mag_mean(cand) / grad_mag_mean(src), "sharp_fft": hf_energy(cand) / hf_energy(src)}


def to_frames(u8, device):
    """Pure function. (F, H, W, 3) uint8 numpy -> (F, 3, H, W) float [0, 1] on device."""
    return einx.rearrange("f h w c -> f c h w", torch.from_numpy(u8).to(device)).float().div(255)


# ---------------------------------------------------------------- VAE

def load_vae(device="cuda"):
    """
    Command. The H3 VAE loaded as the pipeline loads it (torch_dtype bf16: the encoder and the cast-sensitive decoder
    modules stay float32), frozen, with gradient checkpointing per decoder transformer block, on `device`.
    """
    from diffusers import AutoencoderKLMiniMaxH3
    h3.allow_single_frame()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(h3.localize()), subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()
    return vae


def stats(vae):
    """Query. latents_mean, latents_std as (LC, 1, 1, 1) float32 on the VAE's device."""
    mean, std = h3.latent_stats(vae)
    return mean[0].to(vae.device), std[0].to(vae.device)


def to_imagenet(x):
    """
    Pure function. (..., 3, ...) channel-second RGB in [0, 1] -> ImageNet-normalized, channel axis 1.

    >>> round(float(to_imagenet(torch.full((1, 3, 1, 1), 0.485))[0, 0, 0, 0]), 4)
    0.0
    """
    shape = (1, 3) + (1,) * (x.ndim - 2)
    return (x - x.new_tensor(IMAGENET_MEAN).view(shape)) / x.new_tensor(IMAGENET_STD).view(shape)


def from_imagenet(x):
    """
    Pure function. Inverse of to_imagenet.

    >>> round(float(from_imagenet(torch.zeros(1, 3, 1, 1))[0, 1, 0, 0]), 3)
    0.456
    """
    shape = (1, 3) + (1,) * (x.ndim - 2)
    return x * x.new_tensor(IMAGENET_STD).view(shape) + x.new_tensor(IMAGENET_MEAN).view(shape)


@torch.no_grad()
def encode_posterior(vae, frames, tiled=False):
    """
    Query (runs the GPU model). The encoder's posterior of a clip or one image, normalized (h3.encode's recipe:
    fp32 weights, bf16 autocast; untiled by default as the project encodes).

    Args:
        frames (np.ndarray): (T, H, W, 3) uint8 RGB, T = 1 or 17n + 5
        tiled (bool): use the VAE's 256-px spatial tiles for encoding

    Returns:
        tuple[torch.Tensor, torch.Tensor]: mode, std, each (LC, LT, LH, LW) float32 on the device, e.g. (24, 1, 32, 32)
    """
    vae.use_tiling = tiled
    x = einx.rearrange("T H W C -> 1 C T H W", torch.from_numpy(np.ascontiguousarray(frames)).to(vae.device)).float() / 255
    with torch.autocast("cuda", torch.bfloat16):
        posterior = vae.encode(to_imagenet(x), return_dict=False)[0]
    mean, std = stats(vae)
    return (posterior.mode()[0].float() - mean) / std, posterior.std[0].float() / std


def sample_posterior(mode, std, seed=h3.ENCODE_SEED):
    """
    Near-pure function (seeded RNG on the tensor's device). A posterior sample mode + std·ε, as h3.encode draws it.

    >>> z = sample_posterior(torch.zeros(2, 1, 1, 1), torch.zeros(2, 1, 1, 1)); z.flatten().tolist()
    [0.0, 0.0]
    """
    eps = torch.randn(mode.shape, generator=torch.Generator(device=mode.device).manual_seed(seed), device=mode.device)
    return mode + std * eps


def decode_float(vae, z):
    """
    Query (runs the GPU model; differentiable under grad). Pixels of normalized latents, unclamped, tiled as h3.decode.

    Args:
        z (torch.Tensor): (LC, LT, LH, LW) normalized float32, e.g. (24, 1, 32, 32) or (24, 42, 32, 32)

    Returns:
        torch.Tensor: (T, 3, H, W) float32 RGB, ~[0, 1], e.g. (1, 3, 512, 512) or (141, 3, 512, 512)
    """
    vae.use_tiling = True
    mean, std = stats(vae)
    video = vae.decode((z * std + mean)[None], return_dict=False)[0].float()
    return einx.rearrange("1 C T H W -> T C H W", from_imagenet(video))


def decode_images(vae, z):
    """
    Query (runs the GPU model; differentiable under grad). Pixels of a batch of single-frame latents (the 7-repeat decode).

    Args:
        z (torch.Tensor): (B, LC, 1, LH, LW) normalized float32, e.g. (4, 24, 1, 32, 32)

    Returns:
        torch.Tensor: (B, 3, H, W) float32 RGB ~[0, 1] unclamped, e.g. (4, 3, 512, 512)
    """
    vae.use_tiling = True
    mean, std = stats(vae)
    video = vae.decode(z * std + mean, return_dict=False)[0].float()
    return from_imagenet(video[:, :, 0])


def to_u8(x):
    """
    Pure function. (T, 3, H, W) float [0, 1] -> (T, H, W, 3) uint8 numpy, clamped and rounded as h3.decode.

    >>> to_u8(torch.tensor([0.5, 1.2, -0.1]).view(1, 3, 1, 1)).tolist()
    [[[[128, 255, 0]]]]
    """
    return einx.rearrange("T C H W -> T H W C", (x.detach().clamp(0, 1) * 255).round().byte()).cpu().numpy()


def u8_to_float(frames, device):
    """Pure function. (T, H, W, 3) uint8 numpy -> (T, 3, H, W) float [0, 1] on device."""
    return to_frames(np.ascontiguousarray(frames), device)


# ---------------------------------------------------------------- FLUX.2 VAE (FIBO2's image encoder)

def load_flux2(device="cuda"):
    """Command. FIBO2's FLUX.2 VAE (bf16, frozen) and its BatchNorm stats on `device`. Returns (vae, bn_mean, bn_std)."""
    from diffusers import AutoencoderKLFlux2
    from sana_alpha import models
    from sana_alpha.fibo2 import bn_stats
    vae = AutoencoderKLFlux2.from_pretrained(models.localize(FLUX2_VAE_DIR), torch_dtype=torch.bfloat16).to(device).eval()
    vae.requires_grad_(False)
    bn_mean, bn_std = bn_stats(FLUX2_VAE_DIR)
    return vae, bn_mean.to(device), bn_std.to(device)


def flux2_latent(flux2, x):
    """
    Query (runs the GPU model; differentiable under grad). FIBO2-normalized FLUX.2 latents (posterior mode).

    Args:
        x (torch.Tensor): (B, 3, H, W) float RGB [0, 1]

    Returns:
        torch.Tensor: (B, 32, H/8, W/8) float32
    """
    from sana_alpha.fibo2 import normalize_latents
    vae, bn_mean, bn_std = flux2
    return normalize_latents(vae.encode((x * 2 - 1).to(vae.dtype)).latent_dist.mode().float(), bn_mean, bn_std)


# ---------------------------------------------------------------- judges

def load_dists():
    """
    Command (loads weights). DISTS (Ding et al. 2020). DISTS_pytorch reads its weights from sys.prefix/weights.pt (a
    data-files install), which is wrong in a uv overlay environment: point sys.prefix at the installed file meanwhile.
    """
    import importlib.metadata
    from DISTS_pytorch import DISTS
    weights = next(f for f in importlib.metadata.files("DISTS_pytorch") if f.name == "weights.pt").locate()
    prefix, sys.prefix = sys.prefix, str(weights.parent)
    try:
        return DISTS()
    finally:
        sys.prefix = prefix


class Judges:
    """Command (holds GPU nets). LPIPS alex + vgg, DISTS, FLUX.2 VAE: the losses and the judges of a reconstruction."""

    def __init__(self, device="cuda"):
        """Command. Loads every net on `device`, frozen."""
        import lpips
        self.device = device
        self.alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
        self.vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
        self.dists = load_dists().to(device).eval().requires_grad_(False)
        self.flux2 = load_flux2(device)

    def lpips(self, net, x, y):
        """Query (differentiable). Mean LPIPS of (B, 3, H, W) [0, 1] pairs; net 'alex' or 'vgg'."""
        return getattr(self, net)(x * 2 - 1, y * 2 - 1).mean()

    @torch.no_grad()
    def score(self, cand, src, batch=8):
        """
        Query. Every metric of a reconstruction vs its source.

        Args:
            cand, src (torch.Tensor): (T, 3, H, W) float [0, 1]; cand is clamped and uint8-rounded first

        Returns:
            dict: psnr, ssim, lpips_alex, lpips_vgg, dists, de2000, flux_mse, sharp_grad, sharp_fft (+ flicker when T > 1)
        """
        from pytorch_msssim import ssim
        cand = (cand.clamp(0, 1) * 255).round() / 255
        out = {"psnr": psnr(cand, src), "de2000": de2000(cand, src), **sharpness(cand, src)}
        parts = {"ssim": [], "lpips_alex": [], "lpips_vgg": [], "dists": [], "flux_mse": []}
        for i in range(0, len(cand), batch):
            a, b = cand[i:i + batch], src[i:i + batch]
            parts["ssim"].append(ssim(a, b, data_range=1.0, size_average=False))
            parts["lpips_alex"].append(self.alex(a * 2 - 1, b * 2 - 1).flatten())
            parts["lpips_vgg"].append(self.vgg(a * 2 - 1, b * 2 - 1).flatten())
            parts["dists"].append(self.dists(a, b, batch_average=False).flatten())
            parts["flux_mse"].append((flux2_latent(self.flux2, a) - flux2_latent(self.flux2, b)).pow(2).mean((1, 2, 3)))
        out.update({k: float(torch.cat(v).mean()) for k, v in parts.items()})
        if len(cand) > 1:
            out["flicker"] = abs(temporal_diff(cand) - temporal_diff(src))
        return out


# ---------------------------------------------------------------- objectives

OBJECTIVES = {  # name -> {loss term: weight}; terms: l2, l1 (pixels [0, 1]), lpips_vgg, lpips_alex, flux (FLUX.2 latent MSE)
    "l2": {"l2": 1.0},
    "l1": {"l1": 1.0},
    "lpips_vgg": {"lpips_vgg": 1.0},
    "lpips_alex": {"lpips_alex": 1.0},
    "flux": {"flux": 1.0},
    "l1_vgg": {"l1": 1.0, "lpips_vgg": 0.1},  # the usual VAE recipe: pixels + a perceptual term
    "l1_vgg_flux": {"l1": 1.0, "lpips_vgg": 0.1, "flux": 0.01},
}


def objective_loss(judges, terms, pred, src, src_flux=None):
    """
    Query (differentiable). Weighted sum of loss terms of pred vs src.

    Args:
        terms (dict): term -> weight, e.g. OBJECTIVES["l1_vgg"]
        pred, src (torch.Tensor): (B, 3, H, W) float [0, 1] (pred unclamped)
        src_flux (torch.Tensor): FLUX.2 latents of src (B, 32, H/8, W/8), needed by the 'flux' term

    Returns:
        torch.Tensor: scalar
    """
    total = pred.new_zeros(())
    for term, weight in terms.items():
        if term == "l2":
            value = (pred - src).pow(2).mean()
        elif term == "l1":
            value = (pred - src).abs().mean()
        elif term in ("lpips_vgg", "lpips_alex"):
            value = judges.lpips(term.split("_")[1], pred, src)
        elif term == "flux":
            value = (flux2_latent(judges.flux2, pred) - src_flux).pow(2).mean()
        else:
            raise ValueError(f"unknown loss term {term}")
        total = total + weight * value
    return total
