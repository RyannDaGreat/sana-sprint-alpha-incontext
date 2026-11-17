# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "torch==2.15.0", "diffusers==0.43.0", "safetensors==0.8.0", "accelerate==1.10.1"]
# ///
"""
Encodes each Sample's alpha.png and rgb_on_gray.png with the teacher's DC-AE, and measures the RGB round trip.

Writes latent_alpha.safetensors, latent_rgb.safetensors and result.json (vae_psnr_1024) into each Sample folder.
Latent file: one tensor, key "latent", (32, 32, 32) bfloat16 (LC, LH, LW), the raw encoder output: vae.config.scaling_factor
is NOT applied. Multiply by it (0.41407) before diffusion training; divide by it before vae.decode. The metadata repeats this.
"""

import fcntl
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import torch
from diffusers import AutoencoderDC
from PIL import Image
from safetensors.torch import save_file
from _batch import sample_folders

GPU_BATCH = 8  # Images per encoder call. A100 80GB: 13 encodes/s, 9 decodes/s at 8; slower at 32
IO_WORKERS = 16  # Threads that read PNGs and write latents while the GPU works
DTYPE = torch.bfloat16  # Sana runs DC-AE in bfloat16; the latents are stored in bfloat16 too
LOCAL_MODELS = Path("/models/sana-sprint-alpha-incontext")  # Node-local disk: FSx is too slow to load from in every run


def to_model_input(image):
    """
    Pure function. Turns an 8-bit image into DC-AE input: RGB in [-1, 1]. A grayscale matte is copied to 3 channels.

    Args:
        image (np.ndarray): (H, W) uint8 matte or (H, W, 3) uint8 RGB

    Returns:
        np.ndarray: (3, H, W) float32 in [-1, 1]

    Examples:
        >>> to_model_input(np.array([[0, 255]], np.uint8)).tolist()
        [[[-1.0, 1.0]], [[-1.0, 1.0]], [[-1.0, 1.0]]]
        >>> to_model_input(np.full((1, 1, 3), 128, np.uint8)).astype(float).round(3).tolist()
        [[[0.004]], [[0.004]], [[0.004]]]
    """
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return (image.transpose(2, 0, 1).astype(np.float32) / 127.5) - 1


def psnr(decoded, original):
    """
    Pure function. PSNR in dB of decoded against original, both in [-1, 1], measured in [0, 1] units, one per image.

    Args:
        decoded (torch.Tensor): (B, 3, H, W) float
        original (torch.Tensor): (B, 3, H, W) float

    Returns:
        torch.Tensor: (B,) float32

    Examples:
        >>> a = torch.zeros(1, 3, 2, 2)
        >>> round(psnr(a + 0.2, a).item(), 2)  # 0.1 off in [0, 1]: 20 dB
        20.0
    """
    mse = (((decoded.float().clamp(-1, 1) - original.float()) / 2) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse)


def localize(model_dir):
    """
    Command. Copies model_dir to LOCAL_MODELS/<its path in the repo>, as src/sana_alpha/models.py does, the first time this machine needs it, and returns the local copy.
    A lock makes the other loops wait for the one copy; the copy goes to a temporary folder first, then is renamed into place.
    """
    source = Path(model_dir).resolve()
    local = LOCAL_MODELS / source.relative_to(Path(__file__).resolve().parents[2])  # Repo root
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local.parent / f".{source.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not local.exists():
            partial = local.parent / f".{source.name}.partial"
            if partial.exists():  # Left by a copy that was killed
                shutil.rmtree(partial)
            print(f"Copying {source} to {local}", flush=True)
            shutil.copytree(source, partial)
            partial.rename(local)
    return local


def read_inputs(sample):
    """Query. Reads one Sample's matte and RGB on gray. Returns two (3, 1024, 1024) float32 arrays in [-1, 1]."""
    return tuple(to_model_input(np.asarray(Image.open(sample / name))) for name in ("alpha.png", "rgb_on_gray.png"))


def write_outputs(sample, latent_alpha, latent_rgb, vae_psnr, scaling_factor):
    """Command. Writes one Sample's two latent files and its result.json."""
    metadata = {"scaling_factor": str(scaling_factor), "scaling_factor_applied": "false", "layout": "LC, LH, LW"}
    save_file({"latent": latent_alpha.contiguous()}, sample / "latent_alpha.safetensors", metadata)
    save_file({"latent": latent_rgb.contiguous()}, sample / "latent_rgb.safetensors", metadata)
    (sample / "result.json").write_text(json.dumps({"vae_psnr_1024": vae_psnr}))


def main(batch, model_dir):
    """
    Command. Encodes every Sample of batch on the first visible GPU. Reading the next chunk overlaps the GPU work.

    Args:
        batch (str): The $BATCH folder
        model_dir (str): A diffusers model folder with a vae subfolder (AutoencoderDC), TEACHER_DIR
    """
    start = time.time()
    vae = AutoencoderDC.from_pretrained(localize(model_dir), subfolder="vae", torch_dtype=DTYPE).to("cuda").eval()
    samples = sample_folders(batch)
    loaded = time.time()
    chunks = [samples[i:i + GPU_BATCH] for i in range(0, len(samples), GPU_BATCH)]
    with ThreadPoolExecutor(IO_WORKERS) as pool, torch.inference_mode():
        read = lambda chunk: [pool.submit(read_inputs, sample) for sample in chunk]
        pending = read(chunks[0]) if chunks else []
        writes = []
        for index, chunk in enumerate(chunks):
            inputs = [future.result() for future in pending]
            pending = read(chunks[index + 1]) if index + 1 < len(chunks) else []
            alpha = torch.from_numpy(np.stack([a for a, _ in inputs])).to("cuda", DTYPE)
            rgb = torch.from_numpy(np.stack([r for _, r in inputs])).to("cuda")  # float32: PSNR compares against the exact 8-bit image
            latent_alpha = vae.encode(alpha).latent
            latent_rgb = vae.encode(rgb.to(DTYPE)).latent
            vae_psnr = psnr(vae.decode(latent_rgb).sample, rgb).tolist()
            latent_alpha, latent_rgb = latent_alpha.cpu(), latent_rgb.cpu()
            writes += [pool.submit(write_outputs, sample, latent_alpha[i], latent_rgb[i], vae_psnr[i], vae.config.scaling_factor)
                       for i, sample in enumerate(chunk)]
        for future in writes:
            future.result()  # Raises a write error here, loudly
    seconds = time.time() - loaded
    print(f"Loaded the VAE in {loaded - start:.1f} s. Encoded {len(samples)} Samples in {seconds:.1f} s ({len(samples) / seconds:.1f}/s). Mean PSNR {np.mean([json.loads((s / 'result.json').read_text())['vae_psnr_1024'] for s in samples]):.2f} dB", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
