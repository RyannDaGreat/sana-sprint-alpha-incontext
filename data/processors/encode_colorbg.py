# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "torch==2.15.0", "diffusers==0.43.0", "safetensors==0.8.0", "accelerate==1.10.1"]
# ///
"""
Composites each Sample's rgba.png on its own random solid color and encodes it with the teacher's DC-AE (adapter_v2's training data).

Reads input.json (page_id) and rgba.png. Writes latent_rgb_colorbg.safetensors (same format as encode_latents' latent_rgb) and
result.json (bg_color) into each Sample folder.
"""

import hashlib
import json
import time

import fire
import numpy as np
import torch
from PIL import Image
from _batch import sample_folders
from encode_latents import DTYPE, load_vae, run_pipelined, save_latent, to_model_input
from split_alpha import composite_on_gray


def background_color(page_id):
    """
    Pure function. The image's background color: uniform RGB in [0, 1), seeded by the first 8 bytes of sha256(page_id).

    Args:
        page_id (str): MAGICK image ID

    Returns:
        list[float]: [r, g, b]

    Examples:
        >>> [round(c, 3) for c in background_color("Hcx82jcUpM")]  # The same every run, on every machine
        [0.628, 0.306, 0.224]
    """
    seed = int.from_bytes(hashlib.sha256(page_id.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).random(3).tolist()


def read_input(sample):
    """Query. Reads one Sample and composites it on its color. Returns ((3, 1024, 1024) float32 in [-1, 1], bg_color)."""
    color = background_color(json.loads((sample / "input.json").read_text())["page_id"])
    rgba = np.asarray(Image.open(sample / "rgba.png"))
    return to_model_input(composite_on_gray(rgba, np.array(color, np.float32))), color


def write_output(sample, latent, color, scaling_factor):
    """Command. Writes one Sample's latent file and its result.json."""
    save_latent(sample / "latent_rgb_colorbg.safetensors", latent, scaling_factor)
    (sample / "result.json").write_text(json.dumps({"bg_color": color}))


def main(batch, model_dir):
    """
    Command. Encodes every Sample of batch on the first visible GPU.

    Args:
        batch (str): The $BATCH folder
        model_dir (str): TEACHER_DIR
    """
    vae = load_vae(model_dir)
    samples = sample_folders(batch)
    start = time.time()

    def encode_chunk(inputs):
        latent = vae.encode(torch.from_numpy(np.stack([image for image, _ in inputs])).to("cuda", DTYPE)).latent.cpu()
        return [(latent[i], color, vae.config.scaling_factor) for i, (_, color) in enumerate(inputs)]

    run_pipelined(samples, read_input, encode_chunk, write_output)
    print(f"Encoded {len(samples)} Samples in {time.time() - start:.1f} s ({len(samples) / (time.time() - start):.1f}/s)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
