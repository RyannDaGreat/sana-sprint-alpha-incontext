# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "torch==2.15.0", "diffusers==0.43.0", "safetensors==0.8.0", "accelerate==1.10.1"]
# ///
"""
Encodes each Sample's trimask.png with the teacher's DC-AE, copied to 3 channels like the matte (encode_latents).

Writes latent_trimask.safetensors into each Sample folder: the same format as encode_latents' latent_alpha
({"latent": (32, 32, 32) bfloat16 (LC, LH, LW)}, unscaled). Gray 128 (unknown) becomes about 0 in the model input.
"""

import time

import fire
import numpy as np
import torch
from PIL import Image
from _batch import sample_folders
from encode_latents import DTYPE, load_vae, run_pipelined, save_latent, to_model_input


def read_input(sample):
    """Query. Reads one Sample's trimask. Returns (3, 1024, 1024) float32 in [-1, 1]."""
    return to_model_input(np.asarray(Image.open(sample / "trimask.png").convert("L")))


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
        latent = vae.encode(torch.from_numpy(np.stack(inputs)).to("cuda", DTYPE)).latent.cpu()
        return [(latent[i], vae.config.scaling_factor) for i in range(len(inputs))]

    write = lambda sample, latent, scaling_factor: save_latent(sample / "latent_trimask.safetensors", latent, scaling_factor)
    run_pipelined(samples, read_input, encode_chunk, write)
    print(f"Encoded {len(samples)} Samples in {time.time() - start:.1f} s ({len(samples) / (time.time() - start):.1f}/s)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
