"""
Matte + caption -> RGB on 50% gray + RGBA cutout. Shared by the CLI (`run_diffusion.sh`) and the web UI (server.py).

    ./run_diffusion.sh MATTE.png "a red teapot" OUT_PREFIX      writes OUT_PREFIX_gray.png and OUT_PREFIX_rgba.png
"""

import fire
import numpy as np
import torch
from PIL import Image

from sana_alpha.eval_set import GRAY_LEVEL
from sana_alpha.generate import encode_mattes, generate
from sana_alpha.models import load_eval_pipeline

DEFAULT_CHECKPOINT = "models/control/overfit_test/final_step_000400"
SIZE = 1024  # the model's resolution
MAX_SEED = 2**31 - 1
BYTE = 255


def prepare_matte(image):
    """
    Pure function. Any PIL image -> (1024, 1024) float32 matte in [0,1] (1 = opaque). Transparent pixels count as 0.

    Args:
        image (PIL.Image.Image): any mode and size, e.g. a 512x512 RGBA brush drawing

    Returns:
        np.ndarray: (SIZE, SIZE) float32

    Examples:
        >>> prepare_matte(Image.new("L", (64, 32), 255)).shape, float(prepare_matte(Image.new("L", (8, 8), 255)).min())
        ((1024, 1024), 1.0)
        >>> float(prepare_matte(Image.new("RGBA", (8, 8), (255, 255, 255, 0))).max())
        0.0
    """
    flat = Image.alpha_composite(Image.new("RGBA", image.size, "black"), image.convert("RGBA"))
    return np.asarray(flat.convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / BYTE


def uncomposite(gray, matte):
    """
    Pure function. Recovers the straight-alpha RGBA cutout from RGB on 50% gray: rgb = (I - 0.5 (1 - a)) / a where a > 0.

    Args:
        gray (np.ndarray): (H, W, 3) float32 RGB in [0,1], the foreground composited on gray
        matte (np.ndarray): (H, W) float32 in [0,1], the alpha

    Returns:
        np.ndarray: (H, W, 4) uint8 RGBA, straight alpha

    Examples:
        >>> uncomposite(np.full((1, 2, 3), 0.5, np.float32), np.array([[0.0, 1.0]], np.float32))[0].tolist()
        [[0, 0, 0, 0], [128, 128, 128, 255]]
        >>> uncomposite(np.full((1, 1, 3), 0.75, np.float32), np.array([[0.5]], np.float32))[0, 0].tolist()
        [255, 255, 255, 128]
    """
    alpha = matte[..., None]
    rgb = np.where(alpha > 0, (gray - GRAY_LEVEL * (1 - alpha)) / np.maximum(alpha, 1e-6), 0).clip(0, 1)
    return np.round(np.concatenate([rgb, alpha], axis=-1) * BYTE).astype(np.uint8)


def load_model(checkpoint=DEFAULT_CHECKPOINT):
    """
    Query (reads weights from disk, allocates GPU memory). Loads a matte-conditioned teacher checkpoint (control or final).

    Args:
        checkpoint (str): repo-relative folder, see models.load_eval_pipeline

    Returns:
        tuple: (pipeline, default __call__ kwargs)
    """
    pipe, defaults, conditioned = load_eval_pipeline("teacher", checkpoint)
    assert conditioned, f"{checkpoint} is not matte-conditioned (needs gates)"
    return pipe, defaults


def run_model(model, matte, caption, seed=0, **call_kwargs):
    """
    Command (runs the GPU). One generation from a matte.

    Args:
        model (tuple): load_model()
        matte (np.ndarray): (SIZE, SIZE) float32 in [0,1], see prepare_matte
        caption (str): e.g. "a red teapot"
        seed (int): initial-noise seed
        **call_kwargs: override the pipeline defaults, e.g. num_inference_steps=20, guidance_scale=4.5

    Returns:
        tuple: RGB on gray (SIZE, SIZE, 3) uint8, RGBA cutout (SIZE, SIZE, 4) uint8
    """
    pipe, defaults = model
    noise = torch.randn((1, 32, 32, 32), generator=torch.Generator("cpu").manual_seed(seed))  # (N, LC, LH, LW)
    cond = encode_mattes(pipe.vae, matte[None])
    image = generate(pipe, [caption], noise, seeds=[seed], cond=cond, **{**defaults, **call_kwargs})[0][0]
    return np.round(image * BYTE).astype(np.uint8), uncomposite(image, matte)


def diffuse(matte_path, caption, out_prefix, checkpoint=DEFAULT_CHECKPOINT, seed=0):
    """
    Command. Writes OUT_PREFIX_gray.png (RGB on 50% gray) and OUT_PREFIX_rgba.png (cutout).

    Args:
        matte_path (str): matte image, white = opaque
        caption (str): what the foreground is
        out_prefix (str): output path without suffix
        checkpoint (str): repo-relative matte-conditioned checkpoint folder
        seed (int): initial-noise seed
    """
    gray, rgba = run_model(load_model(checkpoint), prepare_matte(Image.open(matte_path)), caption, seed)
    Image.fromarray(gray).save(f"{out_prefix}_gray.png")
    Image.fromarray(rgba).save(f"{out_prefix}_rgba.png")


if __name__ == "__main__":
    fire.Fire(diffuse)
