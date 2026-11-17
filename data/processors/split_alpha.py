# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0"]
# ///
"""Splits each Sample's rgba.png into alpha.png (the matte) and rgb_on_gray.png, and writes the matte's stats to result.json."""

import json
from pathlib import Path

import fire
import numpy as np
from PIL import Image
from _batch import process_samples

WORKERS = 8  # Threads in one batch. run_proc_curate.fish runs many batches at once
PNG_COMPRESS_LEVEL = 3  # zlib level: about as small as the default 6, and faster
IMAGE_SIZE = (1024, 1024)  # Every MAGICK image. Another size is a data error
OPAQUE = 0.5  # green_spill looks only at pixels with alpha above this


def composite_on_gray(rgba, gray_level):
    """
    Pure function. Composites straight-alpha RGBA on a flat gray or color: rgb * alpha + gray_level * (1 - alpha), rounded to 8 bit.

    Args:
        rgba (np.ndarray): (H, W, 4) uint8 RGBA, straight alpha
        gray_level (float | np.ndarray): Background as a fraction of white, in [0, 1]: a gray, or a (3,) RGB color

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB

    Examples:
        >>> pixels = np.array([[[255, 0, 0, 255], [255, 0, 0, 0], [0, 0, 255, 128]]], np.uint8)
        >>> composite_on_gray(pixels, 0.5).tolist()
        [[[255, 0, 0], [128, 128, 128], [63, 63, 192]]]
        >>> composite_on_gray(pixels, np.array([0.0, 1.0, 0.2])).tolist()  # encode_colorbg's solid color
        [[[255, 0, 0], [0, 255, 51], [0, 127, 153]]]
    """
    rgb = rgba[..., :3].astype(np.float32) / 255
    alpha = rgba[..., 3:].astype(np.float32) / 255
    return np.rint((rgb * alpha + gray_level * (1 - alpha)) * 255).astype(np.uint8)


def matte_stats(rgba):
    """
    Pure function. Measures a straight-alpha RGBA image's matte.

    Args:
        rgba (np.ndarray): (H, W, 4) uint8 RGBA

    Returns:
        dict: alpha_coverage (mean alpha), alpha_soft_fraction (fraction with 0 < alpha < 1),
              green_spill (mean of max(0, g - max(r, b)) over pixels with alpha > OPAQUE, in [0, 1]; None when there are none)

    Examples:
        >>> pixels = np.array([[[0, 255, 0, 255], [255, 255, 255, 255], [0, 0, 0, 0], [9, 9, 9, 51]]], np.uint8)
        >>> stats = matte_stats(pixels)
        >>> round(stats["alpha_coverage"], 3), stats["alpha_soft_fraction"], stats["green_spill"]
        (0.55, 0.25, 0.5)
        >>> matte_stats(np.zeros((2, 2, 4), np.uint8))["green_spill"] is None
        True
    """
    rgb = rgba[..., :3].astype(np.float32) / 255
    alpha = rgba[..., 3].astype(np.float32) / 255
    spill = np.maximum(0, rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]))[alpha > OPAQUE]
    return dict(
        alpha_coverage=float(alpha.mean()),
        alpha_soft_fraction=float(((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).mean()),
        green_spill=float(spill.mean()) if spill.size else None,
    )


def main(batch, gray_level):
    """
    Command. Writes alpha.png, rgb_on_gray.png and result.json into each Sample folder of batch.

    Args:
        batch (str): The $BATCH folder
        gray_level (float): GRAY_LEVEL
    """
    def process_sample(sample):
        image = Image.open(sample / "rgba.png")
        if image.mode != "RGBA" or image.size != IMAGE_SIZE:
            raise ValueError(f"Expected a {IMAGE_SIZE} RGBA PNG, got {image.size} {image.mode}")
        rgba = np.asarray(image)
        Image.fromarray(rgba[..., 3], "L").save(sample / "alpha.png", compress_level=PNG_COMPRESS_LEVEL)
        Image.fromarray(composite_on_gray(rgba, float(gray_level)), "RGB").save(sample / "rgb_on_gray.png", compress_level=PNG_COMPRESS_LEVEL)
        (sample / "result.json").write_text(json.dumps(matte_stats(rgba)))

    process_samples(Path(batch), process_sample, WORKERS)


if __name__ == "__main__":
    fire.Fire(main)
