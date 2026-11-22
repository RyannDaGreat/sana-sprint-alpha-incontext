# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0"]
# ///
"""
Makes one h3_magick_images row per Sample (manifest §22): the MAGICK image on 50% gray, area-downsampled 1024 -> 512.
Each Sample folder has input.json {page_id, h3_split, subject} and rgb_on_gray.png. It writes children/0/image.png and
children/0/fields.json (one child row, keyed by page_id, in h3_magick_images).
"""
import json
from pathlib import Path

import fire
import numpy as np
from PIL import Image
from _batch import process_samples

WORKERS = 8  # Threads in one batch
SIZE = 512  # H3 Magic resolution (m00912)
PNG_COMPRESS_LEVEL = 3  # zlib level: about as small as the default 6, and faster


def area_downsample(image, size):
    """
    Pure function. Area-averages a square image down by an integer factor (1024 -> 512 averages each 2 x 2 block).

    Args:
        image (np.ndarray): (H, W, C) uint8, H = W = a multiple of size
        size (int): Output side

    Returns:
        np.ndarray: (size, size, C) uint8

    Examples:
        >>> x = np.array([[[0], [255]], [[255], [255]]], np.uint8)  # (2, 2, 1)
        >>> area_downsample(x, 1).tolist()
        [[[191]]]
    """
    factor = image.shape[0] // size
    assert image.shape[0] == image.shape[1] == size * factor, image.shape
    blocks = image.reshape(size, factor, size, factor, -1).astype(np.float32)
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


def process_sample(sample):
    """Command. Writes the Sample's children/0/image.png (512 x 512 RGB) and children/0/fields.json."""
    fields = json.loads((sample / "input.json").read_text())
    image = np.asarray(Image.open(sample / "rgb_on_gray.png").convert("RGB"))
    child = sample / "children" / "0"
    child.mkdir(parents=True)
    Image.fromarray(area_downsample(image, SIZE)).save(child / "image.png", compress_level=PNG_COMPRESS_LEVEL)
    (child / "fields.json").write_text(json.dumps(dict(page_id=fields["page_id"], split=fields["h3_split"], subject=fields["subject"])))


def main(batch):
    """
    Command. Makes the 512 x 512 image of every Sample of batch.

    Args:
        batch (str): The $BATCH folder
    """
    process_samples(batch, process_sample, WORKERS)


if __name__ == "__main__":
    fire.Fire(main)
