"""
Makes trimask_augmentations rows: the ten trimask_<method> processors all run this file, each with its own method tag.

Not a PEP 723 script: it runs in the repo's own uv project, so it uses src/sana_alpha/trimask (manifest §13), like generate_eval.
Each Sample folder has input.json {page_id, split} and alpha.png (and rgb_on_gray.png for RGB_METHODS). It writes children/<variant>/fields.json and
children/<variant>/trimask.png, one trimask_augmentations row per variant. Every Sample gets at least one row: Bluejay fails a lineage
commit whose first Sample has an empty children/ (concerns.md). So the query, not this file, leaves out the showcase images.
"""

import hashlib
import json
from pathlib import Path

import cv2
import fire
import numpy as np
from PIL import Image

from _batch import process_samples
from sana_alpha.eval_set import page_residue
from sana_alpha.trimask import RGB_METHODS, coverage, generate, to_uint8

WORKERS = 8  # Threads per run: OpenCV and PNG coding release the GIL. run_proc_trimask.fish starts several runs per method
cv2.setNumThreads(1)  # Many runs x WORKERS already fill the CPUs: OpenCV's own pool (one thread per core) made the load ~800


def in_subset(page_id, method, subset_digits):
    """
    Pure function. The rule of each trimask_<method> query: the first hex digit of sha256(page_id + method) is in subset_digits.

    Args:
        page_id (str): magick_samples.page_id
        method (str): A trimask method tag
        subset_digits (str): Hex digits, e.g. "012": 3 of 16, about 26k of the 141k images per method

    Returns:
        bool

    Examples:
        >>> in_subset("O7qN711ot9", "morph_band", "012"), in_subset("HWtdTKVLOk", "morph_band", "012")
        (True, False)
    """
    return hashlib.sha256((page_id + method).encode()).hexdigest()[0] in subset_digits


def process_sample(sample, method, variants, subset_digits):
    """Command. Writes the trimask_augmentations rows of one Sample folder (see the module docstring). Raises if the query broke the subset rule."""
    job = json.loads((sample / "input.json").read_text())
    if not in_subset(job["page_id"], method, subset_digits):
        raise ValueError(f"{job['page_id']} is not in the {method} subset {subset_digits!r}: the query and in_subset disagree")
    if page_residue(job["page_id"]) == 0:
        raise ValueError(f"{job['page_id']} is an eval image: the query must leave it out")
    alpha = np.asarray(Image.open(sample / "alpha.png").convert("L"), np.float32) / 255
    rgb = np.asarray(Image.open(sample / "rgb_on_gray.png").convert("RGB"), np.float32) / 255 if method in RGB_METHODS else None
    for variant in range(variants):
        trimask, params = generate(alpha, job["page_id"], method, variant, rgb)
        stored = to_uint8(trimask)
        fractions = {f"{name}_fraction": value for name, value in coverage(stored).items()}
        child = sample / "children" / str(variant)
        child.mkdir(parents=True)
        Image.fromarray(stored).save(child / "trimask.png")
        fields = {**job, "method": method, "variant": variant, "params": json.dumps(params), **fractions}
        (child / "fields.json").write_text(json.dumps(fields))


def main(batch, method, variants, subset_digits):
    """
    Command. Makes the trimasks of every Sample of batch with one method.

    Args:
        batch (str): The $BATCH folder
        method (str): A tag of sana_alpha.trimask.METHODS
        variants (int): Trimasks per image, $TRIMASK_VARIANTS
        subset_digits (str | int): $TRIMASK_SUBSET, e.g. "012". Fire gives "123" as the int 123; str() restores it
    """
    process_samples(batch, lambda sample: process_sample(sample, method, int(variants), str(subset_digits)), WORKERS)


if __name__ == "__main__":
    fire.Fire(main)
