# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "safetensors==0.8.0"]
# ///
"""
Writes the lc_renders rows of one scored Latent Compositor candidate (manifest §24): children/<k>/ per decoded test sample,
with decoded.png (byte for byte the candidate's decoded/<index>.png, or showcase/<index>.png for a candidate scored before
decoded/ existed), error.png (|decoded - true composite| x ERROR_GAIN per channel) and fields.json (its per-sample
metrics from results.json, and its latent MSE from pred.safetensors). result.json: renders_sha256 and renders.
"""
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
from PIL import Image
from safetensors.numpy import load_file

KEYS = ["alpha", "fg_straight", "fg_premult", "fg_gray", "bg", "comp"]  # order of the 6 latents (LC_DIR/lc.py KEYS)
ERROR_GAIN = 4  # An error of 8 levels shows as 32: small errors become visible
METRICS = ("psnr", "ssim", "lpips", "flux2_mse")  # results.json per_sample keys
WORKERS = 16  # Threads that read and encode PNGs
PNG_COMPRESS_LEVEL = 3  # zlib level: about as small as the default 6, and faster


def composite(fg, alpha, bg):
    """
    Pure function. alpha * fg + (1 - alpha) * bg in 8 bit, rounded: exactly LC_DIR/lc.py composite (the true composite).

    Args:
        fg (np.ndarray): (H, W, 3) uint8 straight RGB
        alpha (np.ndarray): (H, W) uint8, 255 = opaque
        bg (np.ndarray): (H, W, 3) uint8

    Returns:
        np.ndarray: (H, W, 3) uint8

    Examples:
        >>> composite(np.array([[[255, 0, 0]]], np.uint8), np.array([[128]], np.uint8), np.array([[[0, 0, 255]]], np.uint8)).tolist()
        [[[128, 0, 127]]]
    """
    a = alpha[..., None].astype(np.float32) / 255
    return np.rint((fg.astype(np.float32) / 255 * a + bg.astype(np.float32) / 255 * (1 - a)) * 255).astype(np.uint8)


def error_map(decoded, true, gain=ERROR_GAIN):
    """
    Pure function. |decoded - true| x gain per channel, clipped to 255.

    Args:
        decoded, true (np.ndarray): (H, W, 3) uint8

    Returns:
        np.ndarray: (H, W, 3) uint8

    Examples:
        >>> error_map(np.array([[[10, 200, 0]]], np.uint8), np.array([[[13, 100, 0]]], np.uint8)).tolist()
        [[[12, 255, 0]]]
    """
    return np.minimum(np.abs(decoded.astype(np.int16) - true.astype(np.int16)) * gain, 255).astype(np.uint8)


def latent_mse(pred, target):
    """
    Pure function. Mean squared error per sample over the latent's channels and pixels.

    Args:
        pred, target (np.ndarray): (N, 24, 64, 64) float

    Returns:
        np.ndarray: (N,) float64

    Examples:
        >>> latent_mse(np.ones((2, 1, 2, 2)), np.array([[[[1, 1], [1, 1]]], [[[0, 0], [1, 1]]]])).tolist()
        [0.0, 0.5]
    """
    return np.square(pred.astype(np.float64) - target.astype(np.float64)).mean(axis=(1, 2, 3))


def png_bytes(image):
    """
    Pure function. A lossless PNG of an (H, W, 3) uint8 RGB image.

    Examples:
        >>> png_bytes(np.zeros((2, 2, 3), np.uint8))[:4]
        b'\\x89PNG'
    """
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return out.getvalue()


def render_files(folder, decoded_dir="decoded"):
    """
    Query (lists the candidate folder). {test index: PNG path} of its decoded test samples: decoded/ (every EVAL sample),
    or showcase/ for a candidate scored before evaluate.py wrote decoded/.
    """
    pngs = sorted((folder / decoded_dir).glob("*.png")) or (sorted((folder / "showcase").glob("*.png")) if decoded_dir == "decoded" else [])
    return {int(p.stem): p for p in pngs}


def test_split(data_dir, split="test"):
    """
    Query (reads the test shards). The test split in position order: metadata, true-composite latents and pixels.

    Returns:
        tuple[list[dict], np.ndarray, dict]: metas; comp latents (N, 24, 64, 64) float16; alpha (N, H, W), fg and bg (N, H, W, 3) uint8
    """
    shards = sorted(data_dir.glob(f"{split}_[0-9][0-9][0-9].safetensors"))
    metas = [m for s in shards for m in json.loads(s.with_suffix(".json").read_text())]
    target = np.concatenate([load_file(str(s))["latents"][:, KEYS.index("comp")] for s in shards])
    parts = [load_file(str(s.with_name(s.stem + "_pixels.safetensors"))) for s in shards]
    pixels = {k: np.concatenate([p[k] for p in parts]) for k in ("alpha", "fg", "bg")}
    return metas, target, pixels


def render_child(child, png, true, fields):
    """Command. Writes child/decoded.png (copied byte for byte), child/error.png and child/fields.json."""
    child.mkdir(parents=True)
    data = png.read_bytes()
    (child / "decoded.png").write_bytes(data)
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert decoded.shape == true.shape, (png, decoded.shape, true.shape)
    (child / "error.png").write_bytes(png_bytes(error_map(decoded, true)))
    (child / "fields.json").write_text(json.dumps(fields))


def main(sample, root):
    """
    Command. Writes SAMPLE/children/ and SAMPLE/result.json for the candidate in SAMPLE/input.json.

    Args:
        sample (str): The $SAMPLE folder; input.json has method_id, folder (relative to the repo root), results_sha256
        root (str): The repo root (".." from data/)
    """
    sample, root = Path(sample), Path(root)
    job = json.loads((sample / "input.json").read_text())
    folder = root / job["folder"]
    round_dir = next(p for p in [folder, *folder.parents] if re.fullmatch(r"r[0-9]+", p.name))  # LC_DIR/r<k>/<name>[/variants/<v>]
    data_dir = round_dir.parent / "data"  # LC_DIR/data
    tiled = job["method_id"].endswith("@tiled")  # the same model scored on the tiled test latents (testtiled split)
    suffix = "_testtiled" if tiled else ""
    per_sample = json.loads((folder / f"results{suffix}.json").read_text())["per_sample"]
    metas, target, pixels = test_split(data_dir, "testtiled" if tiled else "test")
    position = {m["index"]: i for i, m in enumerate(metas)}
    pred = load_file(str(folder / f"pred{suffix}.safetensors"))["pred"]  # (N, 24, 64, 64) float16
    assert pred.shape == target.shape, (pred.shape, target.shape)
    mse = latent_mse(pred, target)
    files = render_files(folder, f"decoded{suffix}")

    def one(k, index):
        """Command. Writes the child of one decoded test sample."""
        i = position[index]
        sample_id = f"test_{index:06d}"
        fields = dict(render_id=f"{job['method_id']}|{sample_id}", method_id=job["method_id"], sample_id=sample_id,
                      latent_mse=float(mse[i]), **{m: per_sample[m].get(str(i)) for m in METRICS})
        true = composite(pixels["fg"][i], pixels["alpha"][i], pixels["bg"][i])
        render_child(sample / "children" / str(k), files[index], true, fields)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, range(len(files)), sorted(files)))
    (sample / "result.json").write_text(json.dumps(dict(renders_sha256=job["results_sha256"], renders=len(files))))
    print(f"{job['method_id']}: {len(files)} renders", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
