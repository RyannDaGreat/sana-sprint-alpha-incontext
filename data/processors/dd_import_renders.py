# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "pyarrow==25.0.1"]
# ///
"""
Writes rows.parquet: the dd_renders rows of one scored DiffDec candidate (manifest §26, DD_DIR/CONTRACT.md): per test sample,
renders/<sample_id>.png and renders_mean8/<sample_id>.png byte for byte, an error map (|render - source| x ERROR_GAIN per channel)
and the per-sample scores of results.json. result.json: renders_sha256 and renders for dd_methods.
"""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ERROR_GAIN = 4  # An error of 8 levels shows as 32: small errors become visible
WORKERS = 16  # Threads that read and encode PNGs
PNG_COMPRESS_LEVEL = 3  # zlib level: about as small as the default 6, and faster
SCORES = ("psnr", "ssim", "lpips_alex", "lpips_vgg", "dists", "de2000", "flux_mse", "sharp_grad", "sharp_fft", "psnr_mean8", "dists_mean8")

SCHEMA = pa.schema(
    [("render_id", pa.string()), ("method_id", pa.string()), ("sample_id", pa.string())]
    + [(k, pa.binary()) for k in ("render_png", "mean8_png", "error_png")]
    + [(k, pa.float64()) for k in SCORES]
)


def error_map(render, source, gain=ERROR_GAIN):
    """
    Pure function. |render - source| x gain per channel, clipped to 255.

    Args:
        render, source (np.ndarray): (H, W, 3) uint8 RGB

    Returns:
        np.ndarray: (H, W, 3) uint8

    Examples:
        >>> error_map(np.array([[[10, 200, 0]]], np.uint8), np.array([[[13, 100, 0]]], np.uint8)).tolist()
        [[[12, 255, 0]]]
    """
    return np.minimum(np.abs(render.astype(np.int16) - source.astype(np.int16)) * gain, 255).astype(np.uint8)


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


def decode_rgb(data):
    """
    Pure function. (H, W, 3) uint8 RGB of PNG bytes.

    Examples:
        >>> decode_rgb(png_bytes(np.full((2, 3, 3), 7, np.uint8))).shape
        (2, 3, 3)
    """
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def render_row(method_id, sample_id, folder, source, scores):
    """
    Query (reads the candidate's PNGs for the sample). The dd_renders row of one sample.

    Args:
        method_id (str): The candidate
        sample_id (str): The test sample
        folder (Path): The candidate folder
        source (bytes): The sample's source.png
        scores (dict): results.json per_sample[sample_id]

    Returns:
        dict: SCHEMA fields (mean8_png is None when renders_mean8/ has no PNG for the sample)
    """
    render = (folder / "renders" / f"{sample_id}.png").read_bytes()
    mean8_path = folder / "renders_mean8" / f"{sample_id}.png"
    render_rgb, source_rgb = decode_rgb(render), decode_rgb(source)
    assert render_rgb.shape == source_rgb.shape, (method_id, sample_id, render_rgb.shape, source_rgb.shape)
    return dict(render_id=f"{method_id}|{sample_id}", method_id=method_id, sample_id=sample_id, render_png=render,
                mean8_png=mean8_path.read_bytes() if mean8_path.exists() else None,
                error_png=png_bytes(error_map(render_rgb, source_rgb)),
                **{k: scores.get(k) for k in SCORES})


def main(sample, root):
    """
    Command. Writes SAMPLE/rows.parquet and SAMPLE/result.json for the candidate in SAMPLE/input.json.

    Args:
        sample (str): The $SAMPLE folder; input.json has method_id, folder (relative to the repo root), results_sha256
        root (str): The repo root (".." from data/)
    """
    sample, root = Path(sample), Path(root)
    job = json.loads((sample / "input.json").read_text())
    folder = root / job["folder"]
    test_dir = folder.parents[1] / "data" / "test"  # DD_DIR/data/test: a candidate is DD_DIR/g<round>/<name>
    per_sample = json.loads((folder / "results.json").read_text())["per_sample"]

    def one(sample_id):
        """Query. The row of one sample."""
        return render_row(job["method_id"], sample_id, folder, (test_dir / sample_id / "source.png").read_bytes(), per_sample[sample_id])

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, sorted(per_sample)))
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), sample / "rows.parquet")
    (sample / "result.json").write_text(json.dumps(dict(renders_sha256=job["results_sha256"], renders=len(rows))))
    print(f"{job['method_id']}: {len(rows)} renders", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
