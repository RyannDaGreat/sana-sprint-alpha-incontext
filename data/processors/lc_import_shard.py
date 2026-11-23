# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.3.4", "pillow==12.0.0", "pyarrow==25.0.1", "safetensors==0.8.0"]
# ///
"""
Writes rows.parquet: the lc_samples rows of one Latent Compositor dataset shard (manifest §24), and result.json
{imported_stamp, samples}. Every row gets its metadata and its 6 latents. Test rows also get lossless 1024 x 1024 PNGs:
the matte, the straight RGBA foreground, the background, the true composite (recomputed exactly as LC_DIR/lc.py) and the
oracle (the H3 decode of the true composite's latent, from LC_DIR/data/test_cache*.safetensors, when it exists).
Train rows keep their pixels in the shard.
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
from safetensors import safe_open
from safetensors.numpy import load_file, save

KEYS = ["alpha", "fg_straight", "fg_premult", "fg_gray", "bg", "comp"]  # order of the 6 latents (LC_DIR/lc.py KEYS)
WORKERS = 16  # Threads that encode PNGs
PNG_COMPRESS_LEVEL = 3  # zlib level: about as small as the default 6, and faster
PNG_COLUMNS = ("alpha_png", "fg_rgba_png", "bg_png", "comp_png", "oracle_png")

SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("split", pa.string()),
    ("subset", pa.string()),
    ("page_id", pa.string()),
    ("pexels_id", pa.string()),
    ("tags", pa.string()),
    ("placement", pa.string()),
    ("background", pa.string()),
    ("shard", pa.string()),
    ("position", pa.int64()),
    ("rating", pa.int64()),
    ("alpha_mean", pa.float64()),
    ("latents", pa.binary()),
    *[(name, pa.binary()) for name in PNG_COLUMNS],
])


def sample_id(split, index):
    """
    Pure function. The lc_samples key of a sample: its split and its index in LC_DIR's seeded shuffle, 6 digits or more.

    Examples:
        >>> sample_id("train", 300), sample_id("test", 1000000)
        ('train_000300', 'test_1000000')
    """
    return f"{split}_{index:06d}"


def composite(fg, alpha, bg):
    """
    Pure function. alpha * fg + (1 - alpha) * bg in 8 bit, rounded: exactly LC_DIR/lc.py composite.

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


def png_bytes(image):
    """
    Pure function. A lossless PNG of an image: (H, W) gray, (H, W, 3) RGB or (H, W, 4) RGBA, uint8.

    Examples:
        >>> png = png_bytes(np.zeros((2, 2, 3), np.uint8))
        >>> png[:4], np.asarray(Image.open(io.BytesIO(png))).shape
        (b'\\x89PNG', (2, 2, 3))
    """
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return out.getvalue()


def latents_bytes(latents):
    """
    Pure function. safetensors bytes of one sample's latents {"latents": (6, 24, 64, 64) float16}, in KEYS order (also
    stored as the metadata "keys").

    Examples:
        >>> from safetensors.numpy import load
        >>> data = latents_bytes(np.zeros((6, 24, 64, 64), np.float16))
        >>> load(data)["latents"].shape
        (6, 24, 64, 64)
    """
    assert latents.shape[0] == len(KEYS) and latents.dtype == np.float16, (latents.shape, latents.dtype)
    return save({"latents": np.ascontiguousarray(latents)}, metadata={"keys": ",".join(KEYS)})


def shard_offset(data_dir, shard):
    """
    Query (reads the .json of the split's earlier shards). The position of the shard's first sample in its split.

    Examples:
        >>> # shard_offset(Path("data"), "test_002") -> 200 (test_000 and test_001 hold 100 samples each)
    """
    split, k = shard.rsplit("_", 1)
    return sum(len(json.loads((data_dir / f"{split}_{j:03d}.json").read_text())) for j in range(int(k)))


def oracle_images(data_dir):
    """
    Query. The oracle decode of every test position that LC_DIR/evaluate.py has made, as {position: (1024, 1024, 3) uint8}:
    from data/test_cache.safetensors (its key oracle, all positions in order) when it exists, else from its inputs
    data/test_cache_part_<w>.safetensors (keys oracle and idx). {} when neither exists yet.
    """
    cache = data_dir / "test_cache.safetensors"
    if cache.exists():
        with safe_open(str(cache), framework="numpy") as f:
            return dict(enumerate(f.get_tensor("oracle")))
    out = {}
    for part in sorted(data_dir.glob("test_cache_part_*.safetensors")):
        tensors = load_file(str(part))
        out.update(zip(tensors["idx"].tolist(), tensors["oracle"]))
    return out


def test_images(pixels, i, oracle):
    """
    Pure function. The PNG columns of one test sample.

    Args:
        pixels (dict): The shard's pixels: alpha (N, H, W), fg (N, H, W, 3), bg (N, H, W, 3), uint8
        i (int): Row in the shard
        oracle (np.ndarray | None): (H, W, 3) uint8, or None when evaluate.py has not made it

    Returns:
        dict[str, bytes | None]: PNG_COLUMNS

    Examples:
        >>> px = {"alpha": np.full((1, 2, 2), 255, np.uint8), "fg": np.full((1, 2, 2, 3), 9, np.uint8), "bg": np.zeros((1, 2, 2, 3), np.uint8)}
        >>> out = test_images(px, 0, None)
        >>> np.asarray(Image.open(io.BytesIO(out["comp_png"]))).tolist()[0][0], out["oracle_png"]
        ([9, 9, 9], None)
    """
    alpha, fg, bg = pixels["alpha"][i], pixels["fg"][i], pixels["bg"][i]
    return dict(alpha_png=png_bytes(alpha), fg_rgba_png=png_bytes(np.concatenate([fg, alpha[..., None]], -1)),
                bg_png=png_bytes(bg), comp_png=png_bytes(composite(fg, alpha, bg)),
                oracle_png=None if oracle is None else png_bytes(oracle))


def sample_row(meta, split, shard, position, plan, latents_blob, images):
    """
    Pure function. The lc_samples row of one sample.

    Args:
        meta (dict): Its entry of <shard>.json (index, page_id, pexels_id, placement, background, alpha_mean, subset)
        split (str): "train" or "test"
        shard (str): e.g. "test_000"
        position (int): Its place in the split (results.json's per_sample key for a test sample)
        plan (dict | None): Its test_plan.json entry (rating, tags), None for train
        latents_blob (bytes): latents_bytes of its 6 latents
        images (dict): PNG_COLUMNS, all None for train

    Returns:
        dict: SCHEMA fields

    Examples:
        >>> m = {"index": 300, "page_id": "L8zRWJRtjL", "pexels_id": "7671926", "placement": {"scale": 0.95}, "background": {"t": 6.5},
        ...      "alpha_mean": 0.37, "subset": "train"}
        >>> row = sample_row(m, "train", "train_000", 0, None, b"", dict.fromkeys(PNG_COLUMNS))
        >>> row["sample_id"], row["placement"], row["rating"], row["comp_png"]
        ('train_000300', '{"scale": 0.95}', None, None)
    """
    return dict(sample_id=sample_id(split, meta["index"]), split=split, subset=meta["subset"], page_id=meta["page_id"],
                pexels_id=meta["pexels_id"], tags=None if plan is None else json.dumps(plan["tags"]),
                placement=json.dumps(meta["placement"]), background=json.dumps(meta["background"]), shard=shard,
                position=position, rating=None if plan is None else plan["rating"], alpha_mean=meta["alpha_mean"],
                latents=latents_blob, **images)


def main(sample, lc_dir):
    """
    Command. Writes SAMPLE/rows.parquet and SAMPLE/result.json for the shard in SAMPLE/input.json.

    Args:
        sample (str): The $SAMPLE folder; input.json has shard, split, files_stamp
        lc_dir (str): LC_DIR, as a path from the working folder (data/)
    """
    sample, data_dir = Path(sample), Path(lc_dir) / "data"
    job = json.loads((sample / "input.json").read_text())
    shard, split = job["shard"], job["split"]
    metas = json.loads((data_dir / f"{shard}.json").read_text())
    latents = load_file(str(data_dir / f"{shard}.safetensors"))["latents"]  # (N, 6, 24, 64, 64) float16
    assert len(latents) == len(metas), (shard, len(latents), len(metas))
    offset = shard_offset(data_dir, shard)
    images = [dict.fromkeys(PNG_COLUMNS)] * len(metas)
    plans = [None] * len(metas)
    if split == "test":
        pixels = load_file(str(data_dir / f"{shard}_pixels.safetensors"))
        oracle = oracle_images(data_dir)
        by_index = {p["index"]: p for p in json.loads((data_dir / "test_plan.json").read_text())}
        plans = [by_index[m["index"]] for m in metas]
        with ThreadPoolExecutor(WORKERS) as pool:
            images = list(pool.map(lambda i: test_images(pixels, i, oracle.get(offset + i)), range(len(metas))))
    rows = [sample_row(m, split, shard, offset + i, plans[i], latents_bytes(latents[i]), images[i]) for i, m in enumerate(metas)]
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), sample / "rows.parquet")
    (sample / "result.json").write_text(json.dumps(dict(imported_stamp=job["files_stamp"], samples=len(rows))))
    print(f"{shard}: {len(rows)} samples (positions {offset}..{offset + len(rows) - 1}), "
          f"{sum(r['oracle_png'] is not None for r in rows)} with an oracle", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
