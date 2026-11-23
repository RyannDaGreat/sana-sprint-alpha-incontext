# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "pyarrow==25.0.1"]
# ///
"""
Finds the Latent Compositor's files (manifest §24) and writes two output tables. Reads only small files, never pixels:
    shards.parquet   one lc_shards row per dataset shard LC_DIR/data/<split>_<k>.safetensors, with a stamp of its 3 files
    methods.parquet  one lc_methods row per scored candidate LC_DIR/r<k>/<name>/ (a folder with results.json):
                     its genome.json card, the summary metrics of results.json, the cost (cost_clean.json when present)
lc_import_shard and lc_import_renders then import a shard or a candidate when its stamp or results_sha256 changed.
"""
import hashlib
import json
import re
from pathlib import Path

import fire
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_NAME = re.compile(r"^(train|test)_(\d{3})\.safetensors$")  # the latent shards only: not test_cache*, not *_pixels
TRACKS = {"L": (10.0, 5e6), "H": (100.0, 5e7)}  # ms_fwdbwd and params limits (LC_DIR/FITNESS.md), as LC_DIR/lineage.py
SUBSETS = ("hard_soft", "hard_contrast", "random")

SHARD_SCHEMA = pa.schema([("shard", pa.string()), ("split", pa.string()), ("files_stamp", pa.string())])
METHOD_SCHEMA = pa.schema([
    ("method_id", pa.string()),
    ("round", pa.int64()),
    ("name", pa.string()),
    ("parents", pa.string()),
    ("what", pa.string()),
    ("how", pa.string()),
    ("cites", pa.string()),
    ("fg_key", pa.string()),
    ("track", pa.string()),
    ("wildcard", pa.bool_()),
    ("params", pa.int64()),
    ("ms_fwd", pa.float64()),
    ("ms_fwdbwd", pa.float64()),
    ("psnr", pa.float64()),
    ("psnr_hard_soft", pa.float64()),
    ("psnr_hard_contrast", pa.float64()),
    ("psnr_random", pa.float64()),
    ("ssim", pa.float64()),
    ("lpips", pa.float64()),
    ("flux2_mse", pa.float64()),
    ("latent_mse", pa.float64()),
    ("oracle_psnr", pa.float64()),
    ("verdict", pa.string()),
    ("folder", pa.string()),
    ("results_sha256", pa.string()),
])


def files_stamp(paths):
    """
    Query (reads file sizes and times). size:mtime_ns of each file, joined by commas: it changes when a file is rewritten.

    Args:
        paths (list[Path]): Files that exist

    Returns:
        str

    Examples:
        >>> # files_stamp([Path("data/test_000.safetensors"), ...]) -> "117964960:1763899860000000000,734003456:...,45084:..."
    """
    return ",".join(f"{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths)


def shard_files(latents_path):
    """
    Pure function. The 3 files of one shard: latents, pixels, metadata.

    Examples:
        >>> [p.name for p in shard_files(Path("data/train_003.safetensors"))]
        ['train_003.safetensors', 'train_003_pixels.safetensors', 'train_003.json']
    """
    return [latents_path, latents_path.with_name(latents_path.stem + "_pixels.safetensors"), latents_path.with_suffix(".json")]


def shard_rows(data_dir):
    """
    Query (lists data_dir). One row per complete shard (all 3 files exist), sorted.

    Returns:
        list[dict]: shard, split, files_stamp
    """
    rows = []
    for path in sorted(data_dir.iterdir()):
        match = SHARD_NAME.match(path.name)
        if match and all(p.exists() for p in shard_files(path)):
            rows.append(dict(shard=path.stem, split=match.group(1), files_stamp=files_stamp(shard_files(path))))
    return rows


def track_of(params, ms_fwdbwd):
    """
    Pure function. The tightest track a candidate's cost fits ('L', 'H' or 'none'); 'ref' when it has no cost (a reference
    that is not a module, e.g. decode -> blend -> encode). Same rule as LC_DIR/lineage.py.

    Examples:
        >>> track_of(1e6, 4.0), track_of(1e6, 40.0), track_of(9e7, 40.0), track_of(None, None)
        ('L', 'H', 'none', 'ref')
    """
    if ms_fwdbwd is None:
        return "ref"
    for name, (ms, most_params) in TRACKS.items():
        if ms_fwdbwd <= ms and params <= most_params:
            return name
    return "none"


def json_list(value):
    """
    Pure function. A genome list field as JSON text, or None when the card does not have it.

    Examples:
        >>> json_list(["r0/lsq_gray_k3", "r1/agent_2"]), json_list(None)
        ('["r0/lsq_gray_k3", "r1/agent_2"]', None)
    """
    return None if value is None else json.dumps(value)


def method_row(method_id, folder, genome, results, clean, results_sha256):
    """
    Pure function. The lc_methods row of one candidate.

    Args:
        method_id (str): "r<k>/<folder name>"
        folder (str): The candidate folder, relative to the repo root
        genome (dict): genome.json ({} when missing)
        results (dict): results.json of LC_DIR/evaluate.py
        clean (dict): cost_clean.json ({} when missing): a re-timing in a clean window, used before results.json's cost
        results_sha256 (str): sha256 (hex) of results.json

    Returns:
        dict: METHOD_SCHEMA fields

    Examples:
        >>> res = {"fg_key": "fg_gray", "params": 1200, "ms_fwd": 0.5, "ms_fwdbwd": 1.5, "latent_mse": {"all": 0.03},
        ...        "decoded": {"psnr": {"all": 30.1, "hard_soft": 28.0, "hard_contrast": 29.0, "random": 33.0},
        ...                    "ssim": {"all": 0.9}, "lpips": {"all": 0.05}, "flux2_mse": {"all": 0.01}},
        ...        "oracle": {"psnr": {"all": 32.5}}}
        >>> row = method_row("r0/lsq_gray_k3", "agents/.scratchpad/latentcomp/r0/lsq_gray_k3", {"name": "lsq_gray_k3", "parents": []},
        ...                  res, {"ms_fwdbwd": 1.2}, "ab12")
        >>> row["round"], row["track"], row["ms_fwdbwd"], row["psnr_hard_soft"], row["parents"], row["cites"]
        (0, 'L', 1.2, 28.0, '[]', None)
    """
    decoded = results["decoded"]
    cost = {key: clean.get(key, results.get(key)) for key in ("params", "ms_fwd", "ms_fwdbwd")}
    return dict(
        method_id=method_id,
        round=int(method_id.split("/")[0][1:]),
        name=genome.get("name", method_id.split("/")[1]),
        parents=json_list(genome.get("parents")),
        what=genome.get("what"),
        how=genome.get("how"),
        cites=json_list(genome.get("cites")),
        fg_key=results.get("fg_key", genome.get("fg_key")),
        track=track_of(cost["params"], cost["ms_fwdbwd"]),
        wildcard=genome.get("wildcard"),
        **cost,
        psnr=decoded["psnr"]["all"],
        **{f"psnr_{s}": decoded["psnr"].get(s) for s in SUBSETS},
        ssim=decoded["ssim"]["all"],
        lpips=decoded["lpips"]["all"],
        flux2_mse=decoded["flux2_mse"]["all"],
        latent_mse=results["latent_mse"]["all"],
        oracle_psnr=results["oracle"]["psnr"]["all"],
        verdict=genome.get("verdict"),
        folder=folder,
        results_sha256=results_sha256,
    )


def read_json_if_exists(path):
    """Query. The JSON in path, or {} when the file does not exist (genome.json and cost_clean.json are optional)."""
    return json.loads(path.read_text()) if path.exists() else {}


def method_rows(root, lc_dir):
    """
    Query (reads the candidate folders). One lc_methods row per LC_DIR/r<k>/**/results.json with a pred.safetensors
    beside it (method_id = the folder under LC_DIR, e.g. "r3/agent_6/control"), and one per results_testtiled.json with
    pred_testtiled.safetensors (method_id + "@tiled": the same model on the tiled test latents); others are reported and skipped.

    Args:
        root (Path): The repo root
        lc_dir (str): LC_DIR, relative to the repo root

    Returns:
        list[dict]
    """
    rows = []
    kinds = (("results.json", "pred.safetensors", ""), ("results_testtiled.json", "pred_testtiled.safetensors", "@tiled"))  # untiled; tiled split
    for results_name, pred_name, tag in kinds:  # children's variants/ and control/ count too
        for results_path in sorted((root / lc_dir).glob(f"r[0-9]*/**/{results_name}")):
            folder = results_path.parent
            if not (folder / pred_name).exists():  # results copied without their predictions cannot be rendered
                print(f"lc_find: skipping {folder}/{results_name} (no {pred_name})", flush=True)
                continue
            data = results_path.read_bytes()
            rel = folder.relative_to(root / lc_dir).as_posix()
            rows.append(method_row(rel + tag, f"{lc_dir}/{rel}", read_json_if_exists(folder / "genome.json"), json.loads(data),
                                   read_json_if_exists(folder / "cost_clean.json"), hashlib.sha256(data).hexdigest()))
    return rows


def main(sample, root, lc_dir):
    """
    Command. Writes SAMPLE/shards.parquet and SAMPLE/methods.parquet.

    Args:
        sample (str): The $SAMPLE folder
        root (str): The repo root (".." from data/)
        lc_dir (str): LC_DIR, relative to the repo root
    """
    sample, root = Path(sample), Path(root)
    shards = shard_rows(root / lc_dir / "data")
    methods = method_rows(root, lc_dir)
    pq.write_table(pa.Table.from_pylist(shards, SHARD_SCHEMA), sample / "shards.parquet")
    pq.write_table(pa.Table.from_pylist(methods, METHOD_SCHEMA), sample / "methods.parquet")
    print(f"{len(shards)} shards ({sum(s['split'] == 'test' for s in shards)} test), {len(methods)} scored candidates", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
