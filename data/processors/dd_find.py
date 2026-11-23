# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "pyarrow==25.0.1"]
# ///
"""
Finds DiffDec's files (manifest §26, DD_DIR/CONTRACT.md). Reads only small files, never pixels. Writes:
    methods.parquet  one dd_methods row per candidate folder DD_DIR/g<round>/<name>/ with a genome.json: the genome, and
                     (when results.json exists) its means and cost. A folder without results.json is still training: status 'training'
    result.json      samples_stamp for dd_jobs: the sha256 of DD_DIR/data/test/samples.json when every test sample is complete, else null
dd_import_samples and dd_import_renders import when the stamp or results_sha256 changed.
"""
import hashlib
import json
from pathlib import Path

import fire
import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_FILES = ("source.png", "latent.safetensors", "flux2.safetensors")  # per test sample, DD_DIR/data/test/<sample_id>/
GENOME_STR = ("agent", "kind", "paper", "paper_url", "idea", "why_win", "ablation_of", "ablation_change", "space", "conditioning",
              "how_to_load", "verdict", "why_held_or_failed")
GENOME_LISTS = ("parents",)  # stored as JSON text
MEANS = ("psnr", "ssim", "lpips_alex", "lpips_vgg", "dists", "de2000", "flux_mse", "sharp_grad", "sharp_fft", "psnr_mean8", "dists_mean8")

METHOD_SCHEMA = pa.schema(
    [("method_id", pa.string()), ("round", pa.int64()), ("name", pa.string()), ("status", pa.string())]
    + [(k, pa.string()) for k in GENOME_STR[:1]] + [(k, pa.string()) for k in GENOME_LISTS]
    + [(k, pa.string()) for k in GENOME_STR[1:]]
    + [("text_cond", pa.bool_()), ("steps", pa.int64()), ("params", pa.int64()), ("train_gpu_hours", pa.float64()),
       ("train_gpu_type", pa.string())]
    + [(k, pa.float64()) for k in MEANS]
    + [("ms_per_image", pa.float64()), ("gpu_type", pa.string()), ("eval_steps", pa.int64()), ("n_samples", pa.int64()),
       ("latent_kind", pa.string()), ("folder", pa.string()), ("results_sha256", pa.string())]
)


def optional(cast, value):
    """
    Pure function. cast(value), or None when the value is None (a genome field the child has not filled yet).

    Examples:
        >>> optional(int, 12.0), optional(int, None), optional(float, "0.5")
        (12, None, 0.5)
    """
    return None if value is None else cast(value)


def method_row(method_id, folder, genome, results, results_sha256):
    """
    Pure function. The dd_methods row of one candidate folder.

    Args:
        method_id (str): "g<round>/<folder name>"
        folder (str): The candidate folder, relative to the repo root
        genome (dict): genome.json
        results (dict | None): results.json, None while the candidate trains
        results_sha256 (str | None): sha256 (hex) of results.json

    Returns:
        dict: METHOD_SCHEMA fields

    Examples:
        >>> row = method_row("g0/h3_decoder", "agents/.scratchpad/diffdec/g0/h3_decoder", {"parents": [], "steps": 1, "text_cond": False},
        ...                  {"means": {"psnr": 30.5, "dists": 0.1}, "ms_per_image": 20.0, "gpu_type": "A100", "steps": 1}, "ab12")
        >>> row["round"], row["name"], row["status"], row["parents"], row["psnr"], row["lpips_vgg"], row["gpu_type"]
        (0, 'h3_decoder', 'scored', '[]', 30.5, None, 'A100')
        >>> row = method_row("g1/x", "f", {"name": "x"}, None, None)
        >>> row["status"], row["psnr"], row["parents"]
        ('training', None, None)
    """
    results = results or {}
    means = results.get("means", {})
    return dict(
        method_id=method_id,
        round=int(method_id.split("/")[0][1:]),
        name=genome.get("name", method_id.split("/")[1]),
        status="scored" if results else "training",
        **{k: optional(str, genome.get(k)) for k in GENOME_STR},
        **{k: optional(json.dumps, genome.get(k)) for k in GENOME_LISTS},
        text_cond=optional(bool, genome.get("text_cond")),
        steps=optional(int, genome.get("steps")),
        params=optional(int, genome.get("params")),
        train_gpu_hours=optional(float, genome.get("train_gpu_hours")),
        train_gpu_type=genome.get("gpu_type"),
        **{k: optional(float, means.get(k)) for k in MEANS},
        ms_per_image=optional(float, results.get("ms_per_image")),
        gpu_type=results.get("gpu_type"),
        eval_steps=optional(int, results.get("steps")),
        n_samples=optional(int, results.get("n_samples")),
        latent_kind=results.get("latent_kind"),
        folder=folder,
        results_sha256=results_sha256,
    )


def method_rows(root, dd_dir):
    """
    Query (reads the candidate folders). One dd_methods row per DD_DIR/g<round>/<name>/genome.json, sorted.

    Args:
        root (Path): The repo root
        dd_dir (str): DD_DIR, relative to the repo root

    Returns:
        list[dict]
    """
    rows = []
    for genome_path in sorted((root / dd_dir).glob("g[0-9]*/*/genome.json")):
        folder = genome_path.parent
        results_path = folder / "results.json"
        data = results_path.read_bytes() if results_path.exists() else None
        rows.append(method_row(f"{folder.parent.name}/{folder.name}", f"{dd_dir}/{folder.parent.name}/{folder.name}",
                               json.loads(genome_path.read_text()), json.loads(data) if data else None,
                               hashlib.sha256(data).hexdigest() if data else None))
    return rows


def samples_stamp(test_dir):
    """
    Query (reads samples.json and lists the sample folders). sha256 (hex) of samples.json when every listed sample has its 3 files,
    else None. Says loudly why when it is None.
    """
    path = test_dir / "samples.json"
    if not path.exists():
        print(f"NOTHING TO IMPORT: {path} does not exist yet", flush=True)
        return None
    data = path.read_bytes()
    missing = [f"{s['sample_id']}/{f}" for s in json.loads(data) for f in SAMPLE_FILES if not (test_dir / s["sample_id"] / f).exists()]
    if missing:
        print(f"NOT IMPORTING SAMPLES YET: {len(missing)} files missing, first {missing[:3]}", flush=True)
        return None
    return hashlib.sha256(data).hexdigest()


def main(sample, root, dd_dir):
    """
    Command. Writes SAMPLE/methods.parquet and SAMPLE/result.json.

    Args:
        sample (str): The $SAMPLE folder
        root (str): The repo root (".." from data/)
        dd_dir (str): DD_DIR, relative to the repo root
    """
    sample, root = Path(sample), Path(root)
    methods = method_rows(root, dd_dir)
    stamp = samples_stamp(root / dd_dir / "data" / "test")
    pq.write_table(pa.Table.from_pylist(methods, METHOD_SCHEMA), sample / "methods.parquet")
    (sample / "result.json").write_text(json.dumps(dict(samples_stamp=stamp)))
    print(f"{len(methods)} candidates ({sum(m['status'] == 'scored' for m in methods)} scored), samples_stamp {stamp}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
