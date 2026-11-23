# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "pyarrow==25.0.1"]
# ///
"""
Writes rows.parquet: the dd_samples rows of DiffDec's test set (manifest §26, DD_DIR/CONTRACT.md): the metadata of
DD_DIR/data/test/samples.json and, byte for byte, each sample's source.png, latent.safetensors and flux2.safetensors.
result.json: imported_samples_stamp (the stamp dd_find computed) for dd_jobs. With no stamp (no complete test set yet) it
says so and imports nothing.
"""
import json
from pathlib import Path

import fire
import pyarrow as pa
import pyarrow.parquet as pq

META = ("origin", "side", "pexels_id", "page_id", "source_table")
FILES = dict(source_png="source.png", latent="latent.safetensors", flux2="flux2.safetensors")

SCHEMA = pa.schema(
    [("sample_id", pa.string())] + [(k, pa.string()) for k in META] + [("has_upper_bound", pa.bool_())]
    + [(k, pa.binary()) for k in FILES]
)


def sample_row(meta, test_dir):
    """
    Query (reads the sample's 3 files). The dd_samples row of one test sample.

    Args:
        meta (dict): A samples.json entry: sample_id, origin, side, pexels_id, page_id, source_table, has_upper_bound
        test_dir (Path): DD_DIR/data/test

    Returns:
        dict: SCHEMA fields (the files byte for byte)
    """
    folder = test_dir / meta["sample_id"]
    return dict(sample_id=meta["sample_id"], **{k: optional_str(meta.get(k)) for k in META},
                has_upper_bound=optional_bool(meta.get("has_upper_bound")),
                **{k: (folder / name).read_bytes() for k, name in FILES.items()})


def optional_str(value):
    """
    Pure function. str(value), or None for None (a sample has no pexels_id or page_id when it comes from the other source).

    Examples:
        >>> optional_str(8620887), optional_str(None), optional_str("Hcx82jcUpM")
        ('8620887', None, 'Hcx82jcUpM')
    """
    return None if value is None else str(value)


def optional_bool(value):
    """
    Pure function. bool(value), or None for None.

    Examples:
        >>> optional_bool(1), optional_bool(None), optional_bool(False)
        (True, None, False)
    """
    return None if value is None else bool(value)


def main(sample, root, dd_dir):
    """
    Command. Writes SAMPLE/rows.parquet and SAMPLE/result.json for the job in SAMPLE/input.json.

    Args:
        sample (str): The $SAMPLE folder; input.json has job_id and samples_stamp (None: no complete test set yet)
        root (str): The repo root (".." from data/)
        dd_dir (str): DD_DIR, relative to the repo root
    """
    sample, root = Path(sample), Path(root)
    stamp = json.loads((sample / "input.json").read_text())["samples_stamp"]
    test_dir = root / dd_dir / "data" / "test"
    metas = json.loads((test_dir / "samples.json").read_text()) if stamp else []
    rows = [sample_row(m, test_dir) for m in metas]
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), sample / "rows.parquet")
    (sample / "result.json").write_text(json.dumps(dict(imported_samples_stamp=stamp)))
    print(f"{len(rows)} samples imported" if stamp else "NOTHING TO IMPORT: no complete test set yet (samples_stamp is null)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
