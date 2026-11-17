# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "huggingface_hub==0.36.0", "pyarrow==25.0.1"]
# ///
"""Downloads magick_index.tsv into the Sample folder and writes rows.parquet: one row per unique MAGICK image."""

import csv
import hashlib
from pathlib import Path

import fire
import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq

INDEX_FILE = "magick_index.tsv"
SCHEMA = pa.schema([
    ("page_id", pa.string()),
    ("subject", pa.string()),
    ("color", pa.string()),
    ("sim_score", pa.float64()),
    ("nsfw", pa.bool_()),
    ("hand_picked", pa.bool_()),
    ("split", pa.string()),
])


def eval_split(page_id, modulus):
    """
    Pure function. Returns "eval" when sha256(page_id) mod modulus is 0, else "train".

    Args:
        page_id (str): MAGICK image ID
        modulus (int): About 1 in modulus images are eval

    Returns:
        str

    Examples:
        >>> eval_split("Hcx82jcUpM", 4000)
        'train'
        >>> eval_split("Hcx82jcUpM", 1)
        'eval'
    """
    return "eval" if int(hashlib.sha256(page_id.encode()).hexdigest(), 16) % modulus == 0 else "train"


def unique_rows(index_rows, modulus):
    """
    Pure function. Turns magick_index.tsv rows into one row per page_id, sorted by page_id.
    A repeated image is listed once as "hand" and once as "auto" with the other fields equal; it gets hand_picked true.

    Args:
        index_rows (Iterable[dict]): TSV rows with page_id, sim_score, color, nsfw, subject, picked
        modulus (int): EVAL_MODULUS

    Returns:
        list[dict]: Rows with the SCHEMA fields

    Examples:
        >>> index = [dict(page_id="b", sim_score="0.9", color="green", nsfw="0", subject="a cat", picked="auto"),
        ...          dict(page_id="a", sim_score="0.8", color="blue", nsfw="1", subject="a dog", picked="auto"),
        ...          dict(page_id="b", sim_score="0.9", color="green", nsfw="0", subject="a cat", picked="hand")]
        >>> [(r["page_id"], r["hand_picked"], r["nsfw"], r["split"]) for r in unique_rows(index, 1)]
        [('a', False, True, 'eval'), ('b', True, False, 'eval')]
    """
    rows = {}
    for row in index_rows:
        page_id = row["page_id"]
        hand = {"hand": True, "auto": False}[row["picked"]]
        if page_id in rows:
            rows[page_id]["hand_picked"] |= hand
            continue
        rows[page_id] = dict(
            page_id=page_id,
            subject=row["subject"],
            color=row["color"],
            sim_score=float(row["sim_score"]),
            nsfw={"0": False, "1": True}[row["nsfw"]],
            hand_picked=hand,
            split=eval_split(page_id, modulus),
        )
    return [rows[page_id] for page_id in sorted(rows)]


def main(sample, repo, modulus):
    """
    Command. Downloads magick_index.tsv to SAMPLE/index.tsv and writes SAMPLE/rows.parquet.

    Args:
        sample (str): The $SAMPLE folder
        repo (str): Hugging Face dataset, MAGICK_REPO
        modulus (int): EVAL_MODULUS
    """
    sample = Path(sample)
    downloaded = huggingface_hub.hf_hub_download(repo, INDEX_FILE, repo_type="dataset", local_dir=sample / "download")
    index = Path(downloaded).replace(sample / "index.tsv")
    with index.open(newline="") as file:  # QUOTE_NONE: subjects hold bare quote characters
        rows = unique_rows(csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE), int(modulus))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), sample / "rows.parquet")
    print(f"{len(rows)} unique images, {sum(r['split'] == 'eval' for r in rows)} of them eval", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
