# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.2.6", "pyarrow==25.0.1"]
# ///
"""
Chooses the Pexels clips the H3 Magic adapters train on (manifest §22). One run gets every cut clip, one Sample each, and
writes each Sample's result.json {h3_use, birdseye_score}:
- "train": the train_count clips with the best rank average of motion_512 (measured on the 512 x 512 clip) and birdseye_score
  (fork PexelsPick's quality score in the selection parquet), among the clips that are not val and move at least min_motion.
- "val": the val clips that move at least min_motion; static val clips are replaced by the best clips left over.
- "unused": the rest.
"""
import json
from pathlib import Path

import fire
import numpy as np
import pyarrow.parquet as pq
from _batch import sample_folders


def rank_average(*scores):
    """
    Pure function. Mean of each item's rank under every score (rank 0 = the highest score), so lower is better.

    Args:
        *scores (np.ndarray): (N,) float each, higher is better

    Returns:
        np.ndarray: (N,) float

    Examples:
        >>> rank_average(np.array([3.0, 1.0, 2.0]), np.array([1.0, 3.0, 2.0])).tolist()  # item 2 is second under both
        [1.0, 1.0, 1.0]
        >>> rank_average(np.array([3.0, 1.0, 2.0])).tolist()
        [0.0, 2.0, 1.0]
    """
    return np.mean([np.argsort(np.argsort(-s, kind="stable"), kind="stable") for s in scores], axis=0)


def choose(rows, train_count, val_count, min_motion):
    """
    Pure function. h3_use of each clip: "train", "val" or "unused".

    Args:
        rows (list[dict]): pexels_id, split, motion_512, birdseye_score (None when the selection has none)
        train_count (int): Clips to train on
        val_count (int): Val clips
        min_motion (float): Least motion_512 of a used clip

    Returns:
        dict[str, str]: pexels_id -> h3_use

    Examples:
        >>> rows = [dict(pexels_id=str(i), split=s, motion_512=m, birdseye_score=b) for i, (s, m, b) in
        ...         enumerate([("val", 5, 1), ("val", 0.1, 1), ("train", 9, 9), ("spare", 8, 1), ("train", 1, 2), ("spare", 0.5, 9)])]
        >>> choose(rows, 2, 2, 1.0)  # val 1 is static: the best leftover (clip 4) replaces it
        {'0': 'val', '1': 'unused', '2': 'train', '3': 'train', '4': 'val', '5': 'unused'}
    """
    use = {r["pexels_id"]: "unused" for r in rows}
    moving = [r for r in rows if r["motion_512"] >= min_motion]
    vals = [r for r in moving if r["split"] == "val"][:val_count]
    others = [r for r in moving if r["split"] != "val"]
    scores = [np.array([r["motion_512"] for r in others], float)]
    if all(r["birdseye_score"] is not None for r in others):
        scores.append(np.array([r["birdseye_score"] for r in others], float))
    order = [others[i] for i in np.argsort(rank_average(*scores), kind="stable")] if others else []
    assert len(order) >= train_count + val_count - len(vals), f"only {len(order)} moving clips for {train_count} train"
    for r in order[:train_count]:
        use[r["pexels_id"]] = "train"
    for r in vals + order[train_count:train_count + val_count - len(vals)]:
        use[r["pexels_id"]] = "val"
    return use


def main(batch, selection, train_count, val_count, min_motion):
    """
    Command. Writes result.json {h3_use, birdseye_score} into every Sample folder of batch.

    Args:
        batch (str): The $BATCH folder: one Sample per cut clip, input.json {pexels_id, split, motion_512}
        selection (str): H3_PEXELS_SELECTION; its birdseye_score column is used when it has one
        train_count, val_count (int): H3_PEXELS_TRAIN, H3_VAL_COUNT
        min_motion (float): H3_MIN_MOTION
    """
    table = pq.read_table(selection)
    birdseye = dict(zip(table["pexels_id"].to_pylist(), table["birdseye_score"].to_pylist())) if "birdseye_score" in table.column_names else {}
    samples = sample_folders(batch)
    rows = [json.loads((sample / "input.json").read_text()) for sample in samples]
    for row in rows:
        row["birdseye_score"] = birdseye.get(row["pexels_id"])
    use = choose(rows, int(train_count), int(val_count), float(min_motion))
    for sample, row in zip(samples, rows):
        (sample / "result.json").write_text(json.dumps(dict(h3_use=use[row["pexels_id"]], birdseye_score=row["birdseye_score"])))
    counts = {u: list(use.values()).count(u) for u in ("train", "val", "unused")}
    print(f"{len(rows)} clips: {counts}; birdseye_score {'used' if birdseye else 'not in the selection yet: motion only'}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
