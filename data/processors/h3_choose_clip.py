# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "numpy==2.2.6", "pyarrow==25.0.1"]
# ///
"""
Chooses the Pexels clips the H3 Magic adapters train on (manifest §22). One run gets every cut clip, one Sample each, and
writes each Sample's result.json {split, h3_use, birdseye_score}. It runs again every few minutes while clips are being cut
(data/recipes/loops/h3_choose_loop.fish), so training can start before the cut ends:
- provisional (h3_use "provisional"): train = the best `share` of the cut clips; the rest "spare".
- final (h3_use "final", after the last clip is cut): train = exactly train_count clips.
"Best" = the rank average of motion_512 (measured on the 512 x 512 clip) and birdseye_score (fork PexelsPick's score in the
selection parquet), among the clips that are not val and move at least min_motion. Val stays val unless static; a static val
clip becomes spare and the best clip left over takes its place.
"""
import json

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
    Pure function. The new split of each clip: "train", "val" or "spare".

    Args:
        rows (list[dict]): pexels_id, split ("val" marks the val clips), motion_512, birdseye_score (None when unknown)
        train_count (int): Clips to train on
        val_count (int): Val clips
        min_motion (float): Least motion_512 of a used clip

    Returns:
        dict[str, str]: pexels_id -> split

    Examples:
        >>> rows = [dict(pexels_id=str(i), split=s, motion_512=m, birdseye_score=b) for i, (s, m, b) in
        ...         enumerate([("val", 5, 1), ("val", 0.1, 1), ("train", 9, 9), ("spare", 8, 1), ("train", 1, 2), ("spare", 0.5, 9)])]
        >>> choose(rows, 2, 2, 1.0)  # val 1 is static: the best leftover (clip 4) replaces it
        {'0': 'val', '1': 'spare', '2': 'train', '3': 'train', '4': 'val', '5': 'spare'}
    """
    split = {r["pexels_id"]: "spare" for r in rows}
    moving = [r for r in rows if r["motion_512"] >= min_motion]
    vals = [r for r in moving if r["split"] == "val"][:val_count]
    others = [r for r in moving if r["split"] != "val"]
    scores = [np.array([r["motion_512"] for r in others], float)]
    if others and all(r["birdseye_score"] is not None for r in others):
        scores.append(np.array([r["birdseye_score"] for r in others], float))
    order = [others[i] for i in np.argsort(rank_average(*scores), kind="stable")] if others else []
    assert len(order) >= train_count + val_count - len(vals), f"only {len(order)} moving clips for {train_count} train"
    for r in order[:train_count]:
        split[r["pexels_id"]] = "train"
    for r in vals + order[train_count:train_count + val_count - len(vals)]:
        split[r["pexels_id"]] = "val"
    return split


def main(batch, selection, train_count, val_count, min_motion, share, final):
    """
    Command. Writes result.json {split, h3_use, birdseye_score} into every Sample folder of batch.

    Args:
        batch (str): The $BATCH folder: one Sample per cut clip, input.json {pexels_id, split, motion_512}
        selection (str): H3_PEXELS_SELECTION; its birdseye_score column is used when it has one
        train_count, val_count (int): H3_PEXELS_TRAIN, H3_VAL_COUNT
        min_motion (float): H3_MIN_MOTION
        share (float): H3_PROVISIONAL_SHARE: the train share of the cut clips before the final choice
        final (bool): True after the last clip is cut: exactly train_count train clips
    """
    final = str(final).lower() == "true"
    table = pq.read_table(selection)
    birdseye = dict(zip(table["pexels_id"].to_pylist(), table["birdseye_score"].to_pylist())) if "birdseye_score" in table.column_names else {}
    samples = sample_folders(batch)
    rows = [json.loads((sample / "input.json").read_text()) for sample in samples]
    for row in rows:
        row["birdseye_score"] = birdseye.get(row["pexels_id"])
    count = int(train_count) if final else int(float(share) * sum(r["split"] != "val" for r in rows))
    split = choose(rows, count, int(val_count), float(min_motion))
    for sample, row in zip(samples, rows):
        (sample / "result.json").write_text(json.dumps(dict(split=split[row["pexels_id"]], h3_use="final" if final else "provisional",
                                                            birdseye_score=row["birdseye_score"])))
    counts = {u: list(split.values()).count(u) for u in ("train", "val", "spare")}
    print(f"{'final' if final else 'provisional'}: {len(rows)} cut clips: {counts}; "
          f"birdseye_score {'used' if birdseye else 'not in the selection: motion only'}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
