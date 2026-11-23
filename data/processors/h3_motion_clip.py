"""
Synthetic moving cut-out clips of Alpha Video v2 (manifest §23 item 4), a long-lived per-GPU worker committed with the
database's commit (the h3_encode pattern). For each chosen MAGICK row (the first COUNT train rows by page_id, plus every
val row): a random smooth path (h3magic/synth_motion.py, seeded by (PATH_SEED, page_id)), 141 frames at 768² on 0.5 gray,
white and black, H3-VAE-encoded as video latents (h3magic/h3.py encode). One h3_motion_clips row per page_id:
    h3_motion_gray_768, h3_motion_white_768, h3_motion_black_768   {"latent": (24, 42, 48, 48) bfloat16} (LC, LT, LH, LW), untiled encode
    h3_motion_gray_768t, h3_motion_white_768t, h3_motion_black_768t   the same, tiled encode (default since §25: H3's latent statistics)
    split, path_seed, scale_0, scale_1, center_x_0, center_y_0, center_x_1, center_y_1, angle_0, angle_1
The caption is the row's h3_magick_images.h3_text_embed_v3 (joined by page_id at training, h3magic/split_video.py).

    h3_motion_clip.py WORKER WORKERS [--count 600]   (from data/; GPU = $CUDA_VISIBLE_DEVICES; ~45 s per clip, 44 GiB)
    h3_motion_clip.py import_files FOLDER            (CPU) commits clips made earlier as files by make_clips.py
"""
import os
import sys
import time

import fire
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import read_png_rgba, tensors_bytes
from h3magic import h3, synth_motion

SIZE = 768
FRAMES = 141  # = h3.VIDEO_FRAMES: the inference length, so every latent frame position is trained
PATH_SEED = 0
HEADS = ("gray", "white", "black")  # synth_motion.LEVELS order
COLUMNS = [f"h3_motion_{h}_768" for h in HEADS]  # untiled (the first 517 clips)
TILED_COLUMNS = [c + "t" for c in COLUMNS]  # tiled encode (§25)


def path_of(page_id, seed=PATH_SEED):
    """
    Pure function. The motion path parameters of one row (deterministic in (seed, page_id)).

    Examples:
        >>> sorted(path_of("00FAQCWeYv"))
        ['angle', 'center', 'scale']
        >>> bool(np.allclose(path_of("00FAQCWeYv")["scale"], path_of("00FAQCWeYv")["scale"]))
        True
    """
    return synth_motion.path_params(np.random.default_rng([seed, int.from_bytes(page_id.encode()[:8], "little")]))


def path_fields(params, seed=PATH_SEED):
    """
    Pure function. The h3_motion_clips scalar columns of one path.

    Examples:
        >>> f = path_fields(dict(scale=np.array([0.6, 0.9]), center=np.array([[0.4, 0.5], [0.6, 0.5]]), angle=np.array([1.0, -2.0])))
        >>> f["scale_1"], f["center_x_0"], f["angle_1"], f["path_seed"]
        (0.9, 0.4, -2.0, 0)
    """
    s, c, a = params["scale"], params["center"], params["angle"]
    return dict(path_seed=seed, scale_0=float(s[0]), scale_1=float(s[1]), center_x_0=float(c[0, 0]), center_y_0=float(c[0, 1]),
                center_x_1=float(c[1, 0]), center_y_1=float(c[1, 1]), angle_0=float(a[0]), angle_1=float(a[1]))


def stable_worker(page_id, workers):
    """
    Pure function. The worker that owns a row: sha256(page_id) mod workers (stable across restarts).

    Examples:
        >>> stable_worker("00FAQCWeYv", 4) == stable_worker("00FAQCWeYv", 4), 0 <= stable_worker("abc", 3) < 3
        (True, True)
    """
    import hashlib
    return int(hashlib.sha256(page_id.encode()).hexdigest(), 16) % workers


def chosen_rows(db, count, done_column="h3_motion_black_768t"):
    """Query. The rows to make: the first `count` train rows by page_id and every val row, with a fresh v3 embedding, none made yet."""
    rows = db.query("SELECT h.page_id, h.split, blob_info(m.rgba) AS rgba FROM h3_magick_images h JOIN magick_samples m ON m.page_id = h.page_id "
                    "WHERE h.h3_text_v3_sha256 = encode(sha256(h.h3_caption_v3), 'hex') ORDER BY h.page_id").to_pylist()
    done = {r["page_id"] for r in db.query(f"SELECT page_id FROM h3_motion_clips WHERE {done_column} IS NOT NULL").to_pylist()}
    train = [r for r in rows if r["split"] == "train"][:count]
    return [r for r in train + [r for r in rows if r["split"] == "val"] if r["page_id"] not in done]


def composites_of(row):
    """Query (reads the RGBA blob). (path params, (3, T, 768, 768, 3) uint8 composites on gray, white, black) of one row."""
    rgba = read_png_rgba(row["rgba"])
    params = path_of(row["page_id"])
    return params, synth_motion.moving_composites(rgba, synth_motion.affine_path(params, FRAMES, rgba.shape[0], SIZE), SIZE)


def main(worker, workers, count=600, only_split=None, tiled=True):
    """Command. Makes this worker's share of the chosen clips (stable_worker), committing each as it is made."""
    from bluejay import open_database
    db = open_database("project.dtab")
    columns = TILED_COLUMNS if tiled else COLUMNS
    rows = [r for r in chosen_rows(db, count, columns[2]) if stable_worker(r["page_id"], workers) == worker and only_split in (None, r["split"])]
    print(f"worker {worker}/{workers}: {len(rows)} clips to make", flush=True)
    vae = h3.load_vae()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)  # the next clip's composites (CPU, ~half the time) are built while the GPU encodes this one
    pending = pool.submit(composites_of, rows[0]) if rows else None
    for i, row in enumerate(rows):
        t = time.time()
        params, clips = pending.result()  # clips (3, T, S, S, 3)
        pending = pool.submit(composites_of, rows[i + 1]) if i + 1 < len(rows) else None
        with torch.inference_mode():
            blobs = {c: tensors_bytes({"latent": h3.encode(vae, clip, tiled=tiled).to(torch.bfloat16)}, layout="LC, LT, LH, LW", background=level,
                                      seed=h3.ENCODE_SEED, tiled=tiled)
                     for c, clip, level in zip(columns, clips, synth_motion.LEVELS)}
        db.commit("h3_motion_clips", [*columns, "split", *path_fields(params)], [{"page_id": row["page_id"], **blobs, "split": row["split"], **path_fields(params)}],
                  insert_missing=True)
        print(f"{time.strftime('%I:%M:%S %p')} worker {worker}: {i + 1}/{len(rows)} {row['page_id']} ({row['split']}) {time.time() - t:.1f} s", flush=True)


def import_files(folder):
    """Command. Commits the clips that make_clips.py wrote as files (same path seed, same recipe) into h3_motion_clips."""
    from pathlib import Path
    from safetensors.torch import load_file
    from bluejay import open_database
    db = open_database("project.dtab")
    splits = {r["page_id"]: r["split"] for r in db.query("SELECT page_id, split FROM h3_magick_images").to_pylist()}
    files = sorted(Path(folder).glob("*.safetensors"))
    for path in files:
        page_id, d = path.stem, load_file(str(path))
        params = path_of(page_id)
        blobs = {c: tensors_bytes({"latent": d[h]}, layout="LC, LT, LH, LW", background=level, seed=h3.ENCODE_SEED)
                 for c, h, level in zip(COLUMNS, HEADS, synth_motion.LEVELS)}
        db.commit("h3_motion_clips", [*COLUMNS, "split", *path_fields(params)], [{"page_id": page_id, **blobs, "split": splits[page_id], **path_fields(params)}],
                  insert_missing=True)
    print(f"imported {len(files)} clips from {folder}", flush=True)


if __name__ == "__main__":
    fire.Fire({"work": main, "import_files": import_files})
