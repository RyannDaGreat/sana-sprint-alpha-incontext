"""
H3 latents of the H3 Magic project (manifest §22): long-lived VAE workers, one or more per GPU, committed with the
database's commit (the encode_fibo2 pattern: the VAE stays loaded, no Bluejay run per batch).

Each blob is safetensors {"latent": bfloat16}, normalized as h3magic/h3.py encode (the MinimaxH3 training-target recipe):
    h3_magick_images  h3_latent_512     (24, 1, 32, 32)      (LC, LT, LH, LW) of image_512
    h3_pexels_clips   h3_latent_frames  (7, 24, 1, 32, 32)   (K, LC, LT, LH, LW) of frame_0s .. frame_6s (m00917)
    h3_pexels_clips   h3_latent_video   (24, 42, 32, 32)     (LC, LT, LH, LW) of clip_512 (141 frames)
Only captioned rows are encoded (training needs the caption); each round does one query per job: images (cheap) and videos (~5 s each on an A100).

    h3_encode.py WORKER WORKERS [--jobs magick,frames,video] [--poll_seconds 120]   (from data/; GPU = $CUDA_VISIBLE_DEVICES)
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import fire
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import read_mp4_rgb, read_png_rgb, round_robin, shard_sql, tensors_bytes
from h3magic import h3

FRAME_COLUMNS = [f"frame_{s}s" for s in range(7)]  # The m00917 stills at 0..6 s
IMAGE_BATCH = 8  # Single frames per VAE call: 32 ran out of memory beside another 34 GB job
IO_WORKERS = 8  # Threads per worker that read and decode PNG / MP4 blobs
JOBS = {  # job: (table, key, blob columns read, output column, rows per query)
    "magick": ("h3_magick_images", "page_id", ["image_512"], "h3_latent_512", 256),
    "frames": ("h3_pexels_clips", "pexels_id", FRAME_COLUMNS, "h3_latent_frames", 64),
    "video": ("h3_pexels_clips", "pexels_id", ["clip_512"], "h3_latent_video", 16),
}


def job_sql(job, worker, workers):
    """
    Pure function. The query of one job's not-yet-encoded, captioned rows of this worker.

    Examples:
        >>> "h3_latent_512 IS NULL" in job_sql("magick", 0, 2) and "h3_caption IS NOT NULL" in job_sql("magick", 0, 2)
        True
    """
    table, key, blobs, output, limit = JOBS[job]
    infos = ", ".join(f"blob_info({c}) AS {c}" for c in blobs)
    ready = " AND ".join(f"{c} IS NOT NULL" for c in blobs)
    return (f"SELECT {key}, {infos} FROM {table} WHERE {output} IS NULL AND h3_caption IS NOT NULL AND {ready} "
            f"AND {shard_sql(key, worker, workers)} {train_first(table)} LIMIT {limit}")


def train_first(table):
    """
    Pure function. SQL that skips the clips h3_choose_clip left unused and orders train rows first, then val, then the
    rest, each in random order. A clip's role is h3_use when h3_choose_clip has set it, else its split.

    Examples:
        >>> train_first("h3_magick_images")
        "ORDER BY split = 'train' DESC, split = 'val' DESC, random()"
        >>> train_first("h3_pexels_clips").startswith("AND coalesce(h3_use, '') <> 'unused' ORDER BY coalesce(h3_use, split) = 'train' DESC")
        True
    """
    if table == "h3_pexels_clips":
        role = "coalesce(h3_use, split)"
        return f"AND coalesce(h3_use, '') <> 'unused' ORDER BY {role} = 'train' DESC, {role} = 'val' DESC, random()"
    return "ORDER BY split = 'train' DESC, split = 'val' DESC, random()"


def encode_rows(vae, pool, job, rows):
    """
    Query (reads Blob files, runs the VAE). The output blob of each row of one job.

    Returns:
        list[bytes]: safetensors bytes, one per row
    """
    _, _, blobs, _, _ = JOBS[job]
    metadata = dict(vae="MiniMax-H3 video VAE, untiled encode, sampled posterior", seed=h3.ENCODE_SEED, normalized="latents_mean/std")
    if job == "video":
        clips = list(pool.map(lambda row: read_mp4_rgb(row["clip_512"]), rows))
        out = []
        for clip in clips:
            if clip.shape != (h3.VIDEO_FRAMES, h3.SIZE, h3.SIZE, 3):
                raise ValueError(f"clip_512 has shape {clip.shape}, expected {(h3.VIDEO_FRAMES, h3.SIZE, h3.SIZE, 3)}")
            out.append(tensors_bytes({"latent": h3.encode(vae, clip).to(torch.bfloat16)}, layout="LC, LT, LH, LW", **metadata))
        return out
    images = np.stack(list(pool.map(lambda row: np.stack([read_png_rgb(row[c]) for c in blobs]), rows)))  # (B, K, H, W, 3)
    flat = images.reshape(-1, 1, *images.shape[2:])  # (B * K, 1, H, W, 3): one image per latent
    latents = torch.cat([h3.encode(vae, flat[i:i + IMAGE_BATCH]) for i in range(0, len(flat), IMAGE_BATCH)])  # (B * K, LC, 1, LH, LW)
    latents = latents.view(len(rows), len(blobs), *latents.shape[1:]).to(torch.bfloat16)
    if job == "magick":
        return [tensors_bytes({"latent": z[0]}, layout="LC, LT, LH, LW", **metadata) for z in latents]
    return [tensors_bytes({"latent": z}, layout="K, LC, LT, LH, LW; K = frames at 0..6 s", **metadata) for z in latents]


def main(worker, workers, jobs="magick,frames,video", poll_seconds=120):
    """
    Command. Encodes this worker's rows of every job, one query of each job per round (images first), committing as it
    goes, until a round finds no rows twice `poll_seconds` apart (captions still arrive).

    Args:
        worker (int): 0 to workers - 1
        workers (int): Number of workers
        jobs (str | tuple): Comma-separated jobs, in order
        poll_seconds (int): Wait between empty queries
    """
    from bluejay import open_database

    db = open_database("project.dtab")
    vae = h3.load_vae()
    pool = ThreadPoolExecutor(IO_WORKERS)
    names = jobs.split(",") if isinstance(jobs, str) else list(jobs)

    def handler(job):
        table, key, _, output, _ = JOBS[job]

        def handle(rows):
            blobs = encode_rows(vae, pool, job, rows)
            db.commit(table, [output], [{key: row[key], output: blob} for row, blob in zip(rows, blobs)])
        return handle

    with torch.inference_mode():
        round_robin(db, [(job_sql(job, worker, workers), handler(job), f"worker {worker} {job}") for job in names], poll_seconds)
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
