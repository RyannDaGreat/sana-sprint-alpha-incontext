"""
FIBO2 latents (manifest §14): the FLUX.2 VAE of Bria Fibo 2.0 encodes every image the SANA runs train on, at 1024, 768,
512 and 256. Long-lived workers, two per GPU, as encode_multires (same reason: a Bluejay run spends most of its time in
claim, materialize and stage); rows are read from their Blob files and committed with the database's commit.

Writes, each {"latent": (32, S/8, S/8) bfloat16 (LC, LH, LW)} = AutoencoderKLFlux2.encode(x).latent_dist.mode(), raw:
the FIBO2 BatchNorm normalization (2x2 patchify, minus vae.bn.running_mean, over sqrt(running_var + eps)) is NOT applied;
fibo2.normalize_latents applies it at training time. 1024: 1 MB, 768: 576 KB, 512: 256 KB, 256: 64 KB.
    magick_samples         fibo2_gray_{1024,768,512,256}     rgb_on_gray (the control target)
                           fibo2_colorbg_{1024,768,512,256}  rgba on its encode_colorbg color (the adapter target, as adapter_v2)
                           fibo2_white_{1024,768,512,256}    rgba on white, job "split" (a splitter LoRA target, manifest §17)
                           fibo2_black_{1024,768,512,256}    rgba on black, job "split" (a splitter LoRA target, manifest §17)
    trimask_augmentations  fibo2_trimask_{1024,768,512,256}  the trimask, copied to 3 channels (the control condition)
Images are area-downsampled from 1024 in uint8, as in encode_multires.

    encode_fibo2.py WORKER WORKERS [--jobs magick,trimask|split]    (from data/; the GPU is $CUDA_VISIBLE_DEVICES)
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import fire
import numpy as np
import torch
from safetensors.torch import save

sys.path.insert(0, os.path.dirname(__file__))
from encode_colorbg import background_color
from encode_multires import model_input, read_png, resize_area, shard_sql
from split_alpha import composite_on_gray

VAE_DIR = "../models/Fibo-2.0-dev/vae"
SIZES = (1024, 768, 512, 256)
GPU_BATCH = {1024: 8, 768: 16, 512: 32, 256: 64}  # Images per encoder call: A100 bf16 does 18, 36, 86, 354 img/s at any batch; ≤ 12 GB
CHUNK = 128  # Rows per encode and commit: ≈ 0.5 GB of pinned images and 0.5 GB of latents per worker
QUERY_CHUNKS = 4  # Chunks per query: the next chunk is decoded while the GPU encodes this one
IO_WORKERS = 6  # Threads per worker that read, decode and downsample PNGs: 16 workers x 6 fill the 96 CPUs
cv2.setNumThreads(1)  # The IO threads already run in parallel


def read_rgba(info):
    """Query. One stored RGBA PNG, read from its Blob file (blob_info: uri, position, size): uint8 (H, W, 4) RGBA."""
    with open(info["uri"], "rb") as f:
        f.seek(info["position"])
        data = np.frombuffer(f.read(info["size"]), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Cannot decode an RGBA PNG at {info}")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def source_images(row, job):
    """
    Query (reads Blob files). The 1024² images of one row that a job encodes.

    Args:
        row (dict): page_id and the job's blob_info columns
        job (str): "magick", "trimask" or "split"

    Returns:
        dict: {output prefix: uint8 (H, W) or (H, W, 3)}
    """
    if job == "trimask":
        return {"fibo2_trimask": read_png(row["trimask"], gray=True)}
    if job == "split":  # The same composite as split_alpha's rgb_on_gray, on white and on black
        rgba = read_rgba(row["rgba"])
        return {"fibo2_white": composite_on_gray(rgba, 1.0), "fibo2_black": composite_on_gray(rgba, 0.0)}
    return {"fibo2_gray": read_png(row["rgb_on_gray"], gray=False),
            "fibo2_colorbg": composite_on_gray(read_rgba(row["rgba"]), np.array(background_color(row["page_id"]), np.float32))}


def prepare(pool, rows, job):
    """
    Query (reads Blob files). Every image of every row at every size, decoded in the pool.

    Returns:
        dict: {(prefix, size): pinned uint8 tensor (B, S, S) or (B, S, S, 3)}
    """
    def one(row):
        return {(prefix, size): resize_area(image, size) for prefix, image in source_images(row, job).items() for size in SIZES}

    decoded = list(pool.map(one, rows))
    return {key: torch.from_numpy(np.stack([d[key] for d in decoded])).pin_memory() for key in decoded[0]}


def main(worker, workers, batch_divisor=1, jobs="magick,trimask", limit=None):
    """
    Command. Encodes this worker's rows of both jobs on the first visible GPU until none is left. Chunk N + 1 is read in
    the thread pool and chunk N - 1 committed in the background while the GPU encodes chunk N.

    Args:
        worker (int): 0 to workers - 1
        workers (int): Number of workers (two per GPU)
        batch_divisor (int): Divides GPU_BATCH, for a GPU shared with other jobs
        jobs (str | tuple): Comma-separated jobs to run, in order: magick, trimask, split
        limit (int): Stop after about this many rows per job (a smoke test); None = every row
    """
    from bluejay import open_database
    from diffusers import AutoencoderKLFlux2

    db = open_database("project.dtab")
    vae = AutoencoderKLFlux2.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).cuda().eval()
    metadata = {"vae": "FLUX.2 (Fibo-2.0-dev/vae)", "normalized": "false", "layout": "LC, LH, LW"}
    pool, background = ThreadPoolExecutor(IO_WORKERS), ThreadPoolExecutor(2)
    batch_of = {size: max(1, n // batch_divisor) for size, n in GPU_BATCH.items()}  # Images per encoder call
    every_job = {  # job: (table, key columns, sql key text, blob columns, done column)
        "magick": ("magick_samples", ["page_id"], "page_id", ["rgb_on_gray", "rgba"], "fibo2_colorbg_256"),
        "trimask": ("trimask_augmentations", ["page_id", "method", "variant"], "concat(page_id, method)", ["trimask"], "fibo2_trimask_256"),
        "split": ("magick_samples", ["page_id"], "page_id", ["rgba", "fibo2_colorbg_256"], "fibo2_black_256"),  # colorbg: only rows the
        # magick job encoded, i.e. the training rows (3 rows store an RGB PNG as rgba and were never encoded)
    }
    names = jobs.split(",") if isinstance(jobs, str) else list(jobs)  # Fire parses "a,b" as a tuple
    start, encoded, commits = time.time(), 0, []
    with torch.inference_mode():
        for job in names:
            table, keys, key_sql, blobs, done_column = every_job[job]
            job_rows = 0
            while limit is None or job_rows < limit:
                for commit in commits:
                    commit.result()  # Raises a commit error here; and the next query must see the committed rows
                commits = []
                blob_infos = ", ".join(f"blob_info({column}) AS {column}" for column in blobs)
                rows = db.query(f"SELECT {', '.join(keys)}, {blob_infos} FROM {table} "
                                f"WHERE {done_column} IS NULL AND {' AND '.join(f'{c} IS NOT NULL' for c in blobs)} "
                                f"AND {shard_sql(key_sql, worker, workers)} LIMIT {CHUNK * QUERY_CHUNKS}").to_pylist()
                if not rows:
                    break
                chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
                pending = background.submit(prepare, pool, chunks[0], job)
                for index, chunk in enumerate(chunks):
                    images = pending.result()
                    if index + 1 < len(chunks):
                        pending = background.submit(prepare, pool, chunks[index + 1], job)
                    updates = [{k: row[k] for k in keys} for row in chunk]
                    for (prefix, size), batch in images.items():
                        step = batch_of[size]
                        latents = torch.cat([vae.encode(model_input(batch[i:i + step].to("cuda", non_blocking=True))).latent_dist.mode()
                                             for i in range(0, len(batch), step)]).cpu()
                        for update, latent in zip(updates, latents):
                            update[f"{prefix}_{size}"] = save({"latent": latent.contiguous()}, metadata)
                    commits.append(background.submit(db.commit, table, [c for c in updates[0] if c not in keys], updates))
                    encoded += len(chunk)
                    job_rows += len(chunk)
                    print(f"worker {worker} {table}: +{len(chunk)} rows ({encoded / (time.time() - start):.2f} rows/s this worker)", flush=True)
    for commit in commits:
        commit.result()
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
