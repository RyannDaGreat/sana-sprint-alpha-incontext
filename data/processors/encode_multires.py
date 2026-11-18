"""
Multi-resolution DC-AE encoder (manifest §13, m00656, m00657): long-lived workers, two per GPU, load the VAE once and keep it
busy. Not a Bluejay processor: a Bluejay run spent ~5 of its ~6 minutes in claim, materialize and stage, so the worker reads
PNG Blobs straight from their files (blob_info), decodes and area-downsamples them in a thread pool, encodes big batches per
resolution, and commits the latents with the database's commit. Run by data/run_encode_multires.fish (it adds Bluejay).

Writes, each {"latent": (32, S/32, S/32) bfloat16 (LC, LH, LW)} unscaled, like encode_latents (multiply by scaling_factor
before diffusion training):
    magick_samples         latent_rgb_{768,512,256}, latent_alpha_{768,512,256}   (latent_*_1024 come from encode_latents)
    trimask_augmentations  latent_trimask_{1024,768,512,256}
latent_trimask_{768,512,256} for the LOWRES_DIGITS rows (first hex digit of sha256(page_id || method || variant)): now all;
first 5 of 16 (~31%) for GPU time, then all (m00677). Jobs in order: magick rows (the training targets
of every low-resolution step), latent_trimask_1024 of every trimask row (waits for augmentation until STOP_WHEN exists), then
the low-resolution trimask latents. Rows are split among workers by the 2nd hex digit of sha256(key text); a job's row is done
when its done column is set (committed with the others).

    encode_multires.py WORKER WORKERS STOP_WHEN    (from data/; the GPU is $CUDA_VISIBLE_DEVICES)
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
from encode_latents import DTYPE, load_vae

TEACHER_DIR = "../models/Sana_Sprint_1.6B_1024px_teacher_diffusers"  # The dtab's TEACHER_DIR
GPU_BATCH = {1024: 16, 768: 16, 512: 32, 256: 64}  # Images per encoder call: A100 bf16 does 12.9, 22.9, 51.6, 206 img/s
CHUNK = 256  # Rows per encode and commit
QUERY_CHUNKS = 4  # Chunks per query: the next chunk is decoded while the GPU encodes this one
IO_WORKERS = 6  # Threads per worker that read, decode and downsample PNGs: 16 workers x 6 fill the 96 CPUs
IDLE_SECONDS = 30  # Wait between queries while augmentation still adds trimask rows
LOWRES_DIGITS = "0123456789abcdef"  # Trimask rows that also get 768/512/256 latents: all (m00677; was 5 of 16 at first, for GPU time)
cv2.setNumThreads(1)  # The IO threads already run in parallel


def resize_area(image, size):
    """
    Pure function. Area-downsamples a square image to size x size (unchanged at its own size).

    Args:
        image (np.ndarray): (H, H) or (H, H, 3) uint8
        size (int): Output side, <= H

    Returns:
        np.ndarray: (size, size) or (size, size, 3) uint8

    Examples:
        >>> resize_area(np.array([[0, 255], [255, 0]], np.uint8), 1).tolist()
        [[128]]
        >>> resize_area(np.zeros((1024, 1024, 3), np.uint8), 256).shape
        (256, 256, 3)
    """
    return image if image.shape[0] == size else cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def read_png(info, gray):
    """Query. One stored PNG, read from its Blob file (blob_info: uri, position, size) and decoded once by OpenCV: uint8 (H, W) if gray, else (H, W, 3) RGB."""
    with open(info["uri"], "rb") as f:
        f.seek(info["position"])
        data = np.frombuffer(f.read(info["size"]), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode the PNG at {info}")
    return image if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def shard_sql(key_sql, worker, workers):
    """
    Pure function. SQL that keeps the rows of one worker: the 2nd hex digit of sha256(key) mod workers.

    Examples:
        >>> shard_sql("page_id", 1, 8)
        "contains('19', substr(encode(sha256(page_id), 'hex'), 2, 1))"
    """
    digits = "".join(f"{d:x}" for d in range(16) if d % workers == worker)
    return f"contains('{digits}', substr(encode(sha256({key_sql}), 'hex'), 2, 1))"


def prepare(pool, rows, sources):
    """
    Query (reads Blob files). Decodes each source image of each row once and area-downsamples it to every size, in the pool.

    Returns:
        dict: {(source column, size): pinned uint8 tensor (B, S, S) or (B, S, S, 3)}
    """
    def one(row):
        out = {}
        for column, mode, _, sizes in sources:
            image = read_png(row[column], mode == "L")
            out.update({(column, size): resize_area(image, size) for size in sizes})
        return out

    decoded = list(pool.map(one, rows))
    return {key: torch.from_numpy(np.stack([d[key] for d in decoded])).pin_memory() for key in decoded[0]}


def model_input(batch):
    """
    Pure function. DC-AE input on the GPU from uint8 images, as encode_latents.to_model_input: 2 x / 255 - 1, a matte copied to 3 channels.

    Args:
        batch (torch.Tensor): (B, S, S) or (B, S, S, 3) uint8 (on the GPU)

    Returns:
        torch.Tensor: (B, 3, S, S) DTYPE in [-1, 1]

    Examples:
        >>> model_input(torch.tensor([[[0, 255]]], dtype=torch.uint8)).float().tolist()
        [[[[-1.0, 1.0]], [[-1.0, 1.0]], [[-1.0, 1.0]]]]
    """
    x = batch.float() / 127.5 - 1
    x = x[:, None].expand(-1, 3, -1, -1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
    return x.to(DTYPE)


def main(worker, workers, stop_when):
    """
    Command. Encodes this worker's rows on the first visible GPU until all are done (see the module docstring). Chunk N + 1 is
    read and decoded in the thread pool and chunk N - 1 committed in the background while the GPU encodes chunk N.

    Args:
        worker (int): 0 to workers - 1
        workers (int): Number of workers (two per GPU)
        stop_when (str): A file that exists once augmentation is done
    """
    from bluejay import open_database

    db = open_database("project.dtab")
    vae = load_vae(TEACHER_DIR)
    metadata = {"scaling_factor": str(vae.config.scaling_factor), "scaling_factor_applied": "false", "layout": "LC, LH, LW"}
    pool, background = ThreadPoolExecutor(IO_WORKERS), ThreadPoolExecutor(2)
    trimask_keys, lowres = ["page_id", "method", "variant"], f"contains('{LOWRES_DIGITS}', left(encode(sha256(concat(page_id, method, CAST(variant AS VARCHAR))), 'hex'), 1))"
    jobs = [  # (table, key columns, sql key text, [(source column, PNG mode, output prefix, sizes)], done column, extra SQL, waits for augmentation)
        ("magick_samples", ["page_id"], "page_id",
         [("rgb_on_gray", "RGB", "latent_rgb", (768, 512, 256)), ("alpha", "L", "latent_alpha", (768, 512, 256))], "latent_rgb_256", "TRUE", False),
        ("trimask_augmentations", trimask_keys, "concat(page_id, method)", [("trimask", "L", "latent_trimask", (1024,))], "latent_trimask_1024", "TRUE", True),
        ("trimask_augmentations", trimask_keys, "concat(page_id, method)", [("trimask", "L", "latent_trimask", (768, 512, 256))], "latent_trimask_256", lowres, False),
    ]
    start, encoded, commits = time.time(), {}, []
    with torch.inference_mode():
        for table, keys, key_sql, sources, done_column, extra, waits in jobs:
            while True:
                for commit in commits:
                    commit.result()  # Raises a commit error here; and the next query must see the committed rows
                commits = []
                finished = not waits or os.path.exists(stop_when)  # Read before the query: it then sees the last commit
                blob_infos = ", ".join(f"blob_info({column}) AS {column}" for column, *_ in sources)
                rows = db.query(f"SELECT {', '.join(keys)}, {blob_infos} FROM {table} "
                                f"WHERE {done_column} IS NULL AND {' AND '.join(f'{c} IS NOT NULL' for c, *_ in sources)} "
                                f"AND {extra} AND {shard_sql(key_sql, worker, workers)} LIMIT {CHUNK * QUERY_CHUNKS}").to_pylist()
                if not rows:
                    if finished:
                        break
                    time.sleep(IDLE_SECONDS)
                    continue
                chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
                pending = background.submit(prepare, pool, chunks[0], sources)
                for index, chunk in enumerate(chunks):
                    images = pending.result()
                    if index + 1 < len(chunks):
                        pending = background.submit(prepare, pool, chunks[index + 1], sources)
                    updates = [{k: row[k] for k in keys} for row in chunk]
                    for (column, size), batch in images.items():
                        prefix = next(p for c, _, p, _ in sources if c == column)
                        latents = torch.cat([vae.encode(model_input(batch[i:i + GPU_BATCH[size]].to("cuda", non_blocking=True))).latent
                                             for i in range(0, len(batch), GPU_BATCH[size])]).cpu()
                        for update, latent in zip(updates, latents):
                            update[f"{prefix}_{size}"] = save({"latent": latent.contiguous()}, metadata)
                        encoded[size] = encoded.get(size, 0) + len(chunk)
                    commits.append(background.submit(db.commit, table, [c for c in updates[0] if c not in keys], updates))
                    rates = ", ".join(f"{size}: {n / (time.time() - start):.1f}/s" for size, n in sorted(encoded.items()))
                    print(f"worker {worker} {table}: +{len(chunk)} rows ({rates} images per size)", flush=True)
    for commit in commits:
        commit.result()
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
