"""
H3 latents of the H3 Magic project (manifest §22): long-lived VAE workers, one or more per GPU, committed with the
database's commit (the encode_fibo2 pattern: the VAE stays loaded, no Bluejay run per batch).

Each blob is safetensors {"latent": bfloat16}, normalized as h3magic/h3.py encode (the MinimaxH3 training-target recipe):
    h3_magick_images  h3_latent_512     (24, 1, 32, 32)      (LC, LT, LH, LW) of image_512
    h3_pexels_clips   h3_latent_frames  (7, 24, 1, 32, 32)   (K, LC, LT, LH, LW) of frame_0s .. frame_6s (m00917)
    h3_pexels_clips   h3_latent_video   (24, 42, 32, 32)     (LC, LT, LH, LW) of clip_512 (141 frames)
    h3_magick_images  h3_latent_white_512, h3_latent_black_512   (24, 1, 32, 32) of magick_samples.rgba composited on
                      white and on black at 1024 and area-downsampled to 512 exactly as image_512 (the split adapter's targets)
    h3_magick_images  h3_latent_gray_768, h3_latent_white_768, h3_latent_black_768   (24, 1, 48, 48) of magick_samples.rgba
                      composited on 0.5 gray, white and black at 1024 and area-resized to 768 (job v3, Alpha Video v2, manifest §23)
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
from h3_blobs import byte_shard_sql, read_mp4_rgb, read_png_rgb, read_png_rgba, round_robin, shard_range, shard_sql, tensors_bytes
from h3_magick_image import area_downsample
from split_alpha import composite_on_gray
from h3magic import h3

FRAME_COLUMNS = [f"frame_{s}s" for s in range(7)]  # The m00917 stills at 0..6 s
IMAGE_BATCH = 8  # Single frames per VAE call: 32 ran out of memory beside another 34 GB job
IO_WORKERS = 8  # Threads per worker that read and decode PNG / MP4 blobs
JOBS = {  # job: (table, key, blob columns read, output column, rows per query)
    "magick": ("h3_magick_images", "page_id", ["image_512"], "h3_latent_512", 256),
    "frames": ("h3_pexels_clips", "pexels_id", FRAME_COLUMNS, "h3_latent_frames", 64),
    "video": ("h3_pexels_clips", "pexels_id", ["clip_512"], "h3_latent_video", 16),
    "split": ("h3_magick_images", "page_id", ["rgba"], "h3_latent_black_512", 64),  # done column; rgba from magick_samples
    "v3": ("h3_magick_images", "page_id", ["rgba"], "h3_latent_black_768", 32),  # done column; rgba from magick_samples
    "grid": ("h3_magick_images", "page_id", ["rgba"], "h3_latent_grid_768", 32),  # rgba from magick_samples
    "v3t": ("h3_magick_images", "page_id", ["rgba"], "h3_latent_black_768t", 32),  # v3 again, tiled encode (§25); done column
    "grid1024": ("h3_magick_images", "page_id", ["rgba"], "h3_latent_grid_1024", 16),  # split v3-grid with 512 panels (panel-size test)
}
SPLIT_BACKGROUNDS = {"h3_latent_white_512": 1.0, "h3_latent_black_512": 0.0}  # split job: output column -> composite_on_gray level
V3_BACKGROUNDS = {"h3_latent_gray_768": 0.5, "h3_latent_white_768": 1.0, "h3_latent_black_768": 0.0}  # v3 job: output column -> level
V3_SIZE = 768  # Alpha Video v2 resolution (m00983: "512 is a little bit low")
V3T_BACKGROUNDS = {c + "t": level for c, level in V3_BACKGROUNDS.items()}  # the tiled-encode columns of job v3t
GRID_PANEL_LARGE = 512  # job grid1024: 2 x 512 panels = a 1024² frame
RGBA_JOBS = {"split": SPLIT_BACKGROUNDS, "v3": V3_BACKGROUNDS, "grid": {"h3_latent_grid_768": None}, "v3t": V3T_BACKGROUNDS, "grid1024": {"h3_latent_grid_1024": None}}  # jobs that composite magick_samples.rgba themselves
GRID_PANEL = 384  # split v3-grid: 2 x 384 = one 768² frame, the token count of the other 768² runs


def job_sql(job, worker, workers, shards=None):
    """
    Pure function. The query of one job's not-yet-encoded, captioned rows of this worker: its 16-shard part
    (worker of workers), or, when shards is a list, the rows of those byte shards (0-255, h3_blobs.byte_shard_sql).

    Examples:
        >>> "h3_latent_512 IS NULL" in job_sql("magick", 0, 2) and "h3_caption IS NOT NULL" in job_sql("magick", 0, 2)
        True
        >>> "IN ('30', '31')" in job_sql("video", 0, 1, [48, 49])
        True
        >>> "JOIN magick_samples" in job_sql("split", 0, 1, [7]) and "h.h3_latent_black_512 IS NULL" in job_sql("split", 0, 1, [7])
        True
    """
    table, key, blobs, output, limit = JOBS[job]
    if job in RGBA_JOBS:  # The RGBA lives in magick_samples: join it (same page_id)
        shard = shard_sql("h.page_id", worker, workers) if shards is None else byte_shard_sql("h.page_id", shards)
        return (f"SELECT h.page_id, blob_info(m.rgba) AS rgba FROM h3_magick_images h JOIN magick_samples m ON m.page_id = h.page_id "
                f"WHERE h.{output} IS NULL AND h.h3_caption IS NOT NULL AND m.rgba IS NOT NULL AND {shard} LIMIT {limit}")
    shard = shard_sql(key, worker, workers) if shards is None else byte_shard_sql(key, shards)
    infos = ", ".join(f"blob_info({c}) AS {c}" for c in blobs)
    ready = " AND ".join(f"{c} IS NOT NULL" for c in blobs)
    return (f"SELECT {key}, {infos} FROM {table} WHERE {output} IS NULL AND h3_caption IS NOT NULL AND {ready} "
            f"AND {shard} {train_first(table)} LIMIT {limit}")


def train_first(table):
    """
    Pure function. SQL that orders train rows first, then val, then the rest, each in random order. The role is the
    split column (train | val | spare); h3_use holds h3_choose_clip's status (provisional | final), not a role.

    Examples:
        >>> train_first("h3_pexels_clips")
        "ORDER BY split = 'train' DESC, split = 'val' DESC, random()"
    """
    return "ORDER BY split = 'train' DESC, split = 'val' DESC, random()"


def split_composites(rgba):
    """
    Pure function. The split job's images of one MAGICK RGBA: on white and on black (SPLIT_BACKGROUNDS order), each
    composited at 1024 in 8 bit and area-downsampled to 512, the recipe of image_512 (= composite_on_gray(rgba, 0.5)).

    Args:
        rgba (np.ndarray): (1024, 1024, 4) uint8 straight alpha

    Returns:
        np.ndarray: (2, 512, 512, 3) uint8 RGB

    Examples:
        >>> rgba = np.zeros((1024, 1024, 4), np.uint8); rgba[..., 0] = 255; rgba[:512, :, 3] = 255  # top opaque red, bottom clear
        >>> out = split_composites(rgba)
        >>> out.shape, out[0, 0, 0].tolist(), out[0, -1, 0].tolist(), out[1, -1, 0].tolist()  # red; white; black
        ((2, 512, 512, 3), [255, 0, 0], [255, 255, 255], [0, 0, 0])
    """
    return np.stack([area_downsample(composite_on_gray(rgba, level), h3.SIZE) for level in SPLIT_BACKGROUNDS.values()])


def area_resize(image, size):
    """
    Pure function. Area-resamples a square uint8 image to size x size (any ratio; cv2 INTER_AREA in float32, rounded).

    Args:
        image (np.ndarray): (H, H, C) uint8, e.g. (1024, 1024, 3)
        size (int): Output side, e.g. 768

    Returns:
        np.ndarray: (size, size, C) uint8

    Examples:
        >>> x = np.zeros((4, 4, 3), np.uint8); x[:, :2] = 255
        >>> area_resize(x, 2)[..., 0].tolist()
        [[255, 0], [255, 0]]
        >>> area_resize(np.full((1024, 1024, 3), 77, np.uint8), 768).shape, int(area_resize(np.full((1024, 1024, 3), 77, np.uint8), 768).max())
        ((768, 768, 3), 77)
    """
    import cv2
    return np.rint(cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)).clip(0, 255).astype(np.uint8)


def v3_composites(rgba):
    """
    Pure function. The v3 job's images of one MAGICK RGBA: on 0.5 gray, white and black (V3_BACKGROUNDS order), each
    composited at 1024 in 8 bit, then area-resized to V3_SIZE.

    Args:
        rgba (np.ndarray): (1024, 1024, 4) uint8 straight alpha

    Returns:
        np.ndarray: (3, 768, 768, 3) uint8 RGB

    Examples:
        >>> rgba = np.zeros((1024, 1024, 4), np.uint8); rgba[..., 0] = 255; rgba[:512, :, 3] = 255  # top opaque red, bottom clear
        >>> out = v3_composites(rgba)
        >>> out.shape, out[0, 0, 0].tolist(), out[0, -1, 0].tolist(), out[1, -1, 0].tolist(), out[2, -1, 0].tolist()
        ((3, 768, 768, 3), [255, 0, 0], [128, 128, 128], [255, 255, 255], [0, 0, 0])
    """
    return np.stack([area_resize(composite_on_gray(rgba, level), V3_SIZE) for level in V3_BACKGROUNDS.values()])


def grid_image(rgba, panel=GRID_PANEL):
    """
    Pure function. The split v3-grid training image of one MAGICK RGBA: 2x2 panels, each composited at 1024 in 8 bit and
    area-resized to `panel`: top left on 0.5 gray, top right on white, bottom left on black, bottom right the alpha matte
    as a grayscale image (white = opaque).

    Args:
        rgba (np.ndarray): (1024, 1024, 4) uint8 straight alpha

    Returns:
        np.ndarray: (2·panel, 2·panel, 3) uint8, e.g. (768, 768, 3)

    Examples:
        >>> rgba = np.zeros((1024, 1024, 4), np.uint8); rgba[..., 0] = 255; rgba[:512, :, 3] = 255  # top opaque red, bottom clear
        >>> g = grid_image(rgba)
        >>> g.shape, g[0, 0].tolist(), g[383, 0].tolist(), g[383, 384].tolist(), g[767, 0].tolist(), g[384, 767].tolist(), g[767, 767].tolist()
        ((768, 768, 3), [255, 0, 0], [128, 128, 128], [255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0])
    """
    gray, white, black = (area_resize(composite_on_gray(rgba, level), panel) for level in (0.5, 1.0, 0.0))
    alpha = area_resize(np.repeat(rgba[..., 3:], 3, -1), panel)
    return np.concatenate([np.concatenate([gray, white], 1), np.concatenate([black, alpha], 1)], 0)


def encode_rows(vae, pool, job, rows):
    """
    Query (reads Blob files, runs the VAE). The output blobs of each row of one job.

    Returns:
        list[dict[str, bytes]]: {output column: safetensors bytes}, one per row
    """
    _, _, blobs, output, _ = JOBS[job]
    metadata = dict(vae="MiniMax-H3 video VAE, tiled encode (256 px), sampled posterior", seed=h3.ENCODE_SEED, normalized="latents_mean/std")
    if job == "video":
        clips = list(pool.map(lambda row: read_mp4_rgb(row["clip_512"]), rows))
        out = []
        for clip in clips:
            if clip.shape != (h3.VIDEO_FRAMES, h3.SIZE, h3.SIZE, 3):
                raise ValueError(f"clip_512 has shape {clip.shape}, expected {(h3.VIDEO_FRAMES, h3.SIZE, h3.SIZE, 3)}")
            out.append({output: tensors_bytes({"latent": h3.encode(vae, clip).to(torch.bfloat16)}, layout="LC, LT, LH, LW", **metadata)})
        return out
    if job in RGBA_JOBS:
        composites = {"split": split_composites, "v3": v3_composites, "v3t": v3_composites, "grid": lambda rgba: grid_image(rgba)[None],
                      "grid1024": lambda rgba: grid_image(rgba, GRID_PANEL_LARGE)[None]}[job]
        images = np.stack(list(pool.map(lambda row: composites(read_png_rgba(row["rgba"])), rows)))  # (B, K, S, S, 3)
    else:
        images = np.stack(list(pool.map(lambda row: np.stack([read_png_rgb(row[c]) for c in blobs]), rows)))  # (B, K, H, W, 3)
    flat = images.reshape(-1, 1, *images.shape[2:])  # (B * K, 1, H, W, 3): one image per latent
    latents = torch.cat([h3.encode(vae, flat[i:i + IMAGE_BATCH]) for i in range(0, len(flat), IMAGE_BATCH)])  # (B * K, LC, 1, LH, LW)
    latents = latents.view(len(rows), images.shape[1], *latents.shape[1:]).to(torch.bfloat16)
    if job in RGBA_JOBS:
        return [{column: tensors_bytes({"latent": z[k]}, layout="LC, LT, LH, LW", background=str(level), **metadata)
                 for k, (column, level) in enumerate(RGBA_JOBS[job].items())} for z in latents]
    if job == "magick":
        return [{output: tensors_bytes({"latent": z[0]}, layout="LC, LT, LH, LW", **metadata)} for z in latents]
    return [{output: tensors_bytes({"latent": z}, layout="K, LC, LT, LH, LW; K = frames at 0..6 s", **metadata)} for z in latents]


def main(worker, workers, jobs="magick,frames,video", poll_seconds=120, shard_lo=None, shard_hi=None):
    """
    Command. Encodes this worker's rows of every job, one query of each job per round (images first), committing as it
    goes, until a round finds no rows twice `poll_seconds` apart (captions still arrive).

    Args:
        worker (int): 0 to workers - 1
        workers (int): Number of workers
        jobs (str | tuple): Comma-separated jobs, in order
        poll_seconds (int): Wait between empty queries
        shard_lo, shard_hi (int | None): 256-shard mode: this worker takes its turn-dealt part of byte shards
            shard_lo..shard_hi (h3_blobs.shard_range), so nodes split the keys; None = the old 16-shard mode
    """
    from bluejay import open_database

    db = open_database("project.dtab")
    vae = h3.load_vae()
    pool = ThreadPoolExecutor(IO_WORKERS)
    names = jobs.split(",") if isinstance(jobs, str) else list(jobs)
    shards = None if shard_lo is None else shard_range(shard_lo, shard_hi, worker, workers)
    print(f"worker {worker}/{workers}: shards {'16-mode' if shards is None else shards}, jobs {names}", flush=True)

    def handler(job):
        table, key, _, output, _ = JOBS[job]

        def handle(rows):
            blobs = encode_rows(vae, pool, job, rows)
            db.commit(table, list(blobs[0]), [{key: row[key], **blob} for row, blob in zip(rows, blobs)])
        return handle

    with torch.inference_mode():
        round_robin(db, [(job_sql(job, worker, workers, shards), handler(job), f"worker {worker} {job}") for job in names], poll_seconds)
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
