"""
Shared helpers of the H3 Magic long-lived GPU workers (h3_encode, h3_embed, h3_generate, manifest §22): reading Blob
files, safetensors bytes, sharding, and the worker loop over a Bluejay query.
"""

import time

import cv2
import imageio.v3 as iio
import numpy as np
from safetensors.torch import load, save

cv2.setNumThreads(1)


def read_blob(info):
    """Query. The bytes of one stored Blob (blob_info: uri, position, size)."""
    with open(info["uri"], "rb") as f:
        f.seek(info["position"])
        return f.read(info["size"])


def read_png_rgb(info):
    """Query. One stored PNG as uint8 (H, W, 3) RGB."""
    image = cv2.imdecode(np.frombuffer(read_blob(info), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode the PNG at {info}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mp4_rgb(info):
    """Query. One stored MP4 as uint8 (T, H, W, 3) RGB, every frame."""
    return iio.imread(read_blob(info), extension=".mp4", plugin="pyav")


def tensors_bytes(tensors, **metadata):
    """Pure function. safetensors bytes of a dict of CPU tensors (contiguous), with string metadata."""
    return save({k: v.contiguous() for k, v in tensors.items()}, {k: str(v) for k, v in metadata.items()})


def bytes_tensors(data):
    """Pure function. The dict of tensors in safetensors bytes."""
    return load(data)


def shard_sql(key_sql, worker, workers):
    """
    Pure function. SQL that keeps the rows of one worker: the 2nd hex digit of sha256(key) mod workers.

    Examples:
        >>> shard_sql("page_id", 1, 8)
        "contains('19', substr(encode(sha256(page_id), 'hex'), 2, 1))"
    """
    digits = "".join(f"{d:x}" for d in range(16) if d % workers == worker)
    return f"contains('{digits}', substr(encode(sha256({key_sql}), 'hex'), 2, 1))"


def shard_range(lo, hi, worker=0, workers=1):
    """
    Pure function. The byte shards (0-255) of one worker: shards lo..hi (inclusive), dealt out to workers in turn.

    Examples:
        >>> shard_range(0, 47, 1, 4)[:3], len(shard_range(0, 47, 1, 4)), len(shard_range(48, 255, 0, 16))
        ([1, 5, 9], 12, 13)
    """
    return list(range(lo, hi + 1))[worker::workers]


def byte_shard_sql(key_sql, shards):
    """
    Pure function. SQL that keeps the rows whose LAST byte of sha256(key), 0-255, is in shards (256-shard mode, so
    workers on several nodes can split the keys; shard_sql allows only 16 workers). The last byte, because the MAGICK
    rows were chosen as the smallest sha256(page_id), so their first bytes are all small.

    Examples:
        >>> byte_shard_sql("page_id", [0, 171])
        "substr(encode(sha256(page_id), 'hex'), 63, 2) IN ('00', 'ab')"
    """
    if not shards:
        raise ValueError("byte_shard_sql needs at least one shard")
    prefixes = ", ".join("'%02x'" % s for s in shards)
    return f"substr(encode(sha256({key_sql}), 'hex'), 63, 2) IN ({prefixes})"


def round_robin(db, tasks, poll_seconds=0):
    """
    Command. Each round runs every task's query once, in order, and passes its rows to the task's handle (which commits
    them). Ends when a whole round finds no rows; with poll_seconds > 0 only after two empty rounds that far apart
    (other processors still add rows).

    Args:
        db: A Bluejay database
        tasks (list[tuple[str, callable, str]]): (sql, handle(rows), label) per task, highest priority first
        poll_seconds (float): Wait after an empty round

    Returns:
        dict[str, int]: Rows handled per label
    """
    done, start, empty = {label: 0 for _, _, label in tasks}, time.time(), 0
    while True:
        found = False
        for sql, handle, label in tasks:
            rows = db.query(sql).to_pylist()
            if rows:
                found = True
                handle(rows)
                done[label] += len(rows)
                print(f"{time.strftime('%I:%M:%S %p')} {label}: +{len(rows)} ({done[label]} done, {done[label] / (time.time() - start):.3f} rows/s)", flush=True)
        if found:
            empty = 0
            continue
        empty += 1
        if poll_seconds == 0 or empty >= 2:
            return done
        time.sleep(poll_seconds)
