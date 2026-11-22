# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["av==19.0.2", "ffmpeg-binaries==1.1.0", "fire==0.7.1", "numpy==2.2.6", "opencv-python-headless==4.12.0.88", "pillow==12.0.0", "rp==0.1.1430"]
#
# [tool.uv.sources]
# rp = { index = "pypi-direct" }
# av = { index = "pypi-direct" }  # Its wheel bundles FFmpeg; the internal index may hold new releases back
#
# [[tool.uv.index]]
# name = "pypi-direct"
# url = "https://pypi.org/simple"
# explicit = true
# ///
"""
Cuts each Sample's Pexels video (input.json {pexels_id, source_path}) for the H3 Magic project (manifest §22, m00912, m00917).

clip.mp4: the first CLIP_FRAMES frames at FPS (H3 wants 17n+5 frames at 24 fps: 141 = 5.875 s), 512 x 512, no audio.
frame_<k>s.png: the source frames at k = 0..6 s. Frame k of either is the last source frame at or before its time (sample and hold).
Each kept frame is scaled to hold a 512 x 512 box (short side 512, aspect kept) at the size rp.resize_image_to_hold
gives, by area averaging (as rp's "auto" does when it shrinks), then cut by rp.crop_image_to_square (center).
The scaling runs in FFmpeg's swscale on the decoded YUV frame: 4x less CPU than converting the 4K frame to RGB first.
result.json: motion_512, the mean |frame k - frame k + MOTION_GAP| of the clip in 8-bit levels.
"""
import json
import subprocess
import time
from pathlib import Path

import av
import ffmpeg
import fire
import numpy as np
import rp
from PIL import Image
from _batch import process_samples

ffmpeg.add_to_path()
DECODE_THREADS = 4  # Per clip: 48 loops x 4 threads keep 96 cores busy without thrashing
WORKERS = 1  # One clip per run thread: a 4K decode already uses several cores; run_proc_h3magic.fish runs many loops
SIZE = 512
FPS = 24
CLIP_FRAMES = 141  # 17n+5, the longest within the first 6 s
STILL_SECONDS = range(7)  # 0, 1, ..., 6 s (m00917)
MOTION_GAP = 12  # Frames: 0.5 s
CRF = 12  # Near lossless H.264
ENCODE_THREADS = 2  # preset medium on all cores was the biggest CPU user of the cut (concerns.md)
PNG_COMPRESS_LEVEL = 3


def hold_times(times, targets):
    """
    Pure function. For each target time, the index of the last source frame at or before it (sample and hold).

    Args:
        times (list[float]): Source frame times, increasing
        targets (list[float]): Wanted times

    Returns:
        list[int]

    Examples:
        >>> hold_times([0.0, 0.04, 0.08, 0.12], [0.0, 1 / 24, 2 / 24, 0.1])  # a 25 fps source at 24 fps
        [0, 1, 2, 2]
    """
    times = np.asarray(times)
    return [int(np.searchsorted(times, t + 1e-9, side="right") - 1) for t in targets]


def hold_size(height, width):
    """
    Pure function. The (height, width) that rp.resize_image_to_hold(height=SIZE, width=SIZE) gives an image of this size.

    Examples:
        >>> hold_size(2160, 3840), hold_size(3840, 2160), hold_size(720, 1366)
        ((512, 911), (911, 512), (512, 972))
    """
    scale = rp.r._resize_image_to_hold_scale(height, width, SIZE, SIZE, True)
    return rp.r._size_to_height_width(scale, height, width)


def square_512(frame):
    """
    Query (decodes pixels of an av.VideoFrame). Scales the frame to hold_size by area averaging, then rp.crop_image_to_square.

    Args:
        frame (av.VideoFrame): A decoded frame of any size

    Returns:
        np.ndarray: (SIZE, SIZE, 3) uint8 RGB
    """
    height, width = hold_size(frame.height, frame.width)
    held = frame.reformat(width=width, height=height, format="rgb24", interpolation="AREA").to_ndarray()
    square = rp.crop_image_to_square(held)
    assert square.shape == (SIZE, SIZE, 3), (frame.height, frame.width, held.shape, square.shape)
    return square


def motion_score(frames, gap):
    """
    Pure function. Mean absolute difference, in 8-bit levels, between each frame and the frame `gap` later.

    Args:
        frames (np.ndarray): (T, H, W, 3) uint8
        gap (int): Frames apart

    Returns:
        float

    Examples:
        >>> f = np.zeros((3, 2, 2, 3), np.uint8); f[2] = 10
        >>> motion_score(f, 1)
        5.0
    """
    a = frames[:-gap].astype(np.int16)
    b = frames[gap:].astype(np.int16)
    return float(np.abs(a - b).mean())


def read_frames(path, last_time):
    """
    Query. Decodes the video's frames up to last_time s, keeping each av.VideoFrame and its time in seconds.

    Returns:
        tuple[list[float], list[av.VideoFrame]]
    """
    times, frames = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = DECODE_THREADS
        start = stream.start_time * stream.time_base if stream.start_time is not None else 0
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base - start)
            if t > last_time + 1e-6:
                break
            times.append(t)
            frames.append(frame)
    return times, frames


def encode_mp4(frames, path):
    """Command. Writes (T, 512, 512, 3) uint8 RGB frames as an H.264 mp4 at FPS, CRF, yuv420p, preset veryfast, no audio."""
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-threads", str(ENCODE_THREADS), "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, input=frames.tobytes(), check=True)


def process_sample(sample):
    """Command. Writes the Sample's clip.mp4, frame_<k>s.png and result.json."""
    began = time.time()
    fields = json.loads((sample / "input.json").read_text())
    times, raw = read_frames(fields["source_path"], max(STILL_SECONDS))
    clip_times = [k / FPS for k in range(CLIP_FRAMES)]
    clip_index = hold_times(times, clip_times)
    still_index = hold_times(times, list(STILL_SECONDS))
    assert min(clip_index + still_index) >= 0, f"no frame at time 0: first frame at {times[0] if times else None} s"
    assert times[-1] >= max(STILL_SECONDS) - 1 / FPS, f"video ends at {times[-1]:.3f} s, before {max(STILL_SECONDS)} s"
    square = {i: square_512(raw[i]) for i in sorted(set(clip_index + still_index))}
    clip = np.stack([square[i] for i in clip_index])
    encode_mp4(clip, sample / "clip.mp4")
    for k, i in zip(STILL_SECONDS, still_index):
        Image.fromarray(square[i]).save(sample / f"frame_{k}s.png", compress_level=PNG_COMPRESS_LEVEL)
    motion = motion_score(clip, MOTION_GAP)
    (sample / "result.json").write_text(json.dumps(dict(motion_512=motion)))
    print(f"{fields['pexels_id']}: {len(times)} source frames, motion {motion:.2f}, {time.time() - began:.1f} s", flush=True)


def main(batch):
    """
    Command. Cuts every Sample of batch.

    Args:
        batch (str): The $BATCH folder
    """
    process_samples(batch, process_sample, WORKERS)


if __name__ == "__main__":
    fire.Fire(main)
