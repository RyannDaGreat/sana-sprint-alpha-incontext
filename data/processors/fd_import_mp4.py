# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "opencv-python-headless"]
# ///
"""
Writes children/<clip_id>/{clip.mp4, fields.json}: the fd_renders rows of one method, from mp4 files that already exist
(the journal's journal/assets/h3magic_fastdec/<clip>_<tag>.mp4, made by agents/.scratchpad/fastdec_ga/make_videos.py). Round 0 migration.
CPU only. tag: "source" = the source frames, "full" = the original full H3 decode, "T10" / "T30" / "T100" = that tier's winner.
Per-clip scores vs the full decode come from the winner's scores.json rows (SSIM and LPIPS on every 4th frame).
"""
import json
from pathlib import Path

import cv2
import fire
from fd_lib import FD_GA, FD_ROOT, read_json, score_file

ASSETS = FD_ROOT / "journal/assets/h3magic_fastdec"
PAGE_CLIPS = ("8620887", "8106222", "6192775", "856662", "8513880", "11610252")  # the journal page's six clips
REFERENCE_TAGS = {"source": "source", "original_full_decode": "full"}  # method_id -> mp4 tag


def tag_of(method_id, winners):
    """
    Pure function. The mp4 tag of a method: "source", "full", or the tier of the winner named method_id.

    Examples:
        >>> tag_of("source", []), tag_of("original_full_decode", [])
        ('source', 'full')
        >>> tag_of("lead_a5_skip16", [{"name": "lead_a5_skip16", "tier": "T100"}])
        'T100'
    """
    if method_id in REFERENCE_TAGS:
        return REFERENCE_TAGS[method_id]
    return next(w["tier"] for w in winners if w["name"] == method_id)


def probe(path):
    """Query. (n_frames, fps) of an mp4 file, from its header."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), f"cannot open {path}"
    n_frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return n_frames, fps


def main(sample):
    """
    Command. Writes the children of SAMPLE for each page clip of the method in SAMPLE/input.json (method_id, model_dir). Fails when an mp4 is missing.

    Args:
        sample (str): The $SAMPLE folder
    """
    sample = Path(sample)
    method_id = json.loads((sample / "input.json").read_text())["method_id"]
    winners = read_json(FD_GA / "winners.json")
    tag = tag_of(method_id, winners)
    model_dir = json.loads((sample / "input.json").read_text()).get("model_dir")
    per_clip = {}
    if model_dir and score_file(model_dir):
        per_clip = {r["clip"].removeprefix("val_"): r["vs_full"] for r in read_json(score_file(model_dir))["rows"]}
    for clip_id in PAGE_CLIPS:
        path = ASSETS / f"{clip_id}_{tag}.mp4"
        n_frames, fps = probe(path)
        vs_full = per_clip.get(clip_id, {})
        child = sample / "children" / clip_id
        child.mkdir(parents=True)
        (child / "clip.mp4").write_bytes(path.read_bytes())
        (child / "fields.json").write_text(json.dumps({
            "method_id": method_id, "clip_id": clip_id, "n_frames": n_frames, "fps": fps,
            "psnr_vs_full": vs_full.get("psnr"), "ssim_vs_full": vs_full.get("ssim"), "lpips_vs_full": vs_full.get("lpips")}))
        print(f"{method_id} {clip_id}: {n_frames} frames at {fps} fps from {path.name}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
