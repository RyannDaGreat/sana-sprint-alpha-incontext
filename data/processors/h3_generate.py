"""
Eval inference of the H3 Magic project (manifest §22), all through Bluejay so Sparrow shows every result (m00912).

    plan                          (CPU) h3_eval_prompts rows from h3magic/eval_prompts.json, and one h3_eval_generations
                                  row per (h3_eval_models row, prompt) that has none
    work WORKER WORKERS           long-lived GPU worker: H3 loads ONCE (~77 GB), then generates this worker's rows, model
                                  by model; switching models only re-weights the LoRA adapters (h3magic/h3.py set_adapters)

A model is W = W_H3 + a·Δ_pexels + b·Δ_imagegen + c·Δ_graymagic (h3_eval_models a_pexels, b_imagegen, c_graymagic).
mode video: 141 frames (5.875 s, 24 fps) 512² -> video (H.264 CRF 12 MP4); mode image: 1 frame -> image (PNG).
Measured on one A100 (stock H3, 512²): video 100.9 s at 30 steps (30.4 s at 8); image 5.2 s at 30 steps.

    h3_generate.py plan | work WORKER WORKERS [--poll_seconds 60]   (from data/; GPU = $CUDA_VISIBLE_DEVICES)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import fire
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import bytes_tensors, read_blob, round_robin
from h3magic import h3

ROOT = Path(__file__).resolve().parents[2]
EVAL_PROMPTS = ROOT / "h3magic/eval_prompts.json"
VIDEO_CRF = 12  # Near-lossless H.264, as h3_cut_clip's clips
ADAPTERS = (("pexels", "pexels_adapter", "a_pexels"), ("imagegen", "imagegen_adapter", "b_imagegen"), ("graymagic", "graymagic_adapter", "c_graymagic"))


def mp4_bytes(frames, fps=h3.FPS, crf=VIDEO_CRF):
    """Query (runs ffmpeg). uint8 (T, H, W, 3) RGB frames as H.264 MP4 bytes (yuv420p, no audio)."""
    import imageio_ffmpeg
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as out:
        t, h, w, _ = frames.shape
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", out.name], input=frames.tobytes(), check=True)
        return Path(out.name).read_bytes()


def png_bytes(image):
    """Pure function. uint8 (H, W, 3) RGB as PNG bytes."""
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encode failed")
    return data.tobytes()


def plan():
    """Command. Adds the h3_eval_prompts rows and every missing h3_eval_generations row. Idempotent."""
    from bluejay import open_database

    db = open_database("project.dtab")
    prompts = json.loads(EVAL_PROMPTS.read_text())
    have = {r["prompt_id"] for r in db.query("SELECT prompt_id FROM h3_eval_prompts").to_pylist()}
    new = [{"prompt_id": p["id"], "short": p["short"], "prompt": p["prompt"], "seed": p["seed"]} for p in prompts if p["id"] not in have]
    if new:
        db.commit("h3_eval_prompts", ["short", "prompt", "seed"], new, insert_missing=True)
    models = [r["model_tag"] for r in db.query("SELECT model_tag FROM h3_eval_models").to_pylist()]
    done = {(r["model_tag"], r["prompt_id"]) for r in db.query("SELECT model_tag, prompt_id FROM h3_eval_generations").to_pylist()}
    rows = [{"model_tag": m, "prompt_id": p["id"], "seed": p["seed"]} for m in models for p in prompts if (m, p["id"]) not in done]
    if rows:
        db.commit("h3_eval_generations", ["seed"], rows, insert_missing=True)
    print(f"plan: {len(new)} new prompts, {len(rows)} new generations ({len(models)} models x {len(prompts)} prompts)", flush=True)


def work_sql(worker, workers):
    """
    Pure function. The query of this worker's next generations: rows with no output whose prompt has an embedding,
    one model at a time; worker k takes the prompts whose number is k mod workers (an even split of 20 prompts).

    Examples:
        >>> "% 2 = 1" in work_sql(1, 2)
        True
    """
    return ("SELECT g.model_tag, g.prompt_id, g.seed, m.mode, m.steps, m.a_pexels, m.b_imagegen, m.c_graymagic, "
            "m.pexels_adapter, m.imagegen_adapter, m.graymagic_adapter, blob_info(p.h3_text_embed) AS embed "
            "FROM h3_eval_generations g JOIN h3_eval_models m ON m.model_tag = g.model_tag "
            "JOIN h3_eval_prompts p ON p.prompt_id = g.prompt_id "
            "WHERE g.video IS NULL AND g.image IS NULL AND p.h3_text_embed IS NOT NULL "
            f"AND CAST(substr(g.prompt_id, 2) AS INT) % {workers} = {worker} "
            "ORDER BY g.model_tag, g.prompt_id LIMIT 1")


def adapter_mix(row):
    """
    Pure function. The h3.set_adapters argument of one h3_eval_models row: every adapter with a path.

    Examples:
        >>> adapter_mix({"pexels_adapter": "models/h3magic/pexels/step_000400", "a_pexels": 1.0, "imagegen_adapter": None,
        ...              "b_imagegen": 0.0, "graymagic_adapter": None, "c_graymagic": 0.0})
        {'pexels': ('models/h3magic/pexels/step_000400', 1.0)}
    """
    return {name: (row[path], float(row[weight] or 0)) for name, path, weight in ADAPTERS if row[path]}


def work(worker, workers, poll_seconds=60):
    """
    Command. Loads H3 once on this GPU and generates this worker's h3_eval_generations rows until none is left twice
    `poll_seconds` apart; commits each result (video or image, seconds, gpu) at once.
    """
    from bluejay import open_database

    db = open_database("project.dtab")
    start = time.time()
    pipe = h3.load_pipeline()
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"worker {worker}: H3 loaded on GPU {gpu} in {time.time() - start:.0f} s", flush=True)

    def handle(rows):
        for row in rows:
            h3.set_adapters(pipe, adapter_mix(row))
            text = bytes_tensors(read_blob(row["embed"]))
            t = time.time()
            frames = h3.generate(pipe, text, row["seed"], frames=h3.VIDEO_FRAMES if row["mode"] == "video" else 1, steps=row["steps"])
            seconds = time.time() - t
            output = {"video": mp4_bytes(frames)} if row["mode"] == "video" else {"image": png_bytes(frames[0])}
            db.commit("h3_eval_generations", [*output, "seconds", "gpu"],
                      [{"model_tag": row["model_tag"], "prompt_id": row["prompt_id"], **output, "seconds": seconds, "gpu": gpu}])
            print(f"worker {worker}: {row['model_tag']} {row['prompt_id']} {seconds:.1f} s", flush=True)

    with torch.inference_mode():
        round_robin(db, [(work_sql(worker, workers), handle, f"generate worker {worker}")], poll_seconds)
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire({"plan": plan, "work": work})
