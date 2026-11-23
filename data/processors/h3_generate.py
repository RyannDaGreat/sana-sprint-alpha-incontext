"""
Eval inference of the H3 Magic project (manifest §22), all through Bluejay so Sparrow shows every result (m00912).

    plan                          (CPU) h3_eval_prompts rows from the prompt-set files (PROMPT_SETS), and one
                                  h3_eval_generations row per (h3_eval_models row, prompt of its prompt_set) that has none
    work WORKER WORKERS           long-lived GPU worker: H3 loads ONCE (~77 GB), then generates this worker's rows, model
                                  by model; switching models only re-weights the LoRA adapters (h3magic/h3.py set_adapters)

A model is W = W_H3 + a·Δ_pexels + b·Δ_imagegen + c·Δ_graymagic (h3_eval_models a_pexels, b_imagegen, c_graymagic), sampled
with classifier-free guidance scale `cfg` over the null prompt (NULL or 1 = H3's released one-pass sampler; > 1 = two
forwards per step, h3magic/h3.py allow_cfg: plain flow-matching LoRAs de-distill H3, so adapter models need it).
mode video: 141 frames (5.875 s, 24 fps) 512² (or h3_eval_models.size², 768 for Alpha Video v2) -> video (H.264 CRF 12 MP4); mode image: 1 frame -> image (PNG).
A row with split_adapter (the split adapter, h3magic/split.py, m00972) adds the split LoRA at weight γ = g_split and the
gray/white/black head with Δ·γ: video/image stay the gray head (the pipeline's own output), plus the white and black
heads, and the RGBA of the three by the iterative least squares of manifest §17 (sana_alpha.fibo2.rgba_from_composites).
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
# prompt_set of h3_eval_prompts / h3_eval_models (NULL = main): fg = the same 20 subjects with no background described (m00972)
PROMPT_SETS = {"main": EVAL_PROMPTS, "fg": ROOT / "h3magic/eval_prompts_fg.json",
               # Alpha Video v2 (manifest §23): the fg (g01-g20) and main (m01-m20) prompts + the gray suffix
               "fgray": ROOT / "h3magic/eval_prompts_fgray.json", "mgray": ROOT / "h3magic/eval_prompts_mgray.json",
               "vgray": ROOT / "h3magic/eval_prompts_vgray.json",  # 4 fgray prompts (v02, v08, v10, v17): quick video tastings
               "fgrid": ROOT / "h3magic/eval_prompts_fgrid.json"}  # q01-q20: f01-f20 + the 2x2 grid suffix (split v3-grid)
VIDEO_CRF = 12  # Near-lossless H.264, as h3_cut_clip's clips
ADAPTERS = (("pexels", "pexels_adapter", "a_pexels"), ("imagegen", "imagegen_adapter", "b_imagegen"), ("graymagic", "graymagic_adapter", "c_graymagic"),
            ("split", "split_adapter", "g_split"))  # the split adapter's LoRA at γ (its head: h3.set_splitter)
ILS_PROCESSES = 8  # Processes per worker for the RGBA fit: 1.1 s per 512² frame in one process, 141 frames per video
ILS_FRAMES_PER_TASK = 8  # Frames per fit task
CHECKER_SQUARE = 16  # Pixels per checkerboard square behind the RGBA video
CHECKER_LEVELS = (255, 204)  # White and light gray squares (the usual transparency checkerboard)


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


def planned_generations(models, prompts, done):
    """
    Pure function. The missing h3_eval_generations rows: one per (model, prompt) of the same prompt_set (NULL = main)
    that is not in `done`.

    Examples:
        >>> models = [{"model_tag": "stock_video", "prompt_set": None}, {"model_tag": "stock_video_fg", "prompt_set": "fg"}]
        >>> prompts = [{"prompt_id": "p01", "seed": 1001, "prompt_set": None}, {"prompt_id": "p02", "seed": 1002, "prompt_set": "main"},
        ...            {"prompt_id": "f01", "seed": 1001, "prompt_set": "fg"}]
        >>> planned_generations(models, prompts, done={("stock_video", "p01")})
        [{'model_tag': 'stock_video', 'prompt_id': 'p02', 'seed': 1002}, {'model_tag': 'stock_video_fg', 'prompt_id': 'f01', 'seed': 1001}]
    """
    set_of = lambda row: row["prompt_set"] or "main"
    return [{"model_tag": m["model_tag"], "prompt_id": p["prompt_id"], "seed": p["seed"]} for m in models for p in prompts
            if set_of(m) == set_of(p) and (m["model_tag"], p["prompt_id"]) not in done]


def png_rgba_bytes(rgba):
    """Pure function. uint8 (H, W, 4) RGBA (straight alpha) as PNG bytes."""
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError("PNG encode failed")
    return data.tobytes()


def checkerboard(height, width, square=CHECKER_SQUARE, levels=CHECKER_LEVELS):
    """
    Pure function. A transparency checkerboard, uint8 (H, W, 3).

    Examples:
        >>> checkerboard(2, 4, square=2)[..., 0].tolist()
        [[255, 255, 204, 204], [255, 255, 204, 204]]
    """
    import numpy as np
    y, x = np.mgrid[:height, :width]
    return np.repeat(np.where((y // square + x // square) % 2 == 0, levels[0], levels[1]).astype(np.uint8)[..., None], 3, -1)


def over_checkerboard(rgba):
    """
    Pure function. Straight-alpha RGBA frames over the checkerboard: float (T, H, W, 4) in [0, 1] -> uint8 (T, H, W, 3).

    Examples:
        >>> import numpy as np
        >>> over_checkerboard(np.array([[[[1.0, 0, 0, 1.0], [1.0, 0, 0, 0.0]]]]), )[0, 0].tolist()
        [[255, 0, 0], [255, 255, 255]]
    """
    import numpy as np
    board = checkerboard(*rgba.shape[1:3]).astype(np.float32) / 255
    alpha = rgba[..., 3:]
    return np.rint((rgba[..., :3] * alpha + board * (1 - alpha)) * 255).astype(np.uint8)


def fit_rgba(composites):
    """
    Near-pure function (adds src/ to sys.path to import it). sana_alpha.fibo2.rgba_from_composites frame by frame: (K, T, H, W, 3) uint8 heads in
    SPLIT_BACKGROUNDS order -> RGBA (T, H, W, 4) float32 straight alpha, uncertainty (T, H, W) float32.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from sana_alpha.fibo2 import rgba_from_composites
    from h3magic.split import SPLIT_BACKGROUNDS
    fits = [rgba_from_composites(composites[:, t].astype(np.float64) / 255, SPLIT_BACKGROUNDS.values()) for t in range(composites.shape[1])]
    return np.stack([f[0] for f in fits]), np.stack([f[1] for f in fits])


def split_outputs(heads, mode, pool):
    """
    Query (runs the fit in pool, ffmpeg for videos). The h3_eval_generations columns of one split generation.

    Args:
        heads (dict): h3.generate_split's gray, gray_x0, white, black, each uint8 (T, H, W, 3)
        mode (str): video | image
        pool (Executor): processes for fit_rgba

    Returns:
        dict: {mode}_white, {mode}_black, {mode}_alpha (gray), {mode}_rgba (image: RGBA PNG; video: over the checkerboard),
        split_residual (mean RMS residual of the fit), split_x0_psnr (head 0 through the x0 path vs the pipeline's gray, dB)
    """
    import numpy as np
    from h3magic.split import SPLIT_BACKGROUNDS
    # The fit takes all K heads through the same x0 path (gray_x0, not the pipeline's gray: 52 dB apart, bf16 state rounding)
    stack = np.stack([heads["gray_x0" if name == "gray" else name] for name in SPLIT_BACKGROUNDS])  # (K, T, H, W, 3)
    chunks = [stack[:, t:t + ILS_FRAMES_PER_TASK] for t in range(0, stack.shape[1], ILS_FRAMES_PER_TASK)]
    fits = list(pool.map(fit_rgba, chunks))
    rgba, residual = np.concatenate([f[0] for f in fits]), np.concatenate([f[1] for f in fits])
    alpha8 = np.repeat(np.rint(rgba[..., 3:] * 255).astype(np.uint8), 3, -1)
    same = np.array_equal(heads["gray_x0"], heads["gray"])
    out = dict(split_residual=float(residual.mean()), split_x0_psnr=float("inf") if same else h3.psnr(heads["gray_x0"], heads["gray"]))
    if mode == "video":
        return {**out, "video_white": mp4_bytes(heads["white"]), "video_black": mp4_bytes(heads["black"]),
                "video_alpha": mp4_bytes(alpha8), "video_rgba": mp4_bytes(over_checkerboard(rgba))}
    return {**out, "image_white": png_bytes(heads["white"][0]), "image_black": png_bytes(heads["black"][0]),
            "image_alpha": png_bytes(alpha8[0]), "image_rgba": png_rgba_bytes(np.rint(rgba[0] * 255).astype(np.uint8))}


def grid_outputs(frames, mode, pool):
    """
    Query (runs the fit in pool, ffmpeg for videos). The h3_eval_generations columns of one grid generation
    (h3magic/grid.py): white / black = the top-right / bottom-left panels, alpha = the 4th panel (the mask), alpha_ls = the
    least-squares alpha of the three background panels, rgba = colour decontaminated with the panel alpha + the panel alpha,
    split_residual = the fit's mean residual, grid_dalpha_* = |panel alpha − least-squares alpha| statistics.

    Args:
        frames (np.ndarray): (T, 768, 768, 3) uint8 grid frames
        mode (str): video | image
        pool (Executor): processes for fit_rgba
    """
    import numpy as np
    from h3magic import grid
    composites, alpha = grid.panels(frames)  # (3, T, P, P, 3) uint8, (T, P, P)
    chunks = [composites[:, t:t + ILS_FRAMES_PER_TASK] for t in range(0, composites.shape[1], ILS_FRAMES_PER_TASK)]
    fits = list(pool.map(fit_rgba, chunks))
    alpha_ls, residual = np.concatenate([f[0] for f in fits])[..., 3], np.concatenate([f[1] for f in fits])
    color = grid.decontaminate(composites.astype(np.float32) / 255, alpha)
    rgba = np.concatenate([color, alpha[..., None]], -1)  # (T, P, P, 4)
    gray8 = lambda a: np.repeat(np.rint(a[..., None] * 255).astype(np.uint8), 3, -1)
    d = grid.disagreement(alpha, alpha_ls)
    out = dict(split_residual=float(residual.mean()), **{f"grid_dalpha_{k}": v for k, v in d.items() if k != "edge_share"})
    if mode == "video":
        return {**out, "video_white": mp4_bytes(composites[1]), "video_black": mp4_bytes(composites[2]), "video_alpha": mp4_bytes(gray8(alpha)),
                "video_alpha_ls": mp4_bytes(gray8(alpha_ls)), "video_rgba": mp4_bytes(over_checkerboard(rgba))}
    return {**out, "image_white": png_bytes(composites[1][0]), "image_black": png_bytes(composites[2][0]), "image_alpha": png_bytes(gray8(alpha)[0]),
            "image_alpha_ls": png_bytes(gray8(alpha_ls)[0]), "image_rgba": png_rgba_bytes(np.rint(rgba[0] * 255).astype(np.uint8))}


def plan():
    """Command. Adds the h3_eval_prompts rows of every prompt set and every missing h3_eval_generations row. Idempotent."""
    from bluejay import open_database

    db = open_database("project.dtab")
    have = {r["prompt_id"] for r in db.query("SELECT prompt_id FROM h3_eval_prompts").to_pylist()}
    new = [{"prompt_id": p["id"], "short": p["short"], "prompt": p["prompt"], "seed": p["seed"], "prompt_set": name}
           for name, path in PROMPT_SETS.items() for p in json.loads(path.read_text()) if p["id"] not in have]
    if new:
        db.commit("h3_eval_prompts", ["short", "prompt", "seed", "prompt_set"], new, insert_missing=True)
    models = db.query("SELECT model_tag, prompt_set FROM h3_eval_models").to_pylist()
    prompts = db.query("SELECT prompt_id, seed, prompt_set FROM h3_eval_prompts").to_pylist()
    done = {(r["model_tag"], r["prompt_id"]) for r in db.query("SELECT model_tag, prompt_id FROM h3_eval_generations").to_pylist()}
    rows = planned_generations(models, prompts, done)
    if rows:
        db.commit("h3_eval_generations", ["seed"], rows, insert_missing=True)
    print(f"plan: {len(new)} new prompts, {len(rows)} new generations ({len(models)} models, {len(prompts)} prompts)", flush=True)


def work_sql(worker, workers, only="%"):
    """
    Pure function. The query of this worker's next generations: rows with no output whose prompt has an embedding,
    one model at a time; worker k takes the prompts whose number is k mod workers (an even split of 20 prompts).

    Examples:
        >>> "% 2 = 1" in work_sql(1, 2)
        True
        >>> "g.model_tag LIKE 'pexels_video_v2'" in work_sql(0, 7, only="pexels_video_v2")  # one model only (a second node's workers)
        True
    """
    return ("SELECT g.model_tag, g.prompt_id, g.seed, m.mode, m.steps, m.a_pexels, m.b_imagegen, m.c_graymagic, "
            "m.pexels_adapter, m.imagegen_adapter, m.graymagic_adapter, m.cfg, m.split_adapter, m.g_split, m.size, m.layout, "
            "blob_info(p.h3_text_embed) AS embed "
            "FROM h3_eval_generations g JOIN h3_eval_models m ON m.model_tag = g.model_tag "
            "JOIN h3_eval_prompts p ON p.prompt_id = g.prompt_id "
            "WHERE g.video IS NULL AND g.image IS NULL AND p.h3_text_embed IS NOT NULL "
            f"AND CAST(substr(g.prompt_id, 2) AS INT) % {workers} = {worker} AND g.model_tag LIKE '{only}' "
            "ORDER BY g.model_tag, g.prompt_id LIMIT 1")


def adapter_mix(row):
    """
    Pure function. The h3.set_adapters argument of one h3_eval_models row: every adapter with a path.

    Examples:
        >>> adapter_mix({"pexels_adapter": "models/h3magic/pexels/step_000400", "a_pexels": 1.0, "imagegen_adapter": None,
        ...              "b_imagegen": 0.0, "graymagic_adapter": None, "c_graymagic": 0.0, "split_adapter": None, "g_split": None})
        {'pexels': ('models/h3magic/pexels/step_000400', 1.0)}
    """
    return {name: (row[path], float(row[weight] or 0)) for name, path, weight in ADAPTERS if row[path]}


def guidance(row):
    """
    Pure function. The CFG scale of one h3_eval_models row: its cfg, or 1 (H3's released sampler) when NULL.

    Examples:
        >>> guidance({"cfg": None}), guidance({"cfg": 2.5})
        (1.0, 2.5)
    """
    return float(row["cfg"]) if row["cfg"] is not None else 1.0


def work(worker, workers, poll_seconds=60, only="%"):
    """
    Command. Loads H3 once on this GPU and generates this worker's h3_eval_generations rows until none is left twice
    `poll_seconds` apart; commits each result (video or image, seconds, gpu) at once. `only` (SQL LIKE on model_tag) limits it
    to some models, so a second node can take one model without sharing prompt slices with the first node's workers.
    """
    from bluejay import open_database

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    db = open_database("project.dtab")
    start = time.time()
    pipe = h3.load_pipeline()
    pool = ProcessPoolExecutor(ILS_PROCESSES, mp_context=multiprocessing.get_context("spawn"))  # numpy only: never fork a CUDA process
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"worker {worker}: H3 loaded on GPU {gpu} in {time.time() - start:.0f} s", flush=True)

    def handle(rows):
        for row in rows:
            h3.set_adapters(pipe, adapter_mix(row))
            h3.set_splitter(pipe, row["split_adapter"], float(row["g_split"] or 0)) if row["split_adapter"] else h3.set_splitter(pipe, None)
            text = bytes_tensors(read_blob(row["embed"]))
            t = time.time()
            sample = h3.generate_split if row["split_adapter"] else h3.generate
            result = sample(pipe, text, row["seed"], frames=h3.VIDEO_FRAMES if row["mode"] == "video" else 1, steps=row["steps"],
                            size=int(row["size"] or h3.SIZE), cfg=guidance(row))
            seconds = time.time() - t
            frames = result["gray"] if row["split_adapter"] else result
            output = {"video": mp4_bytes(frames)} if row["mode"] == "video" else {"image": png_bytes(frames[0])}
            if row["split_adapter"]:
                output.update(split_outputs(result, row["mode"], pool))
            elif row["layout"] == "grid2x2":
                output.update(grid_outputs(frames, row["mode"], pool))
            db.commit("h3_eval_generations", [*output, "seconds", "gpu"],
                      [{"model_tag": row["model_tag"], "prompt_id": row["prompt_id"], **output, "seconds": seconds, "gpu": gpu}])
            print(f"worker {worker}: {row['model_tag']} {row['prompt_id']} {seconds:.1f} s", flush=True)

    with torch.inference_mode():
        round_robin(db, [(work_sql(worker, workers, only), handle, f"generate worker {worker}")], poll_seconds)
    print(f"worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire({"plan": plan, "work": work})
