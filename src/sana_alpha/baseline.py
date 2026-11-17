"""
Step 1 baseline: stock Sprint student and teacher on the eval prompts, plain and with a gray-background suffix.

    PYTHONPATH=src uv run python -m sana_alpha.baseline run sprint --gpu 6
    PYTHONPATH=src uv run python -m sana_alpha.baseline run teacher --gpu 7
    PYTHONPATH=src uv run python -m sana_alpha.baseline report
"""

import json
from pathlib import Path

import fire
import numpy as np
from PIL import Image

from sana_alpha.eval_set import PROMPT_SUFFIXES as VARIANTS, load_eval_set
from sana_alpha.generate import generate
from sana_alpha.metrics import image_metrics, mean_metrics, montage
from sana_alpha.models import load_pipeline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "agents/.scratchpad/outputs/step1"  # full-resolution PNGs (scratch)
ASSET_DIR = ROOT / "journal/assets/step1"  # grids + metrics for the journal
WARMUP_ITEMS = 1  # first image of each run is excluded from timing (CUDA init, kernel autotune)
GRID_COLUMNS = 6
COMPARE_CELL_PX = 192


def run(kind, gpu=6):
    """
    Command. Generates all eval items for one model (both prompt variants); writes PNGs and per-image metrics json.

    Args:
        kind (str): "sprint" or "teacher"
        gpu (int): CUDA device index
    """
    data = load_eval_set()
    pipe, defaults = load_pipeline(kind, device=f"cuda:{gpu}")
    for variant, suffix in VARIANTS.items():
        name = f"{kind}_{variant}"
        prompts = [item["caption"] + suffix for item in data["items"]]
        images, seconds = generate(pipe, prompts, data["noise"], **defaults)
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
        records = []
        for item, prompt, image, matte, sec in zip(data["items"], prompts, images, data["mattes"], seconds):
            Image.fromarray(np.round(image * 255).astype(np.uint8)).save(OUTPUT_DIR / name / f"{item['page_id']}.png")
            records.append(dict(page_id=item["page_id"], prompt=prompt, seconds=sec, **image_metrics(image, matte)))
        timed = [r["seconds"] for r in records[WARMUP_ITEMS:]]
        summary = dict(config=name, call_kwargs=defaults, seconds_per_image=float(np.mean(timed)),
                       **mean_metrics([{k: v for k, v in r.items() if k not in ("page_id", "prompt", "seconds")} for r in records]))
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        (ASSET_DIR / f"{name}.json").write_text(json.dumps(dict(summary=summary, images=records), indent=1) + "\n")
        print(json.dumps(summary, indent=1))


def load_outputs(name, items):
    """
    Query. Reads one config's generated PNGs.

    Returns:
        np.ndarray: (N, 1024, 1024, 3) float32 RGB in [0,1]
    """
    return np.stack([np.asarray(Image.open(OUTPUT_DIR / name / f"{i['page_id']}.png"), np.float32) / 255 for i in items])


def save_jpeg(grid, path):
    """Command. Saves a (H, W, 3) uint8 grid as a quality-90 JPEG."""
    Image.fromarray(grid).save(path, quality=90)
    print(f"wrote {path} {grid.shape}")


def report():
    """
    Command. Builds the journal grids (one per config + a side-by-side comparison) and metrics.json from `run` outputs.
    """
    data = load_eval_set()
    items = data["items"]
    names = [f"{kind}_{variant}" for kind in ("sprint", "teacher") for variant in VARIANTS]
    outputs = {name: load_outputs(name, items) for name in names}
    reference = [image_metrics(r, m) for r, m in zip(data["references"], data["mattes"])]
    summaries = dict(reference=mean_metrics(reference))
    for name in names:
        summaries[name] = json.loads((ASSET_DIR / f"{name}.json").read_text())["summary"]
    (ASSET_DIR / "metrics.json").write_text(json.dumps(summaries, indent=1) + "\n")

    for name in names:
        cells = [(image, item["caption"]) for image, item in zip(outputs[name], items)]
        rows = [cells[i:i + GRID_COLUMNS] for i in range(0, len(cells), GRID_COLUMNS)]
        save_jpeg(montage(rows), ASSET_DIR / f"grid_{name}.jpg")
    rows = [[(m, item["page_id"]), (r, item["caption"])] + [(outputs[n][i], n) for n in names]
             for i, (item, m, r) in enumerate(zip(items, data["mattes"], data["references"]))]
    save_jpeg(montage(rows, size=COMPARE_CELL_PX), ASSET_DIR / "compare.jpg")


if __name__ == "__main__":
    fire.Fire(dict(run=run, report=report))
