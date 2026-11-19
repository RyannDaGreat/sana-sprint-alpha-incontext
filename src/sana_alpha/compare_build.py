"""
Builds the data of the SANA vs FIBO2 compare reports (manifest §15) from BlueJay's eval_generations.

    PYTHONPATH=src uv run python -m sana_alpha.compare_build build
    PYTHONPATH=src uv run python -m sana_alpha.compare_build methods     # COMPARE_METHODS, comma-separated (for launch scripts)

Writes journal/compare/data/compare.json (items, every evaluated (run, step, method, resolution) with per-item
metrics, means and 95% bootstrap intervals) and journal/compare/data/loss_<run>.json (from the training logs), and
hard-links every eval image into journal/assets/compare/ (git ignores it; publish_pages puts the original PNGs on
journal-assets). Paths in the JSON are relative to journal/ so publish_pages rewrites them. Every report cell shows the
original image: no atlases, thumbnails or downscaled copies (user rule m00705).
CPU only (a few threads): it reads the stored images, it never generates any.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
from PIL import Image

from sana_alpha.compare_metrics import item_metrics, paired_bootstrap_ci
from sana_alpha.gallery import LANCE, trimask_set
from sana_alpha.journal_assets import link
from sana_alpha.trimask import METHODS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "journal/compare/data"
ASSETS = "assets/compare"  # relative to journal/
LOGS = ROOT / "agents/.scratchpad/logs"
COMPARE_METHODS = ("morph_band", "scribbles", "bbox_unknown", "partial_view", "combo_chain", "glass_ripple")  # the cheap eval tier: one band, stroke, box, crop, combination and warp (manifest §15)
RUNS = {"tri_scratch": dict(family="sana", label="SANA-Sprint 1.6B", log="train_control_tri_scratch.log", global_batch=128, rest="sprint_lam0_2step"),  # 16 x 8 GPUs
        "fibo2_tri": dict(family="fibo2", label="FIBO2 8.1B", log="train_fibo2_fibo2_tri.log", global_batch=28, rest="fibo2_30step_full")}  # 4 x 7 GPUs; every item, paired 1:1 with SANA (m00713)
# rest: the one model variant of each run in the reports (its shipped model on the full eval sets); other tags of the run (held-out subsets, 4-step tests) stay out
LOSS_WINDOW = 50  # training-log steps averaged into one loss point (the log has one line per 10 steps)
THREADS = 4  # the GPUs' encode and training jobs need the CPUs
DIGITS = 4
TAG = re.compile(r"^tri_(?P<method>%s)_(?P<run>.+?)_step(?P<step>\d{6})_(?P<rest>.+?)(?:_r(?P<res>\d+))?$"
                 % "|".join(sorted(METHODS, key=len, reverse=True)))
LOG_LINE = re.compile(r"step (\d+) loss ([\d.]+)")


def parse_tag(tag):
    """
    Pure function. The parts of a trimask eval model_tag (gallery.trimask_tag), or None for other tags.

    Examples:
        >>> parse_tag("tri_morph_band_tri_scratch_step003000_sprint_lam0_2step_r768")
        {'method': 'morph_band', 'run': 'tri_scratch', 'step': 3000, 'rest': 'sprint_lam0_2step', 'res': 768}
        >>> parse_tag("tri_op_chain_tri_scratch_step015500_sprint_lam0_2step")["res"], parse_tag("show_v2_step006000_lam0")
        (1024, None)
    """
    match = TAG.match(tag)
    if not match:
        return None
    parts = match.groupdict()
    return dict(parts, step=int(parts["step"]), res=int(parts["res"] or 1024))


def loss_curve(lines, window=LOSS_WINDOW):
    """
    Pure function. Training-log loss averaged over windows of `window` steps: [[window end step, mean loss], ...].

    Examples:
        >>> loss_curve(["[10:01] step 10 loss 0.5 grad 1", "noise", "[10:02] step 20 loss 0.3 grad 1",
        ...             "[10:03] step 60 loss 0.2 grad 1"], window=50)
        [[50, 0.4], [100, 0.2]]
    """
    points = [(int(s), float(v)) for s, v in (LOG_LINE.search(l).groups() for l in lines if LOG_LINE.search(l))]
    groups = {}
    for step, value in points:
        groups.setdefault(-(-step // window) * window, []).append(value)
    return [[end, round(float(np.mean(v)), DIGITS)] for end, v in sorted(groups.items())]


def summarize(per_item, held):
    """
    Near-pure function (seeded bootstrap). Mean and 95% interval of every metric, over all items and over held-out ones.

    Args:
        per_item (dict[str, list[float]]): metric -> values in item order
        held (list[bool]): item is held-out

    Returns:
        dict: metric -> {"all": [mean, low, high], "held": [mean, low, high]}; None for a split with no finished item

    Examples:
        >>> summarize({"band_iou": [0.5, 0.7, 0.9]}, [True, True, False])["band_iou"]["held"]
        [0.6, 0.5, 0.7]
        >>> summarize({"band_iou": [np.nan, np.nan, 0.9]}, [True, True, False])["band_iou"]  # held-out items still rendering
        {'all': [0.9, 0.9, 0.9], 'held': None}
    """
    pick = lambda values, keep: [v for v, k in zip(values, keep) if k]
    ci = lambda values: [round(v, DIGITS) for v in paired_bootstrap_ci(values)] if not np.isnan(values).all() else None
    return {m: {"all": ci(v), "held": ci(pick(v, held))} for m, v in per_item.items()}


def evaluate(tag, parts, blobs, items):
    """
    Command (hard-links images). Metrics and image paths of one eval tag.

    Args:
        tag (str): model_tag
        parts (dict): parse_tag(tag)
        blobs (dict[str, str]): page_id -> blob file relative to the eval_generations dataset (items still rendering are missing)
        items (list[dict]): showcase items in report order

    Returns:
        dict: the eval's record in compare.json (an item still rendering: image null, metrics null)
    """
    read = lambda path: np.asarray(Image.open(path), np.float32) / 255
    sets = ROOT / trimask_set(parts["res"]) / parts["method"]
    images = []
    for item in items:
        image = ROOT / "journal" / ASSETS / "img" / tag / f"{item['page_id']}.png" if item["page_id"] in blobs else None
        if image is not None:
            link(LANCE / blobs[item["page_id"]], image)
        images.append(image)
    scores = [item_metrics(read(image), read(sets / "mattes" / f"{i['page_id']}.png"), read(sets / "originals" / f"{i['page_id']}.png"))
              if image is not None else None for image, i in zip(images, items)]
    names = next(s for s in scores if s is not None)
    per_item = {m: [round(s[m], DIGITS) if s is not None and s[m] == s[m] else None for s in scores] for m in names}
    run = RUNS[parts["run"]]
    return dict(tag=tag, **parts, model=run["family"], samples=parts["step"] * run["global_batch"],
                images=[None if p is None else str(p.relative_to(ROOT / "journal")) for p in images],
                per_item=per_item,
                summary=summarize({m: [np.nan if v is None else v for v in vs] for m, vs in per_item.items()},
                                  [i["split"] == "held-out" for i in items]))


def condition_images(items, folder, kind, name):
    """Command. Links one condition set (trimasks, true mattes or references) into journal/<ASSETS>/cond/<name>/; returns their paths."""
    images = []
    for item in items:
        image = ROOT / "journal" / ASSETS / "cond" / name / f"{item['page_id']}.png"
        link(ROOT / folder / kind / f"{item['page_id']}.png", image)
        images.append(image)
    return dict(images=[str(p.relative_to(ROOT / "journal")) for p in images])


def conditions(items, methods, resolutions):
    """Command. condition_images of the trimasks (per method and resolution), true mattes and MAGICK references (per resolution)."""
    return dict(trimask={f"{m}@{r}": condition_images(items, f"{trimask_set(r)}/{m}", "mattes", f"trimask_{m}_{r}") for m in methods for r in resolutions},
                matte={str(r): condition_images(items, f"{trimask_set(r)}/{METHODS[0]}", "originals", f"matte_{r}") for r in resolutions},
                reference={str(r): condition_images(items, f"{trimask_set(r)}/{METHODS[0]}", "references", f"reference_{r}") for r in resolutions})


def build():
    """Command. Writes journal/compare/data/compare.json and loss_<run>.json and the image links (see the module docstring)."""
    import lance

    table = lance.dataset(str(LANCE)).to_table(columns=["model_tag", "page_id", "image"], filter="image IS NOT NULL").to_pandas()
    items = json.loads((ROOT / trimask_set() / METHODS[0] / "prompts.json").read_text())
    items = sorted(items, key=lambda i: (i["split"] != "held-out", i["page_id"]))  # held-out first, then a fixed order
    tags = {}
    for tag, page_id, image in zip(table.model_tag, table.page_id, table.image):
        parts = parse_tag(tag)
        if parts and parts["run"] in RUNS and parts["rest"] == RUNS[parts["run"]]["rest"]:
            assert image["position"] == 0, f"{tag} {page_id}: blob is packed in {image['blob_uri']}, not a file of its own"
            tags.setdefault(tag, (parts, {}))[1][page_id] = image["blob_uri"]
    every = {i["page_id"] for i in items}
    print(f"{len(tags)} evals, {sum(every <= set(v[1]) for v in tags.values())} complete; partial ones are shown with their missing items as rendering", flush=True)
    with ThreadPoolExecutor(THREADS) as pool:
        evals = list(pool.map(lambda kv: dict(evaluate(kv[0], *kv[1], items), complete=every <= set(kv[1][1])), sorted(tags.items())))
    methods = sorted({e["method"] for e in evals}, key=METHODS.index)
    resolutions = sorted({e["res"] for e in evals}, reverse=True)
    DATA.mkdir(parents=True, exist_ok=True)
    report = dict(items=[dict(page_id=i["page_id"], split=i["split"], caption=i["caption"]) for i in items],
                  runs=RUNS, methods=methods, resolutions=resolutions, evals=evals,
                  conditions=conditions(items, methods, resolutions))
    (DATA / "compare.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    for run, spec in RUNS.items():
        (DATA / f"loss_{run}.json").write_text(json.dumps(loss_curve((LOGS / spec["log"]).read_text().splitlines())) + "\n")
    print(f"wrote {DATA}: {len(evals)} evals, {len(items)} items, methods {methods}, resolutions {resolutions}", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(build=build, methods=lambda: ",".join(COMPARE_METHODS)))
