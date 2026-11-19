"""
The showcase gallery of one control step: every resources/showcase item, two rows (MAGICK caption, detailed prompt),
one column per model: stock Sprint/teacher, the teacher λ sweep, and the Sprint λ sweep at 1 and 2 steps.
λ = fraction of the adapter delta kept (merge_final --adapter_scale): 1 = the control model, 0 = the final model.

    PYTHONPATH=src uv run python -m sana_alpha.gallery rows control_v2 1500 plain detailed   # dtab eval_models rows to paste
    PYTHONPATH=src uv run python -m sana_alpha.gallery build control_v2 1500 [--notes "<p>...</p>"]
The trimask gallery (manifest §13): full-subtraction models of several RUN:STEP on resources/showcase_trimask/<method>:
    PYTHONPATH=src uv run python -m sana_alpha.gallery trimask_rows control_v2:6000 control_tri_ft:6000
    PYTHONPATH=src uv run python -m sana_alpha.gallery build_trimask control_v2:6000 control_tri_ft:6000 [--notes ...]

build writes journal/gallery_<run without "control_">.html and hard-links every full-resolution image into
journal/assets/gallery_<...>/ (git ignores it; run build again to recreate the links).
"""

import html
import json
from pathlib import Path

import fire

import pandas as pd

from sana_alpha.journal_assets import label_of, link

ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = "resources/showcase"
TRIMASK_SET = "resources/showcase_trimask"  # one eval set per trimask method (eval_set.build_trimask)
HELD_SET = "resources/showcase_trimask_held"  # its held-out items only (eval_set.held_out), for FIBO2 (~1 min per image)
PREVIEW_SET = "resources/fibo2_preview"  # the 4 items train_fibo2 previews (eval_set.fibo2_preview), rendered at full resolution
FIBO2_STEPS, FIBO2_GUIDANCE = 30, 5.0  # BriaFibo2Pipeline's defaults (diffusers 0.43)
LANCE = ROOT / "data/lance/eval_generations.lance"
TEACHER_LAMBDAS = (1, 0.75, 0.5, 0.25, 0)
SPRINT_LAMBDAS = (1, 0.75, 0.5, 0)
VARIANTS = dict(plain="MAGICK caption", detailed="detailed prompt")  # prompt_variant -> row label
STOCK = [("show_stock_sprint1", "sprint", "stock", 1, "stock Sprint, 1 step"),
         ("show_stock_sprint2", "sprint", "stock", 2, "stock Sprint, 2 steps"),
         ("show_stock_teacher20", "teacher", "stock", 20, "stock teacher, 20 steps")]


def step_models(run, step):
    """
    Pure function. The showcase models of one control step, as merge_final names its folders.

    Args:
        run (str): control run, e.g. "control_v1"
        step (int): its step

    Returns:
        list[tuple]: (model_tag, pipeline, checkpoint, steps, label), prompt variant "plain" (other variants add "_<variant>")

    Examples:
        >>> [m[:3] for m in step_models("control_v1", 1500)[:2]]
        [('show_v1_step001500_lam1', 'teacher', 'models/control/control_v1/step_001500'), ('show_v1_step001500_lam0.75', 'teacher', 'models/control/control_v1/final_step_001500_adapter0.75')]
        >>> [m[0] for m in step_models("control_v1", 1500)[-2:]]
        ['show_v1_step001500_sprint_lam0.5_1step', 'show_v1_step001500_sprint_lam0_1step']
    """
    folder, name = f"models/control/{run}", f"step_{int(step):06d}"
    tag = f"show_{run.removeprefix('control_')}_step{int(step):06d}"
    suffix = lambda lam: f"_adapter{lam:g}" if lam else ""  # merge_final's folder suffix
    teacher = [(f"{tag}_lam{lam:g}", "teacher", f"{folder}/{name}" if lam == 1 else f"{folder}/final_{name}{suffix(lam)}", 20,
                f"teacher, λ = {lam:g}" + (" (no subtraction)" if lam == 1 else " (full subtraction)" if lam == 0 else "")) for lam in TEACHER_LAMBDAS]
    sprint = [(f"{tag}_sprint_lam{lam:g}_{steps}step", "sprint", f"{folder}/sprint_{name}{suffix(lam)}", steps,
               f"Sprint {steps} step{'s' * (steps > 1)}, λ = {lam:g}" + (" (full subtraction)" if lam == 0 else "")) for steps in (2, 1) for lam in SPRINT_LAMBDAS]
    return teacher + sprint


def dtab_rows(models, variant, eval_set=SHOWCASE, guidance=4.5):
    """
    Pure function. eval_models initial_rows text for models on the showcase set: one DTAB line per row, each cell
    tab-padded to the column the dtab's rows use (tab = 4 columns: 16, 76, 96, 168, 196, 208, 224). Longer model tags
    (the trimask gallery's) move every later column right by the same whole number of tabs.

    Args:
        models (list[tuple]): from step_models or STOCK
        variant (str): prompt_variant
        eval_set (str): eval set folder relative to the repo root
        guidance (float): the rows' guidance (4.5: Sana's; FIBO2_GUIDANCE for FIBO2)

    Returns:
        str

    Examples:
        >>> line = dtab_rows(STOCK[:1], "detailed").expandtabs(4)
        >>> line.split()[1:4], [line.index(k) for k in ("model_tag", "pipeline", "checkpoint", "prompt_variant", "eval_set")]
        (['model_tag', 'show_stock_sprint1_detailed', 'pipeline'], [16, 76, 96, 168, 224])
        >>> line = dtab_rows([("tri_polygon_simplify_tri_scratch_step003000_sprint_lam0_2step_r768", "sprint", "x", 2, "")], "plain")
        >>> [line.expandtabs(4).index(k) for k in ("pipeline", "eval_set")]
        [96, 244]
    """
    keys = "model_tag pipeline checkpoint prompt_variant steps guidance eval_set".split()
    longest = max(len(variant_tag(m[0], variant)) for m in models) + len("model_tag ")
    shift = max(0, 16 + longest + 1 - 76 + 3) // 4 * 4  # the first column needs at least one tab after its cell
    stops = tuple(s + shift for s in (76, 96, 168, 196, 208, 224))
    pad = lambda text, start, stop: text + "\t" * -(-(stop - start - len(text)) // 4)  # start and stop are multiples of 4
    def line(values):
        cells = [f"{k} {v}" for k, v in zip(keys, values)]
        assert all(len(c) < b - a for c, a, b in zip(cells, (16,) + stops, stops)), f"a cell is too wide for its column: {cells}"
        return "\t\t\t,\t" + "".join(pad(c, a, b) for c, a, b in zip(cells, (16,) + stops, stops)) + cells[-1] + "\n"
    return "".join(line((variant_tag(tag, variant), pipeline, checkpoint, variant, steps, guidance, eval_set))
                   for tag, pipeline, checkpoint, steps, _ in models)


def variant_tag(tag, variant):
    """
    Pure function. The model_tag of a showcase model on a prompt variant.

    Examples:
        >>> variant_tag("show_stock_sprint2", "plain"), variant_tag("show_stock_sprint2", "detailed")
        ('show_stock_sprint2', 'show_stock_sprint2_detailed')
    """
    return tag if variant == "plain" else f"{tag}_{variant}"


def rows(run, step, *variants, stock=False):
    """Command. Prints the eval_models rows of one control step (and the STOCK models with --stock) to paste into the dtab."""
    print("".join(dtab_rows(step_models(run, step) + STOCK * stock, v) for v in variants or VARIANTS), end="")


def trimask_models(run_step):
    """
    Pure function. The model of one control step for the trimask gallery: the shipped one, Sprint 2 steps at full
    subtraction (λ = 0). Only it: 19 methods × 4 resolutions are 76 eval_models rows per step, and the eval GPUs are
    shared with training.

    Args:
        run_step (str): "RUN:STEP", e.g. "control_tri_ft:6000"

    Returns:
        list[tuple]: as step_models

    Examples:
        >>> [m[0] for m in trimask_models("control_tri_ft:6000")]
        ['show_tri_ft_step006000_sprint_lam0_2step']
    """
    run, step = run_step.split(":")
    return [m for m in step_models(run, int(step)) if m[0].endswith("_lam0_2step")]


def trimask_tag(tag, method, resolution=1024):
    """
    Pure function. The model_tag of a showcase model on one trimask method's eval set (suffix _r<res> below 1024).

    Examples:
        >>> trimask_tag("show_tri_ft_step006000_lam0", "morph_band"), trimask_tag("show_v2_step006000_lam0", "scribbles", 512)
        ('tri_morph_band_tri_ft_step006000_lam0', 'tri_scribbles_v2_step006000_lam0_r512')
    """
    return tag.replace("show_", f"tri_{method}_", 1) + (f"_r{resolution}" if resolution != 1024 else "")


def trimask_set(resolution=1024):
    """
    Pure function. The trimask eval sets folder of one resolution (eval_set.build_trimask --resolution).

    Examples:
        >>> trimask_set(), trimask_set(256)
        ('resources/showcase_trimask', 'resources/showcase_trimask_256')
    """
    return TRIMASK_SET + (f"_{resolution}" if resolution != 1024 else "")


def trimask_rows(*run_steps, resolution=1024, methods=None):
    """Command. Prints the eval_models rows (MAGICK captions) of RUN:STEP... on the trimask eval sets of one resolution (every method, or the comma-separated `methods`), to paste into the dtab."""
    from sana_alpha.trimask import METHODS

    methods = methods.split(",") if isinstance(methods, str) else methods or METHODS
    assert set(methods) <= set(METHODS), f"unknown methods: {set(methods) - set(METHODS)}"
    print("".join(dtab_rows([(trimask_tag(m[0], method, resolution), *m[1:]) for r in run_steps for m in trimask_models(r)],
                            "plain", f"{trimask_set(resolution)}/{method}") for method in methods), end="")


def fibo2_models(run_step):
    """
    Pure function. The shipped FIBO2 model of one control step (fibo2.load_eval_model) at the pipeline's defaults.

    Examples:
        >>> fibo2_models("fibo2_tri:2000")
        [('show_fibo2_tri_step002000_fibo2_30step', 'fibo2', 'models/fibo2/fibo2_tri/step_002000', 30, 'FIBO2, 30 steps')]
    """
    run, step = run_step.split(":")
    return [(f"show_{run}_step{int(step):06d}_fibo2_{FIBO2_STEPS}step", "fibo2", f"models/fibo2/{run}/step_{int(step):06d}", FIBO2_STEPS, f"FIBO2, {FIBO2_STEPS} steps")]


def fibo2_rows(*run_steps, methods=None):
    """Command. Prints the eval_models rows of FIBO2 control RUN:STEP... on the held-out trimask sets at 1024 (every method, or the comma-separated `methods`), to paste into the dtab."""
    from sana_alpha.trimask import METHODS

    methods = methods.split(",") if isinstance(methods, str) else methods or METHODS
    assert set(methods) <= set(METHODS), f"unknown methods: {set(methods) - set(METHODS)}"
    print("".join(dtab_rows([(trimask_tag(m[0], method), *m[1:]) for r in run_steps for m in fibo2_models(r)],
                            "plain", f"{HELD_SET}/{method}", FIBO2_GUIDANCE) for method in methods), end="")


def fibo2_preview_rows(*run_steps):
    """Command. Prints the eval_models rows of FIBO2 control RUN:STEP... on the training-preview set (PREVIEW_SET, 4 items at 1024), to paste into the dtab."""
    print(dtab_rows([(m[0].replace("show_", "preview_", 1), *m[1:]) for r in run_steps for m in fibo2_models(r)],
                    "plain", PREVIEW_SET, FIBO2_GUIDANCE), end="")


def build_trimask(*run_steps, notes="", name=None, resolution=1024):
    """
    Command. Hard-links the trimask showcase images of the full-subtraction models of RUN:STEP... and writes the page
    journal/<name>.html: one collapsible section per method, rows [matte | trimask | MAGICK on gray | models].
    Metrics per model and method: IoU and gray error against the true matte (eval_generations), and white_miss /
    black_hit against the trimask (metrics.trimask_metrics, computed here from the stored images). Raises if any image is missing.

    Args:
        *run_steps (str): "RUN:STEP", e.g. control_v2:6000 control_tri_ft:6000 control_tri_scratch:6000
        notes (str): HTML of the observations paragraph
        name (str | None): page and assets folder name; default gallery_trimask (+ _<resolution> below 1024)
        resolution (int): eval resolution (trimask_set)
    """
    from concurrent.futures import ThreadPoolExecutor

    import lance
    import numpy as np
    from PIL import Image

    from sana_alpha.metrics import trimask_metrics
    from sana_alpha.trimask import METHODS

    name, sets = name or "gallery_trimask" + (f"_{resolution}" if resolution != 1024 else ""), trimask_set(resolution)
    assets, page = ROOT / f"journal/assets/{name}", ROOT / f"journal/{name}.html"
    models = [(*m[:4], f"{r.split(':')[0]}: {m[4]}", label_of(*m[1:4])) for r in run_steps for m in trimask_models(r)]
    tags = {trimask_tag(m[0], method, resolution): (m[0], method) for m in models for method in METHODS}
    items = json.loads((ROOT / SHOWCASE / "prompts.json").read_text())
    where = "image IS NOT NULL AND model_tag IN (%s)" % ", ".join(f"'{t}'" for t in tags)
    table = lance.dataset(str(LANCE)).to_table(columns=["model_tag", "page_id", "image", "iou", "exterior_gray_error"],
                                               filter=where).to_pandas()
    missing = {(t, i["page_id"]) for t in tags for i in items} - set(zip(table.model_tag, table.page_id))
    assert not missing, f"{len(missing)} trimask showcase images missing, e.g. {sorted(missing)[:3]}: run data/run_proc_eval.fish"
    table["model"], table["method"] = zip(*table.model_tag.map(tags))
    for tag, page_id, image in zip(table.model_tag, table.page_id, table.image):
        assert image["position"] == 0, f"{tag} {page_id}: blob is packed in {image['blob_uri']}, not a file of its own"
        link(LANCE / image["blob_uri"], assets / tag / f"{page_id}.png")
    for item in items:
        for method in METHODS:
            link(ROOT / sets / method / "mattes" / f"{item['page_id']}.png", assets / "trimasks" / method / f"{item['page_id']}.png")
        for kind, target in (("originals", "mattes"), ("references", "references")):  # the true matte and MAGICK, at this resolution
            link(ROOT / sets / METHODS[0] / kind / f"{item['page_id']}.png", assets / target / f"{item['page_id']}.png")

    def score(row):
        """Query. trimask_metrics of one stored image (reads its PNG and its trimask)."""
        read = lambda path: np.asarray(Image.open(path), np.float32) / 255
        return trimask_metrics(read(LANCE / row.image["blob_uri"]), read(ROOT / sets / row.method / "mattes" / f"{row.page_id}.png"))

    with ThreadPoolExecutor(16) as pool:
        table = table.join(pd.DataFrame(list(pool.map(score, table.itertuples())), index=table.index))
    columns = ["iou", "exterior_gray_error", "white_miss", "black_hit"]
    means = pd.concat([table.groupby("model")[columns].mean().assign(method="all"),
                       table.groupby(["model", "method"])[columns].mean().reset_index("method")]).set_index("method", append=True)
    (assets / "metrics.json").write_text(means.reset_index().to_json(orient="records", indent=1) + "\n")
    page.write_text(render_trimask(models, items, METHODS, means, notes, assets.relative_to(page.parent), run_steps, resolution))
    print(f"wrote {page}: {len(items)} items x {len(METHODS)} methods x {len(models)} models, {len(table)} links in {assets}")


def render_trimask(models, items, methods, means, notes, assets, run_steps, resolution=1024):
    """
    Pure function. The trimask gallery page HTML.

    Args:
        models (list[tuple]): (model_tag, pipeline, checkpoint, steps, short label, weight equation), in column order
        items (list[dict]): showcase prompts.json (page_id, caption, split)
        methods (tuple[str]): trimask methods, one section each
        means (pd.DataFrame): index (model_tag, method or "all"); columns iou, exterior_gray_error, white_miss, black_hit
        notes (str): observations HTML
        assets (Path): image folder relative to the page
        run_steps (tuple[str]): RUN:STEP of the models, for the title
        resolution (int): eval resolution (model tags and title)

    Returns:
        str
    """
    esc = html.escape
    label = lambda short, equation="": esc(short) + (f"<br><small>{esc(equation)}</small>" if equation else "")
    figure = lambda src, short, equation="": (f'<figure><a href="{src}" target="_blank"><img loading="lazy" decoding="async" '
                                              f'src="{src}" alt="{esc(short)}"></a><figcaption>{label(short, equation)}</figcaption></figure>')
    body = "".join(f"<tr><td>{label(short, equation)}</td><td>{esc(method)}</td>" + "".join(
        f"<td>{v:.3f}</td>" for v in means.loc[(tag, method)]) + "</tr>"
        for method in ("all", *methods) for tag, *_, short, equation in models)
    sections = ""
    for method in methods:
        rows = ""
        for item in items:
            pid = item["page_id"]
            cells = [figure(f"{assets}/mattes/{pid}.png", "matte"), figure(f"{assets}/trimasks/{method}/{pid}.png", f"trimask ({method})"),
                     figure(f"{assets}/references/{pid}.png", "MAGICK on gray")]
            cells += [figure(f"{assets}/{trimask_tag(tag, method, resolution)}/{pid}.png", short, equation) for tag, *_, short, equation in models]
            rows += (f'<div class="evalrow"><p class="rowcap"><span class="split {item["split"]}">{item["split"]}</span> '
                     f'{esc(item["caption"])} <code>{pid}</code></p>{"".join(cells)}</div>\n')
        sections += (f'<details{" open" * (method == methods[0])}><summary><h2>{esc(method)}</h2></summary>\n'
                     f'<div class="evalgrid" style="--cols: {len(models) + 3}">\n{rows}</div></details>\n')
    return TRIMASK_HTML.format(title=f"Gallery trimask {resolution}", runs=esc(", ".join(run_steps)) + f" at {resolution}&sup2;", n=len(items), k=len(methods),
                               held=sum(i["split"] == "held-out" for i in items), body=body, notes=notes, sections=sections)


def build(run, step, notes="", name=None):
    """
    Command. Hard-links the showcase images of one control step and writes its gallery page. Raises if any is missing.

    Args:
        run (str): control run, e.g. "control_v1"
        step (int): its step
        notes (str): HTML of the observations paragraph
        name (str | None): page and assets folder name; default gallery_<run without "control_">, e.g. gallery_v2_step1500
    """
    import lance
    import pandas as pd

    short = run.removeprefix("control_")
    name = name or f"gallery_{short}"
    assets, page = ROOT / f"journal/assets/{name}", ROOT / f"journal/{name}.html"
    models = [(*m, label_of(*m[1:4])) for m in STOCK + step_models(run, step)]
    tags = {variant_tag(m[0], v): (m, v) for v in VARIANTS for m in models}
    items = json.loads((ROOT / SHOWCASE / "prompts.json").read_text())
    detailed = json.loads((ROOT / SHOWCASE / "prompts_detailed.json").read_text())
    where = "image IS NOT NULL AND model_tag IN (%s)" % ", ".join(f"'{t}'" for t in tags)
    table = lance.dataset(str(LANCE)).to_table(columns=["model_tag", "page_id", "image", "iou", "exterior_gray_error"],
                                               filter=where).to_pandas()
    expected = {(t, i["page_id"]) for t in tags for i in items}
    missing = expected - set(zip(table.model_tag, table.page_id))
    assert not missing, f"{len(missing)} showcase images missing, e.g. {sorted(missing)[:3]}: run data/run_proc_eval.fish"
    for tag, page_id, image in zip(table.model_tag, table.page_id, table.image):
        assert image["position"] == 0, f"{tag} {page_id}: blob is packed in {image['blob_uri']}, not a file of its own"
        link(LANCE / image["blob_uri"], assets / tag / f"{page_id}.png")
    for kind in ("mattes", "references"):
        for item in items:
            link(ROOT / SHOWCASE / kind / f"{item['page_id']}.png", assets / kind / f"{item['page_id']}.png")

    table["split"] = table.page_id.map({i["page_id"]: i["split"] for i in items})
    means = pd.concat([table.groupby("model_tag")[["iou", "exterior_gray_error"]].mean().add_suffix(" all"),
                       table.pivot_table(index="model_tag", columns="split", values=["iou", "exterior_gray_error"])
                       .pipe(lambda d: d.set_axis([f"{a} {b}" for a, b in d.columns], axis=1))], axis=1)
    (assets / "metrics.json").write_text(means.to_json(orient="index", indent=1) + "\n")
    page.write_text(render(run, step, models, items, detailed, means, notes, assets.relative_to(page.parent)))
    print(f"wrote {page}: {len(items)} items x {len(VARIANTS)} prompts x {len(models)} models, {len(table)} links in {assets}")


def render(run, step, models, items, detailed, means, notes, assets):
    """
    Pure function. The gallery page HTML: collapsible sections, items in collapsible groups of GROUP_SIZE.

    Args:
        run, step: the control step, for the title
        models (list[tuple]): (model_tag, pipeline, checkpoint, steps, short label, weight equation), in column order
        items (list[dict]): showcase prompts.json (page_id, caption, split)
        detailed (dict): page_id -> detailed prompt
        means (pd.DataFrame): index model_tag; columns "iou all", "iou held-out", ... "exterior_gray_error train"
        notes (str): observations HTML
        assets (Path): image folder relative to the page

    Returns:
        str
    """
    esc = html.escape
    label = lambda short, equation="": esc(short) + (f"<br><small>{esc(equation)}</small>" if equation else "")
    metric_cols = [f"{m} {s}" for m in ("iou", "exterior_gray_error") for s in ("all", "held-out", "train")]
    head = "".join(f"<th>{esc(c.replace('exterior_gray_error', 'gray error'))}</th>" for c in metric_cols)
    body = "".join(f"<tr><td>{label(short, equation)}</td><td>{esc(VARIANTS[v])}</td>" + "".join(
        f"<td>{means.loc[variant_tag(tag, v), c]:.3f}</td>" for c in metric_cols) + "</tr>"
        for tag, *_, short, equation in models for v in VARIANTS)
    figure = lambda src, short, equation="": (f'<figure><a href="{src}" target="_blank"><img loading="lazy" decoding="async" '
                                              f'src="{src}" alt="{esc(short)}"></a><figcaption>{label(short, equation)}</figcaption></figure>')
    groups = ""
    for start in range(0, len(items), GROUP_SIZE):
        group = items[start:start + GROUP_SIZE]
        rows = ""
        for item in group:
            pid = item["page_id"]
            for v, prompt in dict(plain=item["caption"], detailed=detailed[pid]).items():
                cells = [figure(f"{assets}/mattes/{pid}.png", "matte"), figure(f"{assets}/references/{pid}.png", "MAGICK on gray")]
                cells += [figure(f"{assets}/{variant_tag(tag, v)}/{pid}.png", short, equation) for tag, *_, short, equation in models]
                rows += (f'<div class="evalrow"><p class="rowcap"><span class="split {item["split"]}">{item["split"]}</span> '
                         f'<b>{esc(VARIANTS[v])}:</b> {esc(prompt)} <code>{pid}</code></p>{"".join(cells)}</div>\n')
        splits = ", ".join(sorted({i["split"] for i in group}))
        groups += (f'<details{" open" * (start == 0)}><summary><h2>Items {start + 1}&ndash;{start + len(group)} ({splits})</h2></summary>\n'
                   f'<div class="evalgrid" style="--cols: {len(models) + 2}">\n{rows}</div></details>\n')
    return GALLERY_HTML.format(title=f"Gallery {run.removeprefix('control_')}", run=esc(run), step=int(step), n=len(items),
                               version=f"{run}_sprint_step_{int(step):06d}-int4",
                               held=sum(i["split"] == "held-out" for i in items), head=head, body=body, notes=notes, groups=groups)


GROUP_SIZE = 10  # items per collapsible group of rows
GALLERY_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="eval_grid.css">
<style>
:root {{ --bg: #fdfdfb; --fg: #1d1d1f; --muted: #66666b; --rule: #dcdcd6; --accent: #2f5d8a; --chip: #eef2f6; --train: #b5473a; --held: #2f7d4f;
  --gutter: 16px; --measure: 1100px; --gap: 12px; --radius: 4px; --pad-cell: 4px 8px; --font-body: 17px; --font-small: 14px; --hairline: 1px; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --chip: #26292d; --train: #e0857a; --held: #6cc08b; }} }}
:root[data-theme="dark"] {{ --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --chip: #26292d; --train: #e0857a; --held: #6cc08b; }}
body {{ background: var(--bg); color: var(--fg); font: var(--font-body)/1.6 Georgia, serif; margin: 0; }}
main {{ max-width: var(--measure); margin: 0 auto; padding: 0 var(--gutter); }}
a {{ color: var(--accent); }}
summary h2 {{ display: inline; }}
.metrics {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font: var(--font-small) system-ui, sans-serif; }}
th, td {{ padding: var(--pad-cell); border-bottom: var(--hairline) solid var(--rule); text-align: right; }}
td:first-child, td:nth-child(2) {{ text-align: left; }}
.split {{ font-weight: bold; }} .split.train {{ color: var(--train); }} .split.held-out {{ color: var(--held); }}
</style>
</head>
<body>
<main>
<p><a href="index.html">&larr; journal</a></p>
"""
GALLERY_HTML = GALLERY_HEAD + """<h1>Gallery: {run} step {step}, with and without subtracting the adapter</h1>
<details open><summary><h2>How each image is made</h2></summary>
<ul>
<li><b>W_teacher</b>: the stock SANA-Sprint 1.6B teacher. <b>W_sprint</b>: the stock Sprint student. <b>W_adapter</b>: the teacher fully fine-tuned on MAGICK-on-gray, with no matte (adapter_v1 step 2000).</li>
<li><b>W_control</b>: the trained control model = W_adapter + <b>ΔW_control</b> (the control's LoRA change; the adapter was frozen during control training). <b>gates</b>: GateControl's matte input.</li>
<li>So <b>W_control &minus; W_adapter = ΔW_control</b>: subtracting the adapter leaves only what the control training added. λ is the fraction of the adapter change (W_adapter &minus; W_teacher) kept. λ = 1: no subtraction (the control model). λ = 0: <b>full subtraction</b>.</li>
<li>Sprint columns put the same changes on W_sprint, with no training: nothing was distilled anywhere. Stock columns get no matte. Every label below an image gives its equation and sampler.</li>
<li>Try the 1-step Sprint model (full subtraction, 4-bit) in the browser: <a href="webgpu/?version={version}">WebGPU demo</a>. Only versions exported for the web open there; others fall back to the default model.</li>
</ul>
<p>{n} showcase items: {held} <span class="split held-out">held-out</span> (never trained on); the rest are <span class="split train">train</span>
items (the control saw their mattes in training). Each item has two rows: the MAGICK caption and a detailed prompt. Every model uses the same fixed noise.
Every image is the stored 1024 &times; 1024 PNG: click one to open it.</p>
</details>
<details open><summary><h2>Metrics</h2></summary>
<p>IoU: pixels far from 0.5 gray vs the matte (1 = exact silhouette on clean gray). Gray error: mean |rgb &minus; 0.5| outside the matte (0 = flat gray). Stock models do not see the matte, so their IoU only shows the baseline.</p>
<div class="metrics"><table><tr><th>model</th><th>prompt</th>{head}</tr>{body}</table></div>
</details>
<details open><summary><h2>Observations</h2></summary>
{notes}
</details>
<button class="gridtoggle" onclick="this.setAttribute('aria-pressed', [...document.querySelectorAll('.evalgrid')].map(g => g.classList.toggle('native'))[0])">native size (1:1)</button>
</main>
{groups}
</body>
</html>
"""

TRIMASK_HTML = GALLERY_HEAD + """<h1>Trimask gallery: {runs}</h1>
<details open><summary><h2>How each image is made</h2></summary>
<p>The condition is a <b>trimask</b>: white = surely object, black = surely background, gray = unknown (the model decides).
Each of the {n} showcase items ({held} <span class="split held-out">held-out</span>) has one fixed trimask per method ({k} methods,
<code>resources/showcase_trimask</code>), made from its true matte. Every column is a full-subtraction model
(W_base + W_control &minus; W_adapter + gates); the label under each image gives its equation and sampler. Same fixed noise everywhere.</p>
</details>
<details open><summary><h2>Metrics</h2></summary>
<p>IoU and gray error: against the <b>true</b> matte (as in the other galleries). white miss: share of the trimask's white left gray
(0 = all of it painted). black hit: share of the trimask's black painted (0 = all of it left gray). The first rows average all methods.</p>
<div class="metrics"><table><tr><th>model</th><th>method</th><th>IoU</th><th>gray error</th><th>white miss</th><th>black hit</th></tr>{body}</table></div>
</details>
<details open><summary><h2>Observations</h2></summary>
{notes}
</details>
<button class="gridtoggle" onclick="this.setAttribute('aria-pressed', [...document.querySelectorAll('.evalgrid')].map(g => g.classList.toggle('native'))[0])">native size (1:1)</button>
</main>
{sections}
</body>
</html>
"""

if __name__ == "__main__":
    fire.Fire(dict(rows=rows, build=build, trimask_rows=trimask_rows, fibo2_rows=fibo2_rows, fibo2_preview_rows=fibo2_preview_rows,
                   build_trimask=build_trimask))
