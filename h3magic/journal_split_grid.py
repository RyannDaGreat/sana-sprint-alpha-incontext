# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance", "fire", "numpy", "opencv-python-headless"]
# ///
"""
Builds the split v3-grid page (manifest §23, m01003, m01005, m01006): journal/h3magic_split_grid.html. Every
h3_eval_models row with layout grid2x2: per prompt the whole generated 2x2 frame, the RGBA (colour decontaminated with the
4th-panel alpha), the 4th-panel alpha, the least-squares alpha of the three background panels, and a heat map of their
disagreement |Δα|; per row the |Δα| statistics (all pixels and the edge band) stored by h3_generate. Outputs are copied
byte for byte to journal/assets/h3magic/<tag>/; the heat maps are drawn from the two alphas. Rerun any time:

    uv run --script h3magic/journal_split_grid.py build
"""

import html
import json

import cv2
import fire
import lance
import numpy as np

import journal as j
import journal_split as js

PAGE = j.ROOT / "journal/h3magic_split_grid.html"
NOTES = j.ROOT / "h3magic/split_grid_notes.json"
METRICS = j.ROOT / "h3magic/split_grid_metrics.json"
REGISTER = j.ROOT / "h3magic/split_grid_register.json"  # agents/.scratchpad/alphav2/grid_register.py: per-frame IoU + panel seams of video rows  # agents/.scratchpad/alphav2/grid_missed.py: missed / extra / IoU per prompt
PROMPTS = j.ROOT / "h3magic/eval_prompts_fgrid.json"
KINDS = (("frame", "the generated 2x2 frame"), ("rgba", "RGBA (colour from 3 panels, alpha = 4th panel)"), ("alpha", "4th panel = alpha"),
         ("alpha_ls", "alpha implied by the 3 backgrounds"), ("dalpha", "|Δα| (black 0, white ≥ 0.25)"))
STATS = ("mean", "median", "p95", "edge_mean", "edge_median", "edge_p95")
HEAT_TOP = 0.25  # |Δα| shown at full brightness from this value up
GRID_SENTENCE = (" Shown as a 2x2 grid of the same image: top left on a plain flat medium-gray (#808080) background, top right on a plain white"
                 " background, bottom left on a plain black background, bottom right its grayscale alpha matte, white subject on black.")


def heat(alpha, alpha_ls, top=HEAT_TOP):
    """
    Pure function. A heat map of |Δα| as a uint8 image: 0 black, ≥ top white (inferno colour map).

    Args:
        alpha, alpha_ls (np.ndarray): (H, W) uint8 alphas

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB

    Examples:
        >>> h = heat(np.array([[0, 255]], np.uint8), np.array([[0, 0]], np.uint8)); h.shape, h[0, 0].tolist()
        ((1, 2, 3), [0, 0, 4])
    """
    d = np.abs(alpha.astype(np.float32) - alpha_ls.astype(np.float32)) / 255
    return cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(d / top, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def export(models):
    """Command. Copies every grid output (image rows; video rows: their MP4s) and draws the heat maps. Returns {tag: {prompt_id: row}}."""
    ds = lance.dataset(str(j.LANCE / "h3_eval_generations.lance"))
    found = {}
    for m in models:
        tag, mode = m["model_tag"], m["mode"]
        cols = {"frame": mode, "rgba": f"{mode}_rgba", "alpha": f"{mode}_alpha", "alpha_ls": f"{mode}_alpha_ls"}
        rows = ds.scanner(filter=f"model_tag = '{tag}' AND {cols['rgba']} IS NOT NULL", with_row_id=True,
                          columns=["prompt_id", "seed", "split_residual", *[f"grid_dalpha_{s}" for s in STATS]]).to_table().to_pylist()
        found[tag] = {r["prompt_id"]: r for r in rows}
        data = {}
        for kind, column in cols.items():
            for r, blob in zip(rows, ds.take_blobs(column, ids=[r["_rowid"] for r in rows]) if rows else []):
                out = j.ROOT / "journal" / js.asset(tag, r["prompt_id"], kind, mode)
                out.parent.mkdir(parents=True, exist_ok=True)
                b = blob.read()
                if not out.exists() or out.read_bytes() != b:
                    out.write_bytes(b)
                data[(r["prompt_id"], kind)] = b
        if mode == "image":
            for r in rows:
                a = cv2.imdecode(np.frombuffer(data[(r["prompt_id"], "alpha")], np.uint8), cv2.IMREAD_GRAYSCALE)
                b = cv2.imdecode(np.frombuffer(data[(r["prompt_id"], "alpha_ls")], np.uint8), cv2.IMREAD_GRAYSCALE)
                out = j.ROOT / "journal" / js.asset(tag, r["prompt_id"], "dalpha", mode)
                ok, png = cv2.imencode(".png", cv2.cvtColor(heat(a, b), cv2.COLOR_RGB2BGR))
                assert ok, "PNG encode failed"
                if not out.exists() or out.read_bytes() != png.tobytes():
                    out.write_bytes(png.tobytes())
    return found


def missed_block(tag, metrics, prompts, mode):
    """
    Pure function. The missed-subject table of one row (per prompt: missed, extra, IoU) and its worst 5 prompts' alpha
    panels side by side, or "" when the row has no numbers yet.

    Examples:
        >>> missed_block("t", {}, [], "image")
        ''
    """
    m = metrics.get(tag)
    if not m or not m.get("per_prompt"):
        return ""
    short = {p["id"]: p["short"] for p in prompts}
    per = m["per_prompt"]
    rows = "".join(f'<tr><td>{pid}: {html.escape(short.get(pid, ""))}</td><td>{v["missed"]:.2%}</td><td>{v["extra"]:.2%}</td><td>{v["iou"]:.3f}</td></tr>'
                   for pid, v in sorted(per.items(), key=lambda kv: -kv[1]["missed"]))
    worst = sorted(per, key=lambda pid: -per[pid]["missed"])[:5]
    mean = m["mean"]
    return (f'<details class="sub"><summary>Missed subject, per prompt (mean: missed {mean["missed"]:.2%}, extra {mean["extra"]:.2%}, IoU {mean["iou"]:.3f})</summary>'
            f'<p class="note">missed = share of pixels the three backgrounds call solid (alpha ≥ 0.9) but the 4th panel calls clear (&lt; 0.5); extra = the reverse; '
            f'IoU of the two masks at 0.5. Caveat: on the black panel a dark object (the wok) looks like the background, so the three-background alpha can miss it too.</p>'
            f'<table class="scores"><thead><tr><th>prompt</th><th>missed</th><th>extra</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="note">The worst 5 by "missed": RGBA | 4th-panel alpha | three-background alpha | |Δα|.</p>'
            f'<div class="wide"><table><tbody>' + "".join("<tr>" + "".join(js.cell(js.asset(tag, pid, k, mode), mode, k, short.get(pid, "")) for k in ("rgba", "alpha", "alpha_ls", "dalpha")) + f'<th class="label">{pid}: {html.escape(short.get(pid, ""))}</th></tr>' for pid in worst) + '</tbody></table></div></details>')


def register_block(tag, register, prompts):
    """
    Pure function. The registration table of one video row (per prompt: share of bad frames, mean / min IoU of the panel
    alpha vs the three-background alpha, largest seam offset), or "" when there are no numbers.

    Examples:
        >>> register_block("t", {}, [])
        ''
    """
    r = register.get(tag)
    if not r:
        return ""
    short = {p["id"]: p["short"] for p in prompts}
    per = r["per_prompt"]
    rows = "".join(f'<tr><td>{pid}: {html.escape(short.get(pid, ""))}</td><td>{v["bad_share"]:.0%}</td><td>{v["iou_mean"]:.2f} / {v["iou_min"]:.2f}</td><td>{v["seam_off_max"]} px</td></tr>'
                   for pid, v in sorted(per.items(), key=lambda kv: -kv[1]["bad_share"]))
    clean = len(per) - len(r["failing"])
    return (f'<details class="sub" open><summary>Registration check: {clean} of {len(per)} videos clean</summary>'
            f'<p class="note">Every {r["stride"]}th frame: IoU at 0.5 of the 4th-panel alpha and the alpha implied by the three backgrounds (both cut at the fixed panel '
            f'boundaries), and where the panel seams really are (largest gray-to-black step down the left half, gray-to-white step across the top). A frame is bad when '
            f'IoU &lt; {r["iou_min"]} or a seam is more than {r["seam_tol"]} px off. A misplaced matte still has a clean background, so this is the check that catches it; '
            f'low IoU with the seams in place usually means a coarse matte on smoke, splash or thin parts rather than a shifted layout.</p>'
            f'<table class="scores"><thead><tr><th>prompt</th><th>bad frames</th><th>IoU mean / min</th><th>largest seam offset</th></tr></thead><tbody>{rows}</tbody></table></details>')


def section(model, items, prompts, metrics=None, register=None):
    """Pure function. One <details>: the stats line, the missed-subject block and, per prompt, a row of the KINDS (video rows have no heat map)."""
    tag, mode = model["model_tag"], model["mode"]
    if not items:
        return f'<details class="row"><summary>{html.escape(tag)} <span class="note">(rendering)</span></summary></details>'
    kinds = [k for k in KINDS if mode == "image" or k[0] != "dalpha"]
    mean = {s: np.nanmean([it[f"grid_dalpha_{s}"] for it in items.values()]) for s in STATS}
    head = "".join(f"<th>{html.escape(n)}</th>" for _, n in kinds)
    body = []
    for p in prompts:
        it = items.get(p["id"])
        if it is None:
            body.append(f'<tr><th class="label">{p["id"]}: {html.escape(p["short"])}</th><td class="missing" colspan="{len(kinds)}">rendering</td></tr>')
            continue
        label = (f'{p["id"]}: {html.escape(p["short"])}<br><span class="note">|Δα| mean {it["grid_dalpha_mean"]:.3f}, edge {it["grid_dalpha_edge_mean"]:.3f}</span>')
        body.append(f'<tr><th class="label">{label}</th>' + "".join(js.cell(js.asset(tag, p["id"], k, mode), mode, k, p["short"]) for k, _ in kinds) + "</tr>")
    stats = (f'|Δα| over all pixels: mean {mean["mean"]:.3f}, median {mean["median"]:.3f}, 95th pct {mean["p95"]:.3f}; '
             f'edge band (0.02 &lt; α &lt; 0.98): mean {mean["edge_mean"]:.3f}, median {mean["edge_median"]:.3f}, 95th pct {mean["edge_p95"]:.3f} (means over prompts)')
    video_lora = f" + 1·Δ_{model['pexels_adapter'].split('/')[-2]} ({model['pexels_adapter'].split('/')[-1]})" if model.get("pexels_adapter") else ""
    side = model.get("size") or 768
    eq = f"W_H3 + 1·Δ_graymagic_v3t + 1·Δ_{model['imagegen_adapter'].split('/')[-2]} ({model['imagegen_adapter'].split('/')[-1]}){video_lora}, CFG 1, {side}²"
    label = "UNREQUESTED VARIANT, alpha = max(4th panel, three-background alpha above 0.1): " if model.get("layout") == "grid2x2mix" else ""
    return (f'<details class="row" open><summary>{label}{html.escape(tag)} <span class="note">({len(items)}/{len(prompts)} prompts)</span></summary>'
            f'<p class="eq">{html.escape(eq)}</p><p class="note">{stats}</p>{missed_block(tag, metrics or {}, prompts, mode) if mode == "image" else register_block(tag, register or {}, prompts)}'
            f'<div class="wide"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')


def build():
    """Command. Writes journal/h3magic_split_grid.html."""
    prompts = json.loads(PROMPTS.read_text())
    models = [m for m in j.load_models() if m.get("layout") in ("grid2x2", "grid2x2mix")]
    found = export(models)
    notes = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    order = sorted(models, key=lambda m: (m["mode"] != "image", m["model_tag"]))
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    register = json.loads(REGISTER.read_text()) if REGISTER.exists() else {}
    vprompts = json.loads((j.ROOT / "h3magic/eval_prompts_vgrid.json").read_text())
    mprompts = json.loads((j.ROOT / "h3magic/eval_prompts_mgrid.json").read_text())
    sections = "".join(section(m, found.get(m["model_tag"], {}), {"vgrid": vprompts, "mgrid": mprompts}.get(m.get("prompt_set"), prompts), metrics, register) for m in order)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 split v3-grid</title>
<style>{js.CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: split v3-grid (gray, white / black, alpha in one 2&times;2 frame)</h1>
<p class="note">Part of <a href="h3magic_alphav2.html">Alpha Video v2</a>. Every item is the unmodified output; the |Δα| heat maps are drawn from the two alphas.</p>
<p><b>The idea (m01003, m01005, m01006).</b> Instead of three output heads, H3 draws one 768&times;768 frame made of four 384&times;384 panels:
top left the subject on 50% gray, top right on white, bottom left on black, bottom right its alpha matte in grayscale (white = opaque).
Because the four panels are one image, attention is shared and the subject should match across them. The three background panels give the colour
(decontamination: solve for the foreground colour given the known backgrounds), and the 4th panel is the alpha (the mask).
The alpha that the three backgrounds imply on their own (least squares) is a free check: how much the two disagree, |Δα|, is reported per image.</p>
<p><b>Training.</b> One LoRA on H3 + gray magic v3t (gray magic v3 fine-tuned on tiled latents), trained on the 10,000 MAGICK images laid out as such grids
(tiled VAE encode, manifest &sect;25), guidance loss (scale 4), lr 1e-4. Caption = the subject-only caption + this sentence (also at the end of every test prompt, q01&ndash;q20):
<span class="eq">{html.escape(GRID_SENTENCE.strip())}</span></p>
{notes.get("finding", "")}
{sections}
</main>
<dialog id="lightbox"></dialog>
<script>{js.JS}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(j.ROOT)}: " + ", ".join(f"{t} {len(v)}" for t, v in found.items()), flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build})
