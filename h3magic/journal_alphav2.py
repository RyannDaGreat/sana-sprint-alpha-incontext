# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance", "fire"]
# ///
"""
Builds the Alpha Video v2 page (manifest §23, m00983): journal/h3magic_alphav2.html.

Entry 1, the prompt-only gray suffix search: stock H3 (no adapter, CFG 1) on the 20 subject-only prompts with each
candidate suffix, 768² images (+ test videos), scored by how much of the border is a flat neutral mid-gray. The images
are copied byte for byte from agents/.scratchpad/alphav2/suffix/ to journal/assets/h3magic_alphav2/suffix/.
Later entries (training, tasting sheets, eval rows from h3_eval_models with size 768) are added by SECTIONS.

    uv run --script h3magic/journal_alphav2.py build
"""

import html
import json
import shutil
from pathlib import Path

import fire

import journal as j
import journal_split as js

PAGE = j.ROOT / "journal/h3magic_alphav2.html"
ASSETS = j.ROOT / "journal/assets/h3magic_alphav2"
SUFFIX_DIR = j.ROOT / "agents/.scratchpad/alphav2/suffix"
TASTE_DIR = j.ROOT / "agents/.scratchpad/alphav2/taste"  # tasting sheets and other scratch outputs a note references (copied byte for byte)
FG_PROMPTS = j.ROOT / "h3magic/eval_prompts_fg.json"
CHOSEN = "hex_medium"
NOTES = json.loads((Path(__file__).parent / "alphav2_notes.json").read_text()) if (Path(__file__).parent / "alphav2_notes.json").exists() else {}

INTRO = (
    "<p><b>Goal (m00983).</b> A better cut-out (RGBA) video model than Alpha Video 1 (the split adapter of "
    "<a href=\"h3magic_split.html\">the split page</a>), by 3 PM ET on 2026-11-23.</p>"
    "<p><b>The trick.</b> First ask stock H3, with words alone, to put the subject on a flat gray background. If a prompt "
    "suffix does that, every caption and every test prompt of this version carries the suffix, so the gray magic adapter has "
    "little left to learn: mostly MAGICK's over-saturated look. Then only two adapters, both trained on still images at "
    "768&times;768 (512 was low): the gray magic adapter, and the split adapter on top of it (gray, white and black outputs, "
    "least-squares RGBA). No Pexels and no image-gen adapter: the &quot;basic ass magic&quot; ablation showed they are not needed. "
    "Training data: <a href=\"h3magic_dataset_v3.html\">the dataset page</a>.</p>")

GLOSSARY = tuple((t, d) for t, d in js.GLOSSARY) + (
    ("Suffix", "A sentence added to the end of every prompt and caption, here the one that asks for a flat gray background."),
    ("Border score", "The fraction of pixels in the outer 8% band of the image that are a neutral mid-gray: no hue (max &minus; min channel &le; 12 of 255) and a level between 96 and 176 (50% gray is 128)."),
    ("Subject-only prompts", "The 20 test prompts f01&ndash;f20 that describe only the subject and its motion, never a background."),
)


def copy_asset(src, dst):
    """Command. Copies src to dst byte for byte when dst is missing or differs. Returns dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)
    return dst


def suffix_rows(scores, variants):
    """
    Pure function. The score table rows, best border score first.

    Examples:
        >>> suffix_rows({"a": {"border_neutral": 0.5, "border_std": 20.0, "border_mean": 150.0, "subject_lost": 0.0, "n": 20}},
        ...             {"a": ["", " Gray."]})[0][:2]
        ('a', ' Gray.')
    """
    order = sorted(scores, key=lambda v: -scores[v]["border_neutral"])
    return [(v, (variants[v][0] + " … " if variants[v][0] else "") + variants[v][1] if v in variants else "?", scores[v]) for v in order]


def suffix_section():
    """Command (copies assets). Entry 1: the suffix search table and every image, chosen suffix first."""
    variants = json.loads((SUFFIX_DIR / "variants.json").read_text())
    scores = json.loads((SUFFIX_DIR / "scores_img768.json").read_text())["table"]
    prompts = json.loads(FG_PROMPTS.read_text())
    rows = suffix_rows(scores, variants)
    table = "".join(
        f'<tr{" class=chosen" if v == CHOSEN else ""}><td><b>{html.escape(v)}</b>{" (chosen)" if v == CHOSEN else ""}</td>'
        f'<td>{html.escape(text) if text else "(no suffix: the plain subject-only prompt)"}</td>'
        f'<td>{s["border_neutral"]:.0%}</td><td>{s["border_std"]:.1f}</td><td>{s["border_mean"]:.0f}</td></tr>'
        for v, text, s in rows)
    grids = []
    for v, text, s in rows:
        figures = []
        for p in prompts:
            src = SUFFIX_DIR / "img768" / f"{v}_{p['id']}.png"
            if not src.exists():
                continue
            rel = copy_asset(src, ASSETS / "suffix" / src.name).relative_to(j.ROOT / "journal")
            figures.append(f'<figure><img data-src="{rel}" alt="{html.escape(p["short"])}" loading="lazy"><figcaption>{p["id"]}: {html.escape(p["short"])} &middot; seed {p["seed"]}</figcaption></figure>')
        grids.append(f'<details class="sub"{" open" if v == CHOSEN else ""}><summary>{html.escape(v)}: border score {s["border_neutral"]:.0%}</summary>'
                     f'<p class="eq">W_H3, CFG 1, 768&times;768, prompt = subject-only prompt{" + " + html.escape(repr(text)) if text else ""}</p><div class="grid">{"".join(figures)}</div></details>')
    videos = []
    for src in sorted((SUFFIX_DIR / "vid768").glob("*.mp4")):
        v, pid = src.stem.rsplit("_", 1)
        rel = copy_asset(src, ASSETS / "suffix_video" / src.name).relative_to(j.ROOT / "journal")
        short = next(p["short"] for p in prompts if p["id"] == pid)
        videos.append(f'<figure><video muted loop autoplay playsinline controls preload="none" data-src="{rel}"></video><figcaption>{html.escape(v)} &middot; {pid}: {html.escape(short)} &middot; 141 frames, 768&times;768</figcaption></figure>')
    finding = NOTES.get("suffix_finding", "")
    return (f'<details open><summary>Entry 1: prompt-only gray (stock H3, no adapter)</summary>'
            f'<p>Stock H3 at CFG 1, the 20 subject-only prompts with their fixed seeds, 768&times;768 images, once per candidate suffix '
            f'(round 1: 12 phrasings; round 2: 4 variants of the round-1 winner). Score: share of border pixels that are a flat neutral mid-gray (see the glossary); '
            f'also the spread (std) and mean level of the border, 50% gray = 128.</p>{finding}'
            f'<div class="wide"><table class="scores"><thead><tr><th>variant</th><th>text added</th><th>border score</th><th>border std</th><th>border level</th></tr></thead><tbody>{table}</tbody></table></div>'
            f'{"".join(grids)}'
            f'<details class="sub"><summary>Test videos with the suffix (141 frames, 768&times;768)</summary><div class="grid">{"".join(videos)}</div></details></details>')


DRIFT_FILE = Path(__file__).parent / "alphav2_drift.json"  # video_drift.py results (border-band median alpha per frame)
DRIFT_LABELS = {"split_v2_video_fg": ("Alpha Video 1 (split v2, 512²)", "--c1"), "split_bam_video_fg": ("Alpha Video 1 ablation (split on basic ass magic)", "--c2"),
                "av2_both_video_fgray": ("v2: gray magic v3 + split v3 (768²)", "--c3"), "av2_bothv_video_fgray": ("v2 + video readout (768²)", "--c4")}


def polyline(values, width, height, top):
    """
    Pure function. SVG polyline points of a curve: x spread over width, y = height · (1 − value / top).

    Examples:
        >>> polyline([0.0, 0.5, 1.0], 100, 50, 1.0)
        '0.0,50.0 50.0,25.0 100.0,0.0'
    """
    n = max(len(values) - 1, 1)
    return " ".join(f"{i * width / n:.1f},{height * (1 - min(v, top) / top):.1f}" for i, v in enumerate(values))


def drift_html():
    """Query (reads DRIFT_FILE). The drift chart (SVG, background alpha per frame) and the per-window table."""
    data = json.loads(DRIFT_FILE.read_text())
    w, h, top = 640, 240, 0.4
    lines = "".join(f'<polyline fill="none" stroke="var({DRIFT_LABELS[t][1]})" stroke-width="2" points="{polyline(d["median_curve"], w, h, top)}"/>' for t, d in data.items())
    grid = "".join(f'<line x1="0" x2="{w}" y1="{h * (1 - v / top):.0f}" y2="{h * (1 - v / top):.0f}" stroke="var(--rule)"/><text x="{w + 4}" y="{h * (1 - v / top) + 4:.0f}" class="ax">{v:.1f}</text>' for v in (0.0, 0.1, 0.2, 0.3, 0.4))
    svg = (f'<svg viewBox="0 0 {w + 40} {h + 30}" style="max-width:100%;height:auto" role="img" aria-label="background alpha per frame">{grid}{lines}'
           f'<text x="0" y="{h + 22}" class="ax">frame 0</text><text x="{w - 60}" y="{h + 22}" class="ax">frame 140</text></svg>')
    legend = "".join(f'<li><span style="color:var({DRIFT_LABELS[t][1]})">&#9632;</span> {DRIFT_LABELS[t][0]}</li>' for t in data)
    rows = "".join(f'<tr><td>{DRIFT_LABELS[t][0]}</td>' + "".join(f'<td>{d["table"][k]["median"]:.3f}</td>' for k in ("frame 0", "frames 1-23", "frames 24-69", "frames 70-140"))
                   + f'<td>{d["table"]["per prompt frames 1-140"]["median"]:.3f} / {d["table"]["per prompt frames 1-140"]["p95"]:.3f}</td>'
                   + f'<td>{20 - d["table"]["per prompt frames 1-140"]["above_0_03"]} of {d["prompts"]}</td>'
                   + f'<td>{d["table"]["subject core"]["mean_alpha"]:.3f} ({d["table"]["subject core"]["below_0_9"]:.0%})</td></tr>' for t, d in data.items())
    return (f'<p class="note">Background alpha per frame: in each frame, the median alpha of the outer 8% border band (background, ideally 0), averaged over the 20 subject-only prompts.</p>'
            f'{svg}<ul class="note">{legend}</ul>'
            f'<table class="scores"><thead><tr><th>model</th><th>frame 0</th><th>frames 1&ndash;23</th><th>frames 24&ndash;69</th><th>frames 70&ndash;140</th>'
            f'<th>per prompt (frames 1&ndash;140): median / 95th pct</th><th>prompts below 0.03</th><th>solid subject: mean alpha in the core (share of core pixels below 0.9)</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="note">Solid subject: in every 10th frame, the subject core = pixels with alpha &gt; 0.5, eroded by 15 px (edges, hair and spray excluded); its mean alpha should be 1.</p>')


def note_assets(text):
    """
    Pure function. The scratch files a note's HTML references: assets/h3magic_alphav2/taste/<name> -> <name>.

    Examples:
        >>> note_assets('<img data-src="assets/h3magic_alphav2/taste/a_b.png"> <a href="assets/h3magic_alphav2/taste/c.mp4">')
        ['a_b.png', 'c.mp4']
    """
    import re
    return re.findall(r"assets/h3magic_alphav2/taste/([\w.-]+)", text)


# Test tables (m00983: "tests showing both adapters, zero adapters, and just the split adapter"), each a list of columns
# (title, model_tag, kind); kind gray = the model's own output, alpha / rgba / white / black = split outputs. Alpha Video 1
# columns use the subject-only prompts without the suffix (f01-f20: same subject, motion and seed as g01-g20), at 512².
AV1_V, AV1_I = "Alpha Video 1 (&sect;22 split, full chain, 512&sup2;)", "Alpha Video 1 (&sect;22 split, full chain, 512&sup2;)"
TESTS = (
    ("Video, subject-only prompts + suffix (g01&ndash;g20), 141 frames 768&sup2;", "video", "g", (
        ("zero adapters (stock + suffix)", "av2_stock_video_fgray", "gray"),
        ("gray magic v3", "av2_gm_video_fgray", "gray"),
        ("gray magic v3 + split v3: gray head", "av2_both_video_fgray", "gray"),
        ("gray magic v3 + split v3: alpha", "av2_both_video_fgray", "alpha"),
        ("gray magic v3 + split v3: RGBA", "av2_both_video_fgray", "rgba"),
        ("+ video readout (entry 5): alpha", "av2_bothv_video_fgray", "alpha"),
        ("+ video readout (entry 5): RGBA", "av2_bothv_video_fgray", "rgba"),
        ("split v3 only: RGBA", "av2_split_video_fgray", "rgba"),
        (AV1_V + ": alpha", "split_v2_video_fg", "alpha"),
        (AV1_V + ": RGBA", "split_v2_video_fg", "rgba"))),
    ("Images, subject-only prompts + suffix (g01&ndash;g20), 768&sup2;", "image", "g", (
        ("zero adapters (stock + suffix)", "av2_stock_image_fgray", "gray"),
        ("gray magic v3", "av2_gm_image_fgray", "gray"),
        ("gray magic v3 + split v3: gray head", "av2_both_image_fgray", "gray"),
        ("gray magic v3 + split v3: alpha", "av2_both_image_fgray", "alpha"),
        ("gray magic v3 + split v3: RGBA", "av2_both_image_fgray", "rgba"),
        ("+ video readout (entry 5): RGBA", "av2_bothv_image_fgray", "rgba"),
        ("split v3 only: RGBA", "av2_split_image_fgray", "rgba"),
        (AV1_I + ": RGBA", "split_v2_image_fg", "rgba"))),
    ("Video, main prompts + suffix (m01&ndash;m20; the prompts describe scenes), 141 frames 768&sup2;", "video", "m", (
        ("zero adapters (stock + suffix)", "av2_stock_video_mgray", "gray"),
        ("gray magic v3 + split v3: gray head", "av2_both_video_mgray", "gray"),
        ("gray magic v3 + split v3: RGBA", "av2_both_video_mgray", "rgba"),
        ("+ video readout (entry 5): RGBA", "av2_bothv_video_mgray", "rgba"),
        (AV1_V + ": RGBA", "split_v2_video", "rgba"))),
    ("Images, main prompts + suffix (m01&ndash;m20), 768&sup2;", "image", "m", (
        ("zero adapters (stock + suffix)", "av2_stock_image_mgray", "gray"),
        ("gray magic v3 + split v3: gray head", "av2_both_image_mgray", "gray"),
        ("gray magic v3 + split v3: RGBA", "av2_both_image_mgray", "rgba"),
        (AV1_I + ": RGBA", "split_v2_image", "rgba"))),
)
SET_LETTER = {"fg": "f", "fgray": "g", "mgray": "m", "vgray": "v", None: "p", "main": "p"}  # prompt_set -> prompt id letter


def adapter_name(folder):
    """
    Pure function. Display name of an adapter folder: its run name, plus the step when not final.

    Examples:
        >>> adapter_name("models/h3magic/graymagic_v3/final"), adapter_name("models/h3magic/split_v3/step_00150")
        ('graymagic_v3', 'split_v3 (step 150)')
    """
    run, step = folder.rstrip("/").split("/")[-2:]
    return run if step == "final" else f"{run} (step {int(step.split('_')[-1])})"


def av2_equation(row):
    """
    Pure function. The weight equation of an av2 row: W_H3 + c·Δ_graymagic + γ·Δ_split, with CFG when not 1.

    Examples:
        >>> av2_equation({"c_graymagic": 1.0, "graymagic_adapter": "models/h3magic/graymagic_v3/final", "split_adapter": None, "cfg": None})
        'W_H3 + 1·Δ_graymagic_v3'
        >>> av2_equation({"c_graymagic": 0.0, "graymagic_adapter": None, "split_adapter": "models/h3magic/split_v3/final", "g_split": 1.0, "cfg": None})
        'W_H3 + 1·Δ_split_v3'
    """
    terms = ["W_H3"]
    if row.get("graymagic_adapter"):
        terms.append(f"{row['c_graymagic']:g}·Δ_{adapter_name(row['graymagic_adapter'])}")
    if row.get("split_adapter"):
        terms.append(f"{row['g_split'] or 0:g}·Δ_{adapter_name(row['split_adapter'])}")
    cfg = f", CFG {row['cfg']:g}" if (row.get("cfg") or 1) != 1 else ""
    return " + ".join(terms) + cfg


def prompt_id_for(model, number):
    """
    Pure function. The prompt id of one model row's prompt set for a prompt number ("07").

    Examples:
        >>> prompt_id_for({"prompt_set": "fg"}, "07"), prompt_id_for({"prompt_set": None}, "07"), prompt_id_for({"prompt_set": "fgray"}, "07")
        ('f07', 'p07', 'g07')
    """
    return SET_LETTER[model.get("prompt_set")] + number


def tasting_tables(models_by_tag):
    """
    Pure function. The tasting tables (TESTS format) from the av2 checkpoint rows: gray magic v3 steps (gray output) and
    split v3 steps (alpha and RGBA), images on the subject-only prompts + suffix, in step order.

    Examples:
        >>> rows = {"av2_gm_s00100_image_fgray": {}, "av2_split_s00050_image_fgray": {}, "av2_stock_image_fgray": {}}
        >>> [(t[0][:22], [c[1] for c in t[3]]) for t in tasting_tables(rows)][:2]
        [('Tasting: gray magic v3', ['av2_stock_image_fgray', 'av2_gm_s00100_image_fgray']), ('Tasting: split v3 chec', ['av2_split_s00050_image_fgray', 'av2_split_s00050_image_fgray'])]
    """
    import re
    steps = lambda kind: sorted(t for t in models_by_tag if re.fullmatch(f"av2_{kind}_s\\d{{5}}_image_fgray", t))
    step = lambda tag: int(re.search(r"_s(\d{5})_", tag).group(1))
    gm = [("zero adapters (stock + suffix)", "av2_stock_image_fgray", "gray")] + [(f"step {step(t)}", t, "gray") for t in steps("gm")]
    split = [c for t in steps("split") for c in ((f"step {step(t)}: alpha", t, "alpha"), (f"step {step(t)}: RGBA", t, "rgba"))]
    readout = sorted(t for t in models_by_tag if re.fullmatch(r"av2_bothv_s\d{5}_video_vgray", t))
    video = [c for t in readout for c in ((f"readout step {step(t)}: alpha", t, "alpha"), (f"readout step {step(t)}: RGBA", t, "rgba"))]
    return (("Tasting: gray magic v3 checkpoints (images, g01&ndash;g20, 768&sup2;)", "image", "g", tuple(gm)),
            ("Tasting: split v3 checkpoints (gray magic v3 + split step; images, g01&ndash;g20, 768&sup2;)", "image", "g", tuple(split)),
            ("Tasting: video readout checkpoints (gray magic v3 + split v3 + readout step; 4 videos)", "video", "v", tuple(video)))


def test_tables(models_by_tag, tables=TESTS):
    """Command (copies outputs to journal/assets/h3magic/<tag>/). Comparison tables (TESTS format), pending cells marked."""
    used = {tag for _, _, _, cols in tables for _, tag, _ in cols if tag in models_by_tag}
    plain = [models_by_tag[t] for t in used if not models_by_tag[t].get("split_adapter")]
    split = [models_by_tag[t] for t in used if models_by_tag[t].get("split_adapter")]
    found_plain, found_split = j.export_outputs(plain), js.export_split(split)
    sources = {"g": j.ROOT / "h3magic/eval_prompts_fgray.json", "m": j.ROOT / "h3magic/eval_prompts_mgray.json", "v": j.ROOT / "h3magic/eval_prompts_vgray.json"}
    html_out = []
    for title, mode, letter, cols in tables:
        prompts = json.loads(sources[letter].read_text())
        head = "".join(f"<th>{c}</th>" for c, _, _ in cols)
        eqs = []
        for c, tag, _ in cols:
            m = models_by_tag.get(tag)
            if m and (c.endswith("RGBA") or not m.get("split_adapter")):
                eq = av2_equation(m) if tag.startswith("av2_") else js.split_equation(m)
                eqs.append(f"<li>{c}: <span class=eq>{html.escape(eq)}</span> ({tag})</li>")
        body = []
        for p in prompts:
            number = p["id"][1:]
            cells = []
            for c, tag, kind in cols:
                m = models_by_tag.get(tag)
                pid = prompt_id_for(m, number) if m else None
                done = m and ((pid in found_split.get(tag, {})) if m.get("split_adapter") else (pid in found_plain.get(tag, {})))
                if not done:
                    cells.append('<td class="missing">pending</td>')
                    continue
                path = js.asset(tag, pid, kind, mode) if m.get("split_adapter") else j.media_path(tag, pid, mode)
                cells.append(js.cell(path, mode, kind, p["short"]))
            body.append(f'<tr><th class="label">{number}: {html.escape(p["short"])}<br><span class="note">seed {p["seed"]}</span></th>{"".join(cells)}</tr>')
        html_out.append(f'<details class="row" open><summary>{title}</summary><ul class="note">{"".join(eqs)}</ul>'
                        f'<div class="wide"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')
    return "".join(html_out)


def build():
    """Command. Writes journal/h3magic_alphav2.html from the suffix search outputs and the later entries that exist."""
    entries = [NOTES[k].replace("{{DRIFT}}", drift_html()) for k in sorted(NOTES) if k.startswith("entry_")]
    for name in note_assets("".join(entries)):
        copy_asset(TASTE_DIR / name, ASSETS / "taste" / name)
    models_by_tag = {m["model_tag"]: m for m in j.load_models()}
    tests = f'<h2>Tests (subject-only prompts first)</h2>{NOTES.get("tests_finding", "")}{test_tables(models_by_tag)}'
    tests += f'<h2>Tasting during training</h2>{NOTES.get("tasting_finding", "")}{test_tables(models_by_tag, tasting_tables(models_by_tag))}'
    sections = [suffix_section()] + entries + [tests]
    glossary = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    css = js.CSS + ":root { --media: 384px; --c1: #c0504d; --c2: #e39b3c; --c3: #7f7f7f; --c4: #2f7ed8; } .ax { font: 11px system-ui, sans-serif; fill: var(--muted); } tr.chosen td { font-weight: bold; } table.scores td { font: var(--font-small)/1.4 system-ui, sans-serif; padding: 4px 8px; border-bottom: 1px solid var(--rule); }"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Alpha Video v2</title>
<style>{css}</style>
</head>
<body>
<main>
<h1>H3 Magic: Alpha Video v2 (gray by prompt, 768&times;768, two adapters)</h1>
<p class="note">Every image and video is the unmodified output (768&times;768). Times are Eastern.</p>
{INTRO}
{"".join(sections)}
<details><summary>Glossary</summary><dl>{glossary}</dl></details>
</main>
<dialog id="lightbox"></dialog>
<script>{js.JS}
document.querySelectorAll('figure img, figure video').forEach(el => el.onclick = () => {{
  box.innerHTML = el.tagName === 'VIDEO' ? `<video muted loop autoplay playsinline controls src="${{el.src}}"></video>` : `<img src="${{el.src}}">`;
  box.showModal();
}});</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(j.ROOT)}", flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build})
