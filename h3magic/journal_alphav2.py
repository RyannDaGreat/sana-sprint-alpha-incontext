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


def note_assets(text):
    """
    Pure function. The scratch files a note's HTML references: assets/h3magic_alphav2/taste/<name> -> <name>.

    Examples:
        >>> note_assets('<img data-src="assets/h3magic_alphav2/taste/a_b.png"> <a href="assets/h3magic_alphav2/taste/c.mp4">')
        ['a_b.png', 'c.mp4']
    """
    import re
    return re.findall(r"assets/h3magic_alphav2/taste/([\w.-]+)", text)


def build():
    """Command. Writes journal/h3magic_alphav2.html from the suffix search outputs and the later entries that exist."""
    entries = [NOTES[k] for k in sorted(NOTES) if k.startswith("entry_")]
    for name in note_assets("".join(entries)):
        copy_asset(TASTE_DIR / name, ASSETS / "taste" / name)
    sections = [suffix_section()] + entries
    glossary = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    css = js.CSS + ":root { --media: 384px; } tr.chosen td { font-weight: bold; } table.scores td { font: var(--font-small)/1.4 system-ui, sans-serif; padding: 4px 8px; border-bottom: 1px solid var(--rule); }"
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
