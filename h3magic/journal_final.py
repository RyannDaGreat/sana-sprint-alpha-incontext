# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance", "fire"]
# ///
"""
Builds the one clean H3 Magic page (manifest §22, m00971): journal/h3magic_final.html. Only the working, retrained (v2)
rows, generated at CFG 1: an overview table (20 prompts x 7 stages), then one full-size section per stage.
Reuses the helpers and the byte-for-byte export of h3magic/journal.py. Rerun any time:

    uv run --script h3magic/journal_final.py build
"""

import html
import json
from pathlib import Path

import fire

import journal as j

PAGE = j.ROOT / "journal/h3magic_final.html"
OVERVIEW_PX = 200  # display width of an overview cell; the files are the 512x512 originals
LAST_STEP_NOTE = "step 500 of 544"

# (key, column header in words, section title, model tags in order of preference, open by default, note when a fallback tag is used)
STAGES = (
    ("1", "1: stock H3", "Row 1: stock H3, video", ("stock_video",), False, ""),
    ("2", "2: H3 + Pexels", "Row 2: Pexels adapter, video", ("pexels_video_v2",), False, ""),
    ("3", "3: H3 + Pexels + image-gen (images)", "Row 3: image-gen adapter, images", ("imagegen_image_v2", "imagegen2_s00500_image"), False, LAST_STEP_NOTE),
    ("4", "4: H3 + all adapters, gray magic (images)", "Row 4: gray magic, images", ("graymagic_image_v2",), False, ""),
    ("5a", "5a: H3 + all adapters on", "Row 5a: all adapters on, video", ("all_video_v2",), True, ""),
    ("5b", "5b: H3 + gray magic only", "Row 5b: H3 + gray magic only (Pexels 0, image-gen 0), video", ("graymagic_video_v2",), True, ""),
    ("5c", "5c: H3 + Pexels + gray magic", "Row 5c: H3 + Pexels + gray magic (image-gen 0), video", ("pexels_gray_video_v2",), True, ""),
)
ABLATION_STAGES = (
    ("A1", "Ablation \"basic ass magic\": one LoRA on stock H3, trained on gray MAGICK only, images", "basic_ass_magic_image_v2"),
    ("A2", "Ablation \"basic ass magic\": the same LoRA, video", "basic_ass_magic_video_v2"),
)

INTRO = (
    "<p>We stack adapters (LoRAs) on MiniMax H3, a text-to-video model, so that it draws a subject on a flat 50% gray background, as video and as images. "
    "The adapters are trained one on top of the other: Pexels clips (video look), image-gen (single frames), then gray magic (our MAGICK subjects on gray).</p>"
    "<p>One thing went wrong and was fixed: our first adapters broke H3's built-in guidance (H3 has classifier-free guidance baked in, and plain training removed it, so outputs came out garbled). "
    "We retrained with a guidance loss. Every row on this page is a retrained one, generated at CFG 1. The full history is in <a href=\"h3magic.html\">the long journal page</a>.</p>"
    "<p>How to read this page: the overview table is the same 20 prompts (rows) across 7 stages (columns), shown small. Click any cell for the full 512&times;512 original. "
    "Below it, each stage again at full size, with seed, seconds and weight equation. 5a, 5b and 5c are open; the rest are collapsed.</p>")

EXTRA_GLOSSARY = (
    ("CFG", "Classifier-free guidance scale. CFG 1 means no extra guidance pass: the model runs once per step. H3 is built for CFG 1."),
    ("Guidance loss", "A training loss (musubi-tuner's method) that keeps H3's baked-in guidance while training an adapter, so the adapter works at CFG 1."),
    ("Ablation", "A control experiment with one thing changed. Here: one LoRA trained on gray MAGICK only, directly on stock H3, with no Pexels or image-gen stage before it."),
)
GLOSSARY = tuple((t, d) for t, d in j.GLOSSARY if t in ("H3", "Adapter / LoRA", "Bake", "Pexels", "MAGICK", "Gray magic")) + EXTRA_GLOSSARY

CSS = j.CSS + f"""
:root {{ --overview: {OVERVIEW_PX}px; --lightbox: 94vmin; --head: 40px; }}
.wide {{ overflow-x: auto; }}
table {{ border-collapse: collapse; }}
td {{ min-width: var(--overview); }}
th, td {{ padding: calc(var(--pad) / 2); vertical-align: top; }}
thead th {{ position: sticky; top: 0; background: var(--bg); font: bold var(--font-small)/1.3 system-ui, sans-serif; max-width: var(--overview); text-align: left; z-index: 1; }}
th.label {{ font: var(--font-small)/1.3 system-ui, sans-serif; text-align: left; min-width: 120px; }}
td img, td video {{ width: var(--overview); height: var(--overview); display: block; background: #000; cursor: zoom-in; }}
td.missing {{ color: var(--muted); font: var(--font-small) system-ui, sans-serif; }}
dialog {{ border: 0; padding: 0; background: transparent; max-width: none; }}
dialog::backdrop {{ background: rgba(0, 0, 0, 0.85); }}
dialog img, dialog video {{ width: var(--lightbox); height: var(--lightbox); max-width: 100vw; object-fit: contain; display: block; background: #000; }}
"""

JS = """
const observer = new IntersectionObserver(entries => entries.forEach(e => {
  const el = e.target;
  if (!e.isIntersecting) { if (el.tagName === 'VIDEO') el.pause(); return; }
  if (!el.getAttribute('src')) el.src = el.dataset.src;
  if (el.tagName === 'VIDEO') el.play().catch(err => console.error('play failed', el.src, err));
}), {rootMargin: '200px'});
document.querySelectorAll('[data-src]').forEach(el => observer.observe(el));
const box = document.getElementById('lightbox');
box.onclick = () => { box.close(); box.innerHTML = ''; };
document.querySelectorAll('td[data-full]').forEach(td => td.onclick = () => {
  const video = td.dataset.mode === 'video';
  box.innerHTML = video ? `<video muted loop autoplay playsinline controls src="${td.dataset.full}"></video>` : `<img src="${td.dataset.full}">`;
  box.showModal();
});
"""


def overview_cell(path, mode, label):
    """
    Pure function. One overview table cell: a lazy muted looping video or an image, clickable to the full original.

    Examples:
        >>> overview_cell("a/b.mp4", "video", "Cat").count("data-full")
        1
        >>> 'data-src="a/b.png"' in overview_cell("a/b.png", "image", "Cat")
        True
    """
    el = ('<video muted loop playsinline preload="none" data-src="%s"></video>' if mode == "video"
          else '<img data-src="%s" alt="%s">') % ((path,) if mode == "video" else (path, html.escape(label)))
    return f'<td data-full="{path}" data-mode="{mode}">{el}</td>'


def pick_model(tags, models_by_tag, found, prompts):
    """
    Pure function. The model to show for a stage, as (model, is_fallback): the first tag whose row has an output for every
    prompt; else the first tag with any output (partly rendered); else (None, False).

    Examples:
        >>> pick_model(("a", "b"), {"a": 1, "b": 2}, {"a": {}, "b": {"p1": 1}}, [{"id": "p1"}, {"id": "p2"}])
        (2, True)
        >>> pick_model(("a", "b"), {"a": 1, "b": 2}, {"a": {"p1": 1}, "b": {"p1": 1, "p2": 1}}, [{"id": "p1"}, {"id": "p2"}])
        (2, True)
        >>> pick_model(("a",), {}, {}, [{"id": "p1"}])
        (None, False)
    """
    for need_all in (True, False):
        for i, tag in enumerate(tags):
            have = found.get(tag, {})
            if tag in models_by_tag and (all(p["id"] in have for p in prompts) if need_all else have):
                return models_by_tag[tag], i > 0
    return None, False


def overview_html(columns, prompts):
    """
    Pure function. The overview table. columns: [(header, model or None, found)]; paths come from the model's tag and mode.
    Cells whose output does not exist yet say "not rendered yet".
    """
    head = "".join(f"<th>{html.escape(h)}</th>" for h, _, _ in columns)
    rows = []
    for p in prompts:
        cells = "".join(overview_cell(j.media_path(m["model_tag"], p["id"], m["mode"]), m["mode"], p["short"]) if m and p["id"] in found_of(m, f) else '<td class="missing">not rendered yet</td>'
                        for _, m, f in columns)
        rows.append(f'<tr><th class="label">{html.escape(p["id"])}: {html.escape(p["short"])}</th>{cells}</tr>')
    return f'<div class="wide"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def found_of(model, found):
    """Pure function. The prompt ids with an output for `model`."""
    return found[model["model_tag"]]


def build():
    """
    Command. Exports the shown rows' outputs byte for byte to journal/assets/h3magic/<model_tag>/ and writes journal/h3magic_final.html.
    Rows that are not complete yet get a "rendering" note; run again when they are done.
    """
    prompts = json.loads(j.PROMPTS.read_text())
    wanted = {t for s in STAGES for t in s[3]} | {a[2] for a in ABLATION_STAGES}
    models = [m for m in j.load_models() if m["model_tag"] in wanted]
    by_tag = {m["model_tag"]: m for m in models}
    found = j.export_outputs(models)

    columns, sections, status = [], [], []
    for key, header, title, tags, open_, fallback_note in STAGES:
        model, fallback = pick_model(tags, by_tag, found, prompts)
        columns.append((header + (f" ({fallback_note})" if fallback else ""), model, found))
        if model:
            note = f'<p class="note">Showing {html.escape(model["model_tag"])} ({fallback_note}); the final checkpoint is still rendering, the page updates when done.</p>' if fallback else ""
            if len(found[model["model_tag"]]) < len(prompts):
                note += f'<p class="note">Rendering: {len(found[model["model_tag"]])}/{len(prompts)} outputs so far, the page updates when done.</p>'
            sections.append(j.section_html(title, [model], found, prompts, open_=open_).replace("</summary>", f"</summary>{note}", 1))
        else:
            sections.append(f'<details class="row"{" open" if open_ else ""}><summary>{html.escape(title)}</summary><p class="note">Rendering, the page updates when done.</p></details>')
        status.append((key, model["model_tag"] if model else None, len(found[model["model_tag"]]) if model else 0))
    for key, title, tag in ABLATION_STAGES:
        model = by_tag.get(tag)
        done = len(found.get(tag, {}))
        if model and done:
            sections.append(j.section_html(title, [model], found, prompts, open_=True))
        else:
            sections.append(f'<details class="row" open><summary>{html.escape(title)}</summary><p class="note">Rendering, the page updates when done.</p></details>')
        status.append((key, tag if done else None, done))

    glossary = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Magic: final results</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: final results</h1>
{INTRO}
<details open><summary>Overview: 20 prompts &times; 7 stages</summary><p class="note">Shown at {OVERVIEW_PX} px wide; click a cell for the 512&times;512 original. Videos play when scrolled into view.</p>{overview_html(columns, prompts)}</details>
{"".join(sections)}
<details><summary>Glossary</summary><dl>{glossary}</dl></details>
</main>
<dialog id="lightbox"></dialog>
<script>{JS}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(j.ROOT)}: " + ", ".join(f"{k} {t} {n}" for k, t, n in status), flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build})
