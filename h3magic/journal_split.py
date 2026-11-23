# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance", "fire"]
# ///
"""
Builds the split adapter page (manifest §22, m00972, m00973): journal/h3magic_split.html. For each split model row of
h3_eval_models (split_adapter set) and each of the 20 eval prompts: the gray, white and black heads, the alpha and the
RGBA on a checkerboard, as images and as videos, each labeled with its weight equation. Plus the γ = 0 sanity check
(the split model with γ = 0 must equal its base row exactly). Outputs are copied byte for byte (h3magic/journal.py rule)
to journal/assets/h3magic/<model_tag>/<prompt_id>_<kind>.{mp4,png}. Rerun any time:

    uv run --script h3magic/journal_split.py build
"""

import html
import json

import fire
import lance

import journal as j

PAGE = j.ROOT / "journal/h3magic_split.html"
KINDS = (("gray", "gray head"), ("white", "white head"), ("black", "black head"), ("alpha", "alpha (white = opaque)"), ("rgba", "RGBA on a checkerboard"))
# (section title, model tag, open by default); a tag with no row or no output yet is shown as pending
SECTIONS = (
    ("Split adapter on the full chain, images", "split_v2_image", True),
    ("Split adapter on the full chain, video (like row 5a)", "split_v2_video", True),
    ("Split adapter, gray magic only (Pexels 0, image-gen 0), video (like row 5b)", "split_v2_graymagic_video", True),
    ("Split adapter, gray magic only, images", "split_v2_graymagic_image", False),
    ("Split adapter, Pexels + gray magic (image-gen 0), video (like row 5c)", "split_v2_pexels_gray_video", False),
    ("Split adapter, Pexels + gray magic, images", "split_v2_pexels_gray_image", False),
    ("Split adapter alone on stock H3, CFG 2.5, images (m00973)", "split_v2_only_cfg_image", True),
    ("Split adapter alone on stock H3, CFG 2.5, video (m00973)", "split_v2_only_cfg_video", True),
    ("Split adapter alone on stock H3, CFG 1, images (comparison)", "split_v2_only_image", False),
    ("Split adapter alone on stock H3, CFG 1, video (comparison)", "split_v2_only_video", False),
    ("Ablation: split adapter on \"basic ass magic\" v2, images", "split_bam_image", True),
    ("Ablation: split adapter on \"basic ass magic\" v2, video", "split_bam_video", True),
)
GAMMA0 = (("split_v2_g0_image", "graymagic_image_v2"), ("split_v2_g0_video", "all_video_v2"))  # (γ = 0 row, its base row)

INTRO = (
    "<p>The split adapter adds two more outputs to H3: next to the subject on 50% gray, the same subject on white and on black. "
    "From the three, a per-pixel least-squares fit recovers the alpha (transparency) and the subject's true color, so the model outputs a cut-out (RGBA).</p>"
    "<p>How it works: H3's last layer (24 latent channels &times; a 2&times;2 patch = 96 numbers per patch) gets three copies, gray, white and black. "
    "The copies start equal to the original (a zero-initialized difference &Delta;), and one new adapter (a LoRA plus that difference) learns to make them differ. "
    "All three share one denoising run: the model follows the gray image as before, and at the last step each copy gives its own clean image. "
    "&gamma; scales the whole split adapter: &gamma; = 0 is exactly the model without it.</p>"
    "<p>It was trained twice: on top of the full adapter chain (Pexels + image-gen + gray magic, baked in), and on top of the \"basic ass magic\" ablation adapter. "
    "Training data: the same 10,000 MAGICK images, composited on gray, white and black, all CFG-1 guidance-loss training. Every item is the 512&times;512 original.</p>")

GLOSSARY = tuple((t, d) for t, d in j.GLOSSARY if t in ("H3", "Adapter / LoRA", "Bake", "MAGICK", "Gray magic")) + (
    ("Split adapter (&Delta;_split)", "The adapter on this page: a LoRA plus a difference &Delta; on H3's last layer that makes three outputs (gray, white, black background) instead of one."),
    ("Head", "One of the three copies of the last layer: the gray head (the model's usual output), the white head and the black head."),
    ("&gamma; (gamma)", "How much the split adapter is applied. 0 = off (the three heads are equal and the gray one is exactly the base model), 1 = fully on."),
    ("Alpha", "Transparency per pixel. Shown as gray: white = opaque subject, black = clear background."),
    ("RGBA", "The cut-out: color plus alpha. Shown over a checkerboard, so clear parts show the checkerboard."),
    ("Least squares fit", "Per pixel, the alpha and color that best explain the three images C_k = &alpha;&middot;F + (1 &minus; &alpha;)&middot;b_k for backgrounds b_k = 0.5, 1, 0 (iterative, box-constrained to [0, 1])."),
    ("Residual", "How far the three heads are from one consistent cut-out (RMS, 0 to 1). 0 means they agree perfectly."),
    ("CFG", "Classifier-free guidance scale. CFG 1 = one model pass per step (H3's design). CFG 2.5 = an extra pass with an empty prompt, applied to every head."),
    ("basic ass magic", "The ablation adapter: one LoRA trained on gray MAGICK images directly on stock H3, with no Pexels or image-gen stage."),
)

CSS = j.CSS + """
:root { --cell: 256px; --checker-a: #ffffff; --checker-b: #cccccc; --checker: 16px; --lightbox: 94vmin; --label-min: 120px; --label-max: 160px; }
.wide { overflow-x: auto; }
table { border-collapse: collapse; }
th, td { padding: calc(var(--pad) / 2); vertical-align: top; }
thead th { position: sticky; top: 0; background: var(--bg); font: bold var(--font-small)/1.3 system-ui, sans-serif; text-align: left; z-index: 1; }
th.label { font: var(--font-small)/1.3 system-ui, sans-serif; text-align: left; min-width: var(--label-min); max-width: var(--label-max); }
td img, td video { width: var(--cell); height: var(--cell); display: block; background: #000; cursor: zoom-in; }
td img.rgba { background-color: var(--checker-a); background-image: conic-gradient(var(--checker-b) 25%, var(--checker-a) 0 50%, var(--checker-b) 0 75%, var(--checker-a) 0);
              background-size: var(--checker) var(--checker); }
td.missing { color: var(--muted); font: var(--font-small) system-ui, sans-serif; }
dialog { border: 0; padding: 0; background: transparent; max-width: none; }
dialog::backdrop { background: rgba(0, 0, 0, 0.85); }
dialog img, dialog video { width: var(--lightbox); max-width: 100vw; object-fit: contain; display: block; }
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
  const video = td.dataset.mode === 'video', cls = td.dataset.kind === 'rgba' ? ' class="rgba"' : '';
  box.innerHTML = video ? `<video muted loop autoplay playsinline controls src="${td.dataset.full}"></video>` : `<img${cls} src="${td.dataset.full}">`;
  box.showModal();
});
"""


def split_equation(row):
    """
    Pure function. The weight equation of a split model row: the chain (h3magic/journal.py weight_equation, or the
    basic-ass-magic ablation adapter) + γ·Δ_split, then the CFG.

    Examples:
        >>> split_equation({"model_tag": "split_v2_only_cfg_image", "a_pexels": 0.0, "b_imagegen": 0.0, "c_graymagic": 0.0, "pexels_adapter": "p",
        ...                 "imagegen_adapter": "i", "graymagic_adapter": "g", "split_adapter": "models/h3magic/split_v2/final", "g_split": 1.0, "cfg": 2.5})
        'W_H3 + 0·Δ_pexels + 0·Δ_imagegen + 0·Δ_graymagic + 1·Δ_split, CFG 2.5'
        >>> split_equation({"model_tag": "split_bam_image", "a_pexels": 0.0, "b_imagegen": 0.0, "c_graymagic": 1.0, "pexels_adapter": None,
        ...                 "imagegen_adapter": None, "graymagic_adapter": "models/h3magic/basic_ass_magic_v2/final",
        ...                 "split_adapter": "models/h3magic/split_bam/final", "g_split": 1.0, "cfg": None})
        'W_H3 + 1·Δ_basic_ass_magic_v2 + 1·Δ_split_bam'
    """
    cfg = f", CFG {row['cfg']:g}" if (row.get("cfg") or 1) != 1 else ""
    if "basic_ass_magic" in (row.get("graymagic_adapter") or ""):
        chain = f"W_H3 + {row['c_graymagic']:g}·Δ_{row['graymagic_adapter'].split('/')[-2]}"
    else:
        chain = j.weight_equation({**row, "model_tag": "", "cfg": 1})
    name = "split" if "split_v2" in (row.get("split_adapter") or "") else row["split_adapter"].split("/")[-2]
    return f"{chain} + {row['g_split'] or 0:g}·Δ_{name}{cfg}"


def asset(tag, prompt_id, kind, mode):
    """
    Pure function. Page-relative path of one exported output.

    Examples:
        >>> asset("split_v2_video", "p01", "alpha", "video")
        'assets/h3magic/split_v2_video/p01_alpha.mp4'
    """
    return f"assets/h3magic/{tag}/{prompt_id}_{kind}.{'mp4' if mode == 'video' else 'png'}"


def export_split(models):
    """
    Command. Writes every output of every split model byte for byte to journal/<asset(...)> (rewritten only when its
    bytes differ). Returns {model_tag: {prompt_id: dict(seed, seconds, residual, x0_psnr)}} of the complete outputs.
    """
    ds = lance.dataset(str(j.LANCE / "h3_eval_generations.lance"))
    found = {}
    for model in models:
        tag, mode = model["model_tag"], model["mode"]
        columns = {kind: mode if kind == "gray" else f"{mode}_{kind}" for kind, _ in KINDS}
        rows = ds.scanner(filter=f"model_tag = '{tag}' AND {columns['rgba']} IS NOT NULL", with_row_id=True,
                          columns=["prompt_id", "seed", "seconds", "split_residual", "split_x0_psnr"]).to_table().to_pylist()
        found[tag] = {}
        if not rows:
            continue
        for kind, column in columns.items():
            for row, blob in zip(rows, ds.take_blobs(column, ids=[r["_rowid"] for r in rows])):
                out = j.ROOT / "journal" / asset(tag, row["prompt_id"], kind, mode)
                out.parent.mkdir(parents=True, exist_ok=True)
                data = blob.read()
                if not out.exists() or out.read_bytes() != data:
                    out.write_bytes(data)
        found[tag] = {r["prompt_id"]: dict(seed=r["seed"], seconds=r["seconds"], residual=r["split_residual"], x0_psnr=r["split_x0_psnr"]) for r in rows}
    return found


def cell(path, mode, kind, label):
    """
    Pure function. One table cell: a lazy muted looping video or an image (RGBA over a CSS checkerboard), click to enlarge.

    Examples:
        >>> 'class="rgba"' in cell("a.png", "image", "rgba", "Cat"), 'data-src="a.mp4"' in cell("a.mp4", "video", "gray", "Cat")
        (True, True)
    """
    media = (f'<video muted loop playsinline preload="none" data-src="{path}"></video>' if mode == "video"
             else f'<img{" class=" + chr(34) + "rgba" + chr(34) if kind == "rgba" else ""} data-src="{path}" alt="{html.escape(label)}" loading="lazy">')
    return f'<td data-full="{path}" data-mode="{mode}" data-kind="{kind}">{media}</td>'


def section(title, model, found, prompts, open_):
    """Pure function. One <details>: per prompt a row of gray | white | black | alpha | RGBA, with the equation and fit numbers."""
    tag = model["model_tag"] if model else None
    items = found.get(tag, {})
    if not items:
        return f'<details class="row"><summary>{html.escape(title)} <span class="note">(rendering)</span></summary></details>'
    head = "".join(f"<th>{html.escape(name)}</th>" for _, name in KINDS)
    body = []
    for p in prompts:
        if p["id"] not in items:
            body.append(f'<tr><th class="label">{p["id"]}: {html.escape(p["short"])}</th><td class="missing" colspan="{len(KINDS)}">rendering</td></tr>')
            continue
        it = items[p["id"]]
        label = (f'{p["id"]}: {html.escape(p["short"])}<br><span class="note">seed {it["seed"]} &middot; {it["seconds"]:.0f} s &middot; '
                 f'residual {it["residual"]:.3f}</span>')
        body.append(f'<tr><th class="label">{label}</th>' + "".join(cell(asset(tag, p["id"], kind, model["mode"]), model["mode"], kind, p["short"]) for kind, _ in KINDS) + "</tr>")
    mean = sum(it["residual"] for it in items.values()) / len(items)
    return (f'<details class="row"{" open" if open_ else ""}><summary>{html.escape(title)} <span class="note">({len(items)}/{len(prompts)} prompts)</span></summary>'
            f'<p class="eq">{html.escape(split_equation(model))}</p><p class="note">Model tag {tag}. Mean fit residual {mean:.4f}. Click any cell for the full 512&times;512 original.</p>'
            f'<div class="wide"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')


def gamma0_check(by_tag, prompts):
    """
    Query (reads the generations). The γ = 0 sanity table: for each (γ = 0 row, base row), the prompts whose gray output
    is byte-identical to the base row's output. Returns HTML.
    """
    ds = lance.dataset(str(j.LANCE / "h3_eval_generations.lance"))
    lines = []
    for zero, base in GAMMA0:
        if zero not in by_tag:
            lines.append(f"<li>{zero}: not rendered yet</li>")
            continue
        column = by_tag[zero]["mode"]

        def outputs(tag):
            """Query. prompt_id -> output bytes of one row."""
            rows = ds.scanner(filter=f"model_tag = '{tag}' AND {column} IS NOT NULL", columns=["prompt_id"], with_row_id=True).to_table().to_pylist()
            return {r["prompt_id"]: b.read() for r, b in zip(rows, ds.take_blobs(column, ids=[r["_rowid"] for r in rows]))} if rows else {}

        a, b = outputs(zero), outputs(base)
        both = sorted(set(a) & set(b))
        same = sum(a[p] == b[p] for p in both)
        lines.append(f"<li><b>{zero}</b> ({html.escape(split_equation(by_tag[zero]))}) vs <b>{base}</b>: {same} of {len(both)} prompts byte-identical</li>")
    return "<ul>" + "".join(lines) + "</ul>"


def build():
    """Command. Exports the split rows' outputs and writes journal/h3magic_split.html (pending rows show as rendering)."""
    prompts = json.loads(j.PROMPTS.read_text())
    models = [m for m in j.load_models() if m.get("split_adapter")]
    by_tag = {m["model_tag"]: m for m in models}
    found = export_split(models)
    sections = [section(title, by_tag.get(tag), found, prompts, open_) for title, tag, open_ in SECTIONS]
    extra = [m for m in models if m["model_tag"] not in {t for _, t, _ in SECTIONS} | {z for z, _ in GAMMA0}]
    if extra:  # tasting checkpoints and any other split rows
        inner = "".join(section(f"{m['model_tag']}", m, found, prompts, False) for m in extra)
        sections.append(f'<details class="row"><summary>Training checkpoints and other rows</summary>{inner}</details>')
    glossary = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    prompt_blocks = "".join(f'<details class="sub"><summary>{p["id"]}: {html.escape(p["short"])} (seed {p["seed"]})</summary><p class="prompt">{html.escape(p["prompt"])}</p></details>' for p in prompts)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Magic: split adapter</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: the split adapter (gray, white, black, alpha, RGBA)</h1>
<p class="note">Same 20 prompts and fixed seeds as <a href="h3magic_final.html">the results page</a>. Every item is the unmodified 512&times;512 output; each section shows its weight equation.</p>
{INTRO}
<details open><summary>Sanity check: &gamma; = 0 gives the base model</summary><p class="note">With &gamma; = 0 the split adapter is off, so its gray output must be exactly the base row's.</p>{gamma0_check(by_tag, prompts)}</details>
{"".join(sections)}
<details><summary>The 20 prompts (full text)</summary>{prompt_blocks}</details>
<details><summary>Glossary</summary><dl>{glossary}</dl></details>
</main>
<dialog id="lightbox"></dialog>
<script>{JS}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(j.ROOT)}: " + ", ".join(f"{t} {len(v)}" for t, v in found.items()), flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build})
