# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance", "fire"]
# ///
"""
Builds the H3 Magic results page (manifest §22): journal/h3magic.html plus every output, copied byte for byte
(never re-encoded or resized) to journal/assets/h3magic/<model_tag>/<prompt_id>.{mp4,png}.

Reads the BlueJay tables h3_eval_models and h3_eval_generations from data/lance. Rerun any time:

    uv run --script h3magic/journal.py build
"""

import html
import json
from pathlib import Path

import fire
import lance

ROOT = Path(__file__).resolve().parents[1]
LANCE = ROOT / "data/lance"
PAGE = ROOT / "journal/h3magic.html"
ASSETS = ROOT / "journal/assets/h3magic"
PROMPTS = ROOT / "h3magic/eval_prompts.json"
DATASET_PAGES = {"pexels": "h3magic_pexels_selection.html", "graymagic": "h3magic_dataset_graymagic.html"}
ADAPTERS = (("pexels", "a_pexels", "pexels_adapter"), ("imagegen", "b_imagegen", "imagegen_adapter"), ("graymagic", "c_graymagic", "graymagic_adapter"))

# Main rows: key -> (mode, (a, b, c)), in page order, with the title of the row.
SECTIONS = (
    ("1", "video", (0, 0, 0), "Row 1: stock H3, video"),
    ("1b", "image", (0, 0, 0), "Row 1b (bonus): stock H3, image"),
    ("2", "video", (1, 0, 0), "Row 2: Pexels adapter, video"),
    ("3", "image", (1, 1, 0), "Row 3: image-gen adapter, images"),
    ("4", "image", (1, 1, 1), "Row 4: gray magic, images"),
    ("5a", "video", (1, 1, 1), "Row 5a: all adapters on, video"),
    ("5b", "video", (0, 0, 1), "Row 5b: gray magic only, video"),
    ("5c", "video", (1, 0, 1), "Row 5c: Pexels + gray magic, video"),
    ("A1", "image", None, "Ablation \"basic ass magic\" (m00953): one LoRA on stock H3, gray MAGICK only, images"),
    ("A2", "video", None, "Ablation \"basic ass magic\" (m00953): the same LoRA, video"),
)
ABLATION = "basic_ass_magic"  # model_tag prefix of the ablation rows: its LoRA sits in the graymagic slot but was trained on stock H3
CHECKPOINT_TITLES = {"pexels": "Pexels adapter", "imagegen": "image-gen adapter", "graymagic": "gray magic adapter"}


def section_key(mode, weights, tag):
    """
    Pure function. The page section of one h3_eval_models row: a row key of SECTIONS ("1", "5b", ...), or
    "ckpt_<adapter>" for a training-checkpoint sample (model_tag contains "step"; the adapter is the last one with weight > 0).
    Raises ValueError for a weight mix that has no row.

    Examples:
        >>> section_key("video", (0, 0, 0), "stock_video")
        '1'
        >>> section_key("video", (0, 0, 1), "graymagic_only_video")
        '5b'
        >>> section_key("image", (1, 1, 0), "imagegen_step_000500")
        'ckpt_imagegen'
        >>> section_key("video", (0, 0, 1), "basic_ass_magic_video")
        'A2'
    """
    if tag.startswith(ABLATION):
        return {"image": "A1", "video": "A2"}[mode]
    if "step" in tag:
        named = [name for (name, _, _), w in zip(ADAPTERS, weights) if w > 0]
        if not named:
            raise ValueError(f"checkpoint row {tag} has no adapter weight")
        return "ckpt_" + named[-1]
    for key, sec_mode, sec_weights, _ in SECTIONS:
        if sec_weights is not None and (mode, tuple(float(w) for w in weights)) == (sec_mode, tuple(float(w) for w in sec_weights)):
            return key
    raise ValueError(f"model {tag}: mode {mode} with weights {weights} matches no journal row")


def weight_equation(row):
    """
    Pure function. "W_H3 + 1·Δ_pexels + ..." for an h3_eval_models row; a term is shown when its weight is not 0
    or its adapter folder is set (so row 5b shows its two zero terms).

    Examples:
        >>> weight_equation({"a_pexels": 0.0, "b_imagegen": 0.0, "c_graymagic": 1.0, "pexels_adapter": "p", "imagegen_adapter": "i", "graymagic_adapter": "g"})
        'W_H3 + 0·Δ_pexels + 0·Δ_imagegen + 1·Δ_graymagic'
        >>> weight_equation({"a_pexels": 0.0, "b_imagegen": 0.0, "c_graymagic": 0.0, "pexels_adapter": None, "imagegen_adapter": None, "graymagic_adapter": None})
        'W_H3'
        >>> weight_equation({"model_tag": "basic_ass_magic_image", "c_graymagic": 1.0})
        'W_H3 + 1·Δ_basic_ass_magic'
    """
    if row.get("model_tag", "").startswith(ABLATION):
        return f"W_H3 + {row['c_graymagic']:g}·Δ_{ABLATION}"
    terms = [f"{row[w]:g}·Δ_{name}" for name, w, path in ADAPTERS if row[w] or row[path]]
    return " + ".join(["W_H3", *terms])


def media_path(tag, prompt_id, mode):
    """
    Pure function. Page-relative path of one output.

    Examples:
        >>> media_path("stock_video", "p01", "video")
        'assets/h3magic/stock_video/p01.mp4'
    """
    return f"assets/h3magic/{tag}/{prompt_id}.{'mp4' if mode == 'video' else 'png'}"


def item_html(path, mode, label, seed, seconds, equation):
    """
    Pure function. One figure: a lazy-loaded video or image at its full 512² size with its caption.

    Examples:
        >>> '<img' in item_html("a.png", "image", "Cat", 1, 5.2, "W_H3") and 'data-src="a.png"' in item_html("a.png", "image", "Cat", 1, 5.2, "W_H3")
        True
    """
    tag = ('<video muted loop autoplay playsinline controls preload="none" data-src="%s"></video>' if mode == "video"
           else '<img data-src="%s" alt="%s" loading="lazy">') % ((path,) if mode == "video" else (path, html.escape(label)))
    cap = f"<b>{html.escape(label)}</b> &middot; seed {seed} &middot; {seconds:.1f} s to generate<br><span class=\"eq\">{html.escape(equation)}</span>"
    return f"<figure>{tag}<figcaption>{cap}</figcaption></figure>"


def load_models():
    """Query. h3_eval_models rows ordered by model_tag."""
    return sorted(lance.dataset(str(LANCE / "h3_eval_models.lance")).to_table().to_pylist(), key=lambda r: r["model_tag"])


def export_outputs(models):
    """
    Command. Writes every generated output byte for byte to ASSETS/<model_tag>/<prompt_id>.{mp4,png} (a file is only
    rewritten when its bytes differ). Returns {model_tag: {prompt_id: (seed, seconds)}} of the outputs that exist.
    """
    ds = lance.dataset(str(LANCE / "h3_eval_generations.lance"))
    found = {}
    for model in models:
        tag, mode = model["model_tag"], model["mode"]
        column = "video" if mode == "video" else "image"
        rows = ds.scanner(filter=f"model_tag = '{tag}' AND {column} IS NOT NULL", columns=["prompt_id", "seed", "seconds"], with_row_id=True).to_table().to_pylist()
        blobs = ds.take_blobs(column, ids=[r["_rowid"] for r in rows]) if rows else []
        found[tag] = {}
        for row, blob in zip(rows, blobs):
            data = blob.read()
            out = ROOT / "journal" / media_path(tag, row["prompt_id"], mode)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.read_bytes() != data:
                out.write_bytes(data)
            found[tag][row["prompt_id"]] = (row["seed"], row["seconds"])
    return found


def section_html(title, models, found, prompts, open_=True, sub=False):
    """Pure function. One <details> section holding every output of `models`, in prompt order, at full size."""
    figs = []
    for model in models:
        for p in prompts:
            if p["id"] in found[model["model_tag"]]:
                seed, seconds = found[model["model_tag"]][p["id"]]
                figs.append(item_html(media_path(model["model_tag"], p["id"], model["mode"]), model["mode"], p["short"], seed, seconds, weight_equation(model)))
    done = sum(len(found[m["model_tag"]]) for m in models)
    total = len(models) * len(prompts)
    eq = " | ".join(sorted({weight_equation(m) for m in models}))
    return (f'<details class="{"sub" if sub else "row"}"{" open" if open_ else ""}><summary>{html.escape(title)} '
            f'<span class="note">({done}/{total} outputs)</span></summary><p class="eq">{html.escape(eq)}</p>'
            f'<div class="grid">{"".join(figs)}</div></details>')


GLOSSARY = (
    ("H3", "MiniMax H3, a text-to-video model (480p class). We use it at 512 &times; 512, 141 frames (5.875 s) at 24 fps, 30 steps. It can also make a single frame, which is an image."),
    ("Adapter / LoRA", "A small set of extra weights (rank 16) trained on top of a big model. It is a weight difference &Delta; that gets added to the model."),
    ("Bake", "Add an adapter into the model weights for good (weight 1), then train the next adapter on top. Each adapter here is trained on a model with all earlier adapters baked in."),
    ("&Delta; notation", "&Delta;_pexels, &Delta;_imagegen, &Delta;_graymagic are the weight differences of the three adapters."),
    ("Weight equation", "How a model in a row is built at inference: W = W_H3 + a&middot;&Delta;_pexels + b&middot;&Delta;_imagegen + c&middot;&Delta;_graymagic. Weight 0 turns an adapter off. The equation is printed under every item."),
    ("Pexels", "Pexels is a stock video site. The Pexels adapter is trained on 10,000 real Pexels clips (first 6 s, 512 &times; 512) with H3-style captions. It makes H3 produce Pexels-like video."),
    ("MAGICK", "Our dataset of images of objects, with alpha mattes. Here we use its RGB composited on a 50% gray background, 512 &times; 512."),
    ("Gray magic", "The adapter trained on 10,000 MAGICK images on 50% gray. The captions never mention the gray, so the adapter must learn it. The goal: a subject on a flat gray background."),
)


def lineage_html():
    """Pure function. The adapter lineage as HTML boxes: stock H3 -> +Pexels -> +image-gen -> +gray magic, with links to the dataset pages."""
    steps = (
        ("Stock H3", "W_H3", None),
        ("+ Pexels adapter (baked)", "&Delta;_pexels, trained on 10,000 real Pexels clips (video)", DATASET_PAGES["pexels"]),
        ("+ Image-gen adapter (baked)", "&Delta;_imagegen, trained on 70,000 single frames of the same clips (images)", DATASET_PAGES["pexels"]),
        ("+ Gray magic adapter", "&Delta;_graymagic, trained on 10,000 MAGICK images on gray (images)", DATASET_PAGES["graymagic"]),
    )
    boxes = []
    for title, text, page in steps:
        link = f'<br><a href="{page}">dataset page</a>' if page else ""
        boxes.append(f'<div class="box"><b>{title}</b><br><span class="note">{text}</span>{link}</div>')
    return '<div class="lineage">' + '<span class="arrow">&rarr;</span>'.join(boxes) + "</div>"


CSS = """
:root { --bg: #fdfdfb; --fg: #1d1d1f; --muted: #66666b; --rule: #dcdcd6; --accent: #2f5d8a; --box: #f0f0ea; --gutter: 16px; --measure: 1600px; --font-body: 17px; --font-small: 14px; --pad: 8px; --hairline: 1px; --gap: 24px; --media: 512px; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; } }
:root[data-theme="dark"] { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; }
body { background: var(--bg); color: var(--fg); font: var(--font-body)/1.6 Georgia, serif; margin: 0; }
main { max-width: var(--measure); margin: 0 auto; padding: 0 var(--gutter) var(--gap); }
a { color: var(--accent); }
details { border-top: var(--hairline) solid var(--rule); margin-top: var(--gap); padding-top: var(--pad); }
summary { font: bold 1.25em system-ui, sans-serif; cursor: pointer; }
details.sub { border-top: 0; margin-top: var(--pad); padding-top: 0; }
details.sub summary { font-size: var(--font-body); }
.note, .eq { color: var(--muted); font: var(--font-small)/1.45 system-ui, sans-serif; }
.eq { font-family: ui-monospace, monospace; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--media)), var(--media))); gap: var(--gap); }
figure { margin: 0; width: var(--media); max-width: 100%; }
figure img, figure video { width: var(--media); max-width: 100%; height: auto; aspect-ratio: 1; display: block; background: #000; }
figcaption { font: var(--font-small)/1.45 system-ui, sans-serif; margin-top: var(--pad); }
.lineage { display: flex; flex-wrap: wrap; align-items: stretch; gap: var(--pad); }
.box { background: var(--box); border: var(--hairline) solid var(--rule); padding: var(--pad) var(--gap); flex: 1 1 200px; font: var(--font-small)/1.45 system-ui, sans-serif; }
.arrow { align-self: center; font-size: 1.5em; color: var(--muted); }
dt { font-weight: bold; margin-top: var(--pad); }
dd { margin-left: 0; }
select { font: var(--font-body) system-ui, sans-serif; max-width: 100%; }
.prompt { font: var(--font-small)/1.5 system-ui, sans-serif; margin: var(--pad) 0; }
"""

JS = """
const DATA = %s;
const media = (it, lazy) => it.mode === 'video'
  ? `<video muted loop autoplay playsinline controls preload="none" data-src="${it.path}"></video>`
  : `<img data-src="${it.path}" alt="${it.label}" loading="lazy">`;
const observer = new IntersectionObserver(entries => entries.forEach(e => {
  if (!e.isIntersecting) return;
  const el = e.target; observer.unobserve(el);
  el.src = el.dataset.src;
  if (el.tagName === 'VIDEO') el.play().catch(err => console.error('play failed', el.src, err));
}), {rootMargin: '400px'});
const watch = root => root.querySelectorAll('[data-src]').forEach(el => observer.observe(el));
const select = document.getElementById('cmp-prompt'), out = document.getElementById('cmp-out');
function show() {
  const id = select.value, p = DATA.prompts.find(p => p.id === id);
  const figs = DATA.rows.filter(r => r.items[id]).map(r => {
    const it = r.items[id];
    return `<figure>${media(it)}<figcaption><b>${r.title}</b><br>seed ${it.seed} &middot; ${it.seconds.toFixed(1)} s<br><span class="eq">${r.equation}</span></figcaption></figure>`;
  });
  out.innerHTML = `<p class="prompt">${p.prompt}</p><div class="grid">${figs.join('')}</div>`;
  watch(out);
}
select.innerHTML = DATA.prompts.map(p => `<option value="${p.id}">${p.id}: ${p.short}</option>`).join('');
select.onchange = show;
show();
watch(document);
"""


def compare_data(sections, models_by_key, found, prompts):
    """Pure function. The JSON of the compare view: the prompts and, per row with outputs, its items by prompt id."""
    rows = []
    for key, title in sections:
        for model in models_by_key.get(key, []):
            items = {p["id"]: {"mode": model["mode"], "path": media_path(model["model_tag"], p["id"], model["mode"]), "seed": found[model["model_tag"]][p["id"]][0],
                               "seconds": found[model["model_tag"]][p["id"]][1], "label": p["short"]} for p in prompts if p["id"] in found[model["model_tag"]]}
            if items:
                rows.append({"title": title if key[0] != "c" else f"{title} ({model['model_tag']})", "equation": weight_equation(model), "items": items})
    return {"prompts": [{"id": p["id"], "short": p["short"], "prompt": html.escape(p["prompt"])} for p in prompts], "rows": rows}


def build():
    """
    Command. Exports every output to journal/assets/h3magic and writes journal/h3magic.html. Run it again whenever
    new h3_eval_models rows or generations exist.
    """
    prompts = json.loads(PROMPTS.read_text())
    models = load_models()
    found = export_outputs(models)
    by_key = {}
    for m in models:
        by_key.setdefault(section_key(m["mode"], (m["a_pexels"], m["b_imagegen"], m["c_graymagic"]), m["model_tag"]), []).append(m)

    body = []
    shown = []
    for key, _, _, title in SECTIONS:
        if key in by_key:
            body.append(section_html(title, by_key[key], found, prompts))
            shown.append((key, title))
    ckpts = [(f"ckpt_{name}", f"Training checkpoints: {CHECKPOINT_TITLES[name]}") for name, _, _ in ADAPTERS if f"ckpt_{name}" in by_key]
    if ckpts:
        inner = "".join(section_html(f"{title.split(': ')[1]}: {m['model_tag']}", [m], found, prompts, open_=False, sub=True)
                        for key, title in ckpts for m in by_key[key])
        body.append(f'<details class="row"><summary>Training checkpoints (tasting samples)</summary><p class="note">Samples made during training, to taste the adapter before it finished. Collapsed by default.</p>{inner}</details>')
        shown += ckpts
    if not body:
        raise ValueError("no h3_eval_models row has any output yet")

    prompt_blocks = "".join(f'<details class="sub"><summary>{p["id"]}: {html.escape(p["short"])} (seed {p["seed"]})</summary><p class="prompt">{html.escape(p["prompt"])}</p></details>' for p in prompts)
    glossary = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    data = compare_data(shown, by_key, found, prompts)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Magic: results</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: results, one row per stage</h1>
<p class="note">Same 20 prompts and fixed seeds in every row. Every item is the unmodified 512&times;512 output, labeled with its seed, seconds to generate and weight equation. Dataset pages: <a href="{DATASET_PAGES['pexels']}">Pexels clips</a>, <a href="{DATASET_PAGES['graymagic']}">gray magic images</a>.</p>
<p>We build an RGB-on-gray version of the MiniMax H3 text-to-video model with adapters trained one on top of the other. Each row below is the model <b>W = W_H3 + a&middot;&Delta;_pexels + b&middot;&Delta;_imagegen + c&middot;&Delta;_graymagic</b>. Rows appear as they are evaluated.</p>
<details open><summary>Lineage of the adapters</summary>{lineage_html()}</details>
<details open><summary>Compare one prompt across all rows</summary><p><select id="cmp-prompt"></select></p><div id="cmp-out"></div></details>
{"".join(body)}
<details><summary>The 20 prompts (full text)</summary>{prompt_blocks}</details>
<details><summary>Glossary</summary><dl>{glossary}</dl></details>
</main>
<script>{JS % json.dumps(data)}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)}: " + ", ".join(f"{t} {len(found[t])}" for t in found), flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build})
