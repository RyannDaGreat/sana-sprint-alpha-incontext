"""
Builds the Alpha Video v2 dataset page (manifest §23, m00934 "journal entries for subsets of every dataset we train on,
explaining every adapter"): journal/h3magic_dataset_v3.html. Shows SAMPLES MAGICK training rows exactly as the trainer
sees them before the VAE: magick_samples.rgba composited on 0.5 gray, white and black at 1024 and area-resized to 768
(data/processors/h3_encode.py v3_composites, the inputs of h3_latent_{gray,white,black}_768), saved as lossless PNGs,
with each row's training caption h3_caption_v3 (the subject-only caption + the gray suffix). Run from data/ with $H3_ENV:

    python ../h3magic/journal_dataset_v3.py
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/processors"))
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

PAGE = ROOT / "journal/h3magic_dataset_v3.html"
ASSETS = ROOT / "journal/assets/h3magic_dataset_v3"
SAMPLES = 48
BACKGROUNDS = ("gray", "white", "black")


def png_bytes(image):
    """Pure function. uint8 (H, W, 3) RGB as lossless PNG bytes."""
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok, "PNG encode failed"
    return data.tobytes()


def caption_stats(captions):
    """
    Pure function. Word-count statistics of captions.

    Examples:
        >>> caption_stats(["a b c", "a b"])
        {'n': 2, 'words_mean': 2.5, 'words_min': 2, 'words_max': 3}
    """
    words = [len(c.split()) for c in captions]
    return dict(n=len(words), words_mean=float(np.mean(words)), words_min=min(words), words_max=max(words))


def main():
    """Command. Writes the composites of SAMPLES train rows and the page."""
    import journal as j
    from bluejay import open_database
    from h3_blobs import read_png_rgba
    from h3_encode import v3_composites
    db = open_database("project.dtab")
    counts = db.query("SELECT split, count(*) AS n, count(h3_latent_black_768) AS latents, "
                      "sum(CASE WHEN h3_text_v3_sha256 = encode(sha256(h3_caption_v3), 'hex') THEN 1 ELSE 0 END) AS embedded "
                      "FROM h3_magick_images GROUP BY split ORDER BY split").to_pylist()
    captions = [r["h3_caption_v3"] for r in db.query("SELECT h3_caption_v3 FROM h3_magick_images WHERE split = 'train'").to_pylist()]
    stats = caption_stats(captions)
    rows = db.query(f"SELECT h.page_id, h.h3_caption_v3 AS caption, blob_info(m.rgba) AS rgba FROM h3_magick_images h "
                    f"JOIN magick_samples m ON m.page_id = h.page_id WHERE h.split = 'train' ORDER BY h.page_id LIMIT {SAMPLES}").to_pylist()
    ASSETS.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        comps = v3_composites(read_png_rgba(row["rgba"]))
        cells = []
        for name, image in zip(BACKGROUNDS, comps):
            path = ASSETS / f"{row['page_id']}_{name}.png"
            data = png_bytes(image)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)
            rel = path.relative_to(ROOT / "journal")
            cells.append(f'<figure><img data-src="{rel}" alt="{name}" loading="lazy"><figcaption>{name} ({row["page_id"]})</figcaption></figure>')
        items.append(f'<div class="item"><div class="trio">{"".join(cells)}</div><p class="prompt">{html.escape(row["caption"])}</p></div>')
    count_rows = "".join(f"<tr><td>{c['split']}</td><td>{c['n']:,}</td><td>{c['latents']:,}</td><td>{c['embedded']:,}</td></tr>" for c in counts)
    css = j.CSS + """
:root { --media: 256px; }
.trio { display: flex; flex-wrap: wrap; gap: var(--pad); }
.item { border-top: var(--hairline) solid var(--rule); padding-top: var(--pad); margin-top: var(--gap); }
.prompt { font: var(--font-small)/1.45 system-ui, sans-serif; }
table td, table th { padding: 4px 10px; border-bottom: var(--hairline) solid var(--rule); font: var(--font-small) system-ui, sans-serif; }
"""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Alpha v2 dataset</title>
<style>{css}</style>
</head>
<body>
<main>
<h1>Alpha Video v2: the training data (MAGICK on gray, white and black, 768&times;768)</h1>
<p class="note">Part of <a href="h3magic_alphav2.html">Alpha Video v2</a>. Every image below is a lossless PNG of exactly what the trainer encodes.</p>
<h2>Adapter lineage</h2>
<div class="lineage">
<div class="box"><b>Stock H3</b><br>MiniMax H3, no adapter. With the gray suffix in the prompt it already draws most subjects on flat gray (entry 1 of the Alpha Video v2 page).</div>
<div class="arrow">&rarr;</div>
<div class="box"><b>Gray magic v3</b> (&Delta;_graymagic_v3)<br>A LoRA on stock H3, trained on the <b>gray</b> images below with their captions. Since the captions already ask for gray, it mostly learns MAGICK's look and a flatter, darker gray (tasting: border level 122, i.e. 0.48, vs 144 from the words alone).</div>
<div class="arrow">&rarr;</div>
<div class="box"><b>Split v3</b> (&Delta;_split_v3)<br>Trained with gray magic v3 baked in: a LoRA plus three output heads, taught the <b>gray</b>, <b>white</b> and <b>black</b> images of the same row at once. From the three, least squares gives the cut-out (RGBA).</div>
</div>
<h2>How the images are made</h2>
<p>Each row is a MAGICK image with a true alpha matte (1024&times;1024 RGBA). It is composited in 8 bit on 50% gray (128), on white and on black at 1024, then area-resized to 768&times;768 and encoded by H3's VAE as a single-frame latent (24 &times; 1 &times; 48 &times; 48). The caption is the subject-only caption written for H3 Magic (it never mentions a background) plus the suffix chosen in entry 1: &quot;The subject is isolated on a plain flat medium-gray (#808080) background, like a cut-out on an empty gray canvas.&quot;</p>
<h2>Counts</h2>
<table><thead><tr><th>split</th><th>rows</th><th>with 768 latents</th><th>with caption embedding</th></tr></thead><tbody>{count_rows}</tbody></table>
<p class="note">Train captions: {stats['n']:,}, {stats['words_mean']:.0f} words on average ({stats['words_min']}&ndash;{stats['words_max']}). The val rows are held out for the validation loss.</p>
<h2>{SAMPLES} training rows (the first by page id)</h2>
{"".join(items)}
</main>
<script>
const observer = new IntersectionObserver(entries => entries.forEach(e => {{ if (e.isIntersecting && !e.target.getAttribute('src')) e.target.src = e.target.dataset.src; }}), {{rootMargin: '400px'}});
document.querySelectorAll('[data-src]').forEach(el => observer.observe(el));
</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)} with {len(items)} rows", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "h3magic"))
    main()
