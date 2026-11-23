"""
Builds the encoder-bottleneck page (manifest §25, m00992-m00994): journal/h3magic_encoder.html, from Bluejay (m00998):
h3_enc_samples (sources), h3_enc_runs (the encoder baselines, the latent-optimization runs, fine-tuned encoders) and
h3_enc_results (every decode with its scores). Every image and video blob is written byte for byte to
journal/assets/h3magic_encoder/ (lossless PNGs at full resolution; videos as stored, H.264 CRF 12). Zoom crops are cut
in the browser from those full PNGs (CSS background, nearest neighbor), never resampled files. The conclusions box comes
from h3magic/journal_encoder_verdict.json (written by the agent after it looked at the results). Run from data/:

    source h3_env.fish; $H3_ENV python ../h3magic/journal_encoder.py
"""
import html
import json
import sys
from pathlib import Path

import cv2
import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/processors"))
PAGE = ROOT / "journal/h3magic_encoder.html"
ASSETS = ROOT / "journal/assets/h3magic_encoder"
VERDICT = ROOT / "h3magic/journal_encoder_verdict.json"
ZOOM_SIDE = 96  # side in source pixels of each zoom window (shown at ZOOM_SCALE× nearest neighbor)
ZOOM_SCALE = 3
ZOOMS_PER_IMAGE = 2
BASELINES = ("encoder_sample", "encoder_mode", "encoder_tiled")
BASELINE_LABEL = {"encoder_sample": "H3 encoder (pipeline latent: posterior sample)", "encoder_mode": "H3 encoder (posterior mode)",
                  "encoder_tiled": "H3 encoder, tiled encode (mode)"}
METRICS = (("psnr", "PSNR dB", 1), ("ssim", "SSIM", 1), ("lpips_alex", "LPIPS alex", -1), ("lpips_vgg", "LPIPS vgg", -1), ("dists", "DISTS", -1),
           ("flux_mse", "FLUX.2 latent MSE", -1), ("de2000", "ΔE2000", -1), ("sharp_grad", "sharpness grad", 0), ("sharp_fft", "sharpness HF", 0),
           ("flicker", "flicker", -1), ("z_dev", "mean |z − mode|", 0))
CURVE_KEYS = (("psnr", "PSNR dB"), ("lpips_alex", "LPIPS alex"), ("dists", "DISTS"))


def detail_windows(image, side=ZOOM_SIDE, count=ZOOMS_PER_IMAGE):
    """
    Pure function. Top-left (x, y) of the `count` non-overlapping side × side windows with the most fine detail
    (Laplacian energy of the luma): where an encoder loses detail first.

    Args:
        image (np.ndarray): (H, W, 3) uint8

    Returns:
        list[tuple[int, int]]

    Examples:
        >>> im = np.zeros((512, 512, 3), np.uint8); im[300:340:2, 100:140] = 255
        >>> x, y = detail_windows(im, count=1)[0]; x <= 100 and x + 96 >= 140 and y <= 300 and y + 96 >= 338
        True
    """
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = np.abs(cv2.Laplacian(luma, cv2.CV_32F))
    box = cv2.boxFilter(energy, -1, (side, side), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    h, w = luma.shape
    box = box[:h - side + 1, :w - side + 1].copy()
    out = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(box)), box.shape)
        out.append((int(x), int(y)))
        box[max(0, y - side):y + side, max(0, x - side):x + side] = -1
    return out


def fmt(v, spec=".4f"):
    """
    Pure function. A number formatted, or an em dash for None.

    >>> fmt(1.23456, ".2f"), fmt(None)
    ('1.23', '—')
    """
    return "—" if v is None else format(v, spec)


def spec_of(key):
    """
    Pure function. The number format of a metric column.

    >>> spec_of("psnr"), spec_of("dists")
    ('.2f', '.4f')
    """
    return ".2f" if key == "psnr" else ".3f" if key in ("de2000", "flicker", "sharp_grad", "sharp_fft", "z_dev") else ".4f"


def metric_table(rows, caption, keys=METRICS):
    """
    Pure function. An HTML table: one row per (label, metrics dict); the best value per column in bold.

    >>> "<b>30.00</b>" in metric_table([("a", {"psnr": 30.0}), ("b", {"psnr": 29.0})], "c", keys=(("psnr", "PSNR", 1),))
    True
    """
    keys = [k for k in keys if any(m.get(k[0]) is not None for _, m in rows)]
    best = {}
    for key, _, sign in keys:
        vals = [m[key] for _, m in rows if m.get(key) is not None]
        best[key] = (max(vals) if sign > 0 else min(vals)) if sign else None
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in keys)
    body = ""
    for label, m in rows:
        cells = ""
        for key, _, _ in keys:
            v = m.get(key)
            text = fmt(v, spec_of(key))
            cells += f"<td>{'<b>' + text + '</b>' if v is not None and v == best[key] else text}</td>"
        body += f"<tr><td>{label}</td>{cells}</tr>"
    return f"<div class='scroll'><table><caption>{caption}</caption><tr><th>latent</th>{head}</tr>{body}</table></div>"


def mean_scores(rows):
    """
    Pure function. Mean of every metric over result rows (None values skipped).

    >>> m = mean_scores([{"psnr": 30.0, "flicker": None}, {"psnr": 32.0, "flicker": None}]); m["psnr"], m["flicker"]
    (31.0, None)
    """
    out = {}
    for key, _, _ in METRICS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def run_label(run):
    """
    Pure function. A short human label of an h3_enc_runs row.

    >>> run_label({"run_id": "img_l2", "method": "optimize", "objective": "l2", "prior_weight": 0.0, "steps": 100})
    'optimized latent: l2, 100 Adam steps (img_l2)'
    """
    if run["method"] in BASELINE_LABEL:
        return BASELINE_LABEL[run["method"]]
    if run["method"] == "encoder_finetuned":
        return f"fine-tuned encoder ({run['run_id']}, latent reg {run['prior_weight']})"
    prior = f", prior {run['prior_weight']}" if run.get("prior_weight") else ""
    return f"optimized latent: {run['objective']}{prior}, {run['steps']} Adam steps ({run['run_id']})"


def write_blob(db_read, info, rel):
    """Command. Writes one stored blob byte for byte to ASSETS/rel (only when its bytes differ). Returns the page-relative path."""
    data = db_read(info)
    out = ASSETS / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_bytes() != data:
        out.write_bytes(data)
    return str(out.relative_to(PAGE.parent))


def short_metrics(m):
    """
    Pure function. A caption line with PSNR, LPIPS-alex and DISTS of a metrics dict ('' for None).

    >>> short_metrics({"psnr": 30.0, "lpips_alex": 0.05, "dists": 0.1})
    '<br>PSNR 30.00 dB, LPIPS 0.0500, DISTS 0.1000'
    """
    return "" if m is None else f"<br>PSNR {m['psnr']:.2f} dB, LPIPS {m['lpips_alex']:.4f}, DISTS {m['dists']:.4f}"


def zoom_html(panels, side, windows):
    """
    Pure function. Zoom rows: for each (x, y) window, every panel's crop at ZOOM_SCALE× (CSS background of the full PNG).

    >>> "scale(3)" in zoom_html([("a", "a.png", None)], 512, [(10, 20)])
    True
    """
    out = ""
    for x, y in windows:
        cells = "".join(f"<div><div class='zoom' style='width:{ZOOM_SIDE * ZOOM_SCALE}px;height:{ZOOM_SIDE * ZOOM_SCALE}px'><div style='width:{side}px;height:{side}px;"
                        f"background-image:url({p});background-size:{side}px {side}px;transform:scale({ZOOM_SCALE}) translate(-{x}px,-{y}px)'></div></div>"
                        f"<div class='zcap'>{html.escape(lab)}</div></div>" for lab, p, _ in panels)
        out += f"<p class='note'>Zoom: the {ZOOM_SIDE}&times;{ZOOM_SIDE} window at x={x}, y={y} (the most fine detail in the source), {ZOOM_SCALE}&times; nearest neighbor.</p><div class='zooms'>{cells}</div>"
    return out


def curve_svg(series, label, width=720, height=260):
    """
    Pure function. An SVG line chart: series = [(name, steps, values)], one polyline each.

    Examples:
        >>> svg = curve_svg([("a", [0, 10], [30.0, 31.0])], "PSNR")
        >>> svg.startswith("<svg") and "polyline" in svg
        True
    """
    colors = ("#2f5d8a", "#b3261e", "#2e7d32", "#8e44ad", "#d35400", "#16a085", "#7f8c8d", "#c0392b", "#2c3e50", "#f39c12")
    lo = min(min(v) for _, _, v in series)
    hi = max(max(v) for _, _, v in series)
    xmax = max(max(s) for _, s, _ in series) or 1
    pad = 40
    sx = lambda s: pad + (width - 2 * pad) * s / xmax
    sy = lambda v: height - pad - (height - 2 * pad) * ((v - lo) / ((hi - lo) or 1))
    lines = ""
    for i, (name, steps, values) in enumerate(series):
        pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(steps, values))
        c = colors[i % len(colors)]
        lines += f"<polyline fill='none' stroke='{c}' stroke-width='2' points='{pts}'><title>{html.escape(name)}</title></polyline>"
        lines += f"<text x='{width - pad + 4}' y='{sy(values[-1]):.1f}' style='fill:{c}'>{html.escape(name)}</text>"
    axes = (f"<text x='{pad}' y='{pad - 12}'>{html.escape(label)} vs Adam step (mean over the run's samples)</text>"
            f"<text x='4' y='{sy(hi):.1f}'>{hi:.3f}</text><text x='4' y='{sy(lo):.1f}'>{lo:.3f}</text>"
            f"<text x='{sx(xmax) - 20:.1f}' y='{height - 10}'>{xmax}</text><line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' stroke='currentColor' opacity='0.3'/>")
    return f"<svg viewBox='0 0 {width + 200} {height}' width='100%' style='max-width:{width + 200}px' role='img' aria-label='{html.escape(label)} curves'>{axes}{lines}</svg>"


def run_curves(results, runs, kind):
    """Pure function. Per optimization run of `kind` samples: (name, steps, mean values) for each CURVE_KEYS key."""
    out = {key: [] for key, _ in CURVE_KEYS}
    for run in runs:
        curves = [json.loads(r["curve"]) for r in results if r["run_id"] == run["run_id"] and r["curve"] and r["kind"] == kind]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        steps = [c["step"] for c in curves[0][:n]]
        for key, _ in CURVE_KEYS:
            if all(key in c[0] for c in curves):
                out[key].append((run["run_id"], steps, [float(np.mean([c[i][key] for c in curves])) for i in range(n)]))
    return out


CSS = """
:root { --bg: #fdfdfb; --fg: #1d1d1f; --muted: #66666b; --rule: #dcdcd6; --accent: #2f5d8a; --box: #f0f0ea;
        --gutter: 16px; --measure: 1700px; --font-body: 17px; --font-small: 14px; --pad: 8px; --hairline: 1px; --gap: 24px; --zcap: 288px; --bar: 4px; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; } }
:root[data-theme="dark"] { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; }
body { background: var(--bg); color: var(--fg); font: var(--font-body)/1.6 Georgia, serif; margin: 0; }
main { max-width: var(--measure); margin: 0 auto; padding: 0 var(--gutter) var(--gap); }
a { color: var(--accent); }
.verdict { background: var(--box); border-left: var(--bar) solid var(--accent); padding: var(--pad) calc(2 * var(--pad)); margin: var(--gap) 0; }
details { border-top: var(--hairline) solid var(--rule); margin-top: var(--gap); padding-top: var(--pad); }
summary { font: bold 1.25em system-ui, sans-serif; cursor: pointer; }
details.sub { border-top: 0; margin-top: var(--pad); padding-top: 0; }
details.sub summary { font-size: var(--font-body); }
.note { color: var(--muted); font: var(--font-small)/1.45 system-ui, sans-serif; }
.grid { display: flex; flex-wrap: wrap; gap: var(--gap); }
figure { margin: 0; max-width: 100%; }
figure img, figure video { max-width: 100%; height: auto; display: block; background: #000; }
figcaption { font: var(--font-small)/1.45 system-ui, sans-serif; margin-top: var(--pad); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font: var(--font-small)/1.4 system-ui, sans-serif; margin: var(--pad) 0; }
th, td { border-bottom: var(--hairline) solid var(--rule); padding: calc(var(--pad) / 2) var(--pad); text-align: left; vertical-align: top; }
caption { text-align: left; color: var(--muted); }
.zooms { display: flex; flex-wrap: wrap; gap: var(--pad); }
.zoom { overflow: hidden; position: relative; background: #000; max-width: 100%; }
.zoom div { transform-origin: 0 0; image-rendering: pixelated; }
.zcap { font: var(--font-small)/1.3 system-ui, sans-serif; max-width: var(--zcap); }
svg text { fill: var(--fg); font: 12px system-ui, sans-serif; }
"""


def main(max_image_examples=16):
    """Command. Writes journal/h3magic_encoder.html and its assets from Bluejay (module docstring)."""
    from bluejay import open_database
    from h3_blobs import read_blob
    db = open_database("project.dtab")
    verdict = json.loads(VERDICT.read_text())
    samples = {r["sample_id"]: r for r in db.query("SELECT sample_id, kind, origin, side, pexels_id, blob_info(source) AS source, "
                                                    "blob_info(source_mp4) AS mp4 FROM h3_enc_samples").to_pylist()}
    runs = {r["run_id"]: r for r in db.query("SELECT run_id, method, objective, steps, lr, prior_weight FROM h3_enc_runs").to_pylist()}
    metric_cols = ", ".join(k for k, _, _ in METRICS)
    results = db.query(f"SELECT run_id, sample_id, {metric_cols}, seconds, curve, blob_info(image) AS image, blob_info(video) AS video, "
                       "blob_info(frame_070) AS f070 FROM h3_enc_results WHERE run_id != 'smoke'").to_pylist()
    for r in results:
        r["kind"] = samples[r["sample_id"]]["kind"]
    by = {(r["run_id"], r["sample_id"]): r for r in results}
    order = sorted((r for r in runs.values() if r["run_id"] != "smoke"), key=lambda r: (r["method"] not in BASELINES, r["method"] == "encoder_finetuned", r["run_id"]))
    others = [r for r in order if r["method"] not in BASELINES]

    # tables: one per (kind, set of samples a run covered); baselines restricted to the same samples
    groups = {}
    for run in others:
        for kind in ("image", "video"):
            ids = tuple(sorted(s for (rid, s), r in by.items() if rid == run["run_id"] and r["kind"] == kind))
            if ids:
                groups.setdefault((kind, ids), []).append(run)
    tables = ""
    for (kind, ids), group in sorted(groups.items(), key=lambda g: (g[0][0], -len(g[0][1]))):
        rows = [(BASELINE_LABEL[b], mean_scores([by[(b, s)] for s in ids if (b, s) in by])) for b in BASELINES]
        rows += [(html.escape(run_label(run)), mean_scores([by[(run["run_id"], s)] for s in ids])) for run in group]
        sides = sorted({samples[s]["side"] for s in ids})
        tables += metric_table(rows, f"{len(ids)} held-out {kind}s ({', '.join(f'{s}²' for s in sides)}): "
                                     f"{html.escape(', '.join(ids)) if len(ids) <= 12 else str(len(ids)) + ' samples'}; mean vs the source")

    # image examples: samples with the most runs first
    image_ids = sorted({s for (rid, s), r in by.items() if r["kind"] == "image" and rid not in BASELINES},
                       key=lambda s: (-sum((run["run_id"], s) in by for run in others), s))[:max_image_examples]
    blocks = ""
    for sid in image_ids:
        s = samples[sid]
        src_rel = write_blob(read_blob, s["source"], f"{sid}/source.png")
        side = s["side"]
        panels = [("SOURCE (target)", src_rel, None)]
        for run in order:
            r = by.get((run["run_id"], sid))
            if r and r["image"]:
                panels.append((run_label(run), write_blob(read_blob, r["image"], f"{sid}/{run['run_id']}.png"), r))
        rgb = cv2.cvtColor(cv2.imread(str(ASSETS / sid / "source.png")), cv2.COLOR_BGR2RGB)
        figs = "".join(f"<figure style='width:{side}px'><a href='{p}'><img src='{p}' width='{side}' height='{side}' alt='{html.escape(lab)}' loading='lazy'></a>"
                       f"<figcaption><b>{html.escape(lab)}</b>{short_metrics(m)}</figcaption></figure>" for lab, p, m in panels)
        blocks += (f"<details class='sub' open><summary>{html.escape(sid)} ({side}&times;{side}, {html.escape(s['origin'])})</summary>"
                   f"<div class='grid'>{figs}</div>{zoom_html(panels, side, detail_windows(rgb))}</details>")

    # videos
    vblocks = ""
    for sid in sorted({s for (rid, s), r in by.items() if r["kind"] == "video" and rid not in BASELINES}):
        s = samples[sid]
        vids = [("SOURCE", write_blob(read_blob, s["mp4"], f"{sid}/source.mp4"))]
        frames = [("SOURCE (frame 70)", write_blob(read_blob, s["source"], f"{sid}/source_f070.png"), None)]
        rows = []
        for run in order:
            r = by.get((run["run_id"], sid))
            if r and r["video"]:
                vids.append((run_label(run), write_blob(read_blob, r["video"], f"{sid}/{run['run_id']}.mp4")))
                frames.append((run_label(run) + " (frame 70)", write_blob(read_blob, r["f070"], f"{sid}/{run['run_id']}_f070.png"), r))
                rows.append((html.escape(run_label(run)), r))
        vhtml = "".join(f"<figure style='width:512px'><video src='{p}' width='512' height='512' controls loop muted playsinline preload='metadata'></video>"
                        f"<figcaption>{html.escape(lab)}</figcaption></figure>" for lab, p in vids)
        fhtml = "".join(f"<figure style='width:512px'><a href='{p}'><img src='{p}' width='512' height='512' alt='' loading='lazy'></a>"
                        f"<figcaption><b>{html.escape(lab)}</b></figcaption></figure>" for lab, p, _ in frames)
        rgb = cv2.cvtColor(cv2.imread(str(ASSETS / sid / "source_f070.png")), cv2.COLOR_BGR2RGB)
        vblocks += (f"<details class='sub' open><summary>{html.escape(sid)} (141 frames, 512&times;512)</summary>"
                    f"<p class='note'>Videos are H.264 CRF 12 (the only lossy copies on this page); frame 70 is the lossless PNG of the decode.</p>"
                    f"<div class='grid'>{vhtml}</div><div class='grid'>{fhtml}</div>{zoom_html(frames, 512, detail_windows(rgb))}"
                    f"{metric_table(rows, f'{html.escape(sid)}: all 141 frames vs the source')}</details>")

    curves = ""
    for kind in ("image", "video"):
        for key, label in CURVE_KEYS:
            series = run_curves(results, others, kind)[key]
            if series:
                curves += f"<h3>{kind}s: {label}</h3>" + curve_svg(series, label)
    numbers = "".join(f"<li>{n}</li>" for n in verdict["numbers"])
    extra = "".join(f"<details open><summary>{html.escape(s['title'])}</summary>{s['html']}</details>" for s in verdict.get("sections", []))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Encoder Bottleneck</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Is H3's encoder the bottleneck?</h1>
<p class="note">Manifest §25 (m00992, m00993, m00994). {html.escape(verdict['status'])} Every image is the original lossless PNG at full resolution; zooms are cut from those files in the browser. All results live in Bluejay (h3_enc_samples, h3_enc_runs, h3_enc_results; Sparrow view view_h3_encoder).</p>
<div class="verdict"><p><b>Verdict.</b> {verdict['verdict']}</p><ul>{numbers}</ul></div>
<p>The question (m00992): does a latent exist that the <b>frozen</b> H3 decoder turns into a clearly better picture than the encoder's own latent? We find one by gradient descent on the latent itself, against the source pixels, with several objectives, and judge the result with metrics the optimizer did not see and by eye on zoomed details. If such latents are much better, the encoder (not the decoder) limits H3's reconstructions, and a better encoder could be trained with the decoder frozen.</p>
{extra}
<details open><summary>Image examples: source, H3's encoder, optimized latents (full resolution) and detail zooms</summary>
<p class="note">Each block: the source, the decode of the encoder's latent as the pipeline makes it (a posterior sample), the posterior mode, the tiled encode, then the decode of each optimized latent (one per objective) and of each fine-tuned encoder. Below: the two most detailed {ZOOM_SIDE}&times;{ZOOM_SIDE} windows of the source, {ZOOM_SCALE}&times; in every panel.</p>
{blocks or "<p>No image results yet.</p>"}</details>
<details open><summary>Videos (141 frames, 512&times;512)</summary>{vblocks or "<p>Video runs are in progress.</p>"}</details>
<details open><summary>Optimization curves</summary>{curves}</details>
<details open><summary>Metric tables</summary>
<p class="note">Bold = best in its column. PSNR / SSIM / ΔE2000 / sharpness / flicker use the decoder frenzy's own metric functions (fastdec_ga/score.py, copied into data/processors/h3_enc_lib.py); sharpness = decode / source (1 = as sharp as the source). DISTS and the FLUX.2 latent MSE are held-out judges whenever the objective did not use them. Each table compares runs on exactly the same samples.</p>{tables}</details>
<details><summary>Method</summary>{verdict.get('method', '')}</details>
<details><summary>Glossary</summary><dl>
<dt>Latent</dt><dd>H3's compressed picture: 24 numbers per 16&times;16 pixel cell (and per 4 frames in video). One 512&sup2; image = 24&times;1&times;32&times;32; a 141-frame clip = 24&times;42&times;32&times;32. Normalized: per channel, (z &minus; mean) / std, so each number has about unit spread over the data.</dd>
<dt>Encoder latent (posterior sample / mode)</dt><dd>The encoder outputs a mean and a spread per number. The project's pipeline draws a random sample from it (seeded); the mode is the mean itself, without that noise.</dd>
<dt>Tiled encode</dt><dd>Encoding in overlapping 256&times;256 tiles. The project encodes untiled (3&times; faster); the decoder always decodes in tiles.</dd>
<dt>Optimized latent</dt><dd>A latent found by gradient descent (Adam) through the frozen decoder, starting at the posterior mode, so that its decode matches the source under one objective.</dd>
<dt>Objective</dt><dd>What the optimizer minimizes: l2 / l1 (pixel error), lpips_vgg / lpips_alex (a learned perceptual distance), flux (the distance between FLUX.2-VAE latents of decode and source: FIBO2's image encoder), and mixes, e.g. l1_vgg = L1 + 0.1&middot;LPIPS-vgg.</dd>
<dt>Prior</dt><dd>An extra penalty that keeps the latent inside the encoder's spread: mean(((z &minus; mode) / spread)&sup2;).</dd>
<dt>Fine-tuned encoder</dt><dd>A copy of H3's encoder trained with the decoder frozen to make decodes match their sources, plus a penalty on how far its latents move from the original encoder's.</dd>
<dt>DISTS</dt><dd>A perceptual image distance (structure and texture similarity, Ding et al. 2020) that no objective here optimizes: a held-out judge.</dd>
<dt>Sharpness (grad / HF)</dt><dd>Mean Sobel gradient and high-frequency FFT energy of the decode divided by the source's: below 1 = blurrier than the source.</dd>
<dt>Flicker</dt><dd>|mean frame-to-frame change of the decode &minus; that of the source| on 0&ndash;255: extra or missing temporal change.</dd>
</dl></details>
</main>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)}: {len(results)} results, {len(image_ids)} image examples, {len(groups)} tables")


if __name__ == "__main__":
    fire.Fire(main)
