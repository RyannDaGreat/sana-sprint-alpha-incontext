"""
Builds the DiffDec page (manifest §26, m01001): journal/h3magic_diffdec.html, from Bluejay (m00998): dd_samples (the 68
test sources), dd_methods (every candidate and reference) and dd_renders (every decode with its scores). Every image is
written byte for byte to journal/assets/h3magic_diffdec/ (lossless PNGs, original 512x512, never resized); the texture
crops are pixel slices of those PNGs (no resampling), enlarged only by CSS image-rendering: pixelated. The conclusions
box comes from h3magic/journal_diffdec_verdict.json (written by the coordinator); the method, survey and compute
sections are rendered from DD_DIR's FITNESS.md, research/SURVEY.md and MAKO_JOBS.md. Rerun after every round
(after `fish data/run_proc_diffdec.fish` imported the new candidates). Run from data/:

    source uv.env && uv run --no-project --python 3.13 --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple \\
        --with "$(ls Aviary/.wheel/*.whl)" --with numpy --with opencv-python-headless --with fire --with markdown \\
        python ../h3magic/journal_diffdec.py build
"""
import html
import json
import re
import sys
from pathlib import Path

import cv2
import fire
import markdown
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "h3magic"))
from journal_encoder import CSS as BASE_CSS  # the house style of the H3 Magic pages

PAGE = ROOT / "journal/h3magic_diffdec.html"
ASSETS = ROOT / "journal/assets/h3magic_diffdec"
VERDICT = ROOT / "h3magic/journal_diffdec_verdict.json"
DD_DIR = ROOT / "agents/.scratchpad/diffdec"
H3_ID, OPT_ID, ORACLE_ID = "g0/h3_decoder", "g0/optimized_l1vgg", "g0/flux2_oracle"
GATE1_DB, GATE2_DB = 1.0, 3.0  # FITNESS.md: psnr_mean8 >= h3 - 1.0 dB; psnr >= h3 - 3.0 dB
DISTS_TIE = 0.002  # FITNESS.md: DISTS ties within this are broken by LPIPS-vgg, then |1 - sharp_fft|
TOP_METHODS = 3
CROP_SIDE = 160  # source pixels, shown 1:1
CROP_SAMPLES = 8
CROPS_PER_ORIGIN = 3  # so the 32 spare clips do not fill every crop
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
CI_PERCENT = (2.5, 97.5)
METRICS = ("psnr", "ssim", "lpips_alex", "lpips_vgg", "dists", "de2000", "flux_mse", "sharp_grad", "sharp_fft", "psnr_mean8", "dists_mean8")
SAMPLING_KINDS = ("child", "wildcard", "baseline")  # a "sampling method": one of these with more than one step
METHOD_COLS = ("method_id, round, name, agent, kind, parents, paper, paper_url, idea, why_win, ablation_of, ablation_change, space, conditioning, "
               "text_cond, steps, params, train_gpu_hours, train_gpu_type, verdict, why_held_or_failed, status, ms_per_image, gpu_type, folder")
COMPUTE_NOTE = ("Mako AISI Interpositive queue (aisi-interpositive): its H100s are borrow-only and were preempted; round 1 ran on remote node 2 (8 A100). "
                "The jobs we submitted to Mako are listed below.")
EXTRA_CSS = """
:root { --pass: #2e7d32; --fail: #b3261e; --crop-side: 160px; --crop-scale: 1; --pad-cell: 4px; --col-wide: 28em; }
.pass { color: var(--pass); font-weight: bold; }
.fail { color: var(--fail); font-weight: bold; }
th { cursor: pointer; white-space: nowrap; }
td { overflow-wrap: anywhere; }
td.wide { min-width: var(--col-wide); }
.crops img { image-rendering: pixelated; width: calc(var(--crop-side) * var(--crop-scale)); height: auto; }
.crops figure { max-width: none; }
.crops .row { display: flex; flex-wrap: wrap; gap: var(--pad); margin-bottom: var(--gap); }
.crops figcaption { max-width: calc(var(--crop-side) * var(--crop-scale)); }
button { font: var(--font-small) system-ui, sans-serif; padding: var(--pad-cell) var(--pad); }
.verdict.none { border-left-color: var(--fail); }
"""
SORT_JS = """
document.querySelectorAll('table').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, col) {
    th.addEventListener('click', function () {
      var rows = Array.from(table.querySelectorAll('tr')).filter(function (r) { return r.querySelector('td'); });
      var key = function (r) { var c = r.children[col]; var v = c.dataset.sort !== undefined ? c.dataset.sort : c.textContent.trim(); return v === '' ? NaN : Number(v) ; };
      var numeric = rows.every(function (r) { return !isNaN(key(r)); });
      var dir = th.dataset.dir === 'asc' ? -1 : 1; th.dataset.dir = dir === 1 ? 'asc' : 'desc';
      rows.sort(function (a, b) { return numeric ? dir * (key(a) - key(b)) : dir * (a.children[col].textContent.localeCompare(b.children[col].textContent)); });
      rows.forEach(function (r) { r.parentNode.appendChild(r); });
    });
  });
});
document.querySelectorAll('button[data-crop-scale]').forEach(function (b) {
  b.addEventListener('click', function () { document.documentElement.style.setProperty('--crop-scale', b.dataset.cropScale); });
});
"""


# ---------------------------------------------------------------- pure helpers

def fmt(v, spec=".4f"):
    """
    Pure function. A number formatted, or an em dash for None.

    >>> fmt(30.4149, ".2f"), fmt(None)
    ('30.41', '—')
    """
    return "—" if v is None else format(v, spec)


def num_cell(v, spec=".4f", cls=""):
    """
    Pure function. A table cell with the value and a numeric data-sort key (empty for None).

    >>> num_cell(30.4149, ".2f")
    '<td data-sort="30.4149">30.41</td>'
    """
    sort = "" if v is None else repr(float(v))
    return f"<td data-sort=\"{sort}\"{' class=' + cls if cls else ''}>{fmt(v, spec)}</td>"


def md(text):
    """
    Pure function. Markdown text (tables, fenced code) to HTML, as the trusted project notes write it.

    >>> md("**a**")
    '<p><strong>a</strong></p>'
    >>> md("|a|\\n|-|\\n|1|").startswith("<div class='scroll'><table>")
    True
    """
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])
    return out.replace("<table>", "<div class='scroll'><table>").replace("</table>", "</table></div>")


def md_section(text, start, end=None):
    """
    Pure function. The part of a markdown text from the line starting with `start` to the line starting with `end`.

    >>> md_section("## 1. A\\nx\\n## 2. B\\ny", "## 1.", "## 2.")
    '## 1. A\\nx\\n'
    """
    lines = text.splitlines(keepends=True)
    first = next(i for i, l in enumerate(lines) if l.startswith(start))
    last = next((i for i, l in enumerate(lines) if i > first and end and l.startswith(end)), len(lines))
    return "".join(lines[first:last])


def mean(values):
    """
    Pure function. Mean of the non-None values, None when there are none.

    >>> mean([1.0, None, 3.0]), mean([None])
    (2.0, None)
    """
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def paper_links(paper, urls):
    """
    Pure function. The paper text followed by numbered links to its ' ; '-separated URLs (HTML).

    >>> paper_links("SSDD", "https://a.org/1 ; https://b.org/2")
    'SSDD [<a href="https://a.org/1">1</a>] [<a href="https://b.org/2">2</a>]'
    """
    links = "".join(f' [<a href="{html.escape(u.strip())}">{i}</a>]' for i, u in enumerate((urls or "").split(" ; "), 1) if u.strip())
    return html.escape(paper or "") + links


def rank_key(stat):
    """
    Pure function. FITNESS.md's selection order of a stats dict: DISTS in DISTS_TIE-wide bins (lower is better), then
    LPIPS-vgg, then |1 - sharp_fft|. (Bins approximate "ties within 0.002".)

    >>> rank_key({"dists": 0.0701, "lpips_vgg": 0.15, "sharp_fft": 0.9}) < rank_key({"dists": 0.0701, "lpips_vgg": 0.16, "sharp_fft": 0.9})
    True
    """
    return (round(stat["dists"] / DISTS_TIE), stat["lpips_vgg"], abs(1 - stat["sharp_fft"]))


def gates(stat, h3_stat):
    """
    Pure function. (gate1, gate2) of FITNESS.md: psnr_mean8 >= h3 - 1 dB and psnr >= h3 - 3 dB, both on the same samples.

    >>> gates({"psnr": 28.0, "psnr_mean8": 30.0}, {"psnr": 30.0, "psnr_mean8": 30.0})
    (True, True)
    >>> gates({"psnr": 26.0, "psnr_mean8": 28.0}, {"psnr": 30.0, "psnr_mean8": 30.0})
    (False, False)
    """
    return stat["psnr_mean8"] >= h3_stat["psnr"] - GATE1_DB, stat["psnr"] >= h3_stat["psnr"] - GATE2_DB


def bootstrap_ci(deltas):
    """
    Near-pure function (seeded RNG). 95% bootstrap CI of the mean over images of paired differences.

    >>> lo, hi = bootstrap_ci([1.0] * 10); lo == hi == 1.0
    True
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.asarray(deltas, dtype=float)
    means = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas))).mean(axis=1)
    lo, hi = np.percentile(means, CI_PERCENT)
    return float(lo), float(hi)


def best_crop(image, side=CROP_SIDE):
    """
    Pure function. The (x, y, energy) of the side x side window of an (H, W, 3) uint8 RGB image with the highest
    high-frequency energy (sum of |Laplacian| of the luma): hair, foliage, text.

    >>> im = np.zeros((512, 512, 3), np.uint8); im[300:340:2, 100:140] = 255
    >>> x, y, e = best_crop(im); x <= 100 and x + 160 >= 140 and y <= 300 and y + 160 >= 338 and e > 0
    True
    """
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = np.abs(cv2.Laplacian(luma, cv2.CV_32F))
    box = cv2.boxFilter(energy, -1, (side, side), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    h, w = luma.shape
    box = box[:h - side + 1, :w - side + 1]
    y, x = np.unravel_index(int(np.argmax(box)), box.shape)
    return int(x), int(y), float(box[y, x])


def crop_png(png_bytes, x, y, side=CROP_SIDE):
    """
    Pure function. PNG bytes of the pixel slice [y:y+side, x:x+side] of a PNG (lossless, no resampling).

    >>> import cv2 as c; ok, b = c.imencode(".png", np.zeros((200, 200, 3), np.uint8)); out = crop_png(b.tobytes(), 10, 10, 160)
    >>> c.imdecode(np.frombuffer(out, np.uint8), c.IMREAD_COLOR).shape
    (160, 160, 3)
    """
    image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    ok, out = cv2.imencode(".png", image[y:y + side, x:x + side])
    if not ok:
        raise ValueError("Cannot encode the crop")
    return out.tobytes()


def safe(method_id):
    """
    Pure function. A file-name-safe method id.

    >>> safe("g1/pixflow")
    'g1_pixflow'
    """
    return method_id.replace("/", "_")


def pick_methods(stats, own_h3):
    """
    Pure function. The ids of the TOP_METHODS best scored methods by DISTS (rank_key), H3's decoder excluded. Trained
    candidates (not kind reference) come first; references only fill the remaining places. Gate failures are skipped;
    own_h3[m] is H3's decoder's mean over the images that m decoded.

    >>> h3 = {"psnr": 30.0, "psnr_mean8": 30.0}
    >>> s = {"g0/a": {"kind": "reference", "dists": 0.01, "lpips_vgg": 0.1, "sharp_fft": 1.0, "psnr": 30.0, "psnr_mean8": 30.0},
    ...      "g1/b": {"kind": "child", "dists": 0.05, "lpips_vgg": 0.1, "sharp_fft": 1.0, "psnr": 29.0, "psnr_mean8": 30.0}}
    >>> pick_methods(s, {"g0/a": h3, "g1/b": h3})
    ['g1/b', 'g0/a']
    """
    ok = {m: s for m, s in stats.items() if m != H3_ID and all(gates(s, own_h3[m]))}
    trained = sorted((m for m, s in ok.items() if s["kind"] != "reference"), key=lambda m: rank_key(ok[m]))
    refs = sorted((m for m, s in ok.items() if s["kind"] == "reference"), key=lambda m: rank_key(ok[m]))
    return (trained + refs)[:TOP_METHODS]


def twins_of(method_ids, methods):
    """
    Pure function. The deterministic twins (ablation rows) of the methods: those whose ablation_of is one of them, and the
    method a given ablation isolates. Excludes the ids themselves.

    >>> twins_of(["g1/a"], {"g1/a": {"ablation_of": ""}, "g1/b": {"ablation_of": "g1/a"}})
    ['g1/b']
    """
    found = []
    for m in method_ids:
        found += [k for k, v in methods.items() if v["ablation_of"] == m]
        if methods[m]["ablation_of"] in methods:
            found.append(methods[m]["ablation_of"])
    return [m for m in dict.fromkeys(found) if m not in method_ids]


def paired(by, method_id, base_id, sample_ids=None):
    """
    Pure function. [(method render row, base render row)] over the samples that both decoded.

    >>> paired({("a", "s1"): 1, ("b", "s1"): 2, ("a", "s2"): 3}, "a", "b")
    [(1, 2)]
    """
    return [(by[(method_id, s)], by[(base_id, s)]) for s in sorted({s for (m, s) in by if m == method_id})
            if (base_id, s) in by and (sample_ids is None or s in sample_ids)]


# ---------------------------------------------------------------- HTML pieces

def verdict_box(verdict):
    """
    Pure function. The conclusions box; verdict None = no file yet.

    >>> "No verdict yet" in verdict_box(None)
    True
    """
    if verdict is None:
        return "<div class='verdict none'><p><b>No verdict yet.</b> The coordinator has not written h3magic/journal_diffdec_verdict.json.</p></div>"
    numbers = "".join(f"<li>{html.escape(n)}</li>" for n in verdict["numbers"])
    return (f"<div class='verdict'><p><b>Verdict.</b> {html.escape(verdict['verdict'])}</p><ul>{numbers}</ul>"
            f"<p class='note'>{html.escape(verdict['status'])}</p></div>")


def delta_row(label, pairs, opt_pairs):
    """
    Pure function. One headline table row: method vs H3's decoder on the samples both decoded: deltas (method - H3), the
    95% bootstrap CI over images of the DISTS and LPIPS deltas, and the share of the §25 gap closed.

    Args:
        pairs: [(method row, h3 row)] on the common samples
        opt_pairs: [(method row, h3 row, optimized row)] on the samples that all three decoded (the 12 of §25)

    >>> delta_row("m", [({"psnr": 31.0, "psnr_mean8": 32.0, "dists": 0.06, "lpips_vgg": 0.1}, {"psnr": 30.0, "psnr_mean8": 30.0, "dists": 0.07, "lpips_vgg": 0.15})] * 3, []).count("<td")
    8
    """
    def delta(key):
        return [m[key] - h[key] for m, h in pairs]

    def ci_cell(key, spec):
        d = delta(key)
        lo, hi = bootstrap_ci(d)
        return f"<td data-sort='{np.mean(d)}'>{np.mean(d):+{spec}} <span class='note'>[{lo:+{spec}}, {hi:+{spec}}]</span></td>"

    def closed(key):
        if not opt_pairs:
            return "<td>n/a</td>"
        num = mean([m[key] - h[key] for m, h, _ in opt_pairs])
        den = mean([o[key] - h[key] for _, h, o in opt_pairs])
        return f"<td data-sort='{num / den}'>{num / den:.0%}</td>"

    return (f"<tr><td>{html.escape(label)}</td><td data-sort='{len(pairs)}'>{len(pairs)}</td>"
            f"<td data-sort='{np.mean(delta('psnr'))}'>{np.mean(delta('psnr')):+.2f} dB</td>"
            f"<td data-sort='{np.mean(delta('psnr_mean8'))}'>{np.mean(delta('psnr_mean8')):+.2f} dB</td>"
            f"{ci_cell('dists', '.4f')}{ci_cell('lpips_vgg', '.4f')}{closed('psnr')}{closed('dists')}</tr>")


HEADLINE_FORMULAS = """
<ul>
<li>&Delta;PSNR (single sample) = mean over images of [PSNR(method, seed 0) &minus; PSNR(H3 decoder)]. Higher is better. A perfect posterior sampler loses 3.01 dB here.</li>
<li>&Delta;PSNR (mean of 8) = mean over images of [PSNR(mean of 8 seeds' decodes) &minus; PSNR(H3 decoder)]. This estimates what the decoder keeps of the latent's information.</li>
<li>&Delta;DISTS, &Delta;LPIPS-vgg = mean over images of [score(method) &minus; score(H3 decoder)]. Lower is better, so negative means the method wins. In brackets: 95% bootstrap CI of that mean over images (paired, {n_boot} resamples).</li>
<li>Share of the §25 gap closed (PSNR) = (PSNR<sub>method</sub> &minus; PSNR<sub>H3</sub>) / (PSNR<sub>optimized</sub> &minus; PSNR<sub>H3</sub>), on the images that all three decoded (the 12 with §25's optimized latent). DISTS: (DISTS<sub>H3</sub> &minus; DISTS<sub>method</sub>) / (DISTS<sub>H3</sub> &minus; DISTS<sub>optimized</sub>). 100% = as good as the best latent that regression decoding can reach; negative = worse than H3.</li>
</ul>
""".replace("{n_boot}", str(BOOTSTRAP_RESAMPLES))


def headline_html(by, stats, own_h3):
    """
    Pure function. Section 2: the best gate-passing sampling method (and the §25 reference scale) vs H3's decoder.
    """
    opt_ids = {s for (m, s) in by if m == OPT_ID}
    candidates = {m: s for m, s in stats.items() if s["kind"] in SAMPLING_KINDS and s["steps"] > 1 and all(gates(s, own_h3[m])) and m != H3_ID}
    best = min(candidates, key=lambda m: rank_key(candidates[m]), default=None)
    rows = ""
    for m in ([best] if best else []) + [OPT_ID, ORACLE_ID]:
        if m in stats:
            opt_pairs = [(by[(m, s)], by[(H3_ID, s)], by[(OPT_ID, s)]) for s in sorted(opt_ids) if (m, s) in by and (H3_ID, s) in by]
            rows += delta_row(m + (" (best gate-passing sampling method)" if m == best else " (reference)"), paired(by, m, H3_ID), opt_pairs)
    status = (f"<p>Best gate-passing sampling method by FITNESS.md's rule: <b>{html.escape(best)}</b>.</p>" if best else
              "<p><b>No sampling method has passed both gates yet</b> (none is scored, or none keeps the information): "
              "only the references are shown, as the scale of what is on the table.</p>")
    head = ("<tr><th>method</th><th>images</th><th>&Delta;PSNR single</th><th>&Delta;PSNR mean of 8</th><th>&Delta;DISTS [95% CI]</th>"
            "<th>&Delta;LPIPS-vgg [95% CI]</th><th>gap closed, PSNR</th><th>gap closed, DISTS</th></tr>")
    return (f"{status}{HEADLINE_FORMULAS}<div class='scroll'><table><caption>vs g0/h3_decoder on the same images (each row: the images that method decoded)</caption>"
            f"{head}{rows}</table></div>")


def leaderboard_html(stats, methods, own_h3):
    """
    Pure function. Section 3: every method (references too), best DISTS first, gates as pass/fail, sortable by every column.
    """
    scored = sorted((m for m in stats), key=lambda m: rank_key(stats[m]))
    head = ("<tr><th>method</th><th>round</th><th>kind</th><th>images</th><th>space</th><th>steps</th><th>PSNR dB</th><th>PSNR mean of 8</th>"
            "<th>SSIM</th><th>LPIPS-vgg</th><th>DISTS</th><th>sharp HF</th><th>gate 1</th><th>gate 2</th><th>ms / 512&sup2; image</th><th>params</th><th>paper</th></tr>")
    body = ""
    for m in scored + [m for m in methods if m not in stats]:
        meta, s = methods[m], stats.get(m)
        if s is None:
            body += (f"<tr><td>{html.escape(m)}</td><td>{meta['round']}</td><td>{html.escape(meta['kind'])}</td><td>0</td><td>{html.escape(meta['space'])}</td>"
                     f"<td>{meta['steps']}</td><td colspan='10'>{html.escape(meta['status'])}: no renders imported yet</td><td>{meta['params']}</td>"
                     f"<td>{paper_links(meta['paper'], meta['paper_url'])}</td></tr>")
            continue
        g1, g2 = gates(s, own_h3[m])
        mark = lambda ok: f"<td data-sort='{int(ok)}' class='{'pass' if ok else 'fail'}'>{'pass' if ok else 'fail'}</td>"
        body += (f"<tr><td>{html.escape(m)}</td><td>{meta['round']}</td><td>{html.escape(meta['kind'])}</td><td>{s['n']}</td><td>{html.escape(meta['space'])}</td>"
                 f"<td>{meta['steps']}</td>{num_cell(s['psnr'], '.2f')}{num_cell(s['psnr_mean8'], '.2f')}{num_cell(s['ssim'], '.3f')}{num_cell(s['lpips_vgg'])}"
                 f"{num_cell(s['dists'])}{num_cell(s['sharp_fft'], '.3f')}{mark(g1)}{mark(g2)}{num_cell(meta['ms_per_image'], '.0f')}"
                 f"<td data-sort='{meta['params']}'>{meta['params'] / 1e6:.1f} M</td><td>{paper_links(meta['paper'], meta['paper_url'])}</td></tr>")
    return (f"<p class='note'>Best DISTS first (FITNESS.md's rank; click a header to sort). Gate 1 (information kept): PSNR of the mean of 8 seeds &ge; H3's PSNR &minus; {GATE1_DB} dB. "
            f"Gate 2 (one sample usable): PSNR &ge; H3's PSNR &minus; {GATE2_DB} dB. Both are checked against H3's decoder on the same images as the row (optimized_l1vgg has only the 12 images of §25, "
            f"so its numbers are not comparable to a 68-image row). Speed is reported, not ranked.</p>"
            f"<div class='scroll'><table>{head}{body}</table></div>")


def lineage_html(methods, stats):
    """
    Pure function. Section 6: one table per round (round 0 = references), best DISTS first, every candidate kept.
    """
    out = ""
    for rnd in sorted({m["round"] for m in methods.values()}):
        group = sorted((m for m in methods.values() if m["round"] == rnd),
                       key=lambda m: rank_key(stats[m["method_id"]]) if m["method_id"] in stats else (float("inf"),))
        rows = ""
        for m in group:
            s = stats.get(m["method_id"])
            parents = ", ".join(json.loads(m["parents"])) if m["parents"] else ""
            rows += (f"<tr><td>{html.escape(m['method_id'])}</td><td>{html.escape(m['kind'])}</td><td>{html.escape(parents)}</td>"
                     f"<td>{paper_links(m['paper'], m['paper_url'])}</td><td class='wide'>{html.escape(m['idea'])}</td>"
                     f"<td class='wide'>{html.escape(m['why_win'])}</td><td>{html.escape(m['ablation_of'])}</td>"
                     f"<td>{html.escape(m['ablation_change'])}</td><td>{html.escape(m['verdict'])}</td>"
                     f"<td class='wide'>{html.escape(m['why_held_or_failed'])}</td>"
                     f"{num_cell(s['dists'] if s else None)}{num_cell(s['psnr'] if s else None, '.2f')}<td>{html.escape(m['status'])}</td></tr>")
        title = "Round 0: references and baseline" if rnd == 0 else f"Round {rnd}"
        out += (f"<details class='sub' open><summary>{title} ({len(group)} candidates)</summary><div class='scroll'><table><tr><th>method</th><th>kind</th><th>parents</th>"
                f"<th>paper</th><th>idea</th><th>why it should win (written before running)</th><th>ablation of</th><th>the one change</th><th>verdict</th>"
                f"<th>why held or failed</th><th>DISTS</th><th>PSNR</th><th>status</th></tr>{rows}</table></div></details>")
    return out


def figures(items, side):
    """
    Pure function. Figures of (label, page-relative path, caption html) at their natural size (lazy loading).

    >>> "loading='lazy'" in figures([("a", "x.png", "c")], 512)
    True
    """
    return "".join(f"<figure style='width:{side}px'><a href='{p}'><img src='{p}' width='{side}' height='{side}' alt='{html.escape(lab)}' loading='lazy'></a>"
                   f"<figcaption><b>{html.escape(lab)}</b>{cap}</figcaption></figure>" for lab, p, cap in items)


def caption(row):
    """
    Pure function. '<br>PSNR .. dB, DISTS ..' of a render row (empty for None).

    >>> caption({"psnr": 30.0, "dists": 0.07})
    '<br>PSNR 30.00 dB, DISTS 0.0700'
    """
    return "" if row is None else f"<br>PSNR {row['psnr']:.2f} dB, DISTS {row['dists']:.4f}"


GLOSSARY = """
<dl>
<dt>H3 latent</dt><dd>MiniMax H3's compressed picture: 24 numbers per 16&times;16 pixel cell (24&times;32&times;32 for a 512&sup2; image), normalized per channel. Here always the encoder's posterior sample (seed 20261122), the latent the pipeline trains on and generates.</dd>
<dt>H3 decoder (g0/h3_decoder)</dt><dd>H3's own decoder, a regression decoder trained to minimize a pixel-style error, run in bf16 on 256-px tiles. The baseline every candidate must beat.</dd>
<dt>Sampling decoder</dt><dd>A decoder that starts from random noise and removes it step by step (diffusion or flow matching), conditioned on the latent. The same latent can give many different, equally valid images.</dd>
<dt>Posterior sample</dt><dd>One random draw from all the images that fit a latent. A perfect sampler's single draw has twice the squared error of the average image, i.e. 3.01 dB lower PSNR (Blau &amp; Michaeli 2018).</dd>
<dt>Mean-of-8</dt><dd>The average of the decodes of 8 different noise seeds: an estimate of the posterior mean. Its PSNR says how much information the decoder kept; it is what Gate 1 checks.</dd>
<dt>Flow matching</dt><dd>A way to train a noise-to-image generator: the net predicts the straight-line velocity from noise to image, and sampling integrates it in a few steps.</dd>
<dt>Deterministic twin</dt><dd>The same network and losses as a sampling candidate but without noise (a regression decoder). It separates "sampling helps" from "the loss helps": a candidate must beat its own twin.</dd>
<dt>Refiner / PMRF</dt><dd>A sampling model that starts from H3's decode (not from pure noise) and corrects it (PMRF, Ohayon et al., 2024). The no-latent twin (refiner_nolat) gets no latent input; the gap between the two is the information the refiner takes from the latent beyond H3's decode.</dd>
<dt>FIBO / FLUX.2 latent space</dt><dd>The latent of the FLUX.2 VAE (FIBO2 normalization), 32 channels at 8&times; downsampling, or 128 channels on H3's 16&times; grid after 2&times;2 patchifying. A child can diffuse there and finish with the FLUX.2 decoder. g0/flux2_oracle is the round trip of the source through that VAE: the ceiling of any such decoder.</dd>
<dt>Optimized latent (g0/optimized_l1vgg)</dt><dd>§25: a latent found by gradient descent through H3's frozen decoder to match the source (L1 + LPIPS-vgg). The upper bound of what regression decoding can get; only 12 images have it.</dd>
<dt>PSNR</dt><dd>Peak signal-to-noise ratio in dB: pixel error on a log scale. Higher is better. Rewards the blurry average.</dd>
<dt>DISTS</dt><dd>Deep Image Structure and Texture Similarity (Ding et al. 2020): a learned perceptual distance. Lower is better. No child trains on it, so it is the held-out judge and the rank.</dd>
<dt>LPIPS (alex, vgg)</dt><dd>Learned Perceptual Image Patch Similarity (Zhang et al. 2018) on AlexNet or VGG features. Lower is better. LPIPS-vgg is the second judge (children may train on LPIPS-vgg, so it is a weaker judge).</dd>
<dt>sharp HF</dt><dd>High-frequency FFT energy of the decode divided by the source's. Below 1 = blur; above 1 = grain or ringing.</dd>
<dt>Gate</dt><dd>A pass/fail check before a method may be ranked: Gate 1 keeps the information (mean-of-8), Gate 2 keeps a single sample usable.</dd>
<dt>Genetic frenzy (mini)</dt><dd>Several agents each propose, build and score decoders in rounds; the best are kept as parents. Here: 3 rounds &times; 5 children, each with a prediction written before running and an isolating ablation.</dd>
</dl>
"""


# ---------------------------------------------------------------- data and build

def read_blob(info):
    """Query. The bytes of one stored Blob (blob_info: uri, position, size)."""
    with open(info["uri"], "rb") as f:
        f.seek(info["position"])
        return f.read(info["size"])


def write_asset(data, rel):
    """Command. Writes bytes to ASSETS/rel (only when they differ). Returns the page-relative path."""
    out = ASSETS / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_bytes() != data:
        out.write_bytes(data)
    return str(out.relative_to(PAGE.parent))


def load_verdict(path):
    """Query. The verdict dict from the JSON file; None, with a loud warning on stderr, when the file is missing."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: {path} is missing: the conclusions box will say 'No verdict yet'", file=sys.stderr)
        return None
    verdict = json.loads(path.read_text())
    missing = {"verdict", "numbers", "status"} - set(verdict)
    if missing:
        raise KeyError(f"{path} lacks the keys {sorted(missing)}")
    return verdict


def section(title, body, open_=True):
    """
    Pure function. A collapsible top-level section.

    >>> section("T", "b").startswith("<details open><summary>T")
    True
    """
    return f"<details{' open' if open_ else ''}><summary>{html.escape(title)}</summary>{body}</details>"


def build(verdict_path=VERDICT):
    """Command. Writes journal/h3magic_diffdec.html and journal/assets/h3magic_diffdec/ from Bluejay (module docstring)."""
    from bluejay import open_database
    db = open_database("project.dtab")
    verdict = load_verdict(verdict_path)
    samples = {r["sample_id"]: r for r in db.query("SELECT sample_id, origin, side, has_upper_bound, blob_info(source_png) AS src FROM dd_samples ORDER BY origin, sample_id").to_pylist()}
    methods = {r["method_id"]: r for r in db.query(f"SELECT {METHOD_COLS} FROM dd_methods").to_pylist() if not r["method_id"].endswith("_smoke")}
    cols = ", ".join(METRICS)
    renders = db.query(f"SELECT method_id, sample_id, {cols}, blob_info(render_png) AS png FROM dd_renders").to_pylist()
    by = {(r["method_id"], r["sample_id"]): r for r in renders if r["method_id"] in methods}
    if H3_ID not in methods or not any(m == H3_ID for m, _ in by):
        raise ValueError(f"{H3_ID} has no renders in dd_renders: run fish data/run_proc_diffdec.fish")
    n_smoke = len(db.query("SELECT method_id FROM dd_methods").to_pylist()) - len(methods)

    stats = {}
    for m in methods:
        rows = [r for (mid, _), r in by.items() if mid == m]
        if rows and all(r["dists"] is not None for r in rows):
            stats[m] = {**{k: mean([r[k] for r in rows]) for k in METRICS}, "n": len(rows), "kind": methods[m]["kind"], "steps": methods[m]["steps"]}
    def h3_on(m):
        return {k: mean([by[(H3_ID, s)][k] for (mid, s) in by if mid == m and (H3_ID, s) in by]) for k in METRICS}
    own_h3 = {m: h3_on(m) for m in stats}  # H3's decoder on the same images as m: the gates compare like with like
    top = pick_methods(stats, own_h3)
    twins = twins_of(top, methods)
    show = [H3_ID] + top
    cropped = show + [t for t in twins if t in stats]
    leaderboard = leaderboard_html(stats, methods, own_h3)
    headline = headline_html(by, stats, own_h3)

    # section 4: crops
    order = sorted(samples, key=lambda s: -best_crop(cv2.cvtColor(cv2.imdecode(np.frombuffer(read_blob(samples[s]["src"]), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))[2])
    per_origin, chosen = {}, []
    for s in order:
        if per_origin.get(samples[s]["origin"], 0) < CROPS_PER_ORIGIN and len(chosen) < CROP_SAMPLES:
            per_origin[samples[s]["origin"]] = per_origin.get(samples[s]["origin"], 0) + 1
            chosen.append(s)
    crops_html = ""
    for sid in chosen:
        src = read_blob(samples[sid]["src"])
        x, y, _ = best_crop(cv2.cvtColor(cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        items = [("source", write_asset(crop_png(src, x, y), f"crops/{sid}_source.png"), "")]
        for m in cropped:
            if (m, sid) in by:
                tag = "" if m in show else " (deterministic twin)"
                items.append((m + tag, write_asset(crop_png(read_blob(by[(m, sid)]["png"]), x, y), f"crops/{sid}_{safe(m)}.png"), caption(by[(m, sid)])))
        crops_html += (f"<p class='note'>{html.escape(sid)} ({html.escape(samples[sid]['origin'])}): the {CROP_SIDE}&times;{CROP_SIDE} window at x={x}, y={y}, the highest high-frequency energy of the source.</p>"
                       f"<div class='row'>{figures(items, CROP_SIDE)}</div>")
    crops_block = ("<p class='note'>Pixel slices of the stored full-resolution PNGs, 1:1 (no resampling). The buttons enlarge them in the browser with nearest-neighbor "
                   "(image-rendering: pixelated); the files stay " f"{CROP_SIDE}&times;{CROP_SIDE}. Columns: source, H3's decoder, the top {TOP_METHODS} gate-passing methods, then their deterministic twins.</p>"
                   "<p><button data-crop-scale='1'>1:1</button> <button data-crop-scale='2'>2&times;</button> <button data-crop-scale='3'>3&times;</button></p>"
                   f"<div class='crops'>{crops_html or '<p>No crops yet.</p>'}</div>")

    # section 5: full images
    groups = ""
    for origin in dict.fromkeys(s["origin"] for s in samples.values()):
        ids = [s for s in samples if samples[s]["origin"] == origin]
        blocks = ""
        for sid in ids:
            side = samples[sid]["side"]
            items = [("source", write_asset(read_blob(samples[sid]["src"]), f"{sid}/source.png"), "")]
            items += [(m, write_asset(read_blob(by[(m, sid)]["png"]), f"{sid}/{safe(m)}.png"), caption(by[(m, sid)])) for m in show if (m, sid) in by]
            blocks += f"<details class='sub'><summary>{html.escape(sid)}</summary><div class='grid'>{figures(items, side)}</div></details>"
        groups += f"<details class='sub'><summary>{html.escape(origin)} ({len(ids)} images)</summary>{blocks}</details>"
    full_block = (f"<p class='note'>Every test image, full 512&times;512 lossless PNG: source, H3's decoder and the top {TOP_METHODS} methods ({html.escape(', '.join(top)) or 'none scored yet'}). "
                  "Captions: PSNR and DISTS of that one image. Click an image for the file.</p>" + groups)

    fitness = (DD_DIR / "FITNESS.md").read_text()
    survey = (DD_DIR / "research/SURVEY.md").read_text()
    mako = (DD_DIR / "MAKO_JOBS.md").read_text()
    origins = {o: sum(1 for s in samples.values() if s["origin"] == o) for o in dict.fromkeys(s["origin"] for s in samples.values())}
    method_block = (f"<p>Test set: {len(samples)} held-out 512&sup2; images ({', '.join(f'{n} {o}' for o, n in origins.items())}); "
                    f"{sum(s['has_upper_bound'] for s in samples.values())} have §25's optimized latent. Data: ~80,000 training images (Pexels clips frames 0 to 6 s and MAGICK images, train splits only) "
                    f"with their H3 posterior-sample latents; no test image or test clip was trained on. Everything is in Bluejay (dd_samples, dd_methods, dd_renders; Sparrow views view_dd_*).</p>"
                    f"<h3>The fitness rule (FITNESS.md, written before round 1)</h3>{md(fitness.split(chr(10), 1)[1])}"
                    f"<h3>Compute</h3><p>{html.escape(COMPUTE_NOTE)}</p>{md(md_section(mako, '| job id') if '| job id' in mako else mako)}")
    survey_block = (f"<p class='note'>The research sub-frenzy (10 research agents, 2026-11-23): has anyone built a sampling decoder for a frozen VAE latent? Table 1 of research/SURVEY.md; "
                    f"[U] marks unverified rows. No sampling decoder for H3 (or any frozen 16&times; video latent) exists.</p>{md(md_section(survey, '| Method', '## 2.') if '| Method' in survey else survey)}")

    smoke_note = f"; {n_smoke} code smoke-test folders are not candidates and are hidden" if n_smoke else ""
    genetic = section("The genetic frenzy (mini: 3 rounds &times; 5 children)",
                      "<p class='note'>The user asked for &ldquo;a mini genetic sub-frenzy&rdquo;. Each child states the paper it builds on, its prediction (written before running) "
                      f"and one isolating ablation. Every candidate is kept, best DISTS first within its round{smoke_note}.</p>" + lineage_html(methods, stats))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Magic: sampling decoder</title>
<style>{BASE_CSS}{EXTRA_CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: a sampling decoder for H3's latent</h1>
<p class="note">Manifest §26 (m01001). The question: does a decoder that SAMPLES (diffusion or flow, conditioned on H3's latent) get more out of the same latent than H3's own regression decoder? All numbers are measured on the same 68 held-out images and the same latents, against the source pixels.</p>
{verdict_box(verdict)}
{section("Headline numbers, in plain words", headline)}
{section("Leaderboard", leaderboard)}
{section("Texture crops at 1:1", crops_block)}
{section("Full images: all test samples", full_block, open_=False)}
{genetic}
{section("Method: the fitness rule, the test set, the data, the compute", method_block)}
{section("Survey: what exists", survey_block, open_=False)}
{section("Glossary", GLOSSARY, open_=False)}
</main>
<script>{SORT_JS}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)}: {len(methods)} methods ({len(stats)} scored), {len(samples)} samples, top = {top}, crops = {len(chosen)}")


if __name__ == "__main__":
    fire.Fire({"build": build})
