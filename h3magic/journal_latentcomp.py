# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "fire"]
# ///
"""
Builds the Latent Compositor page (manifest §24, m00983, m00984): journal/h3magic_latentcomp.html. Reads the candidates
of agents/.scratchpad/latentcomp (lineage.json from lineage.py, each candidate's results.json, genome.json and
showcase/<index>.png full-resolution decodes) and the curated test views (test_view/<index>_*.png). Copies every image
byte for byte (full 1024², lossless PNG) to journal/assets/h3magic_latentcomp/ and writes one full-resolution error map
per (method, example): |output − true composite| × ERROR_GAIN per channel. Rerun after every round:

    uv run --script h3magic/journal_latentcomp.py build
"""

import html
import json
import shutil
from pathlib import Path

import cv2
import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LC = ROOT / "agents/.scratchpad/latentcomp"
PAGE = ROOT / "journal/h3magic_latentcomp.html"
ASSETS = ROOT / "journal/assets/h3magic_latentcomp"
ERROR_GAIN = 4  # error maps show |output − truth| × 4 (a 64-level error is white)
ZOOM = 256  # side of the soft-edge zoom window in pixels (shown 2× nearest-neighbor by the page, cut from the full PNG)
REFERENCES = ("r0/dbe_gray", "r0/lsq_gray_k3", "r0/lerp_straight", "r0/conv_ref")  # always shown when scored
TOP_PER_ROUND = 3  # best candidates of each round shown in the image explorer (all of them are in the tables)
INPUTS = (("alpha", "alpha matte (white = opaque)"), ("fg_rgba", "foreground RGBA (straight alpha)"), ("bg", "background"),
          ("comp", "TRUE composite (target pixels)"), ("oracle", "oracle: decode of the true composite latent (the ceiling)"))


def soft_window(alpha, side=ZOOM):
    """
    Pure function. Top-left (x, y) of the side × side window with the most semi-transparency, Σ alpha·(1 − alpha)
    (peaks at half transparency, so faint near-zero haze does not win): where the composite is easiest to get wrong.

    Args:
        alpha (np.ndarray): (H, W) uint8

    Returns:
        tuple[int, int]

    Examples:
        >>> a = np.zeros((1024, 1024), np.uint8); a[600:640, 700:760] = 128
        >>> x, y = soft_window(a); x <= 700 and x + 256 >= 760 and y <= 600 and y + 256 >= 640
        True
    """
    a = alpha.astype(np.float32) / 255
    soft = a * (1 - a)
    box = cv2.boxFilter(soft, -1, (side, side), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    h, w = alpha.shape
    box = box[:h - side + 1, :w - side + 1]
    y, x = np.unravel_index(int(np.argmax(box)), box.shape)
    return int(x), int(y)


def error_map(pred, true):
    """
    Pure function. |pred − true| × ERROR_GAIN per channel, clipped: (H, W, 3) uint8 pair -> (H, W, 3) uint8.

    Examples:
        >>> error_map(np.full((1, 1, 3), 110, np.uint8), np.full((1, 1, 3), 100, np.uint8)).tolist()
        [[[40, 40, 40]]]
    """
    return np.clip(np.abs(pred.astype(np.int16) - true.astype(np.int16)) * ERROR_GAIN, 0, 255).astype(np.uint8)


def read_rgb(path):
    """Query. A PNG as (H, W, 3) uint8 RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def copy_asset(src, name):
    """Command. Copies src byte for byte to ASSETS/name (only when missing or changed). Returns the page-relative path."""
    dst = ASSETS / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)
    return f"assets/h3magic_latentcomp/{name}"


def fmt(v, spec):
    """
    Pure function. A number formatted with spec, or '–' when missing.

    Examples:
        >>> fmt(1.23456, ".2f"), fmt(None, ".1f")
        ('1.23', '–')
    """
    return "–" if v is None else format(v, spec)


def shown_methods(rows):
    """Pure function. Candidate ids for the image explorer: the references, then the best TOP_PER_ROUND of every round ≥ 1."""
    ids = [r for r in REFERENCES if any(row["id"] == r for row in rows)]
    for rnd in sorted({r["round"] for r in rows if r["round"] >= 1}):
        best = sorted([r for r in rows if r["round"] == rnd], key=lambda r: -r["psnr"])[:TOP_PER_ROUND]
        ids += [r["id"] for r in best if r["id"] not in ids]
    return ids


def build_examples(rows, meta_by_index):
    """
    Command (copies PNGs, writes error maps). The explorer data: one entry per showcase example (hardest first) with
    its inputs, its soft-edge window, and per shown method the output, error map and per-sample metrics.
    """
    by_id = {r["id"]: r for r in rows}
    methods = shown_methods(rows)
    indices = sorted({int(p.stem) for m in methods for p in (LC / m / "showcase").glob("*.png")})
    examples = []
    for index in indices:
        meta = meta_by_index[index]
        stem = LC / "test_view" / str(index)
        alpha = cv2.imread(f"{stem}_alpha.png", cv2.IMREAD_GRAYSCALE)
        true = read_rgb(f"{stem}_comp.png")
        ex = dict(index=index, subset=meta["subset"], rating=meta.get("rating"), tags=meta.get("tags") or [], window=soft_window(alpha),
                  inputs={k: copy_asset(Path(f"{stem}_{k}.png"), f"test/{index}_{k}.png") for k, _ in INPUTS}, outputs=[])
        oracle = read_rgb(f"{stem}_oracle.png")
        err_path = ASSETS / f"test/{index}_oracle_err.png"
        if not err_path.exists():
            cv2.imwrite(str(err_path), cv2.cvtColor(error_map(oracle, true), cv2.COLOR_RGB2BGR))
        ex["inputs"]["oracle_err"] = f"assets/h3magic_latentcomp/test/{index}_oracle_err.png"
        for m in methods:
            png = LC / m / "showcase" / f"{index}.png"
            if not png.exists():
                continue
            safe = m.replace("/", "_")
            out = copy_asset(png, f"{safe}/{index}.png")
            err = ASSETS / f"{safe}/{index}_err.png"
            if not err.exists() or err.stat().st_mtime < png.stat().st_mtime:
                cv2.imwrite(str(err), cv2.cvtColor(error_map(read_rgb(png), true), cv2.COLOR_RGB2BGR))
            res = json.loads((LC / m / "results.json").read_text())["per_sample"]
            pos = str(meta["position"])
            ex["outputs"].append(dict(id=m, name=by_id[m]["name"], path=out, err=f"assets/h3magic_latentcomp/{safe}/{index}_err.png",
                                      psnr=res["psnr"].get(pos), lpips=res["lpips"].get(pos), ssim=res["ssim"].get(pos)))
        examples.append(ex)
    order = {"hard_soft": 0, "hard_contrast": 1, "random": 2}
    return sorted(examples, key=lambda e: (order[e["subset"]], -(e["rating"] or 0)))


def leaders(rows):
    """Pure function. The best row of track L and of track H (or None)."""
    out = {}
    for track in ("L", "H"):
        best = [r for r in rows if r["track"] == track]
        out[track] = max(best, key=lambda r: r["psnr"]) if best else None
    return out


def table_html(rows, caption=""):
    """Pure function. A sortable HTML table of candidates, best decoded PSNR first."""
    head = ("<tr><th>candidate</th><th>method</th><th>parents</th><th>track</th><th>PSNR</th><th>hard soft</th><th>hard contrast</th>"
            "<th>random</th><th>SSIM</th><th>LPIPS</th><th>FLUX.2 MSE</th><th>latent MSE</th><th>params</th><th>ms fwd</th><th>ms fwd+bwd</th><th>verdict</th></tr>")
    body = []
    for r in sorted(rows, key=lambda r: -r["psnr"]):
        ms_f = r.get("ms_fwd", r.get("ms_reference"))
        body.append(
            f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['name'])}<br><span class='note'>{html.escape(r['change'][:160])}</span></td>"
            f"<td>{html.escape(', '.join(r['parents']))}</td><td>{r['track']}{' (wildcard)' if r['wildcard'] else ''}</td>"
            + "".join(f"<td data-sort='{r[k] if r.get(k) is not None else -1}'>{fmt(r.get(k), s)}</td>" for k, s in
                      (("psnr", ".2f"), ("psnr_hard", ".2f"), ("psnr_contrast", ".2f"), ("psnr_random", ".2f"), ("ssim", ".4f"), ("lpips", ".4f"), ("flux2", ".4f"), ("latent_mse", ".4f")))
            + f"<td data-sort='{r.get('params', -1)}'>{fmt(r.get('params'), ',')}</td>"
            f"<td data-sort='{ms_f if ms_f is not None else -1}'>{fmt(ms_f, '.2f')}{' (reference)' if 'ms_reference' in r else ''}</td>"
            f"<td data-sort='{r.get('ms_fwdbwd', -1)}'>{fmt(r.get('ms_fwdbwd'), '.2f')}</td><td class='note'>{html.escape(r['verdict'][:220])}</td></tr>")
    return f"<div class='scroll'><table class='sortable'><caption>{caption}</caption>{head}{''.join(body)}</table></div>"


CSS = """
:root { --bg: #fdfdfb; --fg: #1d1d1f; --muted: #66666b; --rule: #dcdcd6; --accent: #2f5d8a; --box: #f0f0ea; --good: #2e7d32; --bad: #b3261e;
        --gutter: 16px; --measure: 1600px; --font-body: 17px; --font-small: 14px; --pad: 8px; --hairline: 1px; --gap: 24px; --media: 512px; --zoom: 512px; --thumb: 256px; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; --good: #81c784; --bad: #ef9a9a; } }
:root[data-theme="dark"] { --bg: #18191b; --fg: #e8e8e6; --muted: #a0a0a5; --rule: #3a3b3e; --accent: #8ab4dc; --box: #232427; --good: #81c784; --bad: #ef9a9a; }
body { background: var(--bg); color: var(--fg); font: var(--font-body)/1.6 Georgia, serif; margin: 0; }
main { max-width: var(--measure); margin: 0 auto; padding: 0 var(--gutter) var(--gap); }
a { color: var(--accent); }
details { border-top: var(--hairline) solid var(--rule); margin-top: var(--gap); padding-top: var(--pad); }
summary { font: bold 1.25em system-ui, sans-serif; cursor: pointer; }
details.sub { border-top: 0; margin-top: var(--pad); padding-top: 0; }
details.sub summary { font-size: var(--font-body); }
.note { color: var(--muted); font: var(--font-small)/1.45 system-ui, sans-serif; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--media)), var(--media))); gap: var(--gap); }
figure { margin: 0; width: var(--media); max-width: 100%; }
figure img { width: var(--media); max-width: 100%; height: auto; aspect-ratio: 1; display: block; background: #000; }
figcaption { font: var(--font-small)/1.45 system-ui, sans-serif; margin-top: var(--pad); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font: var(--font-small)/1.4 system-ui, sans-serif; margin: var(--pad) 0; }
th, td { border-bottom: var(--hairline) solid var(--rule); padding: calc(var(--pad) / 2) var(--pad); text-align: left; vertical-align: top; }
th { cursor: pointer; position: sticky; top: 0; background: var(--bg); }
caption { text-align: left; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: var(--pad); align-items: center; font: var(--font-small) system-ui, sans-serif; margin: var(--pad) 0; }
select, button { font: var(--font-small) system-ui, sans-serif; max-width: 100%; }
.slider { position: relative; width: min(100%, 1024px); aspect-ratio: 1; user-select: none; touch-action: none; background: #000; }
.slider img { position: absolute; inset: 0; width: 100%; height: 100%; }
.slider .top { clip-path: inset(0 0 0 var(--split, 50%)); }
.slider .bar { position: absolute; top: 0; bottom: 0; left: var(--split, 50%); width: 2px; background: #fff; box-shadow: 0 0 2px #000; }
.slider .lab { position: absolute; top: var(--pad); padding: 2px var(--pad); background: rgba(0,0,0,0.6); color: #fff; font: var(--font-small) system-ui, sans-serif; }
.zooms { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--thumb)), var(--thumb))); gap: var(--pad); }
.zoom { width: var(--thumb); height: var(--thumb); overflow: hidden; position: relative; background: #000; }
.zoom div { width: 1024px; height: 1024px; transform-origin: 0 0; image-rendering: pixelated; background-size: 1024px 1024px; }
.zcap { font: var(--font-small)/1.3 system-ui, sans-serif; }
svg text { fill: var(--fg); font: 12px system-ui, sans-serif; }
.big { font: bold 1.1em system-ui, sans-serif; }
"""

JS = """
const D = __DATA__;
const $ = id => document.getElementById(id);
// sortable tables: data-sort first, then text
document.querySelectorAll('table.sortable').forEach(t => t.querySelectorAll('th').forEach((th, c) => th.onclick = () => {
  const rows = [...t.querySelectorAll('tr')].slice(1), asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
  const key = r => { const td = r.children[c]; const v = td.dataset.sort; return v !== undefined ? parseFloat(v) : td.textContent; };
  rows.sort((a, b) => { const x = key(a), y = key(b); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); });
  rows.forEach(r => t.appendChild(r));
}));
// explorer
const exSel = $('ex'), mSel = $('method'), viewSel = $('view'), slider = $('slider');
exSel.innerHTML = D.examples.map((e, i) => `<option value="${i}">${e.subset} · rating ${e.rating ?? '–'} · ${e.tags.join(', ')} · #${e.index}</option>`).join('');
function methods(e) { return [{id: 'oracle', name: 'oracle (decode of the true latent)', path: e.inputs.oracle, err: e.inputs.oracle_err}].concat(e.outputs); }
function fillMethods() {
  const e = D.examples[exSel.value], keep = mSel.value;
  mSel.innerHTML = methods(e).map((m, i) => `<option value="${m.id}">${m.id === 'oracle' ? m.name : m.id + ' · ' + m.name + (m.psnr ? ' · ' + m.psnr.toFixed(2) + ' dB' : '')}</option>`).join('');
  if ([...mSel.options].some(o => o.value === keep)) mSel.value = keep;
}
function render() {
  const e = D.examples[exSel.value], m = methods(e).find(m => m.id === mSel.value), err = viewSel.value === 'err';
  const left = err ? e.inputs.oracle_err : e.inputs.comp, right = err ? m.err : m.path;
  slider.innerHTML = `<img src="${left}"><img class="top" src="${right}"><div class="bar"></div>`
    + `<span class="lab" style="left:8px">${err ? 'oracle error ×4' : 'TRUE composite'}</span><span class="lab" style="right:8px">${m.id}${err ? ' error ×4' : ''}</span>`;
  const [x, y] = e.window, items = [['true composite', e.inputs.comp], ['alpha', e.inputs.alpha], ['background', e.inputs.bg]]
    .concat(methods(e).map(m => [m.id + (m.psnr ? ' (' + m.psnr.toFixed(2) + ' dB)' : ''), err ? m.err : m.path]));
  $('zooms').innerHTML = items.map(([lab, src]) => `<div><div class="zoom"><div style="background-image:url('${src}');transform:scale(2) translate(${-x}px,${-y}px)"></div></div><div class="zcap">${lab}</div></div>`).join('');
  $('inputs').innerHTML = [['alpha', 'alpha matte (white = opaque)'], ['fg_rgba', 'foreground RGBA'], ['bg', 'background'], ['comp', 'TRUE composite'], ['oracle', 'oracle']]
    .map(([k, lab]) => `<figure><a href="${e.inputs[k]}"><img src="${e.inputs[k]}" alt="${lab}" loading="lazy"></a><figcaption>${lab}</figcaption></figure>`).join('');
}
slider.addEventListener('pointermove', ev => { if (ev.buttons !== 1 && ev.pointerType === 'mouse') return; const r = slider.getBoundingClientRect(); slider.style.setProperty('--split', Math.min(100, Math.max(0, (ev.clientX - r.left) / r.width * 100)) + '%'); });
slider.addEventListener('pointerdown', ev => slider.dispatchEvent(new PointerEvent('pointermove', ev)));
exSel.onchange = () => { fillMethods(); render(); }; mSel.onchange = render; viewSel.onchange = render;
fillMethods(); if (D.examples.length) { const best = D.examples[0].outputs.slice(-1)[0]; if (best) mSel.value = best.id; render(); }
// scatter: PSNR vs ms fwd+bwd (log x)
(function scatter() {
  const pts = D.rows.filter(r => r.ms_fwdbwd), W = 720, H = 360, m = 48, svg = $('scatter');
  if (!pts.length) return;
  const xs = pts.map(r => Math.log10(r.ms_fwdbwd)), ys = pts.map(r => r.psnr);
  const x0 = Math.min(...xs, 0) - 0.1, x1 = Math.max(...xs, 2) + 0.1, y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys, D.oracle || 0) + 0.3;
  const X = v => m + (v - x0) / (x1 - x0) * (W - 2 * m), Y = v => H - m - (v - y0) / (y1 - y0) * (H - 2 * m);
  let s = `<line x1="${m}" y1="${H - m}" x2="${W - m}" y2="${H - m}" stroke="currentColor"/><line x1="${m}" y1="${m}" x2="${m}" y2="${H - m}" stroke="currentColor"/>`;
  [1, 10, 100].forEach(v => { if (Math.log10(v) >= x0 && Math.log10(v) <= x1) s += `<line x1="${X(Math.log10(v))}" y1="${m}" x2="${X(Math.log10(v))}" y2="${H - m}" stroke="gray" stroke-dasharray="4"/><text x="${X(Math.log10(v)) + 3}" y="${m + 12}">${v} ms${v === 10 ? ' (track L)' : v === 100 ? ' (track H)' : ''}</text>`; });
  if (D.oracle) s += `<line x1="${m}" y1="${Y(D.oracle)}" x2="${W - m}" y2="${Y(D.oracle)}" stroke="gray"/><text x="${W - m - 150}" y="${Y(D.oracle) - 4}">oracle ${D.oracle.toFixed(2)} dB</text>`;
  if (D.dbe) s += `<line x1="${m}" y1="${Y(D.dbe)}" x2="${W - m}" y2="${Y(D.dbe)}" stroke="gray" stroke-dasharray="2"/><text x="${W - m - 210}" y="${Y(D.dbe) + 14}">decode→blend→encode ${D.dbe.toFixed(2)} dB</text>`;
  const hue = r => `hsl(${(r.round * 53) % 360} 60% 50%)`;
  pts.forEach(r => s += `<circle cx="${X(Math.log10(r.ms_fwdbwd))}" cy="${Y(r.psnr)}" r="5" fill="${hue(r)}"><title>${r.id} ${r.name}\\nPSNR ${r.psnr.toFixed(2)} dB, hard soft ${r.psnr_hard?.toFixed(2)}\\n${r.ms_fwdbwd.toFixed(2)} ms fwd+bwd, ${r.params?.toLocaleString()} params</title></circle>`);
  s += `<text x="${W / 2 - 80}" y="${H - 10}">ms forward + backward (log)</text><text x="4" y="${m - 10}">decoded PSNR on EVAL (dB)</text>`;
  [0, 0.5, 1].forEach(f => { const v = y0 + f * (y1 - y0); s += `<text x="4" y="${Y(v) + 4}">${v.toFixed(1)}</text>`; });
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = s;
})();
"""


def build():
    """Command. Writes journal/h3magic_latentcomp.html and its assets (module docstring)."""
    rows = json.loads((LC / "lineage.json").read_text()) if (LC / "lineage.json").exists() else []
    plan = json.loads((LC / "data/test_plan.json").read_text())
    meta_by_index = {p["index"]: dict(subset=p["subset"], rating=p.get("rating"), tags=p.get("tags"), position=i) for i, p in enumerate(plan)}
    examples = build_examples(rows, meta_by_index) if rows else []
    lead = leaders(rows)
    by_id = {r["id"]: r for r in rows}
    oracle = rows[0]["oracle_psnr"] if rows else None
    dbe = by_id.get("r0/dbe_gray")
    rounds = sorted({r["round"] for r in rows})
    status = f"Rounds scored so far: {', '.join(map(str, rounds))} (of up to 7: round 0 = baselines)." if rows else "Round 0 (baselines) is running."
    head_rows = [r for r in (lead["L"], lead["H"], dbe, by_id.get("r0/lsq_gray_k3"), by_id.get("r0/lerp_straight")) if r]
    headline = table_html(list({r["id"]: r for r in head_rows}.values()), "Best per track vs the references (EVAL = 96 test samples, hardest first; decoded with the H3 decoder)")
    round_tables = "".join(f"<details class='sub' {'open' if rnd == max(rounds) else ''}><summary>Round {rnd}{' (baselines)' if rnd == 0 else ''}: {len([r for r in rows if r['round'] == rnd])} candidates</summary>"
                           f"{table_html([r for r in rows if r['round'] == rnd])}</details>" for rnd in rounds)
    def lead_text(track):
        """Pure function. One sentence on a track leader."""
        r = lead[track]
        return (f"<b>Track {track}</b>: {html.escape(r['id'])} ({html.escape(r['name'])}): {r['psnr']:.2f} dB on EVAL, {fmt(r['psnr_hard'], '.2f')} dB on the hard soft mattes, "
                f"LPIPS {r['lpips']:.4f}; {fmt(r.get('params'), ',')} params, {fmt(r.get('ms_fwdbwd'), '.2f')} ms forward + backward." if r else f"<b>Track {track}</b>: no candidate yet.")
    findings = "<ul>" + f"<li>{lead_text('L')}</li><li>{lead_text('H')}</li>" + (
        f"<li><b>References:</b> decode&rarr;blend&rarr;encode {dbe['psnr']:.2f} dB at {fmt(dbe.get('ms_reference'), '.0f')} ms per sample; "
        f"oracle (the true composite latent, decoded) {oracle:.2f} dB: what a perfect compositor scores (PSNR can still be gamed by blurrier latents, see r0/taeh3_dbe; read LPIPS and latent MSE beside it).</li>" if dbe and oracle else "") + "</ul>"
    data = dict(examples=examples, rows=rows, oracle=oracle, dbe=dbe["psnr"] if dbe else None)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latent Compositor</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>H3 Magic: the Latent Compositor</h1>
<p class="note">Manifest §24 (m00983, m00984, m00985). {status} Every image is the original 1024&times;1024 PNG, byte for byte.</p>
<p>The question: given the H3 video VAE's latents of an <b>alpha matte</b> (encoded as a gray image, white = opaque), a <b>foreground</b> and a <b>background</b>,
can a small network output the latent of the composite <i>&alpha;&middot;fg + (1&minus;&alpha;)&middot;bg</i> directly, without decoding the three latents, blending the pixels and encoding again?
It has to be precise (it will be used inside training losses) and cheap to backpropagate through. Decode&rarr;blend&rarr;encode is the reference to beat on cost and the precision yardstick, never the method.</p>
<p>Why it is hard: one latent cell stands for a 16&times;16 pixel patch. A hair strand, a soft edge or a semi-transparent veil lives <i>inside</i> a cell, and the composite is a product (&alpha;&middot;fg), not a sum, so a plain weighted average of latents is wrong exactly where it matters.</p>
<details open><summary>Findings</summary>{findings}
<svg id="scatter" width="100%" style="max-width:720px" role="img" aria-label="PSNR against cost"></svg>
<p class="note">Each dot is one candidate (color = round; hover for details). Higher is better, left is cheaper.</p>
{headline}</details>
<details open><summary>Examples (hardest first): true composite vs each method, error maps, soft-edge zoom</summary>
<p class="note">Drag across the image to move the divider (left: the TRUE composite, right: the chosen method). "Error" shows |output &minus; truth| &times; {ERROR_GAIN}. The zoom row cuts the 256&times;256 window with the most semi-transparent pixels out of each full-resolution PNG and shows it at 2&times; (nearest neighbor), for every method at once.</p>
<div class="controls"><label>example <select id="ex"></select></label><label>method <select id="method"></select></label>
<label>view <select id="view"><option value="img">images</option><option value="err">error maps (&times;{ERROR_GAIN})</option></select></label></div>
<div class="slider" id="slider"></div>
<h3>Soft-edge zoom (2&times;)</h3><div class="zooms" id="zooms"></div>
<h3>Inputs of this example (full resolution; click to open)</h3><div class="grid" id="inputs"></div>
</details>
<details open><summary>Every candidate, by round (lineage)</summary>
<p class="note">PSNR / SSIM / LPIPS / FLUX.2 MSE: the decoded output vs the true composite on EVAL (48 hard soft + 24 hard contrast + 24 random). Latent MSE: vs the target latent on all 300.
Cost: fp32, batch 1, one 1024&times;1024 sample, A100; "fwd+bwd" = forward and backward of a loss to the three inputs (how it will be used). Track L: &le; 10 ms fwd+bwd and &le; 5 M params; track H: &le; 100 ms and &le; 50 M.
Oracle PSNR on EVAL (decode of the true latent): {fmt(oracle, '.2f')} dB.</p>
{round_tables}</details>
<details><summary>Data</summary>
<p><b>Train:</b> 8,000 composites at 1024&times;1024: a MAGICK foreground (straight-alpha RGBA, half of them rescaled to 35&ndash;100% and moved) over a frame of an opaque Pexels video (originals of at least 1024 pixels on the short side, a random time in the first 8 s, resized to hold 1024&sup2;, center crop, random flips). Composited in 8 bit. Six latents per sample (H3 VAE, one image = one latent frame, 24&times;64&times;64, the deterministic posterior mode): the matte as a gray image, the foreground three ways (straight, premultiplied on black, over 50% gray), the background, and the true composite.</p>
<p><b>Which foreground?</b> Over 50% gray. In every fitted model it gives the lowest error (closed-form least squares, latent MSE 0.036 vs 0.044 premultiplied and 0.050 straight); it has no undefined pixels (MAGICK's straight RGB keeps the original photo under &alpha; = 0); and it is what the H3 alpha model of this project produces.</p>
<p><b>Test (300, never trained on):</b> chosen to be hard (m00984). 1,000 candidate mattes from the whole MAGICK collection (the softest by soft-pixel fraction plus random ones, ranked by edge complexity) were shown on contact sheets to a batch of 8 Sonnet agents, which rated each 1&ndash;5 for "intricate, easy to composite wrong" and tagged it (hair, fur, smoke, glass, lace, motion blur, ...). <b>hard soft</b> = the 150 top-rated mattes on random backgrounds; <b>hard contrast</b> = the next 75 on the most contrasting of 8 background frames (brightness gap and background texture behind the soft edges); <b>random</b> = 75 random pages. The 96-sample EVAL set takes the hardest 48 + 24 + 24.</p>
</details>
<details><summary>Fitness rule (written before round 1)</summary>
<ul><li>Primary score: mean decoded PSNR on EVAL vs the true composite (H3 decoder, bf16, 256-pixel tiles). Ties within 0.1 dB: lower LPIPS.</li>
<li>Hard rules: no H3 decoder or encoder inside the module (latents in, latent out); differentiable; trained on the train split only; at most 45 GPU-minutes of training.</li>
<li>Two tracks, one winner each: L (&le; 10 ms fwd+bwd, &le; 5 M params) and H (&le; 100 ms, &le; 50 M params).</li>
<li>Genetic frenzy: up to 7 rounds of up to 8 Sonnet children; at most 2 children per lineage per round, at least one wildcard per round; every candidate kept.</li></ul></details>
<details><summary>Research (sub-frenzy of 5 Sonnet agents)</summary>
<p>The closest prior work is DecFormer, "Your Latent Mask is Wrong" (<a href="https://arxiv.org/abs/2512.05198">arXiv 2512.05198</a>): a 7.7 M-parameter network predicts the latent of a composite from two latents and a mask, as z = (1&minus;a)&middot;z<sub>A</sub> + a&middot;z<sub>B</sub> + s with a per-channel gate a and a free residual s. On the FLUX VAE (8&times; downsampling) it reports about 41 dB against 33 dB for plain latent blending [arXiv 2512.05198]. Our setting is harder in one way (16&times; cells, the matte only as a latent) and the round 0 baselines measure how far plain blending gets here.</p></details>
<details><summary>Glossary</summary><dl>
<dt>Latent</dt><dd>The VAE's compressed image: 24 numbers per 16&times;16 pixel cell, 24&times;64&times;64 for a 1024&sup2; image.</dd>
<dt>Oracle</dt><dd>The true composite encoded and decoded again: the best any method that outputs a latent can do (the VAE itself loses detail).</dd>
<dt>Decode&rarr;blend&rarr;encode (dbe)</dt><dd>Decode the three latents to pixels, composite there, encode the result: the slow reference.</dd>
<dt>lerp / lsq</dt><dd>Closed-form baselines: a weighted average of latents with an estimated per-cell alpha (lerp), and a least-squares linear map over 1&times;1 or 3&times;3 cells (lsq).</dd>
<dt>EVAL</dt><dd>The 96 test samples decoded for every candidate (the 48 hardest soft mattes, 24 hard-contrast, 24 random).</dd>
<dt>Track L / H</dt><dd>Cost budgets: light (&le; 10 ms forward + backward, &le; 5 M parameters) and heavy (&le; 100 ms, &le; 50 M).</dd>
<dt>Genetic frenzy</dt><dd>Rounds of parallel agents; each round breeds the best candidates of the last (combining and mutating their ideas) plus wildcards; every candidate is kept and scored the same way.</dd>
<dt>FLUX.2 MSE</dt><dd>Distance between the FLUX.2 VAE latents (FIBO2's image encoder) of the decoded output and of the true composite: a second, independent judge.</dd>
</dl></details>
</main>
<script>{JS.replace("__DATA__", json.dumps(data))}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)}: {len(rows)} candidates, {len(examples)} examples")


if __name__ == "__main__":
    fire.Fire({"build": build})
