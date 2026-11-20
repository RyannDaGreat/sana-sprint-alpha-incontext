"""
The fast-decoder journal report (manifest §18 "fast decoder", §19; m00786, m00802, m00811, m00812, m00830, m00835):
every candidate ever measured, re-run on the same 1024² held-out test set, grouped and color-coded by round, sortable
scores, clean-GPU timings against the two preview budgets, the GA lineage, and the sneak peek timeline figures.

    PYTHONPATH=src uv run python -m sana_alpha.fastdec_report          # writes journal/fastdec.html + journal/assets/fastdec/

Images: hard links into journal/assets/fastdec/ (git ignores journal/assets; publish_pages puts the original PNGs on the
journal-assets branch, content-addressed). Every candidate shows test rows 0–7 (the hard images: the cat is row 1, the
astronaut row 6); the LEADERS best per track show all SHOW_ALL rows. 1024² lossless PNGs, shown 1:1.
"""

import json
import os
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "agents/.scratchpad/frenzy/report_all_1024"
LINEAGE = ROOT / "agents/.scratchpad/frenzy/genomes/lineage.jsonl"
OUT_HTML = ROOT / "journal/fastdec.html"
ASSETS = ROOT / "journal/assets/fastdec"
HARD_ROWS, SHOW_ALL, LEADERS = 8, 48, 6
BROWSER_FULL_MS = 4853.0  # the user's WebGPU run of the full decoder at 1024² (m00722)
BUDGET_BROWSER_MS = {"10%": 105.5, "20%": 211.0}  # 0.10 / 0.20 × 99,208 ms / (2 × 46 + 2) previews (m00807, m00835)
HEAD_MS = 0.24  # the highway head per block, clean A100 (§19)
ROUNDS = [(-1, "Round −1: first comparison", "#8a8a8a"), (0, "Round 0", "#3b7dd8"), (1, "Round 1", "#2e9e5b"),
          ("F", "Feature distillation", "#c2410c"), (2, "Round 2", "#8b5cf6"), (3, "Round 3", "#d4a017"),
          (4, "Round 4", "#0ea5e9"), (5, "Round 5", "#db2777"), (6, "Round 6", "#65a30d")]
ROUND_MINUS_1 = {"fastdec__agent_1__lsq32to3", "fastdec__agent_1__lsq_patch_k1", "fastdec__agent_1__lsq_patch_k3",
                 "fastdec__agent_1__lsq_patch_k5_1000img", "fastdec__agent_4__fal_tiny_ae", "fastdec__agent_4__onnx_main__fal_lite_C",
                 "fastdec__agent_4__onnx_main__fal_lite_D", "fastdec__agent_4__academiasd_tae", "fastdec__agent_5__runs__c48"}


def round_of(name):
    """
    Pure function. The GA round of a candidate id (render_all.py names: folder__agent__file).

    Examples:
        >>> round_of("fastdec_r2__agent_1__agent_2__D_cont"), round_of("featdist__agent_3__out_only"), round_of("fastdec__agent_4__fal_tiny_ae")
        (2, 'F', -1)
    """
    if name in ROUND_MINUS_1:
        return -1
    head = name.split("__")[0]
    return 0 if head == "fastdec" else "F" if head == "featdist" else int(head.split("_r")[1])


def table_rows(cands, full_ms):
    """Pure function. One <tr> per candidate, numeric cells with data-sort keys (the house sortable-table rule)."""
    color = {r: c for r, _, c in ROUNDS}
    out = []
    for c in cands:
        gpu = c.get("ms")
        browser = gpu and full_ms and BROWSER_FULL_MS * gpu / full_ms
        fits = "—" if not gpu else ("10% and 20%" if gpu + HEAD_MS <= BUDGET_BROWSER_MS["10%"] * full_ms / BROWSER_FULL_MS else
                                     "20%" if gpu + HEAD_MS <= BUDGET_BROWSER_MS["20%"] * full_ms / BROWSER_FULL_MS else "neither")
        cell = lambda v, f: f"<td data-sort='{v if v is not None else 1e9}'>{f.format(v) if v is not None else '—'}</td>"
        out.append(f"<tr><td><span class='dot' style='background:{color[c['round']]}'></span>{c['name'].replace('__', ' / ')}</td>"
                   f"<td data-sort='{ -1 if c['round'] == -1 else 2.5 if c['round'] == 'F' else c['round']}'>{c['round']}</td>"
                   + cell(c["ssim"], "{:.4f}") + cell(c["lpips"], "{:.4f}") + cell(c["delta_e_48"], "{:.2f}") + cell(c["psnr"], "{:.2f}")
                   + cell(c.get("ms32"), "{:.2f}") + cell(c.get("ms16"), "{:.2f}") + cell(browser, "{:.0f}") + f"<td>{fits}</td></tr>")
    return "\n".join(out)


def build(rows_all=SHOW_ALL):
    """Command. Writes journal/fastdec.html and hard-links its images into journal/assets/fastdec/."""
    scores = json.loads((SRC / "scores.json").read_text())
    clean = json.loads((SRC / "clean_timing.json").read_text()) if (SRC / "clean_timing.json").exists() else {}
    full = scores["FULL_DECODER"]
    full_ms = clean.get("FULL_DECODER", {}).get("ms_ort_cuda_fp32")
    cands = []
    for name, s in scores.items():
        if name == "FULL_DECODER" or "ssim" not in s or "duplicate_of" in s or "error" in s:
            continue
        t = clean.get(name, {})
        ms = [v for v in (t.get("ms_ort_cuda_fp32"), t.get("ms_ort_cuda_fp16")) if v]
        cands.append(dict(name=name, round=round_of(name), ssim=s["ssim"], lpips=s["lpips"], delta_e_48=s["delta_e_48"], psnr=s["psnr"],
                          ms32=t.get("ms_ort_cuda_fp32"), ms16=t.get("ms_ort_cuda_fp16"), ms=min(ms) if ms else None))
    rank = lambda c: -(2 * c["ssim"] - c["lpips"] - c["delta_e_48"] / 50)  # quality order inside a round (ΔE ≈ 1–3 → /50 ≈ LPIPS scale)
    order = {r: i for i, (r, _, _) in enumerate(ROUNDS)}
    cands.sort(key=lambda c: (order[c["round"]], rank(c)))
    budget_ms = {k: v * full_ms / BROWSER_FULL_MS for k, v in BUDGET_BROWSER_MS.items()} if full_ms else {}
    def distinct(cs):  # an fp16 re-export is the same network: one leader entry (the faster timing)
        seen, out = set(), []
        for c in cs:
            base = c["name"][:-5] if c["name"].endswith("_fp16") else c["name"]
            if base not in seen:
                seen.add(base); out.append(c)
        return out
    leaders = {k: distinct(sorted((c for c in cands if c["ms"] and c["ms"] + HEAD_MS <= b), key=rank))[:LEADERS] for k, b in budget_ms.items()}
    shown_all = {c["name"] for v in leaders.values() for c in v}
    # images: hard links (no copies); every candidate rows 0..HARD_ROWS-1, leaders all rows
    def link(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
    columns = [("source", "Original 1024² image", "#000"), ("full_decoder", "Full FLUX.2 decoder", "#000")] + \
              [(c["name"], c["name"].replace("__", " / "), dict((r, col) for r, _, col in ROUNDS)[c["round"]]) for c in cands]
    n_png = 0
    for key, _, _ in columns:
        n = rows_all if key in shown_all or key in ("source", "full_decoder") else HARD_ROWS
        for i in range(n):
            src = SRC / key / f"{i}.png"
            if src.exists():
                link(src, ASSETS / key / f"{i}.png"); n_png += 1
    gb = sum(f.stat().st_size for f in ASSETS.rglob("*.png")) / 1e9
    lineage = [json.loads(l) for l in LINEAGE.read_text().splitlines()] if LINEAGE.exists() else []
    mermaid = "graph LR\n" + "\n".join(f"  {p.split('__')[-1].replace('-', '_')}[{p.split('__')[-1]}] --> {k['child'].replace('-', '_')}[{k['child']}: {k['kind']} {k['track']}]"
                                         for k in lineage for p in (k["parents"] or ["research"]))
    # literal asset paths (publish_pages rewrites each one to its journal-assets URL; a JS-built path would not be rewritten)
    imgs = lambda k: [f"assets/fastdec/{k}/{i}.png" for i in range(rows_all if k in shown_all or k in ("source", "full_decoder") else HARD_ROWS)
                      if (ASSETS / k / f"{i}.png").exists()]
    data = dict(columns=[dict(key=k, label=l, color=c, imgs=imgs(k)) for k, l, c in columns], hard=HARD_ROWS, rows=rows_all)
    legend = "".join(f"<span class='leg'><span class='dot' style='background:{c}'></span>{t}</span>" for _, t, c in ROUNDS)
    leader_html = "".join(f"<h4>Track {k} (≤ {budget_ms[k]:.1f} ms on the A100 ≈ {BUDGET_BROWSER_MS[k]:.0f} ms in the browser)</h4><ol>"
                          + "".join(f"<li>{c['name'].replace('__', ' / ')}: SSIM {c['ssim']:.3f}, LPIPS {c['lpips']:.3f}, ΔE {c['delta_e_48']:.2f}, {c['ms']:.2f} ms</li>" for c in v)
                          + "</ol>" + ("" if v else "<p>No scored candidate fits this budget yet.</p>") for k, v in leaders.items())
    html = TEMPLATE.format(full_ssim=full["ssim"], full_de=full["delta_e_48"], full_ms=full_ms or float("nan"), legend=legend, leaders=leader_html,
                           table=table_rows(cands, full_ms), mermaid=mermaid or "graph LR\n  r2[round 2] --> r3[round 3 pending]",
                           data=json.dumps(data), n=len(cands), n_png=n_png, gb=gb, b10=budget_ms.get("10%", 0), b20=budget_ms.get("20%", 0))
    OUT_HTML.write_text(html)
    print(f"{OUT_HTML}: {len(cands)} candidates, {n_png} PNGs linked, {gb:.2f} GB in journal/assets/fastdec")


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fast decoder for the sneak peek</title>
<style>
:root {{ --bg: #fbfbf9; --fg: #1b1b1f; --muted: #6b6b72; --line: #e3e3e0; --accent: #2563eb; --cell: 1024px; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #121214; --fg: #ececef; --muted: #a3a3ab; --line: #2b2b31; --accent: #6b9cf6; }} }}
body {{ font: 16px/1.55 Inter, system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 70em; padding: 2em max(16px, 3vw); }}
h1 {{ font-weight: 650; letter-spacing: -0.01em; }} summary {{ font-weight: 600; font-size: 1.15em; cursor: pointer; margin: 1em 0 0.4em; }}
p, li {{ max-width: 46em; }} .meta {{ color: var(--muted); font-size: 0.9em; }}
table.sort {{ border-collapse: collapse; font-size: 0.85em; }} table.sort th, table.sort td {{ border-bottom: 1px solid var(--line); padding: 3px 8px; text-align: left; }}
table.sort th {{ cursor: pointer; position: sticky; top: 0; background: var(--bg); }}
.dot {{ display: inline-block; width: 0.75em; height: 0.75em; border-radius: 50%; margin-right: 0.4em; vertical-align: -1px; }}
.leg {{ margin-right: 1.2em; white-space: nowrap; }}
.wide {{ width: 100vw; position: relative; left: 50%; transform: translateX(-50%); overflow-x: auto; }}
.grid {{ display: grid; grid-auto-flow: column; gap: 6px; padding: 0 16px; }}
.col {{ width: var(--cell); }} .col h5 {{ margin: 0 0 4px; font-size: 0.85em; border-top: 6px solid; padding-top: 4px; position: sticky; top: 0; background: var(--bg); }}
.col img {{ display: block; width: var(--cell); height: var(--cell); margin-bottom: 6px; }}
.cmp {{ position: relative; width: var(--cell); height: var(--cell); }} .cmp img {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
</head><body>
<h1>A fast decoder for the sneak peek</h1>
<p class="meta">FIBO2 trimask model · manifest §18 (sneak peek) and §19 (results) · every image is a 1024² original or 1024² decode, lossless PNG, shown 1:1.</p>

<details open><summary>Summary</summary>
<p>The sneak peek shows a preview after every sampler step and every transformer block. A preview needs a <b>fast decoder</b>: latents → pixels far faster than the full FLUX.2 VAE decoder
(here {full_ms:.0f} ms per 1024² on an A100 with ONNX Runtime; 4,853 ms in the user's browser). The user's limit is a 10% slowdown, later also 20% (m00807, m00835): per preview
0.10 × 99,208 ms / (2 passes × 46 blocks + 2 steps) ≈ 105 ms in the browser, which is <b>≈ {b10:.1f} ms</b> on the A100 at the browser/A100 ratio of the full decoder (20%: ≈ {b20:.1f} ms), including the 0.24 ms highway head.</p>
<p>{n} candidates from a genetic search (rounds of 5 Sonnet agents, mutations and crossovers of the best) were all re-run on the same 300 held-out 1024² images (the full decoder: SSIM {full_ssim:.3f}, ΔE2000 {full_de:.2f} against the originals) and timed with the GPU to themselves.</p>
{leaders}
</details>

<details open><summary>Glossary</summary><ul>
<li><b>Sneak peek</b>: previews during a generation, after every step and every transformer block.</li>
<li><b>Fast decoder</b>: a small network latent (1, 32, LH, LW) → image (1, 3, 8·LH, 8·LW) in [0, 1], replacing the full VAE decoder for previews only.</li>
<li><b>Sneak peek highway</b>: one small head per transformer block that predicts the final layer's output from that block, through a gradient stop.</li>
<li><b>Tweedie estimate x̂0</b>: x_t − σ·v, the clean latent the model implies at a step; what a preview shows.</li>
<li><b>SSIM / LPIPS / ΔE2000 / PSNR</b>: structural similarity (1 = identical), learned perceptual distance (alex; 0 = identical), CIEDE2000 color difference (mean over pixels, first 48 test rows), peak signal-to-noise ratio, each against the original image.</li>
<li><b>Clean-GPU timing</b>: every other process on the GPU paused, ONNX Runtime CUDA, one 1024² decode, median of 25 after 5 warm-ups (m00806).</li>
<li><b>Round / track</b>: one generation of the genetic search; the 10% and 20% budgets, each with its own winners.</li>
<li><b>λ</b>: the adapter fraction; the decoders do not depend on it (they decode latents); the timeline figures state it.</li>
</ul></details>

<details open><summary>All candidates, sortable</summary>
<p class="meta">Click a header to sort. ms = clean A100 ORT per 1024² (fp32 / fp16 export); browser ms = estimate 4,853 × candidate / full decoder (not measured in a browser). {legend}</p>
<div style="overflow-x:auto"><table class="sort"><thead><tr><th>Candidate</th><th>Round</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>ΔE ↓</th><th>PSNR ↑</th><th>ms fp32</th><th>ms fp16</th><th>browser ms (est.)</th><th>Fits</th></tr></thead>
<tbody>{table}</tbody></table></div>
</details>

<details open><summary>Side by side, 1024² at 1:1</summary>
<p class="meta">Columns grouped by round (color bar), best first inside a round. Rows 1–8 are the hard images (cat: row 2, astronaut: row 7) for every candidate; the track leaders, the original and the full decoder show all rows. {n_png} PNGs ({gb:.2f} GB).
Compare two: <select id="ca"></select> vs <select id="cb"></select> on row <input id="cr" type="number" min="0" value="1" style="width:4em"> (drag over the image).</p>
<div class="cmp" id="cmp"><img id="ia"><img id="ib" style="clip-path: inset(0 0 0 50%)"></div>
<div class="wide"><div class="grid" id="grid"></div></div>
</details>

<details open><summary>Family tree (genetic search)</summary>
<pre class="mermaid">{mermaid}</pre>
</details>

<details open><summary>The sneak peek timeline</summary>
<p>Per-block previews jump back at every new step (a sawtooth) unless blended with the previous step's x̂0 (weights fitted on held-out generations); with the previous x̂0 as a head input the highway is monotone by itself. The sneak peek runs in the app end to end (headless Chrome, wasm, 256², turbo 4 steps): +2.3% wall time with it on, 80 previews.</p>
<p><img src="assets/sneakpeek/live_test_256/peek_02.png" width="256" height="256"> <img src="assets/sneakpeek/live_test_256/peek_05.png" width="256" height="256"> <img src="assets/sneakpeek/live_test_256/timeline_accordion.png"></p>
<p class="meta">Held-out generations, ridge heads; gray/blue = plain head alone / blended; dashed = with the previous x̂0 as an input. λ = 0 is the user's setting, λ = 1 the training setting (m00834).</p>
<h4>Turbo, 1024², 4 steps (the shipped mode), λ = 0</h4><img src="assets/sneakpeek/timeline_turbo1024_4step_lam0.png" style="max-width:100%">
<h4>Turbo, 1024², 4 steps, λ = 1</h4><img src="assets/sneakpeek/timeline_turbo1024_4step_lam1.png" style="max-width:100%">
<h4>Dev, 512², 8 steps, CFG 5, λ = 0</h4><img src="assets/sneakpeek/timeline_dev512_8step_cfg5_lam0.png" style="max-width:100%">
<h4>Dev, 512², 8 steps, CFG 5, λ = 1</h4><img src="assets/sneakpeek/timeline_dev512_8step_cfg5_lam1.png" style="max-width:100%">
<h4>Blend-weight heatmap (dev, 512², 8 steps, λ = 0)</h4><img src="assets/sneakpeek/timeline_8step_512.png" style="max-width:100%">
</details>

<details open><summary>Feature distillation: the verdict</summary>
<p>Five Sonnet agents distilled the full FLUX.2 decoder's intermediate activations into small students (FitNets hints through 1×1 adapters, progressive stage-wise training, attention/Gram/relational losses, weight inheritance by slicing the full decoder, hints plus a late hinge/R1 GAN), each with an output-only ablation of the same student and steps. Re-measured on the 300 held-out 1024² images: <b>the feature losses never beat output-only distillation on SSIM or LPIPS (5 of 5)</b>; only one improved ΔE (1.57 vs 1.94). Slicing the full decoder's channels destroyed it (SSIM 0.79) and the recovery was too short. Caveat: about an hour of training each; hint losses may pay off with far longer schedules (BK-SDM trains much longer) [FitNets; BK-SDM].</p>
</details>

<details><summary>Sources cited</summary><ul>
<li>madebyollin/taesd, taef2 (MIT): tiny autoencoders. github.com/madebyollin/taesd</li>
<li>fal/FLUX.2-Tiny-AutoEncoder (Apache-2.0). huggingface.co/fal/FLUX.2-Tiny-AutoEncoder</li>
<li>Romero et al., FitNets, arXiv 1412.6550; Kim et al., BK-SDM, arXiv 2305.15798; Zagoruyko &amp; Komodakis, attention transfer, arXiv 1612.03928; Tung &amp; Mori, arXiv 1907.09682: feature distillation.</li>
<li>Mescheder et al., arXiv 1801.04406 (R1); Wortsman et al., WiSE-FT, arXiv 2109.01903; Belrose et al., tuned lens, arXiv 2303.08112.</li>
<li>Sharma et al. 2005, the CIEDE2000 color-difference formula.</li>
</ul></details>

<script>
const D = {data};
const byKey = Object.fromEntries(D.columns.map((c) => [c.key, c]));
const src = (k, i) => byKey[k].imgs[i];
const grid = document.getElementById('grid');
for (const c of D.columns) {{
  const col = document.createElement('div'); col.className = 'col';
  col.innerHTML = `<h5 style="border-color:${{c.color}}">${{c.label}}</h5>` + c.imgs.map((u, i) =>
    `<img loading="lazy" width="1024" height="1024" src="${{u}}" alt="${{c.label}}, test row ${{i}}">`).join('');
  grid.appendChild(col);
}}
const ca = document.getElementById('ca'), cb = document.getElementById('cb'), cr = document.getElementById('cr');
for (const s of [ca, cb]) for (const c of D.columns) s.add(new Option(c.label, c.key));
cb.selectedIndex = 1;
const show = () => {{ const i = Math.min(Number(cr.value), D.hard - 1); document.getElementById('ia').src = src(ca.value, i); document.getElementById('ib').src = src(cb.value, i); }};
[ca, cb, cr].forEach((e) => e.addEventListener('input', show)); show();
document.getElementById('cmp').addEventListener('pointermove', (e) => {{
  const r = e.currentTarget.getBoundingClientRect(); document.getElementById('ib').style.clipPath = `inset(0 0 0 ${{(100 * (e.clientX - r.left)) / r.width}}%)`; }});
document.querySelectorAll('table.sort th').forEach((th, j) => th.addEventListener('click', () => {{
  const tb = th.closest('table').tBodies[0], rows = [...tb.rows], asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
  const key = (r) => {{ const c = r.cells[j]; const v = c.dataset.sort ?? c.textContent; return isNaN(+v) ? v : +v; }};
  rows.sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * (asc ? 1 : -1)).forEach((r) => tb.appendChild(r));
}}));
mermaid.initialize({{ startOnLoad: true, theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default' }});
</script>
</body></html>
"""

if __name__ == "__main__":
    fire.Fire(build)
