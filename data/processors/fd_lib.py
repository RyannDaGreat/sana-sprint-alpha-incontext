"""
Shared helpers of the FastDec GA processors (fd_register, fd_score, fd_render, fd_import_mp4; manifest §22, m00998): the genome
card -> fd_methods row, the scorer's output -> fd_scores rows, the speed tiers, and the paths. No dependencies.

Paths in the tables are relative to the repo root. FD_ROOT is the repo root, found from this file (data/processors/fd_lib.py).
"""
import json
from datetime import datetime
from pathlib import Path

FD_ROOT = Path(__file__).resolve().parents[2]
FD_GA = FD_ROOT / "agents/.scratchpad/fastdec_ga"  # the search's scorer, timings and winners
SCORE_FILES = ("scores.json", "score.json")  # score.py writes the first; one variant dir has the second
TIER_MIN_SPEEDUP = (("T100", 100), ("T30", 30), ("T10", 10))  # FITNESS.md: the highest tier a candidate reaches
BASELINE_EVAL = "val20_stride4"  # BASELINE.json and most scores.json: 20 held-out clips, SSIM/LPIPS on every 4th frame
GENOME_CARD_EVAL = "genome_card"  # the numbers the agent wrote into its genome.json (no per-clip rows, no de2000 for most)
METRICS = ("psnr", "ssim", "lpips", "de2000")


def now_iso():
    """Query (reads the clock). The local time, ISO 8601, to the second: "2026-11-23T08:15:02-05:00"."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    """Query. The parsed JSON file; raises FileNotFoundError when it is missing."""
    return json.loads(Path(path).read_text())


def tier_of(speedup):
    """
    Pure function. The highest speed tier (FITNESS.md) that a speedup over the full H3 decode reaches.

    Examples:
        >>> tier_of(161.6), tier_of(40.1), tier_of(16.3), tier_of(5.0), tier_of(None)
        ('T100', 'T30', 'T10', 'slow', 'untimed')
    """
    if speedup is None:
        return "untimed"
    return next((tier for tier, least in TIER_MIN_SPEEDUP if speedup >= least), "slow")


def score_file(model_dir):
    """Query. The scorer's output file in a model dir (relative to the repo root), or None when it has none."""
    return next((p for p in (FD_ROOT / model_dir / name for name in SCORE_FILES) if p.exists()), None)


def find_timing(method_id, model_dir, timings, winners):
    """
    Pure function. (ms_clean, speedup, ms_note) of a method from the coordinator's clean timings. timings: timings.json
    (entries have "dir", "ms", "clean", "speedup"), winners: winners.json (list of {"name", "ms", "speedup"}, the final clean timing).
    (None, None, "no clean timing") when neither has it.

    Examples:
        >>> timings = {"a": {"dir": "d/a", "ms": 230.0, "clean": True, "speedup": 25.3}}
        >>> find_timing("a_x", "d/a", timings, [])
        (230.0, 25.3, 'clean timing, timings.json')
        >>> find_timing("lead_w", "lead/w", {}, [{"name": "lead_w", "ms": 36, "speedup": 161.6}])
        (36, 161.6, 'final clean timing, winners.json')
        >>> find_timing("b", "d/b", timings, [])
        (None, None, 'no clean timing')
    """
    for winner in winners:
        if winner["name"] == method_id:
            return winner["ms"], winner["speedup"], "final clean timing, winners.json"
    for entry in timings.values():
        if entry["dir"] == model_dir and entry["clean"]:
            return entry["ms"], entry["speedup"], "clean timing, timings.json"
    return None, None, "no clean timing"


def score_row(method_id, eval_id, mean, ms_clean, speedup, ms_note, steps):
    """
    Pure function. One fd_scores row (a dict of the table's columns) from a mean block of score.py's output:
    {"vs_source": {psnr, ssim, lpips, de2000?, flicker?}, "vs_full": {...}?, "sharp_grad"?, "sharp_fft"?}. Metrics that the block lacks are None.
    flicker is the temporal-difference gap to the full decode (FITNESS.md order); flicker_vs_source is the gap to the source (the TQ rule).

    Examples:
        >>> mean = {"vs_full": {"psnr": 28.0, "ssim": 0.82, "lpips": 0.15, "flicker": 0.88}, "vs_source": {"psnr": 27.5, "ssim": 0.8, "lpips": 0.18, "flicker": 1.0}}
        >>> r = score_row("m", "val20_stride4", mean, 230.0, 25.3, "note", 4500)
        >>> r["tier"], r["ssim_vs_full"], r["lpips_vs_source"], r["flicker"], r["flicker_vs_source"], r["de2000"], r["sharp_grad"]
        ('T10', 0.82, 0.18, 0.88, 1.0, None, None)
    """
    full, source = mean.get("vs_full") or {}, mean.get("vs_source") or {}
    row = {"method_id": method_id, "eval_id": eval_id, "ms_clean": ms_clean, "ms_note": ms_note, "speedup": speedup,
           "tier": tier_of(speedup), "de2000": source.get("de2000"), "flicker": full.get("flicker"),
           "flicker_vs_source": source.get("flicker"), "sharp_grad": mean.get("sharp_grad"), "sharp_fft": mean.get("sharp_fft"), "steps": steps}
    for metric in METRICS[:3]:
        row[f"{metric}_vs_full"], row[f"{metric}_vs_source"] = full.get(metric), source.get(metric)
    return row


def method_fields(genome, model_dir, status):
    """
    Pure function. The fd_methods columns that fd_register writes (all but the key, round and the claim) from a genome card.
    ckpt_path is the card's checkpoint text up to its first " (" note. "params" is a JSON text of the parameter count and of the agent's own note on the checkpoint.

    Examples:
        >>> g = {"name": "r1a1", "agent": 1, "tier_target": "T100", "kind": "mutation", "parents": ["seed"], "change": "c", "params": 5,
        ...      "checkpoint": "x/ckpt.pt (step 4)", "how_to_load": "h", "verdict": "v"}
        >>> f = method_fields(g, "x", "")
        >>> f["name"], f["status"], f["ckpt_path"], json.loads(f["params"]), f["paper"]
        ('r1a1', 'registered', 'x/ckpt.pt', {'n_params': 5, 'checkpoint_note': 'x/ckpt.pt (step 4)'}, '')
    """
    return {"agent": genome["agent"], "name": genome["name"], "tier_target": genome["tier_target"], "kind": genome["kind"],
            "parents": genome["parents"], "paper": genome.get("paper", ""), "paper_url": genome.get("paper_url", ""),
            "hypothesis": genome.get("hypothesis", ""), "change": genome["change"],
            "params": json.dumps({"n_params": genome["params"], "checkpoint_note": genome["checkpoint"]}),
            "ckpt_path": genome["checkpoint"].split(" (")[0].strip(), "model_dir": model_dir, "how_to_load": genome["how_to_load"],
            "genome_json": json.dumps(genome, indent=1), "verdict": genome["verdict"], "status": status or "registered",
            "created": now_iso()}
