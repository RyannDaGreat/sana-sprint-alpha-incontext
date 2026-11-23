# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1"]
# ///
"""
Writes children/<eval_id>/fields.json: the fd_scores rows of one method.
Where a candidate's scores.json exists, it is read (FD_RESCORE=1 runs the scorer again). Where it does not, agents/.scratchpad/fastdec_ga/score.py
runs in a subprocess on the GPU (FD_SCORE_DEVICE, default cuda; stride FD_SCORE_STRIDE, default 4) and writes it. Where the scorer has not run
and the genome card holds scores, those become the row "genome_card". The original decoder (kind reference) reads BASELINE.json.
Timing comes from the coordinator's clean timings only (timings.json, winners.json): the scorer's own timing is marked "shared".
"""
import json
import os
import subprocess
from pathlib import Path

import fire
from fd_lib import BASELINE_EVAL, FD_GA, FD_ROOT, GENOME_CARD_EVAL, find_timing, read_json, score_file, score_row

SCORER_DEPS = ("torch", "einx", "numpy", "lpips", "fire")  # score.py also imports pytorch_msssim, which it takes from fastdec_ga/remote_libs
DEFAULT_STRIDE = 4  # FITNESS.md: SSIM and LPIPS on every 4th frame is the search's usual setting


def scorer_rows(method_id, scores, ms, speedup, ms_note, steps):
    """
    Pure function. One row of the scorer's output (its "mean" block, eval val20_stride<N>), plus its shared-GPU timing in the note.

    Examples:
        >>> s = {"stride_ssim_lpips": 4, "mean": {"vs_source": {"psnr": 27.0, "ssim": 0.8, "lpips": 0.17}}, "ms_fp16_autocast": 266.0}
        >>> r = scorer_rows("m", s, None, None, "no clean timing", 7)[0]
        >>> r["eval_id"], r["psnr_vs_source"], r["tier"], r["ms_note"]
        ('val20_stride4', 27.0, 'untimed', 'no clean timing; scorer timing 266 ms is shared, not comparable')
    """
    if "ms_fp16_autocast" in scores:
        ms_note = f"{ms_note}; scorer timing {scores['ms_fp16_autocast']:.0f} ms is shared, not comparable"
    return [score_row(method_id, f"val20_stride{scores['stride_ssim_lpips']}", scores["mean"], ms, speedup, ms_note, steps)]


def run_scorer(model_dir):
    """Command. Runs score.py on the model dir with the GPU settings of the environment; it writes the dir's scores.json."""
    command = ["uv", "run", "--no-project", *[f"--with={dep}" for dep in SCORER_DEPS], "python", str(FD_GA / "score.py"), str(FD_ROOT / model_dir), "--stride", os.environ.get("FD_SCORE_STRIDE", str(DEFAULT_STRIDE)),
               "--device", os.environ.get("FD_SCORE_DEVICE", "cuda")]
    subprocess.run(command, check=True)


def main(sample):
    """
    Command. Writes SAMPLE/children/<eval_id>/fields.json for each fd_scores row of the method in SAMPLE/input.json (method_id, model_dir, kind).
    May run the scorer on the GPU (see the module docstring).

    Args:
        sample (str): The $SAMPLE folder
    """
    sample = Path(sample)
    job = json.loads((sample / "input.json").read_text())
    method_id, model_dir, kind = job["method_id"], job.get("model_dir"), job["kind"]
    timings, winners = read_json(FD_GA / "timings.json"), read_json(FD_GA / "winners.json")
    if kind == "reference":
        baseline = read_json(FD_GA / "BASELINE.json")
        ms = read_json(FD_GA / "timing_baseline.json")["ms"]
        rows = [score_row(method_id, BASELINE_EVAL, baseline["mean"], ms, 1.0, "timing_baseline.json: the full H3 decode, bf16, the reference", None)]
        rows[0]["tier"] = "reference"
    else:
        genome = read_json(FD_ROOT / model_dir / "genome.json")
        ms, speedup, ms_note = find_timing(method_id, model_dir, timings, winners)
        steps = next((w.get("steps") for w in winners if w["name"] == method_id), genome.get("steps"))
        if score_file(model_dir) is None and not genome.get("scores"):
            run_scorer(model_dir)
        elif os.environ.get("FD_RESCORE"):
            run_scorer(model_dir)
        if score_file(model_dir) is not None:
            rows = scorer_rows(method_id, read_json(score_file(model_dir)), ms, speedup, ms_note, steps)
        else:
            note = f"{ms_note}; {genome['ms_note']}" if genome.get("ms_note") else ms_note
            rows = [score_row(method_id, GENOME_CARD_EVAL, genome["scores"], ms, speedup, note, steps)]
            rows[0]["flicker"] = genome.get("flicker")  # the card keeps flicker beside its scores, not inside
    for row in rows:
        child = sample / "children" / row["eval_id"]
        child.mkdir(parents=True)
        (child / "fields.json").write_text(json.dumps(row))
        print(f"{method_id} {row['eval_id']}: {row['tier']}, ssim vs full {row['ssim_vs_full']}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
