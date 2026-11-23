# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1"]
# ///
"""
Writes result.json: the fd_methods columns of one candidate, read from MODEL_DIR/genome.json (the genome card of the FastDec GA).
The row exists already (a seeded fd_methods row: method_id, round, model_dir); this fills in the rest. Runs on CPU, needs no GPU.
To register a new candidate: add an fd_methods row (initial_rows in project.dtab, or a commit) with method_id, round, model_dir and
status (empty), put genome.json in model_dir, run `bluejay run project.dtab fd_register`.
"""
import json
from pathlib import Path

import fire
from fd_lib import FD_ROOT, method_fields, read_json


def main(sample):
    """
    Command. Writes SAMPLE/result.json from the genome card of the method in SAMPLE/input.json. Fails when the card is missing,
    lacks a field, has another name than method_id, or the checkpoint ckpt.pt is missing.

    Args:
        sample (str): The $SAMPLE folder; input.json has method_id, model_dir, status
    """
    sample = Path(sample)
    job = json.loads((sample / "input.json").read_text())
    genome = read_json(FD_ROOT / job["model_dir"] / "genome.json")
    assert genome["name"] == job["method_id"], (genome["name"], job["method_id"])
    fields = method_fields(genome, job["model_dir"], job.get("status"))
    assert (FD_ROOT / fields["ckpt_path"]).exists(), f"no checkpoint at {fields['ckpt_path']}"
    (sample / "result.json").write_text(json.dumps(fields))
    print(f"registered {job['method_id']}: {fields['tier_target']} {fields['kind']}, parents {fields['parents']}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
