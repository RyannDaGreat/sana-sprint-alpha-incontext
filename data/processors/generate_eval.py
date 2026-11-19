"""
Eval inference for Bluejay: plan_eval makes one eval_generations row per (model, eval image); generate_eval fills them.

Not a PEP 723 script: it runs in the repo's own uv project, so it uses the same src/sana_alpha code (models,
generate, metrics) and the same torch/diffusers as training and the plain eval scripts.

    plan SAMPLE              SAMPLE/input.json {model_tag, eval_set} -> SAMPLE/rows.jsonl, one row per eval image
    start                    start this GPU's server unless it runs; it keeps the models loaded between runs
    batch BATCH              ask this GPU's server to fill every Sample folder of BATCH: image.png + result.json
    summarize STEP TAG...    (by hand) metrics table + comparison montage of models in journal/assets/STEP/
This GPU = $CUDA_VISIBLE_DEVICES, one index; its server listens on BASE_PORT + that index.
"""

import json
import os
import sys
from pathlib import Path

import fire
import rp.web_evaluator

ROOT = Path(__file__).resolve().parents[2]
GPU_BATCH = 4  # images per pipeline call inside the server
BASE_PORT = 31600  # GPU n serves on BASE_PORT + n; far from the ports in use on the node (23947, 28527, 35057)
LANCE = ROOT / "data/lance"  # the dtab's database root
SUMMARY_COLUMNS = ["iou", "exterior_gray_error", "border_gray_error", "border_std", "foreground_fraction", "seconds"]


def plan(sample):
    """
    Command. Writes SAMPLE/rows.jsonl: the eval_generations rows of one eval_models row, one per item of its eval set.

    Args:
        sample (str): the $SAMPLE folder; input.json holds model_tag and eval_set (a folder relative to the repo root)
    """
    from safetensors import safe_open

    job = json.loads((Path(sample) / "input.json").read_text())
    model_tag, eval_set = job["model_tag"], job["eval_set"]
    items = json.loads((ROOT / eval_set / "prompts.json").read_text())
    noise_seed = int(safe_open(ROOT / eval_set / "noise.safetensors", "pt").metadata()["seed"])
    rows = [dict(model_tag=model_tag, page_id=item["page_id"], caption=item["caption"], noise_index=index, noise_seed=noise_seed)
            for index, item in enumerate(items)]
    (Path(sample) / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def gpu_port():
    """Query. The port of this GPU's server: BASE_PORT + $CUDA_VISIBLE_DEVICES (one GPU index, or KeyError/ValueError)."""
    return BASE_PORT + int(os.environ["CUDA_VISIBLE_DEVICES"])


def serve():
    """Command. Answers generate_batch(BATCH) on gpu_port() until killed, keeping each model loaded."""
    import numpy as np
    import torch
    from PIL import Image
    from sana_alpha import fibo2
    from sana_alpha.eval_set import PROMPT_SUFFIXES, load_eval_set
    from sana_alpha.generate import STEP_NOISE_SEED, encode_mattes, generate
    from sana_alpha.metrics import image_metrics
    from sana_alpha.models import load_eval_pipeline

    eval_sets = {}  # EVAL_SET -> load_eval_set(EVAL_SET), plus "cond": its matte latents once a GateControl model needs them
    pipes = {}  # pipeline kind -> (checkpoint, pipe, default call kwargs, conditioned); one per kind, so at most two

    def get_pipe(kind, checkpoint):
        """Command. Returns the loaded pipeline of kind with checkpoint's weights, loading it if needed (fibo2: fibo2.load_eval_model, no defaults, conditioned)."""
        if kind not in pipes or pipes[kind][0] != checkpoint:
            pipes.pop(kind, None)
            torch.cuda.empty_cache()
            load = (lambda k, c: (fibo2.load_eval_model(c, torch.device("cuda")), {}, True)) if kind == "fibo2" else load_eval_pipeline
            pipes[kind] = (checkpoint, *load(kind, checkpoint))
        return pipes[kind][1:]

    def make_prompt(row):
        """Query. The row's prompt: its caption + PROMPT_SUFFIXES, or for "detailed" EVAL_SET/prompts_detailed.json[page_id]."""
        if row["prompt_variant"] == "detailed":
            return json.loads((ROOT / row["eval_set"] / "prompts_detailed.json").read_text())[row["page_id"]]
        return row["caption"] + PROMPT_SUFFIXES[row["prompt_variant"]]

    def generate_batch(batch):
        """Command. Generates every Sample of BATCH, grouped by eval set and model settings; writes image.png and result.json."""
        samples = sorted(p for p in Path(batch).iterdir() if p.is_dir())
        rows = [json.loads((s / "input.json").read_text()) for s in samples]
        groups = {}
        for sample, row in zip(samples, rows):
            key = (row["eval_set"], row["pipeline"], row["checkpoint"], row["steps"], row["guidance"])
            groups.setdefault(key, []).append((sample, row))
        for (eval_set, kind, checkpoint, steps, guidance), members in groups.items():
            if eval_set not in eval_sets:
                eval_sets[eval_set] = load_eval_set(ROOT / eval_set)
            data = eval_sets[eval_set]
            for _, row in members:
                assert data["items"][row["noise_index"]]["page_id"] == row["page_id"], f"noise_index does not match {row}"
            pipe, defaults, conditioned = get_pipe(kind, checkpoint)
            prompts = [make_prompt(row) for _, row in members]
            indices = [row["noise_index"] for _, row in members]
            seeds = [STEP_NOISE_SEED + i for i in indices]
            if kind == "fibo2":  # its own sampler; the seed is the item's noise (the set's noise is Sana-shaped)
                kwargs = dict(sampler="fibo2.sample", num_inference_steps=int(steps), guidance_scale=float(guidance))
                images, seconds = fibo2.generate_images(pipe, prompts, data["mattes"][indices], seeds, int(steps), float(guidance))
                prompts = [fibo2.caption_prompt(p) for p in prompts]
            else:
                if conditioned and "cond" not in data:
                    data["cond"] = encode_mattes(pipe.vae, data["mattes"])
                kwargs = dict(defaults, num_inference_steps=int(steps), guidance_scale=float(guidance))
                if kind == "sprint" and int(steps) != 2:
                    kwargs["intermediate_timesteps"] = None  # 2 steps only; popping it leaves the pipeline default 1.3
                cond = data["cond"][indices] if conditioned else None
                images, seconds = generate(pipe, prompts, data["noise"][indices], seeds=seeds, batch_size=GPU_BATCH, cond=cond, **kwargs)
            for (sample, row), prompt, image, sec, seed in zip(members, prompts, images, seconds, seeds):
                Image.fromarray(np.round(image * 255).astype(np.uint8)).save(sample / "image.png")
                result = dict(prompt=prompt, call_kwargs=json.dumps(kwargs), step_noise_seed=seed, seconds=sec,
                              **image_metrics(image, data["originals"][row["noise_index"]]))  # true matte, also for trimask sets
                (sample / "result.json").write_text(json.dumps(result))
            print(f"{kind} {checkpoint} {len(members)} images, {np.mean(seconds):.2f} s/image", flush=True)

    rp.web_evaluator.run_server(gpu_port(), scope={"generate_batch": generate_batch})


def start():
    """Command. Starts serve() for this GPU unless something already serves on its port; prints its PID once it listens."""
    print(rp.start_local_server_and_wait([sys.executable, __file__, "serve"], gpu_port()))


def batch(batch):
    """Command. Asks this GPU's server to fill every Sample folder of BATCH; raises the server's error if it fails."""
    evaluation = rp.web_evaluator.Client("localhost", gpu_port()).evaluate("generate_batch(b)", b=str(Path(batch).resolve()))
    if evaluation.errored:  # rp returns a server error instead of raising it
        raise evaluation.error


def summarize(step, *model_tags, eval_set="resources/eval"):
    """
    Command. Prints each model's mean metrics; writes journal/assets/STEP/metrics.json and hard-links the full-resolution
    images into journal/assets/eval/full/ (see sana_alpha.journal_assets). Reads the eval_generations rows that
    generate_eval filled; raises if a model has none.

    Args:
        step (str): journal asset folder name, e.g. "step2_adapter"
        *model_tags (str): eval_models rows to compare, e.g. teacher_plain adapter_run1_step004000
        eval_set (str): eval set folder relative to the repo root (EVAL_SET in the dtab)
    """
    import lance
    from sana_alpha.journal_assets import link_all

    ds = lance.dataset(str(LANCE / "eval_generations.lance"))
    tags = ", ".join(f"'{tag}'" for tag in model_tags)
    rows = ds.to_table(columns=["model_tag", "page_id", *SUMMARY_COLUMNS], with_row_id=True,
                       filter=f"image IS NOT NULL AND model_tag IN ({tags})").to_pandas()
    missing = set(model_tags) - set(rows.model_tag)
    assert not missing, f"no generated images for {sorted(missing)}: run data/run_proc_eval.fish first"
    means = rows.groupby("model_tag")[SUMMARY_COLUMNS].mean().loc[list(model_tags)]
    means.insert(0, "images", rows.groupby("model_tag").size())
    print(means.to_string(float_format="%.3f"))
    out = ROOT / "journal/assets" / step
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(means.to_json(orient="index", indent=1) + "\n")

    link_all(*model_tags)


if __name__ == "__main__":
    fire.Fire(dict(plan=plan, serve=serve, start=start, batch=batch, summarize=summarize))
