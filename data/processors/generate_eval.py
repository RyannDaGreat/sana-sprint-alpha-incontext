"""
Eval inference for Bluejay: plan_eval makes one eval_generations row per (model, eval image); generate_eval fills them.

Not a PEP 723 script: it runs in the repo's own uv project, so it uses the same src/sana_alpha code (models,
generate, metrics) and the same torch/diffusers as training and the plain eval scripts.

    plan SAMPLE EVAL_SET     SAMPLE/input.json {model_tag} -> SAMPLE/rows.jsonl, one row per eval image
    start                    start this GPU's server unless it runs; it keeps the models loaded between runs
    batch BATCH EVAL_SET     ask this GPU's server to fill every Sample folder of BATCH: image.png + result.json
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


def plan(sample, eval_set):
    """
    Command. Writes SAMPLE/rows.jsonl: the eval_generations rows of one eval_models row, one per eval image.

    Args:
        sample (str): the $SAMPLE folder; input.json holds model_tag
        eval_set (str): eval set folder relative to the repo root ($EVAL_SET)
    """
    from safetensors import safe_open

    model_tag = json.loads((Path(sample) / "input.json").read_text())["model_tag"]
    items = json.loads((ROOT / eval_set / "prompts.json").read_text())
    noise_seed = int(safe_open(ROOT / eval_set / "noise.safetensors", "pt").metadata()["seed"])
    rows = [dict(model_tag=model_tag, page_id=item["page_id"], caption=item["caption"], noise_index=index, noise_seed=noise_seed)
            for index, item in enumerate(items)]
    (Path(sample) / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def gpu_port():
    """Query. The port of this GPU's server: BASE_PORT + $CUDA_VISIBLE_DEVICES (one GPU index, or KeyError/ValueError)."""
    return BASE_PORT + int(os.environ["CUDA_VISIBLE_DEVICES"])


def serve():
    """Command. Answers generate_batch(BATCH, EVAL_SET) on gpu_port() until killed, keeping each model loaded."""
    import numpy as np
    import torch
    from PIL import Image
    from sana_alpha.eval_set import PROMPT_SUFFIXES, load_eval_set
    from sana_alpha.generate import STEP_NOISE_SEED, generate
    from sana_alpha.metrics import image_metrics
    from sana_alpha.models import load_pipeline, load_transformer_weights

    eval_sets = {}  # EVAL_SET -> load_eval_set(EVAL_SET)
    pipes = {}  # pipeline kind -> (checkpoint, pipe, default call kwargs); one per kind, so the GPU holds at most two

    def get_pipe(kind, checkpoint):
        """Command. Returns the loaded pipeline of kind with checkpoint's weights, loading it if needed."""
        if kind not in pipes or pipes[kind][0] != checkpoint:
            pipes.pop(kind, None)
            torch.cuda.empty_cache()
            pipe, defaults = load_pipeline(kind)
            if checkpoint != "stock":
                load_transformer_weights(pipe, checkpoint)
            pipes[kind] = (checkpoint, pipe, defaults)
        return pipes[kind][1:]

    def generate_batch(batch, eval_set):
        """Command. Generates every Sample of BATCH, grouped by model settings; writes image.png and result.json."""
        if eval_set not in eval_sets:
            eval_sets[eval_set] = load_eval_set(ROOT / eval_set)
        data = eval_sets[eval_set]
        samples = sorted(p for p in Path(batch).iterdir() if p.is_dir())
        rows = [json.loads((s / "input.json").read_text()) for s in samples]
        groups = {}
        for sample, row in zip(samples, rows):
            assert data["items"][row["noise_index"]]["page_id"] == row["page_id"], f"noise_index does not match {row}"
            groups.setdefault((row["pipeline"], row["checkpoint"], row["steps"], row["guidance"]), []).append((sample, row))
        for (kind, checkpoint, steps, guidance), members in groups.items():
            pipe, defaults = get_pipe(kind, checkpoint)
            kwargs = dict(defaults, num_inference_steps=int(steps), guidance_scale=float(guidance))
            if kind == "sprint" and int(steps) != 2:
                kwargs.pop("intermediate_timesteps")  # SCMScheduler defines it for 2 steps only
            prompts = [row["caption"] + PROMPT_SUFFIXES[row["prompt_variant"]] for _, row in members]
            indices = [row["noise_index"] for _, row in members]
            seeds = [STEP_NOISE_SEED + i for i in indices]
            images, seconds = generate(pipe, prompts, data["noise"][indices], seeds=seeds, batch_size=GPU_BATCH, **kwargs)
            for (sample, row), prompt, image, sec, seed in zip(members, prompts, images, seconds, seeds):
                Image.fromarray(np.round(image * 255).astype(np.uint8)).save(sample / "image.png")
                result = dict(prompt=prompt, call_kwargs=json.dumps(kwargs), step_noise_seed=seed, seconds=sec,
                              **image_metrics(image, data["mattes"][row["noise_index"]]))
                (sample / "result.json").write_text(json.dumps(result))
            print(f"{kind} {checkpoint} {len(members)} images, {np.mean(seconds):.2f} s/image", flush=True)

    rp.web_evaluator.run_server(gpu_port(), scope={"generate_batch": generate_batch})


def start():
    """Command. Starts serve() for this GPU unless something already serves on its port; prints its PID once it listens."""
    print(rp.start_local_server_and_wait([sys.executable, __file__, "serve"], gpu_port()))


def batch(batch, eval_set):
    """Command. Asks this GPU's server to fill every Sample folder of BATCH; raises the server's error if it fails."""
    evaluation = rp.web_evaluator.Client("localhost", gpu_port()).evaluate(
        "generate_batch(b, e)", b=str(Path(batch).resolve()), e=eval_set)
    if evaluation.errored:  # rp returns a server error instead of raising it
        raise evaluation.error


if __name__ == "__main__":
    fire.Fire(dict(plan=plan, serve=serve, start=start, batch=batch))
