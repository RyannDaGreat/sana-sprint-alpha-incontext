"""
Loading the two base models.

- Teacher: `SanaPipeline` + `DPMSolverMultistepScheduler` (flow_prediction, flow_shift 3.0), 20 steps, true CFG 4.5.
  The diffusers teacher weights are a plain flow-matching (velocity) model with timesteps in [0, 1000] that the
  pipeline multiplies by `transformer.config.timestep_scale` (0.001). TrigFlow only appears in Sprint distillation,
  as a wrapper that converts TrigFlow inputs to flow inputs; sampling the teacher needs no TrigFlow code.
- Student: `SanaSprintPipeline` + `SCMScheduler` (trigflow, sigma_data 0.5), 2 steps (max_timesteps 1.5708,
  intermediate 1.3), embedded guidance 4.5. The 2-step sampler draws fresh noise at the intermediate step, so it
  needs a seeded generator as well as the fixed initial latents.
"""

import os
import shutil
import uuid
from pathlib import Path

import torch
from diffusers import SanaPipeline, SanaSprintPipeline
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = Path("/models/sana-sprint-alpha-incontext")  # node-local disk (system-wide /models/ convention)
TEACHER_DIR = "models/Sana_Sprint_1.6B_1024px_teacher_diffusers"
SPRINT_DIR = "models/Sana_Sprint_1.6B_1024px_diffusers"
WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"  # the weights file of a diffusers transformer folder


def localize(path):
    """
    Command. Copies a repo file or folder (on FSx) to LOCAL_ROOT once per machine and returns the local copy.

    Safe when many processes call it at once: each copies to its own temp name beside the target, then renames it
    into place; a loser of the race deletes its copy and uses the winner's. A file is copied again when its size or
    mtime differs from the source (a checkpoint overwritten under the same name); a folder is assumed immutable.

    Args:
        path (str | Path): relative to the repo root, or absolute inside it, e.g. "models/Sana_Sprint_1.6B_1024px_diffusers"

    Returns:
        Path: e.g. /models/sana-sprint-alpha-incontext/models/Sana_Sprint_1.6B_1024px_diffusers
    """
    source = (ROOT / path).resolve()
    target = LOCAL_ROOT / source.relative_to(ROOT)
    stamp = lambda p: (p.stat().st_size, p.stat().st_mtime_ns)
    if target.is_dir() if source.is_dir() else (target.exists() and stamp(target) == stamp(source)):
        return target
    temp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}")
    temp.parent.mkdir(parents=True, exist_ok=True)
    print(f"localize: copying {source} -> {target}", flush=True)
    shutil.copytree(source, temp) if source.is_dir() else shutil.copy2(source, temp)
    try:
        temp.rename(target)  # atomic; a file replaces a stale copy, a folder fails if another process won
    except OSError:
        if not target.is_dir():
            raise
        shutil.rmtree(temp)
    return target

TEACHER_DEFAULTS = dict(num_inference_steps=20, guidance_scale=4.5)
SPRINT_DEFAULTS = dict(num_inference_steps=2, max_timesteps=1.5708, intermediate_timesteps=1.3, guidance_scale=4.5)


def load_pipeline(kind, device="cuda", dtype=torch.bfloat16):
    """
    Query. Loads the teacher or the Sprint student pipeline from models/ (reads weights from disk, allocates GPU memory).

    Args:
        kind (str): "teacher" or "sprint"
        device (str): torch device
        dtype (torch.dtype): weight dtype for all components

    Returns:
        tuple: (pipeline, default __call__ kwargs dict)
    """
    if kind == "teacher":
        pipe, defaults = SanaPipeline.from_pretrained(localize(TEACHER_DIR), dtype=dtype), TEACHER_DEFAULTS
    elif kind == "sprint":
        pipe, defaults = SanaSprintPipeline.from_pretrained(localize(SPRINT_DIR), dtype=dtype), SPRINT_DEFAULTS
    else:
        raise ValueError(f"kind must be 'teacher' or 'sprint', got {kind!r}")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device), dict(defaults)


def load_transformer_weights(pipe, checkpoint):
    """
    Command. Replaces the pipeline transformer's weights in place (strict: every key must match).

    Args:
        pipe: a Sana pipeline
        checkpoint (str | Path | dict): a SanaTransformer2DModel .safetensors path or diffusers transformer folder
            (repo-relative; localized first), or a state dict, e.g. a fine-tuned or `base - adapter + control` model

    Returns:
        the same pipeline
    """
    path = None if isinstance(checkpoint, dict) else localize(checkpoint)
    state_dict = checkpoint if path is None else load_file(path / WEIGHTS_FILE if path.is_dir() else path)
    pipe.transformer.load_state_dict(state_dict, strict=True)  # float32 checkpoints are rounded once to the pipe dtype
    return pipe


def load_eval_pipeline(kind, checkpoint):
    """
    Query (reads weights from disk, allocates GPU memory). The pipeline of one eval_models row. checkpoint is one of:

        "stock"                                       the stock `kind` model
        a .safetensors file, or a diffusers           its weights replace the transformer's (adapter: models/adapter/<run>/step_XXXXXX)
            transformer folder
        a control step folder                         the base it trained on (metadata "adapter"; "None" = teacher)
            (control_lora.safetensors + gates)        + PEFT control LoRA + gates (models/control/<run>/step_XXXXXX)
        a merge_final folder                          W_base + (W_control - W_adapter) in float32, then gates
            (transformer + gates.safetensors)         (models/control/<run>/final_step_XXXXXX)

    Args:
        kind (str): "teacher" or "sprint"
        checkpoint (str): see above, repo-relative

    Returns:
        tuple: (pipeline, default __call__ kwargs, conditioned: True when the model needs the matte latents)
    """
    from sana_alpha import gatecontrol as gc

    pipe, defaults = load_pipeline(kind)
    path = ROOT / checkpoint
    if (path / "control_lora.safetensors").exists():
        adapter = gc.load_control_files(path)[2]["adapter"]
        if adapter != "None":
            load_transformer_weights(pipe, adapter)
        gc.load_control(pipe.transformer, path)
    elif checkpoint != "stock":
        load_transformer_weights(pipe, checkpoint)
        if (path / "gates.safetensors").exists():
            gc.attach_gates(pipe.transformer).load_state_dict(load_file(path / "gates.safetensors"), strict=True)
            gc.enable_gatecontrol(pipe.transformer)
    conditioned = hasattr(pipe.transformer, "gatecontrol_gates")
    assert kind == "teacher" or not conditioned, f"GateControl models run on the teacher pipeline, not {kind!r}"
    return pipe, defaults, conditioned
