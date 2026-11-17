"""
Journal images at full 1024x1024 resolution: hard-links the files BlueJay stored (no copies, no disk cost).

    uv run python -m sana_alpha.journal_assets          # recreates journal/assets/eval/full/ (git ignores it)

journal/assets/eval/full/<model_tag>/<page_id>.png   every eval_generations image (blob files in data/lance)
journal/assets/eval/full/matte/<page_id>.png         resources/eval/mattes
journal/assets/eval/full/reference/<page_id>.png     resources/eval/references (the MAGICK image on gray)
journal/assets/eval/full/samples/<run>/step_N.png    models/adapter/<run>/samples (training sample sheets, 2048 x 1024)
"""

import os
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "journal/assets/eval/full"
LANCE = ROOT / "data/lance/eval_generations.lance"


def link(source, destination):
    """Command. Hard-links source to destination (replacing a stale file there), creating the folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    os.link(source, destination)


def model_label(pipeline, checkpoint, steps, prompt_variant="plain", on_adapter=True):
    """
    Pure function. The weight equation and sampler of one eval_models row, in the journal's notation: W_teacher,
    W_sprint (stock models), W_adapter, W_control = W_adapter + ΔW_control (the adapter was frozen while the control
    trained), gates (GateControl's matte input). Reads only the checkpoint path, as merge_final names its folders.

    Args:
        pipeline (str): "teacher" or "sprint"
        checkpoint (str): "stock" or a repo path (see models.load_eval_pipeline)
        steps (int): sampling steps
        prompt_variant (str): "plain", "gray" or "detailed"
        on_adapter (bool): False when the control trained on the plain teacher (its metadata "adapter" is "None")

    Returns:
        str: "equation · sampler · source"

    Examples:
        >>> model_label("teacher", "models/control/control_v1/final_step_001500", 20)
        'W_teacher + W_control − W_adapter (= W_teacher + ΔW_control) + gates · teacher sampler, 20 steps, CFG 4.5 · control_v1 step 1500'
        >>> model_label("sprint", "models/control/control_v1/sprint_step_001500_adapter0.5", 1)
        'W_sprint + 0.5·(W_adapter − W_teacher) + (W_control − W_adapter) + gates · Sprint sampler, 1 step · control_v1 step 1500'
        >>> model_label("teacher", "models/control/control_v1/step_001500", 20).split(" · ")[0]
        'W_control (= W_adapter + ΔW_control) + gates'
        >>> model_label("teacher", "models/adapter/adapter_v1/step_002000", 20)
        'W_adapter (no matte) · teacher sampler, 20 steps, CFG 4.5 · adapter_v1 step 2000'
        >>> model_label("sprint", "stock", 2, "gray")
        'W_sprint (no matte) · Sprint sampler, 2 steps · caption + gray words'
    """
    import re

    base = "W_" + pipeline
    sampler = "teacher sampler, %d steps, CFG 4.5" % steps if pipeline == "teacher" else "Sprint sampler, %d step%s" % (steps, "s" * (steps > 1))
    path = Path(checkpoint)
    match = re.fullmatch(r"(final_|sprint_)?step_(\d+)(?:_adapter([\d.]+))?", path.name)
    source = "%s step %d" % (path.parent.name, int(match[2])) if match else None
    if checkpoint == "stock":
        equation = base + " (no matte)"
    elif match and path.parent.parent.name == "adapter":
        equation = "W_adapter (no matte)"
    elif match:
        lam = float(match[3] or 0)
        if not match[1] and pipeline == "teacher":
            equation = "W_control (= %s + ΔW_control) + gates" % ("W_adapter" if on_adapter else "W_teacher")
        elif not on_adapter:
            equation = "%s + ΔW_control + gates (control trained on W_teacher, no adapter)" % base
        elif lam == 0:
            equation = "%s + W_control − W_adapter (= %s + ΔW_control) + gates" % (base, base)
        else:
            equation = "%s + %g·(W_adapter − W_teacher) + (W_control − W_adapter) + gates" % (base, lam)
    else:
        raise ValueError("no label rule for checkpoint %r" % checkpoint)
    prompt = dict(plain=None, gray="caption + gray words", detailed="detailed prompt")[prompt_variant]
    return " · ".join(part for part in (equation, sampler, source, prompt) if part)


def label_of(pipeline, checkpoint, steps, prompt_variant="plain"):
    """Query. model_label of one eval_models row; reads a control step folder's metadata to see if it trained on an adapter."""
    from safetensors import safe_open

    lora = ROOT / checkpoint / "control_lora.safetensors"
    on_adapter = not lora.exists() or safe_open(lora, "pt").metadata()["adapter"] != "None"
    return model_label(pipeline, checkpoint, int(steps), prompt_variant, on_adapter)


def write_labels(path=ROOT / "journal/eval_labels.js"):
    """Command. Writes MODEL_LABELS = {model_tag: label_of(row)} for every eval_models row, for journal/eval_grid.js."""
    import json

    import lance

    rows = lance.dataset(str(LANCE.with_name("eval_models.lance"))).to_table(
        columns=["model_tag", "pipeline", "checkpoint", "steps", "prompt_variant"]).to_pylist()
    labels = {r["model_tag"]: label_of(r["pipeline"], r["checkpoint"], r["steps"], r["prompt_variant"]) for r in rows}
    path.write_text("// Written by src/sana_alpha/journal_assets.py write_labels: the weight equation of each eval model\n"
                    "const MODEL_LABELS = %s;\n" % json.dumps(labels, indent=1, ensure_ascii=False))
    print("wrote %d labels to %s" % (len(labels), path))


def link_all(*tags, out=FULL):
    """
    Command. Hard-links the images of the given eval_models tags (default: all) into OUT/<tag>/<page_id>.png, plus
    the mattes, MAGICK references and adapter training sample sheets. Raises if a blob file is missing.

    Args:
        *tags (str): eval_models tags, e.g. adapter_v1_step002000. Default: every tag with generated images.
        out (Path): destination folder
    """
    import lance

    where = "image IS NOT NULL" + (" AND model_tag IN (%s)" % ", ".join("'%s'" % tag for tag in tags) if tags else "")
    rows = lance.dataset(str(LANCE)).to_table(columns=["model_tag", "page_id", "image"], filter=where).to_pylist()
    for row in rows:
        link(LANCE / row["image"]["blob_uri"], Path(out) / row["model_tag"] / (row["page_id"] + ".png"))
    for kind, folder in dict(matte="mattes", reference="references").items():
        for path in (ROOT / "resources/eval" / folder).glob("*.png"):
            link(path, Path(out) / kind / path.name)
    for path in (ROOT / "models/adapter").glob("*/samples/*.png"):
        link(path, Path(out) / "samples" / path.parent.parent.name / path.name)
    print("linked %d generations into %s" % (len(rows), out))
    write_labels()


if __name__ == "__main__":
    fire.Fire(link_all)
