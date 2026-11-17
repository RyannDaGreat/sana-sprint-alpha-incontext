"""
The fixed eval set: MAGICK images where sha256(page_id) mod 4000 is 0, with fixed noise.

Build it once (writes resources/eval/):
    PYTHONPATH=src uv run python -m sana_alpha.eval_set build

The rule must match data/project.dtab (EVAL_MODULUS, the `split` column of import_index).
"""

import hashlib
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "resources/eval"
DOWNLOAD_DIR = ROOT / "agents/.scratchpad/download/magick"  # untrusted downloads, own folder
MAGICK_REPO = "OneOverZero/MAGICK"
EVAL_MODULUS = 4000  # must equal EVAL_MODULUS in data/project.dtab
GRAY_LEVEL = 0.5
NOISE_SEED = 20261117  # the date the eval set was fixed
LATENT_SHAPE = (32, 32, 32)  # (LC, LH, LW): DC-AE f32c32 latent of a 1024x1024 image
PROMPT_SUFFIXES = dict(plain="", gray=", on a plain 50% gray background")  # prompt variant -> text added to the caption


def is_eval_page(page_id):
    """
    Pure function. The eval rule: sha256 of the UTF-8 page_id, read as a big integer, mod EVAL_MODULUS is 0.

    Args:
        page_id (str): MAGICK image ID

    Returns:
        bool

    Examples:
        >>> is_eval_page("6NqAw3m1ss"), is_eval_page("Hcx82jcUpM")
        (True, False)
    """
    return int(hashlib.sha256(page_id.encode("utf-8")).hexdigest(), 16) % EVAL_MODULUS == 0


def select_eval_rows(index):
    """
    Pure function. Dedupes the MAGICK index by page_id, keeps eval rows that are not nsfw, sorted by page_id.

    Args:
        index (pd.DataFrame): magick_index.tsv as read with quoting=3 (columns page_id, subject, nsfw, ...)

    Returns:
        pd.DataFrame: columns page_id, caption, sim_score

    Examples:
        >>> df = pd.DataFrame(dict(page_id=["6NqAw3m1ss", "6NqAw3m1ss", "Hcx82jcUpM"],
        ...                        subject=["two dice", "two dice", "crayfish"], nsfw=[0, 0, 0], sim_score=[0.99, 0.99, 0.98]))
        >>> select_eval_rows(df).to_dict("records")
        [{'page_id': '6NqAw3m1ss', 'caption': 'two dice', 'sim_score': 0.99}]
    """
    unique = index.drop_duplicates("page_id")
    keep = unique.page_id.map(is_eval_page) & (unique.nsfw == 0)
    rows = unique[keep].sort_values("page_id")
    return rows.rename(columns=dict(subject="caption"))[["page_id", "caption", "sim_score"]].reset_index(drop=True)


def split_rgba(rgba):
    """
    Pure function. Splits a straight-alpha RGBA image into its matte and its RGB composited on gray.

    Args:
        rgba (np.ndarray): (H, W, 4) uint8 RGBA, straight alpha

    Returns:
        tuple: matte (H, W) float32 in [0,1] (1 = opaque), gray composite (H, W, 3) float32 RGB in [0,1]

    Examples:
        >>> m, g = split_rgba(np.array([[[255, 0, 0, 255], [255, 0, 0, 0]]], np.uint8))
        >>> m.tolist(), g.tolist()
        ([[1.0, 0.0]], [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]]])
    """
    x = rgba.astype(np.float32) / 255
    alpha = x[..., 3]
    return alpha, x[..., :3] * alpha[..., None] + GRAY_LEVEL * (1 - alpha[..., None])


def make_noise(count, seed=NOISE_SEED):
    """
    Pure function. Fixed standard-normal initial latents, one per eval item, made on CPU so they are device independent.

    Args:
        count (int): N, number of eval items
        seed (int): torch CPU generator seed

    Returns:
        torch.Tensor: (N, LC, LH, LW) float32, e.g. (29, 32, 32, 32)

    Examples:
        >>> make_noise(2).shape, bool((make_noise(2) == make_noise(2)).all())
        (torch.Size([2, 32, 32, 32]), True)
    """
    return torch.randn((count, *LATENT_SHAPE), generator=torch.Generator("cpu").manual_seed(seed))


def load_eval_set(eval_dir=EVAL_DIR):
    """
    Query. Reads the eval set from disk.

    Args:
        eval_dir (Path): folder written by `build`

    Returns:
        dict: items (list of {page_id, caption, sim_score}), mattes (N, H, W) float32 in [0,1],
              references (N, H, W, 3) float32 RGB gray composites, noise (N, 32, 32, 32) float32
    """
    eval_dir = Path(eval_dir)
    items = json.loads((eval_dir / "prompts.json").read_text())
    mattes = np.stack([np.asarray(Image.open(eval_dir / "mattes" / f"{i['page_id']}.png"), np.float32) / 255 for i in items])
    refs = np.stack([np.asarray(Image.open(eval_dir / "references" / f"{i['page_id']}.png"), np.float32) / 255 for i in items])
    noise = load_file(eval_dir / "noise.safetensors")["noise"]
    assert len(noise) == len(items), f"noise has {len(noise)} rows but prompts.json has {len(items)} items"
    return dict(items=items, mattes=mattes, references=refs, noise=noise)


def build(eval_dir=str(EVAL_DIR), download_dir=str(DOWNLOAD_DIR)):
    """
    Command. Downloads magick_index.tsv and the eval PNGs, writes mattes/, references/, prompts.json, noise.safetensors.

    Args:
        eval_dir (str): output folder (resources/eval)
        download_dir (str): where the raw Hugging Face files go (scratch, untrusted)
    """
    eval_dir, download_dir = Path(eval_dir), Path(download_dir)
    (eval_dir / "mattes").mkdir(parents=True, exist_ok=True)
    (eval_dir / "references").mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download(MAGICK_REPO, "magick_index.tsv", repo_type="dataset", local_dir=download_dir)
    rows = select_eval_rows(pd.read_csv(tsv, sep="\t", quoting=3))
    print(f"eval rows: {len(rows)} (sha256(page_id) mod {EVAL_MODULUS} == 0, nsfw excluded)")
    for page_id in rows.page_id:
        png = hf_hub_download(MAGICK_REPO, f"images/{page_id[:2]}/{page_id}.png", repo_type="dataset", local_dir=download_dir)
        rgba = np.asarray(Image.open(png).convert("RGBA"))
        assert rgba.shape == (1024, 1024, 4), f"{page_id}: unexpected shape {rgba.shape}"
        matte, gray = split_rgba(rgba)
        Image.fromarray(np.round(matte * 255).astype(np.uint8), "L").save(eval_dir / "mattes" / f"{page_id}.png")
        Image.fromarray(np.round(gray * 255).astype(np.uint8), "RGB").save(eval_dir / "references" / f"{page_id}.png")
        print(f"  {page_id}  alpha mean {matte.mean():.3f}")
    (eval_dir / "prompts.json").write_text(json.dumps(rows.to_dict("records"), indent=1) + "\n")
    meta = dict(seed=str(NOISE_SEED), generator="torch.Generator('cpu').manual_seed(seed); torch.randn((N,32,32,32))",
                order="same as prompts.json")
    save_file(dict(noise=make_noise(len(rows))), eval_dir / "noise.safetensors", metadata=meta)
    print(f"wrote {eval_dir}")


if __name__ == "__main__":
    fire.Fire(dict(build=build))
