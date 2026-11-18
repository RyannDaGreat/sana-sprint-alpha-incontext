"""
The fixed eval set: MAGICK images where sha256(page_id) mod 4000 is 0, with fixed noise.

Build it once (writes resources/eval/):
    PYTHONPATH=src uv run python -m sana_alpha.eval_set build
The showcase set adds training images (residue 1, split "train") after the held-out ones, so its first noise rows
equal the eval set's (writes resources/showcase/):
    PYTHONPATH=src uv run python -m sana_alpha.eval_set build resources/showcase --residues 0,1
The showcase trimask sets (one eval set per method, conditions = fixed trimasks; writes resources/showcase_trimask/):
    PYTHONPATH=src uv run python -m sana_alpha.eval_set build_trimask

The rule must match data/project.dtab (EVAL_MODULUS, the `split` column of import_index).
"""

import hashlib
import json
import os
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
TRIMASK_SHARED = ("prompts.json", "prompts_detailed.json", "noise.safetensors")  # files a trimask set links from its source
PROMPT_SUFFIXES = dict(plain="", gray=", on a plain 50% gray background")  # prompt variant -> text added to the caption


def page_residue(page_id):
    """
    Pure function. sha256 of the UTF-8 page_id, read as a big integer, mod EVAL_MODULUS. 0 = eval, never trained on.

    Args:
        page_id (str): MAGICK image ID

    Returns:
        int

    Examples:
        >>> page_residue("6NqAw3m1ss"), page_residue("Hcx82jcUpM")
        (0, 3229)
    """
    return int(hashlib.sha256(page_id.encode("utf-8")).hexdigest(), 16) % EVAL_MODULUS


def select_eval_rows(index, residues=(0,)):
    """
    Pure function. Dedupes the MAGICK index by page_id, keeps the non-nsfw rows whose page_residue is in residues,
    labels them split "held-out" (residue 0) or "train", and sorts them held-out first, then by page_id.

    Args:
        index (pd.DataFrame): magick_index.tsv as read with quoting=3 (columns page_id, subject, nsfw, ...)
        residues (tuple[int]): page_residue values to keep

    Returns:
        pd.DataFrame: columns page_id, caption, sim_score, split

    Examples:
        >>> df = pd.DataFrame(dict(page_id=["6NqAw3m1ss", "6NqAw3m1ss", "Hcx82jcUpM"],
        ...                        subject=["two dice", "two dice", "crayfish"], nsfw=[0, 0, 0], sim_score=[0.99, 0.99, 0.98]))
        >>> select_eval_rows(df).to_dict("records")
        [{'page_id': '6NqAw3m1ss', 'caption': 'two dice', 'sim_score': 0.99, 'split': 'held-out'}]
        >>> select_eval_rows(df, residues=(0, 3229)).split.tolist()
        ['held-out', 'train']
    """
    unique = index.drop_duplicates("page_id")
    residue = unique.page_id.map(page_residue)
    rows = unique.assign(split=np.where(residue == 0, "held-out", "train"))[residue.isin(residues) & (unique.nsfw == 0)]
    rows = rows.sort_values(["split", "page_id"]).rename(columns=dict(subject="caption"))
    return rows[["page_id", "caption", "sim_score", "split"]].reset_index(drop=True)


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


def make_noise(count, seed=NOISE_SEED, shape=LATENT_SHAPE):
    """
    Pure function. Fixed standard-normal initial latents, one per eval item, made on CPU so they are device independent.

    Args:
        count (int): N, number of eval items
        seed (int): torch CPU generator seed
        shape (tuple[int]): (LC, LH, LW) of one latent; (32, r/32, r/32) at resolution r

    Returns:
        torch.Tensor: (N, LC, LH, LW) float32, e.g. (29, 32, 32, 32)

    Examples:
        >>> make_noise(2).shape, bool((make_noise(2) == make_noise(2)).all())
        (torch.Size([2, 32, 32, 32]), True)
    """
    return torch.randn((count, *shape), generator=torch.Generator("cpu").manual_seed(seed))


def load_eval_set(eval_dir=EVAL_DIR):
    """
    Query. Reads the eval set from disk.

    Args:
        eval_dir (Path): folder written by `build`

    Returns:
        dict: items (list of {page_id, caption, sim_score}), mattes (N, H, W) float32 in [0,1] (the condition: 8-bit
              PNG / 255, so a trimask's unknown 128 stays 0.502), originals (N, H, W) the true mattes (originals/ of a
              trimask set, else = mattes), references (N, H, W, 3) float32 RGB gray composites, noise (N, 32, 32, 32) float32
    """
    eval_dir = Path(eval_dir)
    items = json.loads((eval_dir / "prompts.json").read_text())
    read = lambda kind: np.stack([np.asarray(Image.open(eval_dir / kind / f"{i['page_id']}.png"), np.float32) / 255 for i in items])
    noise = load_file(eval_dir / "noise.safetensors")["noise"]
    assert len(noise) == len(items), f"noise has {len(noise)} rows but prompts.json has {len(items)} items"
    mattes = read("mattes")
    originals = read("originals") if (eval_dir / "originals").is_dir() else mattes
    return dict(items=items, mattes=mattes, originals=originals, references=read("references"), noise=noise)


def build(eval_dir=str(EVAL_DIR), download_dir=str(DOWNLOAD_DIR), residues=(0,)):
    """
    Command. Downloads magick_index.tsv and the eval PNGs, writes mattes/, references/, prompts.json, noise.safetensors.

    Args:
        eval_dir (str): output folder (resources/eval)
        download_dir (str): where the raw Hugging Face files go (scratch, untrusted)
        residues (tuple[int]): page_residue values to include: 0 = held-out eval images, others are training images
    """
    eval_dir, download_dir = Path(eval_dir), Path(download_dir)
    (eval_dir / "mattes").mkdir(parents=True, exist_ok=True)
    (eval_dir / "references").mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download(MAGICK_REPO, "magick_index.tsv", repo_type="dataset", local_dir=download_dir)
    rows = select_eval_rows(pd.read_csv(tsv, sep="\t", quoting=3), tuple(residues))
    print(f"rows: {rows.split.value_counts().to_dict()} (sha256(page_id) mod {EVAL_MODULUS} in {residues}, nsfw excluded)")
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


def build_trimask(source="resources/showcase", out="resources/showcase_trimask", methods=None, variant=0, resolution=1024):
    """
    Command. Writes one eval set per trimask method: OUT/<method>/mattes/<page_id>.png = the 8-bit trimask (0.5 = 128)
    of SOURCE's matte, made by trimask.generate (seeded by page_id, method, variant, so it is fixed). Everything else
    (originals/ = SOURCE's mattes, references/, prompts, noise) is a relative symlink into SOURCE. Raises unless every
    stored 1024 trimask passes trimask.check_consistent (m00647: no incorrect trimasks).
    Below 1024 (m00656) the folder is OUT_<resolution>: trimasks, originals/ and references/ are area-downsampled from
    1024 (uint8 cv2.INTER_AREA, as data/processors/encode_multires.py makes the training latents; resampling can move
    single edge pixels off the rule, so the rule is checked at 1024), and the noise is (32, r/32, r/32) seeded
    NOISE_SEED + resolution. originals/, references/ and noise are written once in OUT_<resolution>/ and linked.

    Args:
        source (str): eval set folder relative to the repo root (resources/showcase)
        out (str): output folder relative to the repo root
        methods (tuple[str] | None): trimask.METHODS tags; None = all of trimask.METHODS (m00646, m00648: training uses all)
        variant (int): trimask.generate variant
        resolution (int): 1024, 768, 512 or 256
    """
    import cv2

    from sana_alpha import trimask

    source, low = ROOT / source, resolution != 1024
    base = ROOT / (f"{out}_{resolution}" if low else out)
    area = lambda image: cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA) if low else image
    items = json.loads((source / "prompts.json").read_text())
    read = lambda kind, item, mode: np.asarray(Image.open(source / kind / f"{item['page_id']}.png").convert(mode))
    shared = {f: source / f for f in TRIMASK_SHARED}
    shared.update(originals=source / "mattes", references=source / "references")
    if low:  # the per-resolution originals, references and noise, once
        for kind, folder, mode in (("mattes", "originals", "L"), ("references", "references", "RGB")):
            (base / folder).mkdir(parents=True, exist_ok=True)
            for item in items:
                Image.fromarray(area(read(kind, item, mode)), mode).save(base / folder / f"{item['page_id']}.png")
        noise = make_noise(len(items), NOISE_SEED + resolution, (LATENT_SHAPE[0], resolution // 32, resolution // 32))
        save_file(dict(noise=noise), base / "noise.safetensors", metadata=dict(seed=str(NOISE_SEED + resolution), order="same as prompts.json"))
        shared.update({k: base / k for k in ("originals", "references", "noise.safetensors")})
    for method in methods or trimask.METHODS:
        folder = base / method
        (folder / "mattes").mkdir(parents=True, exist_ok=True)
        for link, target in shared.items():
            if not (folder / link).is_symlink():
                (folder / link).symlink_to(os.path.relpath(target, folder))  # relative: the dump stays portable
        for item in items:
            alpha = read("mattes", item, "L").astype(np.float32) / 255
            rgb = read("references", item, "RGB").astype(np.float32) / 255 if method in trimask.RGB_METHODS else None  # bilateral_blur follows the RGB on gray
            stored = trimask.to_uint8(trimask.generate(alpha, item["page_id"], method, variant, rgb)[0])
            trimask.check_consistent(alpha, stored.astype(np.float32) / 255)  # m00647: the stored 8-bit values, as read
            Image.fromarray(area(stored), "L").save(folder / "mattes" / f"{item['page_id']}.png")
        print(f"wrote {folder}: {len(items)} trimasks")


if __name__ == "__main__":
    fire.Fire(dict(build=build, build_trimask=build_trimask))
