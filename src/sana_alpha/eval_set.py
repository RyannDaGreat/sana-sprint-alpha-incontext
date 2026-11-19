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


def held_out(source="resources/showcase_trimask", out="resources/showcase_trimask_held"):
    """
    Command. Writes the held-out-only copy of every method's eval set in SOURCE: OUT/prompts.json (SOURCE's items whose
    split is "held-out", same order) and OUT/noise.safetensors (their noise rows), linked into OUT/<method>/ next to
    relative symlinks to SOURCE/<method>'s mattes/, originals/, references/ and prompts_detailed.json. For models too
    slow for the whole set (FIBO2: about a minute per image, manifest §15), so plan_eval makes only these rows.

    Args:
        source (str): trimask eval sets folder relative to the repo root (build_trimask's OUT)
        out (str): output folder relative to the repo root
    """
    from safetensors import safe_open

    source, base = ROOT / source, ROOT / out
    methods = sorted(p.name for p in source.iterdir() if (p / "mattes").is_dir())
    items = json.loads((source / methods[0] / "prompts.json").read_text())
    keep = [k for k, item in enumerate(items) if item["split"] == "held-out"]
    with safe_open(source / methods[0] / "noise.safetensors", "pt") as f:
        noise, seed = f.get_tensor("noise")[keep], f.metadata()["seed"]
    base.mkdir(parents=True, exist_ok=True)
    (base / "prompts.json").write_text(json.dumps([items[k] for k in keep], indent=1) + "\n")
    save_file(dict(noise=noise.contiguous()), base / "noise.safetensors", metadata=dict(seed=seed, order="same as prompts.json (held-out rows of the source)"))
    for method in methods:
        folder = base / method
        folder.mkdir(exist_ok=True)
        links = {k: source / method / k for k in ("mattes", "originals", "references", "prompts_detailed.json")}
        links.update({k: base / k for k in ("prompts.json", "noise.safetensors")})
        for link, target in links.items():
            if not (folder / link).is_symlink():
                (folder / link).symlink_to(os.path.relpath(target, folder))  # relative: the dump stays portable
    print(f"wrote {base}: {len(keep)} held-out items x {len(methods)} methods")


def fibo2_preview(checkpoint="models/fibo2/fibo2_tri/step_002000", out="resources/fibo2_preview"):
    """
    Command. Writes the eval set of a FIBO2 control run's training previews (train_fibo2.control_preview_rows on rank 0
    of the checkpoint's world size, seed and lance versions): OUT/prompts.json (page_id, caption, method, split
    "train"), mattes/ (the trimask, 1024 8-bit), originals/ (the true alpha), references/ (MAGICK on gray), and
    noise.safetensors (make_noise rows; FIBO2 draws its own noise from the item's seed), so BlueJay renders the same
    4 items the trainer previews, at full resolution.

    Args:
        checkpoint (str): a control step folder; its meta.json gives world, lance_versions
        out (str): output folder relative to the repo root
    """
    import lance

    from sana_alpha.train_control import LANCE, TRIMASK_LANCE
    from sana_alpha.train_fibo2 import control_preview_rows, load_index
    from sana_alpha.trimask import METHODS

    meta = json.loads((ROOT / checkpoint / "meta.json").read_text())
    versions = tuple(meta["lance_versions"])
    index = load_index("control", versions, 0, meta["world"], 0)  # rank 0 writes the previews; train's seed default 0
    idx, rows = control_preview_rows(index)
    magick, tri = lance.dataset(str(LANCE), version=versions[0]), lance.dataset(str(TRIMASK_LANCE), version=versions[1])
    record_ids, cond_ids = index["rowids"][idx.numpy()].tolist(), index["cond_rowids"][rows.numpy()].tolist()
    pages = magick._take_rows(record_ids, columns=["page_id"]).column("page_id").to_pylist()  # by _rowid (take() is by position)
    base = ROOT / out
    for kind, dataset, column, ids in (("mattes", tri, "trimask", cond_ids), ("originals", magick, "alpha", record_ids),
                                       ("references", magick, "rgb_on_gray", record_ids)):
        (base / kind).mkdir(parents=True, exist_ok=True)
        for page_id, blob in zip(pages, dataset.take_blobs(column, ids=ids)):
            (base / kind / f"{page_id}.png").write_bytes(blob.read())
    items = [dict(page_id=p, caption=index["captions"][i], method=METHODS[int(index["cond_method"][r])], split="train")
             for p, i, r in zip(pages, idx.tolist(), rows.tolist())]
    (base / "prompts.json").write_text(json.dumps(items, indent=1) + "\n")
    save_file(dict(noise=make_noise(len(items))), base / "noise.safetensors", metadata=dict(seed=str(NOISE_SEED)))
    print(f"wrote {base}: {len(items)} preview items {[(i['page_id'], i['method']) for i in items]}")


if __name__ == "__main__":
    fire.Fire(dict(build=build, build_trimask=build_trimask, held_out=held_out, fibo2_preview=fibo2_preview))
