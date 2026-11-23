"""
A better H3 encoder with the decoder FROZEN (manifest §25 step 4, m00992), through Bluejay (m00998).

    train NAME [options]     fine-tunes a copy of H3's encoder (encoder + quant_conv) end to end through the frozen decoder
    evaluate NAME [--where]  encodes the held-out h3_enc_samples with the fine-tuned encoder (untiled, posterior mode, as
                             the project encodes) and commits run "encft_<NAME>" to h3_enc_results (same scores as the
                             latent-optimization runs)

Training data (Bluejay, never the held-out samples): h3_pexels_clips split 'train' clip_512 (H.264 CRF 12, 141 frames) and
h3_magick_images split 'train' image_512. A step takes VIDEO_WINDOWS windows of 34 frames (two 17-frame encoder chunks
starting at a multiple of 17, as inside a 141-frame clip) and IMAGE_CROPS single frames, each a random 256² crop (one
decoder tile, so a crop decodes exactly as a tile of the full tiled decode). The 7 latent frames of a window decode to
its first 22 frames (one decoder chunk); an image latent is repeated 7× (h3.allow_single_frame).
Loss: L1 + 0.1·LPIPS-vgg of the decode vs the source, plus reg_weight · mean((z − z_H3)²): z_H3 is the original
encoder's latent of the same pixels (normalized units), which keeps the new latents in the distribution H3's diffusion
model was trained on.

Saves models/h3magic/encoder_ft/<NAME>/encoder.safetensors (encoder.* and quant_conv.* weights, loadable into the VAE)
and commits the run row (method encoder_finetuned, its training log).

    h3_enc_train.py train NAME [--steps 600] [--lr 1e-5] [--reg_weight 1.0] [--clips 160] [--images 2000]
    h3_enc_train.py evaluate NAME [--where "kind = 'image'"] [--tiled]
    (from data/; GPU = $CUDA_VISIBLE_DEVICES; started by run_proc_h3_encoder.fish)
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from h3_blobs import bytes_tensors, read_blob, read_mp4_rgb, read_png_rgb  # noqa: E402
import h3_enc_lib as L  # noqa: E402
import h3_enc_opt as O  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models/h3magic/encoder_ft"
CROP = 256  # = the decoder's tile: a crop decodes like one tile of the tiled decode
CLIP_LENGTH = 17  # encoder chunk (vae.config.clip_length)
WINDOW = 2 * CLIP_LENGTH  # two encoder chunks -> 10 latent frames, of which the decoder's first chunk reads 7
DECODE_LATENTS = 7  # latent frames of one decoder chunk
DECODE_FRAMES = 22  # pixel frames of one decoder chunk (7 latents)
VIDEO_WINDOWS = 2
IMAGE_CROPS = 4
TERMS = {"l1": 1.0, "lpips_vgg": 0.1}  # = h3_enc_lib.OBJECTIVES["l1_vgg"]
LOG_EVERY = 10
SAVE_EVERY = 100


def moments(encoder, quant_conv, x):
    """
    Query (runs the encoder; differentiable under grad). The encoder's raw moments exactly as the VAE's untiled _encode:
    one frame through the spatial path alone, else 17-frame chunks (the last padded by repeating its last frame), and the
    token_drop = 3 trailing latent frames dropped.

    Args:
        x (torch.Tensor): (B, 3, T, H, W) ImageNet-normalized, T = 1 or any length

    Returns:
        torch.Tensor: (B, 48, LT, H/16, W/16): mean (24) and logvar (24)
    """
    if x.shape[2] == 1:
        return quant_conv(encoder(x))
    if x.shape[2] % CLIP_LENGTH:
        x = torch.cat([x, x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % CLIP_LENGTH, 1, 1)], dim=2)
    out = torch.cat([quant_conv(encoder(x[:, :, i:i + CLIP_LENGTH])) for i in range(0, x.shape[2], CLIP_LENGTH)], dim=2)
    return out[:, :, :-3]


def normalized_mode(vae, m):
    """Pure function. Normalized posterior mode (B, 24, LT, LH, LW) of raw moments (B, 48, ...) (h3.encode's normalization)."""
    mean, std = L.stats(vae)
    return (m[:, :24].float() - mean) / std


def to_input(frames, device):
    """
    Pure function. uint8 (B, T, H, W, 3) -> ImageNet-normalized float (B, 3, T, H, W) on device.

    >>> to_input(np.zeros((1, 1, 2, 2, 3), np.uint8), "cpu").shape
    torch.Size([1, 3, 1, 2, 2])
    """
    import einx
    x = einx.rearrange("B T H W C -> B C T H W", torch.from_numpy(np.ascontiguousarray(frames)).to(device)).float() / 255
    return L.to_imagenet(x)


def random_crop(frames, rng, side=CROP, align=16):
    """
    Near-pure function (advances rng). A random side × side crop of (..., H, W, 3) frames, its corner on a multiple of align.

    >>> random_crop(np.zeros((2, 512, 512, 3), np.uint8), np.random.default_rng(0)).shape
    (2, 256, 256, 3)
    """
    h, w = frames.shape[-3:-1]
    y = int(rng.integers(0, (h - side) // align + 1)) * align
    x = int(rng.integers(0, (w - side) // align + 1)) * align
    return frames[..., y:y + side, x:x + side, :]


def load_training_data(db, clips, images, exclude):
    """
    Query (reads Bluejay blobs). Train clips (decoded MP4s) and train MAGICK images, never a held-out pexels_id.

    Returns:
        tuple[list[np.ndarray], np.ndarray]: clips (each (141, 512, 512, 3) uint8), images (N, 512, 512, 3) uint8
    """
    ex = ", ".join(f"'{e}'" for e in exclude) or "''"
    rows = db.query(f"SELECT pexels_id, blob_info(clip_512) AS clip FROM h3_pexels_clips WHERE split = 'train' AND clip_512 IS NOT NULL "
                    f"AND pexels_id NOT IN ({ex}) ORDER BY encode(sha256(pexels_id), 'hex') LIMIT {clips}").to_pylist()
    t = time.time()
    videos = [read_mp4_rgb(r["clip"]) for r in rows]
    print(f"train data: {len(videos)} clips in {time.time() - t:.0f} s", flush=True)
    rows = db.query(f"SELECT page_id, blob_info(image_512) AS image FROM h3_magick_images WHERE split = 'train' AND image_512 IS NOT NULL "
                    f"ORDER BY encode(sha256(page_id), 'hex') LIMIT {images}").to_pylist()
    pics = np.stack([read_png_rgb(r["image"]) for r in rows])
    print(f"train data: {len(pics)} MAGICK images", flush=True)
    return videos, pics


def training_batch(videos, pics, rng, video_windows=VIDEO_WINDOWS, image_crops=IMAGE_CROPS):
    """
    Near-pure function (advances rng). One step's crops.

    Returns:
        list[np.ndarray]: the non-empty of windows (video_windows, 34, 256, 256, 3) and images (image_crops, 1, 256, 256, 3) uint8;
        half of the image crops are MAGICK images, half single clip frames
    """
    windows = []
    for _ in range(video_windows):
        v = videos[int(rng.integers(len(videos)))]
        start = CLIP_LENGTH * int(rng.integers(0, (len(v) - WINDOW) // CLIP_LENGTH + 1))
        windows.append(random_crop(v[start:start + WINDOW], rng))
    singles = []
    for k in range(image_crops):
        if k % 2 == 0:
            singles.append(random_crop(pics[int(rng.integers(len(pics)))][None], rng))
        else:
            v = videos[int(rng.integers(len(videos)))]
            singles.append(random_crop(v[int(rng.integers(len(v)))][None], rng))
    return [np.stack(group) for group in (windows, singles) if group]


def decode_tile(vae, z):
    """
    Query (differentiable under grad). One decoder tile: normalized (B, 24, LT, 16, 16) -> (B, 3, T, 256, 256) RGB ~[0, 1].
    LT = 1 is decoded with the 7-repeat (frame 0 kept), LT = 7 gives the chunk's 22 frames.
    """
    mean, std = L.stats(vae)
    vae.use_tiling = True
    return L.from_imagenet(vae.decode(z * std[None] + mean[None], return_dict=False)[0].float())


def train(name, steps=600, lr=1e-5, reg_weight=1.0, clips=160, images=2000, seed=0, video_windows=VIDEO_WINDOWS, image_crops=IMAGE_CROPS):
    """Command. Fine-tunes the encoder (module docstring), saves it, commits its run row with the training log. image_crops 0 = video windows only."""
    from bluejay import open_database
    from safetensors.torch import save_file
    db = open_database("project.dtab")
    exclude = [r["pexels_id"] for r in db.query("SELECT pexels_id FROM h3_enc_samples WHERE pexels_id IS NOT NULL").to_pylist()]
    videos, pics = load_training_data(db, clips, images, exclude)
    vae, judges = L.load_vae(), L.Judges()
    teacher_enc, teacher_qc = vae.encoder, vae.quant_conv  # frozen originals (z_H3)
    student_enc, student_qc = copy.deepcopy(teacher_enc).requires_grad_(True), copy.deepcopy(teacher_qc).requires_grad_(True)
    params = list(student_enc.parameters()) + list(student_qc.parameters())
    print(f"{name}: student encoder {sum(p.numel() for p in params) / 1e6:.1f} M params, {steps} steps lr {lr} reg {reg_weight}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    rng = np.random.default_rng(seed)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        groups = training_batch(videos, pics, rng, video_windows, image_crops)
        total = {"loss": 0.0, "rec": 0.0, "reg": 0.0, "rec_h3": 0.0}
        opt.zero_grad(set_to_none=True)
        for frames in groups:
            keep = DECODE_LATENTS if frames.shape[1] > 1 else 1
            x = to_input(frames, "cuda")
            src = x.new_tensor(frames[:, :DECODE_FRAMES if keep > 1 else 1]).float().div(255)  # (B, T, H, W, 3)
            src = src.permute(0, 1, 4, 2, 3).flatten(0, 1)  # (B·T, 3, H, W)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z_h3 = normalized_mode(vae, moments(teacher_enc, teacher_qc, x))[:, :, :keep]
            with torch.autocast("cuda", torch.bfloat16):
                z = normalized_mode(vae, moments(student_enc, student_qc, x))[:, :, :keep]
            pred = decode_tile(vae, z)[:, :, :src.shape[0] // len(frames)].permute(0, 2, 1, 3, 4).flatten(0, 1)
            rec = L.objective_loss(judges, TERMS, pred, src)
            reg = (z - z_h3).pow(2).mean()
            loss = (rec + reg_weight * reg) / len(groups)  # the groups (video windows, single frames) weigh the same
            loss.backward()
            if step % LOG_EVERY == 0:
                with torch.no_grad():
                    rec_h3 = L.objective_loss(judges, TERMS, decode_tile(vae, z_h3)[:, :, :src.shape[0] // len(frames)].permute(0, 2, 1, 3, 4).flatten(0, 1), src)
                total["rec_h3"] += float(rec_h3) / len(groups)
            total["loss"] += float(loss.detach())
            total["rec"] += float(rec.detach()) / len(groups)
            total["reg"] += float(reg.detach()) / len(groups)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % LOG_EVERY == 0:
            row = {"step": step, **total, "seconds": time.time() - t0}
            log.append(row)
            print(f"  step {step:4d} loss {row['loss']:.5f} rec {row['rec']:.5f} (H3 encoder {row['rec_h3']:.5f}) reg {row['reg']:.5f} "
                  f"{row['seconds'] / step:.1f} s/step", flush=True)
        if step % SAVE_EVERY == 0 or step == steps:
            weights = {f"encoder.{k}": v.detach().float().cpu().contiguous() for k, v in student_enc.state_dict().items()}
            weights.update({f"quant_conv.{k}": v.detach().float().cpu().contiguous() for k, v in student_qc.state_dict().items()})
            save_file(weights, out / "encoder.safetensors")
            (out / "train_log.json").write_text(json.dumps(log))
    db.commit("h3_enc_runs", ["method", "objective", "terms", "steps", "lr", "prior_weight", "checkpoint", "train_log"],
              [{"run_id": f"encft_{name}", "method": "encoder_finetuned", "objective": "l1_vgg", "terms": json.dumps(TERMS), "steps": steps,
                "lr": lr, "prior_weight": reg_weight, "checkpoint": str((out / "encoder.safetensors").relative_to(ROOT)), "train_log": json.dumps(log)}],
              insert_missing=True)
    print(f"{name}: saved {out / 'encoder.safetensors'}", flush=True)


def evaluate(name, where="kind IN ('image', 'video') AND origin != 'pexels_train'", tiled=False):
    """
    Command. Scores the fine-tuned encoder's latents on held-out samples; commits run encft_<name> rows to h3_enc_results.
    tiled: encode in the VAE's 256-px spatial tiles (the crop size the encoder was fine-tuned on), run encft_<name>_tiled.
    """
    from bluejay import open_database
    from safetensors.torch import load_file
    db = open_database("project.dtab")
    run = f"encft_{name}" + ("_tiled" if tiled else "")
    if tiled:
        base = db.query(f"SELECT * FROM h3_enc_runs WHERE run_id = 'encft_{name}'").to_pylist()[0]
        db.commit("h3_enc_runs", ["method", "objective", "terms", "steps", "lr", "prior_weight", "checkpoint", "train_log"],
                  [{**{k: base[k] for k in ("method", "objective", "terms", "steps", "lr", "prior_weight", "checkpoint", "train_log")}, "run_id": run}],
                  insert_missing=True)
    vae, judges, gpu = L.load_vae(), L.Judges(), O.gpu_index()
    teacher_enc, teacher_qc = copy.deepcopy(vae.encoder), copy.deepcopy(vae.quant_conv)
    state = load_file(OUT / name / "encoder.safetensors")
    vae.encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    vae.quant_conv.load_state_dict({k[len("quant_conv."):]: v for k, v in state.items() if k.startswith("quant_conv.")})
    todo = db.query(O.todo_sql(run, where)).to_pylist()
    print(f"{run}: {len(todo)} samples", flush=True)
    for r in todo:
        if r["kind"] == "video":
            source = bytes_tensors(read_blob(r["frames"]))["frames"].numpy()
        else:
            source = read_png_rgb(r["src"])[None]
        src = L.u8_to_float(source, "cuda")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            x = to_input(source[None], "cuda")
            m_h3 = moments(teacher_enc, teacher_qc, x)
            vae.use_tiling = tiled
            m_new = vae._encode(x) if tiled else moments(vae.encoder, vae.quant_conv, x)  # _encode = moments, tiled
        z, z_h3 = normalized_mode(vae, m_new)[0], normalized_mode(vae, m_h3)[0]
        std = (m_h3[0, 24:].float().clamp(-30, 20) * 0.5).exp() / L.stats(vae)[1]  # the original posterior's std, normalized
        with torch.no_grad():
            decoded = L.decode_float(vae, z)
            scores = judges.score(decoded, src)
            row = O.result_row(run, r["sample_id"], scores, L.to_u8(decoded), z, z_h3, std, gpu)
        db.commit("h3_enc_results", O.RESULT_COLUMNS, [row], insert_missing=True)
        print(f"  {r['sample_id']} {run}: " + " ".join(f"{k} {v:.4f}" for k, v in scores.items()) +
              f" |z - z_H3| {float((z - z_h3).abs().mean()):.4f}", flush=True)
        del decoded


if __name__ == "__main__":
    fire.Fire({"train": train, "evaluate": evaluate})
