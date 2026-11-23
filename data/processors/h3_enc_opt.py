"""
Is H3's encoder the bottleneck? (manifest §25, m00992), through Bluejay so Sparrow shows every result (m00998).

    register                      (CPU, once) h3_enc_samples rows: the held-out images and clips, sources as lossless blobs
    images RUN [options]          long-lived GPU worker: latent optimization of single images through the FROZEN H3 decoder
    video RUN [options]           long-lived GPU worker: the same for whole 141-frame clips

For each sample: the encoder's posterior (untiled, as the project encodes), then Adam on the latent, starting from the
posterior mode, against the source pixels with one objective of h3_enc_lib.OBJECTIVES, optionally plus a prior that
keeps the latent inside the encoder's posterior: prior_weight · mean(((z − mode) / std)²).
h3_enc_runs gets one row per run, plus the three encoder baselines: encoder_sample (the posterior sample the pipeline
trains on), encoder_mode, encoder_tiled (the mode of a tiled encode). h3_enc_results gets one row per (run, sample): the
decode (PNG / MP4 + 3 lossless frames), the latent, every score vs the source, the optimization curve.

Video memory: the decoder runs with gradient checkpointing per transformer block; the loss is taken on a detached
copy of the decoded clip in frame batches, then one backward through the decoder. Time per Adam step on one A100:
~3.2 s per 512² image, ~26 s per 141-frame clip.

    h3_enc_opt.py register | images RUN --objective l1_vgg --where "kind = 'image' AND side = 512" [--steps 100] [--lr 0.01]
                  [--prior_weight 0] [--batch 4] | video RUN --where "kind = 'video'" [--steps 60] [--lr 0.01]
    (from data/; GPU = $CUDA_VISIBLE_DEVICES; started by run_proc_h3_encoder.fish)
"""
import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from h3_blobs import bytes_tensors, read_blob, read_png_rgb, tensors_bytes  # noqa: E402
from h3_generate import mp4_bytes, png_bytes  # noqa: E402
import h3_enc_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FASTDEC_DATA = ROOT / "agents/.scratchpad/fastdec_h3/data"  # val_*.npz: the decoder frenzy's 20 held-out clips (source of BASELINE.json)
LC_DATA = ROOT / "agents/.scratchpad/latentcomp/data"  # Latent Compositor test shard: MAGICK fg + alpha + Pexels bg at 1024²
LC_SHARD = "test_000"
LC_INDICES = (0, 1, 2, 3, 5, 8, 13, 21)  # MAGICK test samples (the shard is ordered hardest matte first)
SIDES = (512, 768)
IMAGE_FRAME = 70  # the middle frame of a clip is the pexels_val image
SHOW_FRAMES = (0, 70, 140)
BASELINES = ("encoder_sample", "encoder_mode", "encoder_tiled")
LOG_EVERY = 10
SCORE_COLUMNS = ["psnr", "ssim", "lpips_alex", "lpips_vgg", "dists", "de2000", "flux_mse", "sharp_grad", "sharp_fft", "flicker"]


# ---------------------------------------------------------------- register

def area_resize(image, side):
    """
    Pure function. (H, W, 3) uint8 -> (side, side, 3) uint8, cv2 INTER_AREA (the project's downsampling).

    >>> area_resize(np.full((4, 4, 3), 7, np.uint8), 2).tolist()[0][0]
    [7, 7, 7]
    """
    import cv2
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)


def lc_composite(fg, alpha, level):
    """
    Pure function. alpha·fg + (1 − alpha)·level·255 in 8 bit, rounded (the Latent Compositor's lc.composite on a flat level).

    >>> lc_composite(np.full((1, 1, 3), 200, np.uint8), np.full((1, 1), 128, np.uint8), 0.5).tolist()
    [[[164, 164, 164]]]
    """
    a = alpha[..., None].astype(np.float32) / 255
    return np.rint((fg.astype(np.float32) / 255 * a + np.float32(level) * (1 - a)) * 255).astype(np.uint8)


def register():
    """
    Command. Commits every h3_enc_samples row (idempotent: rows that exist are skipped): the 20 pexels_val frames
    (frame IMAGE_FRAME, 512²) and clips (141 frames), and the LC_INDICES MAGICK foregrounds over 50% gray and their
    Pexels backgrounds at each of SIDES.
    """
    from bluejay import open_database
    from safetensors.numpy import load_file
    db = open_database("project.dtab")
    have = {r["sample_id"] for r in db.query("SELECT sample_id FROM h3_enc_samples").to_pylist()}
    columns = ["kind", "origin", "pexels_id", "side", "frame", "source", "source_frames", "source_mp4"]
    empty = dict.fromkeys(columns)
    for path in sorted(FASTDEC_DATA.glob("val_*.npz")):
        pid = path.stem[4:]
        rows = []
        with np.load(path) as data:
            source = data["source"]
        if f"pexels_{pid}_f070" not in have:
            rows.append({**empty, "sample_id": f"pexels_{pid}_f070", "kind": "image", "origin": "pexels_val", "pexels_id": pid, "side": 512,
                         "frame": IMAGE_FRAME, "source": png_bytes(source[IMAGE_FRAME])})
        if f"pexels_{pid}_clip" not in have:
            rows.append({**empty, "sample_id": f"pexels_{pid}_clip", "kind": "video", "origin": "pexels_val", "pexels_id": pid, "side": 512,
                         "source": png_bytes(source[IMAGE_FRAME]), "source_frames": tensors_bytes({"frames": torch.from_numpy(source)}),
                         "source_mp4": mp4_bytes(source)})
        if rows:
            db.commit("h3_enc_samples", columns, rows, insert_missing=True)
        print(f"register: {path.stem} +{len(rows)}", flush=True)
    pixels = load_file(LC_DATA / f"{LC_SHARD}_pixels.safetensors")
    rows = []
    for i in LC_INDICES:
        for origin, image in (("magick", lc_composite(pixels["fg"][i], pixels["alpha"][i], 0.5)), ("pexbg", pixels["bg"][i])):
            for side in SIDES:
                sid = f"{origin}_{LC_SHARD}_{i}_{side}"
                if sid not in have:
                    rows.append({**empty, "sample_id": sid, "kind": "image", "origin": origin, "side": side, "source": png_bytes(area_resize(image, side))})
    if rows:
        db.commit("h3_enc_samples", columns, rows, insert_missing=True)
    print(f"register: {len(rows)} MAGICK / background images", flush=True)


# ---------------------------------------------------------------- shared worker parts

def run_row(run, objective, steps, lr, prior_weight):
    """
    Pure function. The h3_enc_runs row of an optimization run.

    >>> run_row("r", "l1", 10, 0.01, 0.0)["terms"]
    '{"l1": 1.0}'
    """
    return {"run_id": run, "method": "optimize", "objective": objective, "terms": json.dumps(L.OBJECTIVES[objective]), "steps": steps,
            "lr": lr, "prior_weight": prior_weight}


def commit_runs(db, run, objective, steps, lr, prior_weight):
    """Command. Commits the run's h3_enc_runs row and the three baseline rows."""
    rows = [run_row(run, objective, steps, lr, prior_weight)] + [
        {"run_id": b, "method": b, "objective": None, "terms": None, "steps": None, "lr": None, "prior_weight": None} for b in BASELINES]
    db.commit("h3_enc_runs", ["method", "objective", "terms", "steps", "lr", "prior_weight"], rows, insert_missing=True)


def todo_sql(run, where):
    """
    Pure function. The samples matching `where` that have no result of `run` yet.

    >>> "run_id = 'x'" in todo_sql("x", "kind = 'image'")
    True
    """
    return ("SELECT sample_id, side, blob_info(source) AS src, blob_info(source_frames) AS frames FROM h3_enc_samples "
            f"WHERE ({where}) AND sample_id NOT IN (SELECT sample_id FROM h3_enc_results WHERE run_id = '{run}') ORDER BY sample_id")


def baselines_done(db, sample_ids):
    """Query. The sample ids among `sample_ids` that already have all three baseline results."""
    ids = ", ".join(f"'{s}'" for s in sample_ids)
    rows = db.query(f"SELECT sample_id, count(*) AS n FROM h3_enc_results WHERE sample_id IN ({ids}) "
                    f"AND run_id IN ({', '.join(repr(b) for b in BASELINES)}) GROUP BY sample_id").to_pylist()
    return {r["sample_id"] for r in rows if r["n"] == len(BASELINES)}


def result_row(run, sample_id, scores, decoded_u8, latent, mode, std, gpu, seconds=None, curve=None):
    """
    Query (encodes PNG / MP4). One h3_enc_results row.

    Args:
        scores (dict): Judges.score output
        decoded_u8 (np.ndarray): (T, H, W, 3) uint8, T = 1 (image) or 141 (video)
        latent, mode, std (torch.Tensor): (LC, LT, LH, LW)
    """
    video = len(decoded_u8) > 1
    row = {"run_id": run, "sample_id": sample_id, **{k: scores.get(k) for k in SCORE_COLUMNS},
           "z_dev": float((latent - mode).abs().mean()), "z_dev_std": float(((latent - mode) / std).abs().mean()),
           "latent": tensors_bytes({"latent": latent.float().cpu()}), "seconds": seconds,
           "curve": None if curve is None else json.dumps(curve), "gpu": gpu,
           "image": None if video else png_bytes(decoded_u8[0]), "video": mp4_bytes(decoded_u8) if video else None}
    for f in SHOW_FRAMES:
        row[f"frame_{f:03d}"] = png_bytes(decoded_u8[f]) if video else None
    return row


RESULT_COLUMNS = SCORE_COLUMNS + ["z_dev", "z_dev_std", "latent", "seconds", "curve", "gpu", "image", "video", "frame_000", "frame_070", "frame_140"]


def gpu_index():
    """Query. The GPU index of this process (the first of CUDA_VISIBLE_DEVICES)."""
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


# ---------------------------------------------------------------- images

def quick_metrics(judges, pred, src):
    """Query. Per-image PSNR, LPIPS-alex, DISTS of clamped uint8 pred vs src, (B, 3, H, W) -> dict of lists."""
    with torch.no_grad():
        p = (pred.clamp(0, 1) * 255).round() / 255
        return {"psnr": [L.psnr(a[None], b[None]) for a, b in zip(p, src)],
                "lpips_alex": judges.alex(p * 2 - 1, src * 2 - 1).flatten().tolist(),
                "dists": judges.dists(p, src, batch_average=False).flatten().tolist()}


def optimize_images(vae, judges, src, mode, std, terms, steps, lr, prior_weight):
    """
    Command (runs Adam on the GPU). The optimized latents of one batch of images.

    Args:
        src (torch.Tensor): (B, 3, H, W) float [0, 1]
        mode, std (torch.Tensor): (B, LC, 1, LH, LW) the encoder's posterior
        terms (dict): loss terms (L.OBJECTIVES)

    Returns:
        tuple[torch.Tensor, list[list[dict]]]: z (B, LC, 1, LH, LW); per image, its curve rows {step, loss, psnr, lpips_alex, dists, z_dev}
    """
    z = mode.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 10)
    src_flux = L.flux2_latent(judges.flux2, src).detach() if "flux" in terms else None
    curves = [[] for _ in range(len(src))]
    for step in range(steps + 1):
        pred = L.decode_images(vae, z)
        loss = L.objective_loss(judges, terms, pred, src, src_flux)
        if prior_weight:
            loss = loss + prior_weight * ((z - mode) / std).pow(2).mean()
        if step % LOG_EVERY == 0 or step == steps:
            q = quick_metrics(judges, pred, src)
            for j in range(len(src)):
                curves[j].append({"step": step, "loss": float(loss.detach()), "psnr": q["psnr"][j], "lpips_alex": q["lpips_alex"][j],
                                  "dists": q["dists"][j], "z_dev": float((z[j] - mode[j]).detach().abs().mean())})
            print(f"  step {step:4d} loss {float(loss.detach()):.5f} psnr {np.mean(q['psnr']):.2f} lpips {np.mean(q['lpips_alex']):.4f} "
                  f"dists {np.mean(q['dists']):.4f} |dz| {float((z - mode).detach().abs().mean()):.3f}", flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return z.detach(), curves


def images(run, objective="l1_vgg", where="kind = 'image' AND side = 512", steps=100, lr=0.01, prior_weight=0.0, batch=4):
    """Command. Optimizes the latents of the matching image samples (batches of the same side) and commits each batch."""
    from bluejay import open_database
    db = open_database("project.dtab")
    commit_runs(db, run, objective, steps, lr, prior_weight)
    vae, judges, gpu = L.load_vae(), L.Judges(), gpu_index()
    terms = L.OBJECTIVES[objective]
    todo = db.query(todo_sql(run, where)).to_pylist()
    print(f"{run}: {len(todo)} images, objective {objective} {terms}, {steps} steps lr {lr} prior {prior_weight}", flush=True)
    for side in sorted({r["side"] for r in todo}):
        rows = [r for r in todo if r["side"] == side]
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            frames = np.stack([read_png_rgb(r["src"]) for r in chunk])
            src = L.u8_to_float(frames, "cuda")
            posts = [L.encode_posterior(vae, img[None]) for img in frames]
            mode, std = torch.stack([m for m, _ in posts]), torch.stack([s for _, s in posts])
            t0 = time.time()
            z, curves = optimize_images(vae, judges, src, mode, std, terms, steps, lr, prior_weight)
            seconds = (time.time() - t0) / len(chunk)
            kinds = {run: z}
            missing = {r["sample_id"] for r in chunk} - baselines_done(db, [r["sample_id"] for r in chunk])
            if missing:
                kinds.update({"encoder_mode": mode, "encoder_sample": torch.stack([L.sample_posterior(m, s) for m, s in zip(mode, std)]),
                              "encoder_tiled": torch.stack([L.encode_posterior(vae, img[None], tiled=True)[0] for img in frames])})
            out = []
            with torch.no_grad():
                for name, lat in kinds.items():
                    decoded = L.decode_images(vae, lat)
                    for j, r in enumerate(chunk):
                        if name != run and r["sample_id"] not in missing:
                            continue
                        scores = judges.score(decoded[j:j + 1], src[j:j + 1])
                        opt_run = name == run
                        out.append(result_row(name, r["sample_id"], scores, L.to_u8(decoded[j:j + 1]), lat[j], mode[j], std[j], gpu,
                                              seconds if opt_run else None, curves[j] if opt_run else None))
                        print(f"  {r['sample_id']} {name}: PSNR {scores['psnr']:.2f} SSIM {scores['ssim']:.4f} LPIPS {scores['lpips_alex']:.4f} "
                              f"DISTS {scores['dists']:.4f} FLUX {scores['flux_mse']:.4f} sharp {scores['sharp_grad']:.3f}", flush=True)
            db.commit("h3_enc_results", RESULT_COLUMNS, out, insert_missing=True)
            print(f"{time.strftime('%I:%M:%S %p')} {run}: committed {len(out)} results ({seconds:.0f} s per image)", flush=True)


# ---------------------------------------------------------------- video

def loss_and_grad(judges, terms, pred, src, src_flux, frame_batch):
    """
    Command (backward on frame batches). The objective over all frames, and its gradient w.r.t. pred.

    Args:
        pred, src (torch.Tensor): (T, 3, H, W); pred is the decoder output (its graph is not touched here)
        src_flux (torch.Tensor | None): (T, 32, H/8, W/8) FLUX.2 latents of src

    Returns:
        tuple[float, torch.Tensor]: mean loss over frames, d loss / d pred (T, 3, H, W)
    """
    leaf = pred.detach().requires_grad_(True)
    total = 0.0
    for i in range(0, len(src), frame_batch):
        sl = slice(i, i + frame_batch)
        part = L.objective_loss(judges, terms, leaf[sl], src[sl], None if src_flux is None else src_flux[sl])
        weight = (min(i + frame_batch, len(src)) - i) / len(src)
        (part * weight).backward()
        total += float(part.detach()) * weight
    return total, leaf.grad


def video(run, objective="l1_vgg", where="kind = 'video'", steps=60, lr=0.01, prior_weight=0.0, frame_batch=12):
    """Command. Optimizes the whole latent of each matching clip and commits each clip's results."""
    from bluejay import open_database
    db = open_database("project.dtab")
    commit_runs(db, run, objective, steps, lr, prior_weight)
    vae, judges, gpu = L.load_vae(), L.Judges(), gpu_index()
    terms = L.OBJECTIVES[objective]
    todo = db.query(todo_sql(run, where)).to_pylist()
    print(f"{run}: {len(todo)} clips, objective {objective} {terms}, {steps} steps lr {lr} prior {prior_weight}", flush=True)
    for r in todo:
        source = bytes_tensors(read_blob(r["frames"]))["frames"].numpy()
        src = L.u8_to_float(source, "cuda")
        mode, std = L.encode_posterior(vae, source)
        src_flux = None
        if "flux" in terms:
            with torch.no_grad():
                src_flux = torch.cat([L.flux2_latent(judges.flux2, src[i:i + frame_batch]) for i in range(0, len(src), frame_batch)])
        z = mode.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 10)
        curve, t0 = [], time.time()
        for step in range(steps + 1):
            pred = L.decode_float(vae, z)
            loss, grad = loss_and_grad(judges, terms, pred, src, src_flux, frame_batch)
            if step % LOG_EVERY == 0 or step == steps:
                with torch.no_grad():
                    p = (pred.clamp(0, 1) * 255).round() / 255
                    row = {"step": step, "loss": loss, "psnr": L.psnr(p, src), "flicker": abs(L.temporal_diff(p) - L.temporal_diff(src)),
                           "lpips_alex": float(torch.cat([judges.alex(p[i:i + 16] * 2 - 1, src[i:i + 16] * 2 - 1).flatten() for i in range(0, len(p), 16)]).mean()),
                           "z_dev": float((z - mode).detach().abs().mean()), "seconds": time.time() - t0}
                curve.append(row)
                print(f"  {r['sample_id']} step {step:3d} loss {loss:.5f} psnr {row['psnr']:.2f} lpips {row['lpips_alex']:.4f} "
                      f"flicker {row['flicker']:.3f} |dz| {row['z_dev']:.3f} {row['seconds']:.0f} s", flush=True)
            if step == steps:
                break
            if prior_weight:
                prior = prior_weight * ((z - mode) / std).pow(2).mean()
                torch.autograd.backward([pred, prior], [grad, None])
            else:
                pred.backward(grad)
            del pred
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        del pred
        seconds = time.time() - t0
        z = z.detach()
        kinds = {run: z}
        if r["sample_id"] not in baselines_done(db, [r["sample_id"]]):
            kinds.update({"encoder_mode": mode, "encoder_sample": L.sample_posterior(mode, std),
                          "encoder_tiled": L.encode_posterior(vae, source, tiled=True)[0]})
        out = []
        with torch.no_grad():
            for name, lat in kinds.items():
                x = L.decode_float(vae, lat)
                scores = judges.score(x, src)
                out.append(result_row(name, r["sample_id"], scores, L.to_u8(x), lat, mode, std, gpu,
                                      seconds if name == run else None, curve if name == run else None))
                print(f"  {r['sample_id']} {name}: " + " ".join(f"{k} {v:.4f}" for k, v in scores.items()), flush=True)
                del x
        db.commit("h3_enc_results", RESULT_COLUMNS, out, insert_missing=True)
        print(f"{time.strftime('%I:%M:%S %p')} {run}: committed {r['sample_id']} ({seconds / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    fire.Fire({"register": register, "images": images, "video": video})
