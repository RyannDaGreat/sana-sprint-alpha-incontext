"""
Does stock H3 expect TILED-encoded latents? (manifest §25 "latent distribution" finding), through Bluejay (m00998).

The project encodes untiled (h3magic/h3.py encode, 3x faster). On 19 held-out clips the untiled latents had per-channel
std ~0.69 in channels 16-23 after H3's own normalization, a tiled encode ~0.98. Two checks:

    stats [--count 300]   (CPU) per-channel mean / std of the project's STORED training latents (h3_pexels_clips.h3_latent_video
                          and h3_magick_images.h3_latent_512, train rows) -> h3_enc_stats
    fmloss [--clips 30]   (GPU) re-encodes CLIPS train clips untiled and tiled (the project's recipe otherwise: posterior sample,
                          seed h3.ENCODE_SEED) and TILE_VARIANTS on the first VARIANT_CLIPS (stats only) -> h3_enc_stats;
                          then stock H3's flow-matching loss (the clip's caption embedding, CFG off) at FM_SIGMAS with noise
                          shared by both latents -> h3_enc_results runs fm_untiled, fm_tiled (fm_loss, fm_loss_by_sigma)

Flow matching as h3magic/train.py (MinimaxH3's verified recipe): t = 1 − σ is CLEAN, x_t = t·x_0 + (1 − t)·noise,
target x_0 − noise; video σ and its paired audio σ come from one base grid point (VIDEO_SHIFT 12, AUDIO_SHIFT 3);
audio rows hold the latent of silence and are not scored.

    h3_enc_fmloss.py stats | fmloss   (from data/; GPU = $CUDA_VISIBLE_DEVICES; run_proc_h3_encoder.fish routes them)
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import bytes_tensors, read_blob, read_mp4_rgb, tensors_bytes  # noqa: E402
from h3_generate import png_bytes  # noqa: E402
import h3_enc_lib as L  # noqa: E402
import h3_enc_opt as O  # noqa: E402

FM_SIGMAS = (0.1, 0.3, 0.5, 0.6, 0.75, 0.9)  # video σ (noise fraction) of the loss probe
TILE_VARIANTS = ((128, 32), (384, 96))  # (tile, min overlap) besides the default 256 / 64 and untiled
VARIANT_CLIPS = 10
NOISE_SEED = 20261123
VIDEO_AUDIO_LATENTS = 235  # round(141 / 24 · 40): the soundtrack of a 141-frame clip (h3magic/train.py)


def channel_stats(latents):
    """
    Pure function. Per-channel mean and std over every position of a list of (LC, ...) latents.

    >>> m, s = channel_stats([torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)]); m.tolist(), s.tolist()
    ([0.5, 0.5], [0.5, 0.5])
    """
    flat = torch.cat([z.float().flatten(1) for z in latents], dim=1)
    return flat.mean(1), flat.std(1, unbiased=False)


def stat_row(stat_id, source, recipe, latents):
    """Pure function. One h3_enc_stats row of a list of latents."""
    m, s = channel_stats(latents)
    return {"stat_id": stat_id, "source": source, "recipe": recipe, "count": len(latents), "channel_mean": m.double().tolist(), "channel_std": s.double().tolist()}


def describe(row):
    """
    Pure function. One printable line of an h3_enc_stats row: channel groups 0-15 and 16-23.

    >>> describe({"stat_id": "a", "count": 2, "channel_mean": [0.0] * 24, "channel_std": [1.0] * 16 + [0.5] * 8})
    'a (n=2): std ch0-15 1.000, ch16-23 0.500; |mean| ch0-15 0.000, ch16-23 0.000'
    """
    s, m = np.array(row["channel_std"]), np.abs(np.array(row["channel_mean"]))
    return (f"{row['stat_id']} (n={row['count']}): std ch0-15 {s[:16].mean():.3f}, ch16-23 {s[16:].mean():.3f}; "
            f"|mean| ch0-15 {m[:16].mean():.3f}, ch16-23 {m[16:].mean():.3f}")


STAT_COLUMNS = ["source", "recipe", "count", "channel_mean", "channel_std"]


def stats(count=300):
    """Command. Commits h3_enc_stats of the stored training latents (first `count` train rows by sha256 of the key)."""
    from bluejay import open_database
    db = open_database("project.dtab")
    rows = []
    for table, key, column in (("h3_pexels_clips", "pexels_id", "h3_latent_video"), ("h3_magick_images", "page_id", "h3_latent_512")):
        found = db.query(f"SELECT {key}, blob_info({column}) AS z FROM {table} WHERE split = 'train' AND {column} IS NOT NULL "
                         f"ORDER BY encode(sha256({key}), 'hex') LIMIT {count}").to_pylist()
        latents = [bytes_tensors(read_blob(r["z"]))["latent"] for r in found]
        rows.append(stat_row(f"stored_{column}", f"{table}.{column}", "untiled posterior sample (h3.encode), as trained on", latents))
        print(describe(rows[-1]), flush=True)
    db.commit("h3_enc_stats", STAT_COLUMNS, rows, insert_missing=True)


def train_clips(db, n):
    """Query. n train clips with a video latent and a current caption embedding: pexels_id, clip, embed, stored latent."""
    import hashlib
    found = db.query("SELECT pexels_id, h3_caption, h3_text_sha256, blob_info(clip_512) AS clip, blob_info(h3_text_embed) AS embed, "
                     "blob_info(h3_latent_video) AS z FROM h3_pexels_clips WHERE split = 'train' AND clip_512 IS NOT NULL "
                     "AND h3_text_embed IS NOT NULL AND h3_latent_video IS NOT NULL ORDER BY encode(sha256(pexels_id), 'hex') "
                     f"LIMIT {4 * n}").to_pylist()
    fresh = [r for r in found if r["h3_text_sha256"] == hashlib.sha256(r["h3_caption"].encode("utf-8")).hexdigest()]
    return fresh[:n]


def encode_sample(vae, frames, tiled, tile=None):
    """Query (runs the encoder). The project's latent of a clip (posterior sample, h3.ENCODE_SEED), untiled or tiled (tile, overlap)."""
    if tile:
        vae.tile_sample_min_height = vae.tile_sample_min_width = tile[0]
        vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = tile[1]
    mode, std = L.encode_posterior(vae, frames, tiled=tiled)
    vae.tile_sample_min_height = vae.tile_sample_min_width = 256  # diffusers' defaults
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 64
    return L.sample_posterior(mode, std).cpu()


def base_of(video_sigma, shift):
    """
    Pure function. The base grid point u whose shifted σ (shift·u / (1 + (shift − 1)·u)) is video_sigma.

    >>> round(base_of(12 * 0.5 / (1 + 11 * 0.5), 12.0), 6)
    0.5
    """
    return video_sigma / (shift - (shift - 1) * video_sigma)


def fm_losses(transformer, z, embed, audio, layouts, device, seed):
    """
    Query (runs the model, no grad). Stock H3's video flow-matching MSE of one latent at each FM_SIGMAS, fixed noise.

    Args:
        z (torch.Tensor): (24, 42, 32, 32) normalized latent
        embed (dict): prompt_embeds (1, N, 5120), text_token_tags (N,)
        audio (torch.Tensor): (2, 32, 235) silent audio latent

    Returns:
        dict[float, float]: σ -> MSE
    """
    from h3magic import train as T
    import einx
    out = {}
    x0 = T.video_rows(z[None].to(device))
    a0 = einx.rearrange("R C -> 1 R C", T.audio_rows(audio).to(device))
    for k, sigma in enumerate(FM_SIGMAS):
        gen = torch.Generator(device=device).manual_seed(seed * 100 + k)
        noise = torch.randn(x0.shape, generator=gen, device=device)
        audio_noise = torch.randn(a0.shape, generator=gen, device=device)
        video_sigma, audio_sigma = T.paired_sigmas(base_of(sigma, T.VIDEO_SHIFT))
        video_t, audio_t = 1.0 - video_sigma, 1.0 - audio_sigma
        key = (int(embed["text_token_tags"].numel()),)
        if key not in layouts:
            layouts[key] = T.Layout(embed["text_token_tags"], z.shape[1], audio.shape[-1], device, side=z.shape[-1])
        timestep, timestep_indices = layouts[key].row_timesteps(video_t, audio_t)
        pred = transformer(hidden_states=T.noise_sample(x0, noise, video_t).to(T.DTYPE), audio_hidden_states=T.noise_sample(a0, audio_noise, audio_t).to(T.DTYPE),
                           encoder_hidden_states=embed["prompt_embeds"].to(device, T.DTYPE), timestep=timestep, timestep_indices=timestep_indices,
                           return_dict=False, **layouts[key].tensors)[0]
        out[sigma] = float((pred.float() - T.flow_target(x0, noise)).pow(2).mean())
    return out


def fmloss(clips=30):
    """Command. Re-encodes `clips` train clips untiled / tiled (+ TILE_VARIANTS on a few), then stock H3's FM loss on both (module docstring)."""
    from bluejay import open_database
    from diffusers import MiniMaxH3Transformer3DModel
    from h3magic import train as T
    db = open_database("project.dtab")
    rows = train_clips(db, clips)
    print(f"fmloss: {len(rows)} train clips", flush=True)
    vae = L.load_vae()
    latents = {"untiled": [], "tiled": []}
    variants = {f"tiled_{t}_{o}": [] for t, o in TILE_VARIANTS}
    stored, embeds = [], []
    for i, r in enumerate(rows):
        frames = read_mp4_rgb(r["clip"])[:141]
        latents["untiled"].append(encode_sample(vae, frames, False))
        latents["tiled"].append(encode_sample(vae, frames, True))
        if i < VARIANT_CLIPS:
            for (t, o), name in zip(TILE_VARIANTS, variants):
                variants[name].append(encode_sample(vae, frames, True, (t, o)))
        stored.append(bytes_tensors(read_blob(r["z"]))["latent"].float())
        embeds.append(bytes_tensors(read_blob(r["embed"])))
        print(f"  {r['pexels_id']}: |untiled - stored| {float((latents['untiled'][-1] - stored[-1]).abs().mean()):.4f}, "
              f"|tiled - untiled| {float((latents['tiled'][-1] - latents['untiled'][-1]).abs().mean()):.4f}", flush=True)
    stat_rows = [stat_row(f"reencode_{name}", f"{len(rows)} train clips (h3_pexels_clips.clip_512)", f"{name} posterior sample", zs) for name, zs in latents.items()]
    stat_rows += [stat_row(f"reencode_{name}", f"first {VARIANT_CLIPS} of those clips", f"{name.replace('_', ' ')} posterior sample", zs) for name, zs in variants.items()]
    stat_rows += [stat_row(f"reencode_untiled_first{VARIANT_CLIPS}", f"first {VARIANT_CLIPS} of those clips", "untiled posterior sample", latents["untiled"][:VARIANT_CLIPS]),
                  stat_row(f"reencode_tiled_first{VARIANT_CLIPS}", f"first {VARIANT_CLIPS} of those clips", "tiled 256 / 64 posterior sample", latents["tiled"][:VARIANT_CLIPS])]
    db.commit("h3_enc_stats", STAT_COLUMNS, stat_rows, insert_missing=True)
    for s in stat_rows:
        print(describe(s), flush=True)
    del vae
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    audio = T.silent_audio(device, VIDEO_AUDIO_LATENTS)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(T.MODEL_DIR / "transformer", torch_dtype=T.DTYPE, device_map={"": "cuda"}).eval()
    transformer.requires_grad_(False)
    db.commit("h3_enc_runs", ["method", "objective", "terms", "steps", "lr", "prior_weight"],
              [{"run_id": f"fm_{name}", "method": f"fm_{name}", "objective": None, "terms": None, "steps": None, "lr": None, "prior_weight": None}
               for name in latents], insert_missing=True)
    layouts, results = {}, {name: [] for name in latents}
    for i, r in enumerate(rows):
        sid = f"pexels_{r['pexels_id']}_train_clip"
        if not db.query(f"SELECT sample_id FROM h3_enc_samples WHERE sample_id = '{sid}'").to_pylist():
            frames = read_mp4_rgb(r["clip"])
            db.commit("h3_enc_samples", ["kind", "origin", "pexels_id", "side", "frame", "source", "source_frames", "source_mp4"],
                      [{"sample_id": sid, "kind": "video", "origin": "pexels_train", "pexels_id": r["pexels_id"], "side": 512, "frame": None,
                        "source": png_bytes(frames[O.IMAGE_FRAME]), "source_frames": None, "source_mp4": read_blob(r["clip"])}], insert_missing=True)
        out = []
        t = time.time()
        with torch.no_grad():
            for name in latents:
                losses = fm_losses(transformer, latents[name][i], embeds[i], audio, layouts, device, NOISE_SEED + i)
                results[name].append(losses)
                out.append({"run_id": f"fm_{name}", "sample_id": sid, "fm_loss": float(np.mean(list(losses.values()))),
                            "fm_loss_by_sigma": json.dumps(losses), "latent": tensors_bytes({"latent": latents[name][i]})})
        db.commit("h3_enc_results", ["fm_loss", "fm_loss_by_sigma", "latent"], out, insert_missing=True)
        print(f"  {sid}: FM loss untiled {out[0]['fm_loss']:.4f} tiled {out[1]['fm_loss']:.4f} ({time.time() - t:.0f} s)  "
              + " ".join(f"σ{s}: {results['untiled'][-1][s]:.4f}/{results['tiled'][-1][s]:.4f}" for s in FM_SIGMAS), flush=True)
    for s in FM_SIGMAS:
        u, tl = np.array([x[s] for x in results["untiled"]]), np.array([x[s] for x in results["tiled"]])
        print(f"σ {s}: untiled {u.mean():.4f} tiled {tl.mean():.4f} (tiled lower on {int((tl < u).sum())}/{len(u)} clips, mean diff {np.mean(tl - u):+.4f} ± {np.std(tl - u) / np.sqrt(len(u)):.4f})", flush=True)


if __name__ == "__main__":
    fire.Fire({"stats": stats, "fmloss": fmloss})
