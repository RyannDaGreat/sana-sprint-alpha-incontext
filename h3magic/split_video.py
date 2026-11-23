# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch>=2.8", "torchvision",
#   "diffusers @ git+https://github.com/huggingface/diffusers.git@750b3be2c0198e194c87e931a6904066de167a5f",
#   "transformers>=4.57", "accelerate", "peft>=0.17", "safetensors", "einx", "fire", "numpy",
#   "pylance", "pyarrow", "pandas",
# ]
# ///
"""
Video readout of the split adapter (Alpha Video v2, manifest §23 item 4). The split adapter is trained on still images
(T = 1), and in 141-frame video its white / black heads drift toward gray after the first latent frame (§22: ~0.2-0.3
alpha left in the background). Here ONLY the head's Δ (Linear 5120 -> 3·96, h3magic/split.py) is trained further, on
synthetic moving cut-outs (h3magic/synth_motion.py; agents/.scratchpad/alphav2/make_clips.py) whose gray, white and black
composites are exact at every frame. The trunk (H3 + baked gray magic + baked split LoRA) is frozen and runs without
gradients, so the gray output (head 0) cannot change and a full 141-frame 768² clip fits in memory.

Loss: per white / black head k, at the gray trajectory's x_t (H3 sign, t = 1 clean), the head's difference from the
gray head must be the clean difference: (v_k − sg(v_gray)) = x0_k − x0_gray, i.e. Δ_k(x) − sg(Δ_0(x)) = x0_k − x0_gray,
so that at sampling x0_k = x_t − t·v_gray + v_k = x0_gray_estimate + (x0_k − x0_gray). Head 0's rows of Δ get no
gradient and no weight decay (they stay as trained on images). Noise: pre-shift base ~ U(BASE_LO, BASE_HI), around the
last sampling step (30 steps, shift 12: base 1/30, sigma 0.29), the only step whose heads the sampler reads.

Saves step folders that are drop-in split adapter folders: splitter_head.safetensors (new Δ) + the split LoRA's
pytorch_lora_weights.safetensors (copied) + meta.json.

    uv run --script h3magic/split_video.py launch NAME --gpus 0,1,2,3,4,5,6,7 \
        --graymagic models/h3magic/graymagic_v3/final --split models/h3magic/split_v3/final
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import einx
import fire
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (h3magic/train.py: layout, noise, rows, bake, status)
from h3magic import split as splitter  # noqa: E402

CLIP_COLUMNS = ("h3_motion_gray_768", "h3_motion_white_768", "h3_motion_black_768")  # BlueJay h3_motion_clips (data/processors/h3_motion_clip.py)
VAL_CLIPS = 20  # at most this many val clips (the val rows of h3_magick_images)
BASE_LO, BASE_HI = 0.015, 0.08  # pre-shift base range: video sigma 0.15-0.51, around the last sampling step's 0.29
VAL_BASES = (0.02, 0.0333, 0.06)
FRAMES = 141
HEAD_KEYS = ("gray", "white", "black")  # = splitter.SPLIT_BACKGROUNDS order


def readout_loss(delta_rows, x0_rows):
    """
    Pure function. Per-head loss of the video readout: mean over rows of (Δ_k − sg(Δ_0) − (x0_k − x0_0))² for k ≥ 1.

    Args:
        delta_rows (Tensor): (B, R, K, C) γ·Δ(x) at the video rows (the base proj_out cancels in the difference)
        x0_rows (Tensor): (B, R, K, C) clean targets per head, head 0 = gray

    Returns:
        Tensor: (K − 1,) one MSE per white / black head

    Examples:
        >>> x0 = torch.tensor([[[[0.0], [1.0], [-1.0]]]])          # (1, 1, 3, 1): gray 0, white 1, black -1
        >>> exact = torch.tensor([[[[5.0], [6.0], [4.0]]]])         # Δ_0 = 5 and the right differences
        >>> readout_loss(exact, x0).tolist(), readout_loss(torch.zeros(1, 1, 3, 1), x0).tolist()
        ([0.0, 0.0], [1.0, 1.0])
    """
    difference = delta_rows[..., 1:, :] - delta_rows[..., :1, :].detach()
    return einx.mean("B R K C -> K", (difference - (x0_rows[..., 1:, :] - x0_rows[..., :1, :])) ** 2)


def clip_records(split, limit=None):
    """
    Query. The finished h3_motion_clips rows of one split, by page_id: (clip row id, h3_magick_images row id) pairs, the
    second for the caption embedding h3_text_embed_v3 (only rows whose embedding matches h3_caption_v3).
    """
    import hashlib
    import lance
    clips = lance.dataset(str(T.LANCE_DIR / "h3_motion_clips.lance"))
    magick = lance.dataset(str(T.LANCE_DIR / "h3_magick_images.lance"))
    done = clips.to_table(columns=["page_id"], filter=f"split = '{split}' AND h3_motion_black_768 IS NOT NULL", with_row_id=True).to_pylist()
    mag = magick.to_table(columns=["page_id", "h3_caption_v3", "h3_text_v3_sha256"], filter="h3_text_embed_v3 IS NOT NULL", with_row_id=True).to_pylist()
    fresh = {r["page_id"]: r["_rowid"] for r in mag if r["h3_caption_v3"] and r["h3_text_v3_sha256"] == hashlib.sha256(r["h3_caption_v3"].encode()).hexdigest()}
    pairs = sorted((r["page_id"], r["_rowid"], fresh[r["page_id"]]) for r in done if r["page_id"] in fresh)
    return [(c, m) for _, c, m in pairs][:limit]


def load_clip(record):
    """Query. One clip: latents (1, LC, LT, LH, LW) gray, targets (1, K, LC, LT, LH, LW), prompt_embeds (1, N, 5120), tags (N,)."""
    import lance
    import safetensors.torch
    clip_id, magick_id = record
    clips = lance.dataset(str(T.LANCE_DIR / "h3_motion_clips.lance"))
    heads = [safetensors.torch.load(clips.take_blobs(c, ids=[clip_id])[0].read())["latent"].float() for c in CLIP_COLUMNS]
    text = safetensors.torch.load(lance.dataset(str(T.LANCE_DIR / "h3_magick_images.lance")).take_blobs("h3_text_embed_v3", ids=[magick_id])[0].read())
    targets = torch.stack(heads)[None]
    return dict(latents=targets[:, 0], targets=targets, prompt_embeds=text["prompt_embeds"], text_token_tags=text["text_token_tags"])


def readout_forward(transformer, head, batch, audio, base, generator, device, layouts):
    """
    Near-pure function (advances generator; runs the frozen model without gradients, then Δ with gradients). Per-head
    readout loss (K − 1,) of one clip at one base noise level.
    """
    latents = batch["latents"].to(device)
    lt, a, side = latents.shape[2], audio.shape[-1], latents.shape[-1]
    video_sigma, audio_sigma = T.paired_sigmas(base)
    video_t, audio_t = 1.0 - video_sigma, 1.0 - audio_sigma
    x0 = T.video_rows(latents)
    noise = torch.randn(x0.shape, generator=generator, device=device)
    a0 = einx.rearrange("R C -> B R C", T.audio_rows(audio).to(device), B=1)
    noisy_audio = T.noise_sample(a0, torch.randn(a0.shape, generator=generator, device=device), audio_t).to(T.DTYPE)
    key = (int(batch["text_token_tags"].numel()), lt, a, side)
    if key not in layouts:
        layouts[key] = T.Layout(batch["text_token_tags"], lt, a, device, side=side)
    timestep, timestep_indices = layouts[key].row_timesteps(video_t, audio_t)
    captured = {}
    hook = head.register_forward_pre_hook(lambda module, args: captured.__setitem__("x", args[0]))
    try:
        with torch.no_grad():
            transformer(hidden_states=T.noise_sample(x0, noise, video_t).to(T.DTYPE), audio_hidden_states=noisy_audio,
                        encoder_hidden_states=batch["prompt_embeds"].to(device, T.DTYPE), timestep=timestep,
                        timestep_indices=timestep_indices, return_dict=False, **layouts[key].tensors)
    finally:
        hook.remove()
    x = captured["x"].index_select(1, layouts[key].tensors["video_indices"]).float()  # (1, R, 5120), no grad
    delta = einx.rearrange("B R (K C) -> B R K C", head.gamma * head.delta(x), K=head.heads)
    return readout_loss(delta, T.split_rows(batch["targets"].to(device)))


def save(head, folder, split_folder, meta):
    """Command. A drop-in split step folder: the new Δ, the split LoRA copied, meta.json; mirrored node-locally."""
    folder.mkdir(parents=True, exist_ok=True)
    splitter.save_head(head, folder)
    shutil.copyfile(T.ROOT / split_folder / "pytorch_lora_weights.safetensors", folder / "pytorch_lora_weights.safetensors")
    base_meta = json.loads((T.ROOT / split_folder / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps({**base_meta, "video_readout": meta}, indent=1))
    local = T.LOCAL_ADAPTER_ROOT / folder.relative_to(T.ADAPTER_ROOT)
    local.mkdir(parents=True, exist_ok=True)
    for name in (splitter.HEAD_FILE, "pytorch_lora_weights.safetensors", "meta.json"):
        shutil.copyfile(folder / name, local / name)


def train(name, graymagic, split, lr=1e-3, max_steps=400, save_every=25, val_every=25, seed=0, stop_at="", val_clips=VAL_CLIPS):
    """
    Command (torchrun, one process per GPU, one clip per GPU per step). Trains the split head's Δ on the motion clips.

    Args:
        name (str): output folder models/h3magic/<name>/
        graymagic (str): the gray magic adapter folder (baked at weight 1)
        split (str): the image-trained split adapter folder (its LoRA baked at γ = 1, its Δ the starting point)
        lr (float): AdamW lr of Δ (no weight decay)
        max_steps, save_every, val_every, seed (int)
        stop_at (str): "HH:MM" Eastern hard stop; agents/.scratchpad/h3train/STOP_<name> stops early
        val_clips (int): at most this many val clips (h3_motion_clips split val)
    """
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    out, stop_file = T.ADAPTER_ROOT / name, T.STATUS_DIR / f"STOP_{name}"
    audio = T.silent_audio(device, T.audio_latents_for(FRAMES))
    from diffusers import MiniMaxH3Transformer3DModel
    for turn in range(0, world, 2):  # stagger the 66 GB loads (RAM guard)
        if rank // 2 == turn // 2:
            transformer = MiniMaxH3Transformer3DModel.from_pretrained(T.MODEL_DIR / "transformer", torch_dtype=T.DTYPE, device_map={"": str(device)})
            transformer.requires_grad_(False)
            for folder in (graymagic, split):
                print(f"rank {rank}: bake {folder}: {T.bake(transformer, T.ROOT / folder / 'pytorch_lora_weights.safetensors')} layers", flush=True)
        dist.barrier()
    transformer.eval()
    head = splitter.attach_splitter(transformer)
    splitter.load_head(head, T.ROOT / split)
    head.gamma = 1.0
    params = list(head.delta.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, betas=T.ADAM_BETAS, eps=T.ADAM_EPS, weight_decay=0.0)
    train_files, val_files = clip_records("train"), clip_records("val", val_clips)
    draws = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    layouts = {}
    meta = dict(name=name, graymagic=graymagic, split=split, lr=lr, world=world, base_range=[BASE_LO, BASE_HI], clips=len(train_files),
                val_clips=len(val_files), frames=FRAMES, loss="(Δ_k − sg(Δ_0)) vs (x0_k − x0_gray), k = white, black; trunk frozen")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        log = open(out / "log.csv", "a")
        log.write("step,base,loss_white,loss_black,val_white,val_black,seconds,time\n")

    def validate():
        """Query (runs the model). Mean val loss per white / black head over this rank's val clips, all ranks averaged."""
        total, count = torch.zeros(2, device=device), torch.zeros((), device=device)
        val_files = clip_records("val", val_clips)  # val clips made after the start join (the same list on every rank: sorted, then cut to the ranks' minimum)
        n = torch.tensor(len(val_files), device=device)
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
        val_files = val_files[:int(n)]
        with torch.no_grad():
            for i, path in enumerate(val_files[rank::world]):
                batch = load_clip(path)
                for k, base in enumerate(VAL_BASES):
                    total += readout_forward(transformer, head, batch, audio, base, torch.Generator(device=device).manual_seed(1000 * i + k + 17 * rank), device, layouts)
                    count += 1
        dist.all_reduce(total)
        dist.all_reduce(count)
        return (total / count.clamp(min=1)).tolist()

    val = validate()
    if rank == 0:
        print(f"{name} step 0 val white/black {val}", flush=True)
        log.write(f"0,,,,{val[0]:.5f},{val[1]:.5f},,{time.time():.0f}\n")
    step, last = 0, time.time()
    while True:
        train_files = clip_records("train")  # clips the h3_motion_clip workers finished since the last pass join
        count = torch.tensor(len(train_files), device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MIN)  # every rank sees the same list length
        train_files = train_files[:int(count)]
        order = draws.permutation(len(train_files))
        for i in range(0, len(order) - world + 1, world):
            path = train_files[order[i + rank]]
            base = float(draws.uniform(BASE_LO, BASE_HI))
            losses = readout_forward(transformer, head, load_clip(path), audio, base, generator, device, layouts)
            losses.mean().backward()
            T.all_reduce_grads(params, world)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = losses.detach().clone()
            dist.all_reduce(value)
            value = (value / world).tolist()
            step += 1
            if step % val_every == 0:
                val = validate()
            if rank == 0:
                seconds, last = time.time() - last, time.time()
                log.write(f"{step},{base:.4f},{value[0]:.5f},{value[1]:.5f},{val[0]:.5f},{val[1]:.5f},{seconds:.1f},{time.time():.0f}\n")
                log.flush()
                print(f"{name} step {step} base {base:.3f} loss white/black {value[0]:.4f} {value[1]:.4f} val {val[0]:.4f} {val[1]:.4f} "
                      f"{seconds:.1f} s/step {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB", flush=True)
                T.write_status(name, f"step {step} loss {np.mean(value):.4f} val {np.mean(val):.4f} {seconds:.1f} s/step")
                if step % save_every == 0:
                    save(head, out / f"step_{step:05d}", split, {**meta, "step": step, "val": val})
            flag = torch.tensor(int(stop_file.exists() or step >= max_steps or T.past_eastern(stop_at, time.time())), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag):
                if rank == 0:
                    save(head, out / "final", split, {**meta, "step": step, "val": val})
                    T.write_status(name, f"FINISHED at step {step}, val {val}")
                dist.destroy_process_group()
                return


def launch(name, gpus="0,1,2,3,4,5,6,7", **kwargs):
    """Command. Runs `train` under torchrun on gpus with this interpreter, tee'd to agents/.scratchpad/logs/splitvideo_<name>.log."""
    import subprocess
    gpus = ",".join(map(str, gpus)) if isinstance(gpus, (list, tuple)) else str(gpus)
    args = " ".join(f"--{k}={v}" for k, v in kwargs.items())
    log = T.ROOT / f"agents/.scratchpad/logs/splitvideo_{name}.log"
    command = (f"CUDA_VISIBLE_DEVICES={gpus} {sys.executable} -m torch.distributed.run --standalone --nproc_per_node={len(gpus.split(','))} "
               f"{__file__} train {name} {args} 2>&1 | tee -a {log}")
    print(command, flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True)


if __name__ == "__main__":
    fire.Fire(dict(train=train, launch=launch))
