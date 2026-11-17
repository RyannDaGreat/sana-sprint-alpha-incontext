"""
Step 2: train the adapter model. Full fine-tune of the SANA-Sprint 1.6B TEACHER, text-only, on MAGICK gray composites.

Why full fine-tune (not LoRA): the user wants the highest LoRA rank that fits; a full fine-tune is that limit, and it fits
easily (fp32 weights + grads + AdamW moments ≈ 26 GB of 80 GB). Master weights stay fp32 (bf16 autocast for compute), and
checkpoints are saved in fp32, so `W_base − W_adapter + W_control` is exact in fp32 and there is nothing to bake.

Objective: the teacher's own rectified-flow v-prediction recipe (see train_objective.py). Caption dropout 0.1 to the
null ("") embedding keeps CFG working. Each DDP rank owns a fixed shard of records and encodes only those captions once.

Single GPU / toy:
    PYTHONPATH=src uv run python -m sana_alpha.train_adapter --run toy --source toy
Multi GPU:
    PYTHONPATH=src uv run torchrun --nproc_per_node 8 -m sana_alpha.train_adapter --run <name> --source <src> ...
"""

import json
import math
import os
import time
from pathlib import Path

import fire
import torch
import torch.distributed as dist
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

from sana_alpha.eval_set import load_eval_set
from sana_alpha.models import TEACHER_DIR, localize
from sana_alpha.train_data import epoch_batches, load_lance_records, load_toy_records
from sana_alpha.train_objective import add_flow_noise, flow_velocity_target, sample_sigmas
from sana_alpha.train_text import encode_captions, pad_embeddings

ROOT = Path(__file__).resolve().parents[2]
SOURCES = dict(toy=lambda *shard: load_toy_records(ROOT / "agents/.scratchpad/toy_train", *shard),
               lance=lambda *shard: load_lance_records(ROOT / "data/lance/magick_samples.lance", *shard))
SIGMA_BUCKETS = (0.0, 0.5, 0.8, 1.0)  # loss is logged per noise-level band; median training sigma is 0.75
SAMPLE_TILE = 512  # px per image in the sample grid


def lr_at(step, lr, warmup_steps):
    """
    Pure function. Linear warmup to lr, then constant (Sana teacher fine-tune uses a constant schedule).

    Args:
        step (int): optimizer steps taken so far
        lr (float): peak learning rate
        warmup_steps (int): warmup length

    Returns:
        float

    Examples:
        >>> lr_at(0, 1e-3, 100), lr_at(49, 1e-4, 100), lr_at(500, 1e-4, 100)
        (1e-05, 5e-05, 0.0001)
    """
    return lr * min(1.0, (step + 1) / max(1, warmup_steps))


def bucket_losses(per_sample, sigmas, edges=SIGMA_BUCKETS):
    """
    Pure function. Sums and counts per-sample losses inside noise-level bands.

    Args:
        per_sample (torch.Tensor): (B,) losses
        sigmas (torch.Tensor): (B,) noise levels
        edges (tuple): band edges, len K+1

    Returns:
        torch.Tensor: (2, K) float64 rows = (sum, count)

    Examples:
        >>> bucket_losses(torch.tensor([1.0, 2.0, 4.0]), torch.tensor([0.1, 0.6, 0.9])).tolist()
        [[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]]
    """
    out = torch.zeros(2, len(edges) - 1, dtype=torch.float64)
    for k in range(len(edges) - 1):
        m = (sigmas >= edges[k]) & (sigmas <= edges[k + 1]) if k == len(edges) - 2 else (sigmas >= edges[k]) & (sigmas < edges[k + 1])
        out[0, k], out[1, k] = per_sample[m].double().sum(), m.sum()
    return out


def image_grid(images, columns, tile):
    """
    Pure function. Tiles PIL images into one grid image, each resized to tile × tile.

    Args:
        images (list[PIL.Image.Image]): N images
        columns (int): images per row
        tile (int): tile size in px

    Returns:
        PIL.Image.Image: (ceil(N / columns)·tile, columns·tile) RGB

    Examples:
        >>> image_grid([Image.new("RGB", (8, 8))] * 3, 2, 4).size
        (8, 8)
    """
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * tile, rows * tile))
    for i, im in enumerate(images):
        grid.paste(im.convert("RGB").resize((tile, tile), Image.LANCZOS), ((i % columns) * tile, (i // columns) * tile))
    return grid


def setup_distributed():
    """
    Command. Joins the torchrun process group (NCCL) when WORLD_SIZE is set; picks this rank's GPU.

    Returns:
        tuple: (rank, world_size, device)
    """
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def barrier(world):
    """Command. Waits for all ranks (no-op on one process)."""
    if world > 1:
        dist.barrier()


def log_line(path, record):
    """Command. Prints one JSON record and appends it to the log file."""
    line = json.dumps(record)
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


@torch.no_grad()
def save_samples(pipe, transformer, samples, path, steps, guidance, chunk=4):
    """
    Command. Generates the fixed sample prompts from fixed noise with the current weights and writes one grid PNG.

    Args:
        pipe (SanaPipeline): vae + scheduler, no text encoder
        transformer (SanaTransformer2DModel): unwrapped training model (fp32 weights; run under bf16 autocast)
        samples (dict): embeds, mask, null_embeds, null_mask (CPU, padded to 300), noise (N, 32, 32, 32)
        path (Path): output PNG
        steps (int): sampler steps (flow DPM-Solver++, the teacher's scheduler)
        guidance (float): CFG scale
        chunk (int): images per pipeline call
    """
    transformer.eval()
    pipe.transformer = transformer
    device = pipe.vae.device
    images = []
    for i in range(0, len(samples["noise"]), chunk):
        s = slice(i, i + chunk)
        with torch.autocast("cuda", torch.bfloat16):
            out = pipe(negative_prompt=None, prompt_embeds=samples["embeds"][s].to(device), prompt_attention_mask=samples["mask"][s].to(device),
                       negative_prompt_embeds=samples["null_embeds"][s].to(device),
                       negative_prompt_attention_mask=samples["null_mask"][s].to(device),
                       latents=samples["noise"][s].to(device), use_resolution_binning=False,
                       num_inference_steps=steps, guidance_scale=guidance, output_type="latent")
            for latent in out.images:  # decode one at a time: a batched 1024² DC-AE decode can OOM next to the optimizer state
                decoded = pipe.vae.decode(latent[None] / pipe.vae.config.scaling_factor, return_dict=False)[0]
                images += pipe.image_processor.postprocess(decoded, output_type="pil")
    transformer.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    image_grid(images, 4, SAMPLE_TILE).save(path)


def save_checkpoint(transformer, optimizer, step, run_dir, keep_optimizer):
    """
    Command. Writes run_dir/step_XXXXXX/ (diffusers format, fp32 safetensors); optionally the resume state.

    Args:
        transformer (SanaTransformer2DModel): unwrapped model
        optimizer (torch.optim.Optimizer): its state is written to run_dir/resume.pt when keep_optimizer
        step (int): optimizer steps taken
        run_dir (Path): run folder
        keep_optimizer (bool): also write the resume state (overwrites the previous one)
    """
    out = run_dir / f"step_{step:06d}"
    transformer.save_pretrained(out, safe_serialization=True)
    if keep_optimizer:
        torch.save(dict(optimizer=optimizer.state_dict(), step=step, checkpoint=out.name), run_dir / "resume.pt.tmp")
        os.replace(run_dir / "resume.pt.tmp", run_dir / "resume.pt")
    return out


def main(run, source="toy", batch_size=16, lr=2e-5, warmup_steps=250, max_steps=1000, grad_clip=0.1,
         caption_dropout=0.1, weight_decay=0.0, grad_checkpointing=False, seed=0, log_every=10,
         sample_every=250, num_samples=8, sample_steps=20, sample_guidance=4.5, save_every=1000, resume=True,
         compile=False):
    """
    Command. Trains the adapter (full fine-tune of the teacher) and writes models/adapter/<run>/.

    Args:
        run (str): run name; output in models/adapter/<run>/, log in agents/.scratchpad/logs/train_adapter_<run>.log
        source (str): record source, a key of SOURCES
        batch_size (int): per-GPU batch
        lr (float): AdamW peak learning rate
        warmup_steps (int): linear warmup
        max_steps (int): optimizer steps
        grad_clip (float): global grad-norm clip (0.1 as in the teacher fine-tune)
        caption_dropout (float): probability a caption is replaced by the null "" embedding
        weight_decay (float): AdamW decoupled weight decay (0, as in the teacher fine-tune: it would bias the delta)
        grad_checkpointing (bool): recompute block activations in backward
        seed (int): data order / noise seed
        log_every (int): steps between log records
        sample_every (int): steps between sample grids (also at step 0 and at the end); 0 = never
        num_samples (int): first N eval-set prompts with their fixed noise
        sample_steps (int): sampler steps
        sample_guidance (float): CFG scale for samples
        save_every (int): steps between fp32 checkpoints (also at the end); 0 = never
        resume (bool): continue from models/adapter/<run>/resume.pt if it exists
        compile (bool): torch.compile each transformer block
    """
    rank, world, device = setup_distributed()
    teacher = localize(TEACHER_DIR)  # node-local copy; checkpoints still go to FSx
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = ROOT / "models/adapter" / run
    log_path = ROOT / "agents/.scratchpad/logs" / f"train_adapter_{run}.log"
    config = dict(run=run, source=source, batch_size=batch_size, world=world, lr=lr, warmup_steps=warmup_steps,
                  max_steps=max_steps, grad_clip=grad_clip, caption_dropout=caption_dropout, weight_decay=weight_decay,
                  grad_checkpointing=grad_checkpointing, seed=seed, compile=compile)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.json").write_text(json.dumps(config, indent=1))
        log_line(log_path, dict(event="start", **config))

    # Records and caption embeddings (this rank's shard only).
    records = SOURCES[source](rank, world, seed)
    latents, captions = records["latents"], records["captions"]
    unique = sorted(set(captions))
    t0 = time.time()
    text_pipe = SanaPipeline.from_pretrained(teacher, transformer=None, vae=None, dtype=torch.bfloat16).to(device)
    cache_path = run_dir / "text_cache" / f"rank{rank}_of{world}.pt"  # caption -> embedding; makes resume fast
    embeds = torch.load(cache_path) if cache_path.exists() else {}
    missing = [c for c in unique if c not in embeds]
    embeds.update(zip(missing, encode_captions(text_pipe, missing)))
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeds, cache_path)
    null_embed = encode_captions(text_pipe, [""], null=True)[0]
    caption_embeds = [embeds[c] for c in captions]
    if rank == 0:
        eval_set = load_eval_set()
        prompts = [item["caption"] for item in eval_set["items"][:num_samples]]
        # encode_prompt(do_classifier_free_guidance=True) leaves ~32 GB allocated on the GPU (diffusers 0.43); avoid it.
        both, both_mask = pad_embeddings(encode_captions(text_pipe, prompts) + [null_embed] * len(prompts))  # pipeline needs equal shapes
        n = len(prompts)
        samples = dict(embeds=both[:n], mask=both_mask[:n], null_embeds=both[n:], null_mask=both_mask[n:],
                       noise=eval_set["noise"][:num_samples])
        (run_dir / "sample_prompts.json").write_text(json.dumps(prompts, indent=1))
    del text_pipe
    torch.cuda.empty_cache()
    lengths = torch.tensor([len(e) for e in caption_embeds], dtype=torch.float32)
    print(f"[rank {rank}] {len(latents)} records, {len(unique)} unique captions ({len(missing)} newly encoded) in {time.time() - t0:.1f}s, "
          f"tokens mean {lengths.mean():.1f} max {lengths.max():.0f}, null tokens {len(null_embed)}", flush=True)

    # Model: fp32 master weights.
    transformer = SanaTransformer2DModel.from_pretrained(teacher, subfolder="transformer", dtype=torch.float32)
    step = 0
    resume_path = run_dir / "resume.pt"
    resume_state = None
    if resume and resume_path.exists():
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        transformer = SanaTransformer2DModel.from_pretrained(run_dir / resume_state["checkpoint"], dtype=torch.float32)
        step = resume_state["step"]
    transformer.to(device).train()
    if grad_checkpointing:
        transformer.enable_gradient_checkpointing()
    if compile:
        for block in transformer.transformer_blocks:
            block.compile()
    model = DistributedDataParallel(transformer, device_ids=[device.index], gradient_as_bucket_view=True) if world > 1 else transformer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), fused=True)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        if rank == 0:
            log_line(log_path, dict(event="resume", step=step, checkpoint=resume_state["checkpoint"]))
    scaling = AutoencoderDC.load_config(teacher / "vae")["scaling_factor"]  # 0.41407
    if rank == 0:
        sample_pipe = SanaPipeline.from_pretrained(teacher, transformer=None, text_encoder=None, tokenizer=None,
                                                   dtype=torch.float32).to(device)
        sample_pipe.set_progress_bar_config(disable=True)
        params = sum(p.numel() for p in transformer.parameters())
        log_line(log_path, dict(event="model", params=params, trainable=sum(p.numel() for p in transformer.parameters() if p.requires_grad)))

    def maybe_sample(at_step):
        if rank == 0:
            t = time.time()
            save_samples(sample_pipe, transformer, samples, run_dir / "samples" / f"step_{at_step:06d}.png", sample_steps, sample_guidance)
            log_line(log_path, dict(event="samples", step=at_step, seconds=round(time.time() - t, 1)))
        barrier(world)

    if step == 0 and sample_every:
        maybe_sample(0)
    generator = torch.Generator().manual_seed(seed * 1000 + rank + step)
    epoch = step * batch_size // len(latents)
    window = dict(loss=0.0, n=0, time=time.time(), buckets=torch.zeros(2, len(SIGMA_BUCKETS) - 1, dtype=torch.float64))
    if rank == 0:
        log_line(log_path, dict(event="ready", step=step, alloc_gb=round(torch.cuda.memory_allocated(device) / 2**30, 1)))
    torch.cuda.reset_peak_memory_stats(device)
    while step < max_steps:
        batches = epoch_batches(len(latents), batch_size, epoch, seed)
        assert batches, f"shard of {len(latents)} records is smaller than batch_size {batch_size}"
        for idx in batches:
            if step >= max_steps:
                break
            x0 = latents[idx].to(device, torch.float32, non_blocking=True) * scaling
            drop = (torch.rand(len(idx), generator=generator) < caption_dropout).tolist()
            text, text_mask = pad_embeddings([null_embed if d else caption_embeds[i] for i, d in zip(idx.tolist(), drop)])
            noise = torch.randn(x0.shape, generator=generator).to(device)
            sigmas = sample_sigmas(len(idx), generator).to(device)
            xt = add_flow_noise(x0, noise, sigmas)
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(hidden_states=xt, encoder_hidden_states=text.to(device), encoder_attention_mask=text_mask.to(device),
                             timestep=sigmas, return_dict=False)[0]
            per_sample = (pred.float() - flow_velocity_target(x0, noise)).pow(2).mean(dim=(1, 2, 3))
            loss = per_sample.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), grad_clip)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, lr, warmup_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window["loss"] += loss.item()
            window["n"] += 1
            window["buckets"] += bucket_losses(per_sample.detach().cpu(), sigmas.cpu())
            if step % log_every == 0:
                stats = torch.tensor([window["loss"], window["n"], *window["buckets"].flatten().tolist()], device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(stats)
                seconds = time.time() - window["time"]
                if rank == 0:
                    b = stats[2:].view(2, -1).cpu()
                    log_line(log_path, dict(
                        event="train", step=step, epoch=epoch, loss=round(stats[0].item() / stats[1].item(), 5),
                        loss_by_sigma={f"{SIGMA_BUCKETS[k]}-{SIGMA_BUCKETS[k + 1]}": round((b[0, k] / max(b[1, k], 1)).item(), 5) for k in range(b.shape[1])},
                        grad_norm=round(grad_norm.item(), 4), lr=optimizer.param_groups[0]["lr"],
                        it_per_s=round(window["n"] / seconds, 3), img_per_s=round(window["n"] * batch_size * world / seconds, 2),
                        max_mem_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 1)))
                window.update(loss=0.0, n=0, time=time.time(), buckets=torch.zeros_like(window["buckets"]))
            if save_every and (step % save_every == 0 or step == max_steps):
                if rank == 0:
                    out = save_checkpoint(transformer, optimizer, step, run_dir, keep_optimizer=True)
                    log_line(log_path, dict(event="checkpoint", step=step, path=str(out.relative_to(ROOT))))
                barrier(world)
            if sample_every and (step % sample_every == 0 or step == max_steps):
                maybe_sample(step)
                window["time"] = time.time()
        epoch += 1
    if rank == 0:
        log_line(log_path, dict(event="done", step=step))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
