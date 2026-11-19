"""
FIBO2 trimask model (manifest §14): the §13 recipe on Bria Fibo 2.0, in two modes of one trainer.

    adapter  LoRA on the FIBO2 transformer, text only, on MAGICK composited on random solid colors (fibo2_colorbg_<S>),
             as adapter_v2: it learns MAGICK's look, not the gray.
    control  LoRA on W_adapter (the adapter merged in, frozen), condition = the trimask latent (fibo2_trimask_<S>) as
             FIBO2's own clean context image, target = the RGB on gray (fibo2_gray_<S>). Methods drawn uniformly, as §13.
    final    W_base + (W_control − W_adapter) = FIBO2 + the control LoRA alone: the adapter is never merged into it.
Gates (--gates, control only): off by default. FIBO2 was trained to read clean context images (its edit mode), so the
condition path exists already; §13's GateControl gates start a new path at zero. --gates True adds them anyway
(fibo2.attach_context_gates: one zero-initialized tanh gate per block on the attention to the trimask tokens). Multi-resolution as §13 (train_control.RESOLUTION_PROBS, step_resolution).
Objective: v = ε − x_0 on the image tokens, σ = FIBO2's resolution-shifted logit-normal (fibo2.sample_sigmas). Caption
dropout 0.1 to FIBO2's null caption; condition dropout 0.1 to the all-gray trimask (unknown everywhere). Prodigy, as §13.

Data: the latents (≈ 1.2 TB) are not preloaded. Each rank keeps its shard's row ids and captions and reads each batch's
Blobs from a pinned lance version in a background thread, one step ahead.

Outputs in models/fibo2/<run>/: step_XXXXXX/lora.safetensors (float32) + meta.json every --save_every steps, latest.txt,
previews/step_XXXXXX.png ([trimask | generated | reference] rows, control; [generated | reference] rows, adapter).
Launch: ./train_fibo2.sh adapter <run> | ./train_fibo2.sh control <run> --adapter models/fibo2/<adapter run>/step_XXXXXX
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel

from sana_alpha import fibo2, models
from sana_alpha.train_adapter import barrier, setup_distributed
from sana_alpha.train_control import LANCE, METHOD_LOG_EVERY, RESOLUTION_PROBS, TRIMASK_LANCE, method_uniform_draws, step_resolution
from sana_alpha.train_data import epoch_batches, rank_shard, read_latent_blobs
from sana_alpha.train_objective import add_flow_noise, flow_velocity_target
from sana_alpha.trimask import METHODS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/fibo2"
VAE_DIR = f"{fibo2.FIBO2_DIR}/vae"
PREVIEW_COUNT = 4  # rows per preview grid
PREVIEW_SIZE = 1024  # previews at the main training resolution
PREVIEW_STEPS = 20  # sampler steps for previews (the pipeline default is 30; 20 keeps a preview under ~3 min)
PREVIEW_CFG = 5.0  # FIBO2's default guidance


def load_index(mode, versions, rank, world, seed):
    """
    Query. This rank's shard of training rows; no latents. Rows: MAGICK train, not nsfw, all FIBO2 latents written.

    Args:
        mode (str): "adapter" or "control"
        versions (tuple[int, int]): lance versions of magick_samples and trimask_augmentations (equal on every rank)
        rank, world, seed (int): see train_data.rank_shard

    Returns:
        dict: rowids (N,), captions list[str]; control also: cond_rowids (M,), cond_owner (M,) record index,
              cond_method (M,) index into trimask.METHODS
    """
    import lance

    done = "fibo2_colorbg_256" if mode == "adapter" else "fibo2_gray_256"  # the last column encode_fibo2 writes
    ds = lance.dataset(str(LANCE), version=versions[0])
    rows = ds.to_table(columns=["page_id", "subject"], with_row_id=True,
                       filter=f"{done} IS NOT NULL AND split = 'train' AND NOT nsfw").to_pandas()
    if mode == "control":
        tri = lance.dataset(str(TRIMASK_LANCE), version=versions[1])
        options = tri.to_table(columns=["page_id", "method"], with_row_id=True, filter="fibo2_trimask_256 IS NOT NULL").to_pandas()
        rows = rows[rows.page_id.isin(options.page_id)]
    rows = rows.sort_values("page_id").reset_index(drop=True)
    rows = rows.iloc[rank_shard(len(rows), rank, world, seed).numpy()].reset_index(drop=True)
    index = dict(rowids=rows._rowid.to_numpy(), captions=rows.subject.astype(str).tolist())
    if mode == "control":
        options = options[options.page_id.isin(rows.page_id)].sort_values("_rowid")
        per_method = options.method.value_counts().reindex(METHODS, fill_value=0)
        assert (per_method > 0).all(), f"trimask methods without rows in rank {rank}'s shard: {per_method.to_dict()}"
        record_of, method_of = pd.Series(rows.index, index=rows.page_id), {m: i for i, m in enumerate(METHODS)}
        index.update(cond_rowids=options._rowid.to_numpy(), cond_owner=torch.tensor(options.page_id.map(record_of).to_numpy()),
                     cond_method=torch.tensor(options.method.map(method_of).to_numpy()))
    return index


def draws(index, mode, batch_size, seed, gen):
    """
    Near-pure function (advances gen). Endless (record indices, condition rows) batches: adapter = shuffled epochs of
    the records (condition rows unused), control = method_uniform_draws, one epoch's worth at a time.

    Examples:
        >>> it = draws(dict(captions="abcd"), "adapter", 2, 0, None)
        >>> [next(it)[0].tolist() for _ in range(3)]
        [[0, 1], [3, 2], [1, 3]]
    """
    epoch = 0
    while True:
        if mode == "adapter":
            yield from ((b, b) for b in epoch_batches(len(index["captions"]), batch_size, epoch, seed))
        else:
            yield from method_uniform_draws(index["cond_owner"], index["cond_method"], len(index["captions"]), batch_size, gen)
        epoch += 1


def control_preview_rows(index):
    """
    Pure function. The control previews' items: the first condition row of each of PREVIEW_COUNT methods spread
    evenly over METHODS, and its record.

    Args:
        index (dict): load_index("control", ...): cond_method (M,), cond_owner (M,)

    Returns:
        tuple: record indices (PREVIEW_COUNT,), condition rows (PREVIEW_COUNT,)

    Examples:
        >>> idx, rows = control_preview_rows(dict(cond_method=torch.arange(19).repeat(2), cond_owner=torch.arange(38) // 2))
        >>> rows.tolist(), idx.tolist()
        ([0, 6, 12, 18], [0, 3, 6, 9])
    """
    rows = torch.tensor([int(torch.nonzero(index["cond_method"] == m)[0]) for m in torch.linspace(0, len(METHODS) - 1, PREVIEW_COUNT).long()])
    return index["cond_owner"][rows], rows


def read_batch(tables, index, mode, res, idx, rows):
    """
    Query (reads Blobs). One batch's raw latents at resolution res.

    Returns:
        dict: res, idx (B,), rows (B,), target (B, 32, res/8, res/8) bf16; control also cond (B, 32, res/8, res/8) bf16
    """
    target = "fibo2_colorbg" if mode == "adapter" else "fibo2_gray"
    batch = dict(res=res, idx=idx, rows=rows, target=read_latent_blobs(tables[0], f"{target}_{res}", index["rowids"][idx.numpy()].tolist()))
    if mode == "control":
        batch["cond"] = read_latent_blobs(tables[1], f"fibo2_trimask_{res}", index["cond_rowids"][rows.numpy()].tolist())
    return batch


@torch.no_grad()
def gray_latents(device, bn):
    """Query (loads the VAE briefly). The normalized latent of the all-gray trimask at every training resolution: {S: (1, 32, S/8, S/8)}."""
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(models.localize(VAE_DIR), torch_dtype=torch.bfloat16).to(device)
    out = {s: fibo2.normalize_latents(vae.encode(fibo2.gray_image(s).to(device, torch.bfloat16)).latent_dist.mode(), *bn) for s in RESOLUTION_PROBS}
    del vae
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def write_preview(transformer, vae, text, batch, captions, bn, path):
    """
    Command. Samples PREVIEW_COUNT rows (fixed noise per row) and writes the grid at full resolution as a lossless PNG (m00705).

    Args:
        transformer: the model being trained (eval mode during the call)
        vae: AutoencoderKLFlux2 (rank 0)
        text (tuple): (text_encoder, processor)
        batch (dict): read_batch at PREVIEW_SIZE: target, and cond for control
        captions (list[str]): the rows' MAGICK captions
        bn (tuple): fibo2.bn_stats
        path (Path): output PNG
    """
    device = vae.device
    embeds, mask = fibo2.encode_prompts(*text, [fibo2.caption_prompt(c) for c in captions])
    null, null_mask = fibo2.encode_prompts(*text, [fibo2.NULL_PROMPT])
    target = fibo2.normalize_latents(batch["target"].to(device), *bn)
    cond = fibo2.normalize_latents(batch["cond"].to(device), *bn) if "cond" in batch else None
    transformer.eval()
    out = torch.cat([fibo2.sample(transformer, embeds[i:i + 1], mask[i:i + 1], null, null_mask,
                                  None if cond is None else cond[i:i + 1], PREVIEW_STEPS, PREVIEW_CFG, seed=i,
                                  shape=target[i:i + 1].shape) for i in range(len(captions))])
    transformer.train()
    decode = lambda x: ((vae.decode(fibo2.denormalize_latents(x, *bn).to(vae.dtype)).sample.float().clamp(-1, 1) + 1) / 2)
    columns = ([decode(cond)] if cond is not None else []) + [decode(out), decode(target)]
    grid = torch.cat([torch.cat(list(col), dim=1) for col in columns], dim=2)  # (3, rows·S, columns·S)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    image.save(path)


def load_checkpoint(transformer, folder, gates):
    """
    Command (loads weights into transformer). A step folder's LoRA (lora.safetensors, as fibo2.lora_state wrote it) and,
    with gates, its gates.safetensors, into the trainable modules made by fibo2.add_lora / attach_context_gates.

    Returns:
        int: the folder's step (meta.json)
    """
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    lora = {k: v for k, v in load_file(folder / "lora.safetensors").items() if not k.endswith(".alpha")}
    result = set_peft_model_state_dict(transformer, lora)
    assert not result.unexpected_keys, f"unexpected LoRA keys: {result.unexpected_keys[:5]}"
    missing = [k for k in result.missing_keys if "lora_" in k]
    assert not missing, f"LoRA keys missing from {folder}: {missing[:5]}"
    if gates:
        saved = load_file(folder / "gates.safetensors")
        with torch.no_grad():
            for i, g in enumerate(transformer.context_gates):
                g.copy_(saved[f"context_gates.{i}"])
    return json.loads((folder / "meta.json").read_text())["step"]


def train(mode, run, adapter=None, rank=64, batch_size=2, max_steps=20000, warmup=500, lr=1.0, weight_decay=0.01,
          max_grad_norm=1.0, cond_drop=0.1, text_drop=0.1, save_every=500, preview_every=500, log_every=10, seed=0,
          multires=True, gates=False, init_from=None):
    """
    Command. Trains the adapter or control LoRA with DDP (torchrun) or on one GPU; writes models/fibo2/<run>/.

    Args:
        mode (str): "adapter" or "control"
        run (str): output folder name under models/fibo2/
        adapter (str | None): control only: the adapter checkpoint folder (step_XXXXXX)
        rank (int): LoRA rank (alpha = rank)
        batch_size (int): per GPU, at every resolution (as §13)
        max_steps, warmup (int): optimizer steps, linear warmup steps
        lr, weight_decay, max_grad_norm (float): Prodigy (lr 1 = its own estimate), on LoRA weights, gradient clipping
        cond_drop, text_drop (float): condition -> all-gray trimask, caption -> null caption
        save_every, preview_every, log_every (int): steps (preview_every 0 = none)
        seed (int): data order and noise
        multires (bool): step_resolution over RESOLUTION_PROBS; False = 1024 only
        gates (bool): control only: per-block zero-initialized context gates (fibo2.attach_context_gates), saved as gates.safetensors
        init_from (str | None): a step folder of this run to resume from: its LoRA (and gates), its optimizer.pt when present
            (Prodigy state incl. d; else the optimizer starts fresh), and its step count. Seeds are offset by that step, so
            the resumed run draws new batches; schedules keyed on the step (resolution, warmup) continue unchanged
    """
    import lance
    import prodigyopt

    assert mode in ("adapter", "control") and (mode == "adapter") == (adapter is None), f"mode {mode!r} with adapter {adapter!r}"
    assert not gates or mode == "control", "gates are for the control mode"
    rank_id, world, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    main = rank_id == 0
    out_dir = OUT_DIR / run
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True) if main else None

    versions = [[lance.dataset(str(t)).version for t in (LANCE, TRIMASK_LANCE)] if main else None]
    if world > 1:
        dist.broadcast_object_list(versions, src=0)
    tables = [lance.dataset(str(t), version=v) for t, v in zip((LANCE, TRIMASK_LANCE), versions[0])]
    index = load_index(mode, versions[0], rank_id, world, seed)
    log(f"world {world}, mode {mode}, shard {len(index['captions'])} records" +
        (f", {len(index['cond_rowids'])} trimasks" if mode == "control" else "") + f", lance versions {versions[0]}")

    if main:  # one node-local copy, made once, before the other ranks read it
        for part in ("transformer", "text_encoder", "processor", "vae"):
            models.localize(f"{fibo2.FIBO2_DIR}/{part}")
    barrier(world)
    bn = fibo2.bn_stats(VAE_DIR)
    gray = gray_latents(device, bn)
    text = fibo2.load_text_encoder(device)
    null, null_mask = fibo2.encode_prompts(*text, [fibo2.NULL_PROMPT])
    transformer = fibo2.load_transformer(device, adapter)
    fibo2.add_lora(transformer, rank)
    if gates:
        fibo2.attach_context_gates(transformer)
    start_step = 0
    if init_from:
        start_step = load_checkpoint(transformer, Path(init_from), gates)
        log(f"resumed from {init_from} at step {start_step}")
    transformer.enable_gradient_checkpointing()
    transformer.train()
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    gate_params = list(transformer.context_gates) if gates else []
    lora_params = [p for p in trainable if all(p is not g for g in gate_params)]
    log(f"trainable params {sum(p.numel() for p in trainable) / 1e6:.1f}M (LoRA rank {rank}{' + 46 context gates' if gates else ''})")
    model = DistributedDataParallel(transformer, device_ids=[device.index]) if world > 1 else transformer
    groups = [dict(params=lora_params, weight_decay=weight_decay)] + ([dict(params=gate_params, weight_decay=0.0)] if gates else [])
    optimizer = prodigyopt.Prodigy(groups, lr=lr, betas=(0.9, 0.999),
                                   decouple=True, use_bias_correction=True, safeguard_warmup=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + start_step + 1) / max(warmup, 1)))
    if init_from and (Path(init_from) / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(Path(init_from) / "optimizer.pt", map_location=device, weights_only=True))
        log(f"optimizer state restored (prodigy_d {optimizer.param_groups[0]['d']:.2e})")
    elif init_from:
        log("no optimizer.pt in init_from: Prodigy starts fresh (its d re-estimates during the warmup)")

    vae = None
    if main and preview_every:
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(models.localize(VAE_DIR), torch_dtype=torch.bfloat16).to(device)
        if mode == "control":
            preview_idx, preview_rows = control_preview_rows(index)
        else:
            preview_idx = preview_rows = torch.arange(PREVIEW_COUNT)
        preview = read_batch(tables, index, mode, PREVIEW_SIZE, preview_idx, preview_rows)
        preview_captions = [index["captions"][i] for i in preview_idx.tolist()]
        if not init_from:
            write_preview(transformer, vae, text, preview, preview_captions, bn, out_dir / "previews" / "step_000000.png")

    data_seed = seed + 7919 * start_step  # a resumed run draws new batches (7919: any prime; the step is the offset)
    gen = torch.Generator().manual_seed(data_seed * 1000 + rank_id)
    batches = draws(index, mode, batch_size, data_seed, gen)
    step_res = lambda s: step_resolution(s, seed) if multires else 1024
    reader = ThreadPoolExecutor(1)
    pending = reader.submit(read_batch, tables, index, mode, step_res(start_step), *next(batches))
    meta = dict(run=run, mode=mode, adapter=str(adapter), rank=rank, batch_size=batch_size, world=world, multires=multires, gates=gates,
                lance_versions=versions[0], objective="flow v=eps-x0, FIBO2-shifted logit-normal sigma",
                init_from=str(init_from), start_step=start_step)
    drawn = torch.zeros(len(METHODS), dtype=torch.long)
    res_steps, res_secs = dict.fromkeys(RESOLUTION_PROBS, 0), dict.fromkeys(RESOLUTION_PROBS, 0.0)
    step, t_last, seen = start_step, time.time(), 0
    torch.cuda.reset_peak_memory_stats(device)
    while step < max_steps:
        t_step, batch = time.time(), pending.result()
        res, idx, rows = batch["res"], batch["idx"], batch["rows"]
        pending = reader.submit(read_batch, tables, index, mode, step_res(step + 1), *next(batches))  # read one step ahead
        x0 = fibo2.normalize_latents(batch["target"].to(device), *bn)
        b = len(x0)
        drop_text = (torch.rand(b, generator=gen) < text_drop).tolist()
        captions = [fibo2.NULL_PROMPT if d else fibo2.caption_prompt(index["captions"][j]) for d, j in zip(drop_text, idx.tolist())]
        embeds, mask = fibo2.encode_prompts(*text, captions)
        context = None
        if mode == "control":
            cond = fibo2.normalize_latents(batch["cond"].to(device), *bn)
            keep = (torch.rand(b, generator=gen) >= cond_drop).to(device).view(-1, 1, 1, 1)
            context = [torch.where(keep, cond, gray[res].to(device)).to(torch.bfloat16)]
            fibo2.set_context_tokens(transformer, cond)
        sigmas = fibo2.sample_sigmas(b, x0.shape[2] * x0.shape[3] // 4, gen).to(device)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        xt = add_flow_noise(x0, noise, sigmas)
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(hidden_states=xt.to(torch.bfloat16), timestep=sigmas, encoder_hidden_states=embeds,
                         encoder_attention_mask=mask, context_latents=context, return_dict=False)[0]
        loss = ((pred.float() - flow_velocity_target(x0, noise)) ** 2).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step, seen = step + 1, seen + b * world
        torch.cuda.synchronize(device)
        res_steps[res], res_secs[res] = res_steps[res] + 1, res_secs[res] + time.time() - t_step
        if mode == "control":
            drawn += torch.bincount(index["cond_method"][rows], minlength=len(METHODS))
        if step % log_every == 0:
            dt = time.time() - t_last
            log(f"step {step} res {res} loss {loss.item():.4f} grad {grad_norm.item():.3f} {seen / dt:.2f} img/s "
                f"mem {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GB prodigy_d {optimizer.param_groups[0]['d']:.2e}"
                + (f" mean tanh(gate) {torch.cat(gate_params).tanh().mean().item():.4f}" if gates else ""))
            t_last, seen = time.time(), 0
        if step % METHOD_LOG_EVERY == 0:
            log("resolutions (last %d steps): %s" % (METHOD_LOG_EVERY, ", ".join(
                f"{r}: {n} steps {res_secs[r] / max(n, 1):.2f} s/step" for r, n in res_steps.items())))
            res_steps, res_secs = dict.fromkeys(RESOLUTION_PROBS, 0), dict.fromkeys(RESOLUTION_PROBS, 0.0)
            if mode == "control":
                log(f"trimask methods drawn (rank 0, last {METHOD_LOG_EVERY} steps): {dict(zip(METHODS, drawn.tolist()))}")
                drawn.zero_()
        if main and (step % save_every == 0 or step == max_steps):
            folder = out_dir / f"step_{step:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            save_file(fibo2.lora_state(transformer, rank), folder / "lora.safetensors")
            if gates:
                save_file({f"context_gates.{i}": g.detach().float().cpu() for i, g in enumerate(gate_params)}, folder / "gates.safetensors")
            torch.save(optimizer.state_dict(), folder / "optimizer.pt")  # ≈ 3.2 GB: Prodigy's 4 float32 states per LoRA weight, and d
            (folder / "meta.json").write_text(json.dumps(dict(meta, step=step)) + "\n")
            (out_dir / "latest.txt").write_text(f"step_{step:06d}\n")
            log(f"saved {folder}")
        if main and preview_every and (step % preview_every == 0 or step == max_steps):
            write_preview(transformer, vae, text, preview, preview_captions, bn, out_dir / "previews" / f"step_{step:06d}.png")
            log(f"preview {out_dir / 'previews' / f'step_{step:06d}.png'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(dict(train=train))
