"""
Step 3: GateControl matte control on the SANA-Sprint 1.6B teacher with the adapter baked in (frozen).

Trained: a LoRA ("control", rank 16 by default, GateControl's targets) + the per-block gates. Condition = the clean
DC-AE latent of the 3-channel matte, as a second token stream (see gatecontrol.py). Objective = the teacher's own
(train_objective.py, shared with the adapter): logit-normal + shift-3 sigmas, x_t = (1−σ)x_0 + σε, v = ε − x_0,
timestep = σ; MSE on the target stream only. Condition dropout 0.1, caption dropout 0.1 (null = "" like the pipeline's
negative prompt). Optimizer as GateControl: Prodigy lr 1, weight decay 0.01 (0 on gates), constant with warmup.

Outputs, in models/control/<run>/:
    step_XXXXXX/control_lora.safetensors, gates.safetensors   float32 (LoRA delta + gates), every --save_every steps
    previews/step_XXXXXX.png                                  [matte | generated | reference] rows, rank 0
    final_step_XXXXXX/ fp32 transformer + gates.safetensors   from `merge_final`: W_base + (W_control − W_adapter)

Launch (8 GPUs):
    PYTHONPATH=src uv run torchrun --nproc_per_node 8 -m sana_alpha.train_control train --run <name> --adapter models/adapter/<run>/step_XXXXXX
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from sana_alpha import gatecontrol as gc, models
from sana_alpha.trimask import METHODS
from sana_alpha.train_adapter import setup_distributed
from sana_alpha.train_data import epoch_batches, rank_shard, read_latent_blobs
from sana_alpha.train_objective import add_flow_noise, flow_velocity_target, sample_sigmas
from sana_alpha.train_text import encode_captions, pad_embeddings

ROOT = Path(__file__).resolve().parents[2]
TEACHER = models.TEACHER_DIR  # repo-relative; always loaded through models.localize (node-local copy)
CONTROL_DIR = ROOT / "models/control"
OVERFIT_DIR = ROOT / "agents/.scratchpad/control_overfit"
LANCE = ROOT / "data/lance/magick_samples.lance"
TRIMASK_LANCE = ROOT / "data/lance/trimask_augmentations.lance"
PREVIEW_COUNT = 4  # rows in each preview grid
METHOD_LOG_EVERY = 250  # steps between logs of the trimask methods drawn and the resolutions used
RESOLUTION_PROBS = {1024: 0.5, 768: 1 / 6, 512: 1 / 6, 256: 1 / 6}  # m00656: mostly 1024, half the steps split over the rest
PREVIEW_STEPS = 20  # teacher sampling steps (models.TEACHER_DEFAULTS)
PREVIEW_CFG = 4.5  # teacher text CFG (models.TEACHER_DEFAULTS)


def load_control_records(folder=OVERFIT_DIR):
    """
    Query. Reads a latent cache folder: latents.safetensors (latent_rgb_1024, latent_alpha_1024) + rows.tsv.

    Latents are raw DC-AE outputs (scaling_factor NOT applied), as written by data/processors/encode_latents.py.

    Args:
        folder (str | Path): e.g. agents/.scratchpad/control_overfit

    Returns:
        dict: latents (N, 32, 32, 32) RGB-on-gray, cond_latents (N, 32, 32, 32) matte, captions list[str],
              page_ids list[str]
    """
    folder = Path(folder)
    rows = pd.read_csv(folder / "rows.tsv", sep="\t", quoting=3)
    tensors = load_file(folder / "latents.safetensors")
    rgb, alpha = tensors["latent_rgb_1024"], tensors["latent_alpha_1024"]
    assert len(rows) == len(rgb) == len(alpha), f"{len(rows)} rows, {len(rgb)} rgb, {len(alpha)} alpha latents"
    assert (rows.split == "train").all() and (rows.nsfw == 0).all(), "cache must hold train, non-nsfw rows only"
    return dict(latents=rgb, cond_latents=alpha, captions=rows.subject.astype(str).tolist(),
                page_ids=rows.page_id.tolist())


def load_lance_shard(path, version, rank, world, seed):
    """
    Query. This rank's shard of the curated training rows from the BlueJay lance table magick_samples.

    Rows: train split, not nsfw, both latents written. Every rank must pass the same table version, so the
    row list (and so the shard split) is identical across ranks while curation keeps appending.

    Args:
        path (str | Path): data/lance/magick_samples.lance
        version (int): lance table version to read
        rank, world (int): this process and the process count
        seed (int): shard permutation seed (train_data.rank_shard)

    Returns:
        dict: same keys as load_control_records, for this rank's ≈N/world rows (raw, unscaled bf16 latents)
    """
    import lance
    import safetensors.torch

    ds = lance.dataset(str(path), version=version)
    rows = ds.to_table(columns=["page_id", "subject"], with_row_id=True, filter="latent_rgb_1024 IS NOT NULL AND "
                       "latent_alpha_1024 IS NOT NULL AND split = 'train' AND NOT nsfw")
    keep = rank_shard(rows.num_rows, rank, world, seed).tolist()
    ids = [rows["_rowid"][i].as_py() for i in keep]
    read = lambda column: torch.stack([safetensors.torch.load(b.read())["latent"] for b in ds.take_blobs(column, ids=ids)])
    return dict(latents=read("latent_rgb_1024"), cond_latents=read("latent_alpha_1024"),
                captions=[rows["subject"][i].as_py() for i in keep], page_ids=[rows["page_id"][i].as_py() for i in keep])


def step_resolution(step, seed, probs=RESOLUTION_PROBS):
    """
    Pure function. The image resolution of one training step, drawn from probs by an rng seeded with (seed, step),
    so every DDP rank picks the same one without communicating.

    Args:
        step (int): optimizer step about to run
        seed (int): run seed
        probs (dict): resolution -> probability

    Returns:
        int

    Examples:
        >>> [step_resolution(s, 0) for s in range(8)]
        [768, 256, 1024, 256, 256, 1024, 1024, 1024]
        >>> from collections import Counter; sorted(Counter(step_resolution(s, 0) for s in range(6000)).items())
        [(256, 970), (512, 1011), (768, 973), (1024, 3046)]
    """
    return int(np.random.default_rng([seed, step]).choice(list(probs), p=list(probs.values())))


def load_trimask_shard(versions, rank, world, seed, resolutions=(1024,)):
    """
    Query. This rank's shard for trimask control: targets from magick_samples, conditions from trimask_augmentations.

    Records = train, non-nsfw magick_samples rows with latent_rgb at every resolution and at least one trimask with
    latent_trimask_1024, sorted by page_id, then rank_shard. Condition rows = every such trimask of those records.
    Each lower resolution r has its own pool: the rows that also have latent_trimask_<r> (the encoder covers ~1/3).
    Raises unless every trimask.METHODS tag has rows in every pool of this shard.

    Args:
        versions (tuple[int, int]): lance versions of magick_samples and trimask_augmentations (equal on all ranks)
        rank, world, seed (int): see load_lance_shard
        resolutions (tuple[int]): 1024 and the lower resolutions to load

    Returns:
        dict: captions, page_ids (N records) and the 1024 pool: latents (N, 32, 32, 32), cond_latents (M, 32, 32, 32),
              cond_owner (M,) record index, cond_method (M,) index into trimask.METHODS (raw, unscaled bf16 latents);
              lowres {r: pool with the same four keys, latents (N, 32, r/32, r/32)} for the resolutions below 1024
    """
    import lance

    ds, tri = lance.dataset(str(LANCE), version=versions[0]), lance.dataset(str(TRIMASK_LANCE), version=versions[1])
    options = tri.to_table(columns=["page_id", "method"], with_row_id=True,
                           filter="latent_trimask_1024 IS NOT NULL").to_pandas()
    rows = ds.to_table(columns=["page_id", "subject"], with_row_id=True, filter=" AND ".join(
        f"latent_rgb_{r} IS NOT NULL" for r in resolutions) + " AND split = 'train' AND NOT nsfw").to_pandas()
    rows = rows[rows.page_id.isin(options.page_id)].sort_values("page_id").reset_index(drop=True)
    rows = rows.iloc[rank_shard(len(rows), rank, world, seed).numpy()].reset_index(drop=True)
    options = options[options.page_id.isin(rows.page_id)].sort_values("_rowid")
    record_of, method_of = pd.Series(rows.index, index=rows.page_id), {m: i for i, m in enumerate(METHODS)}

    def pool(r):
        """Query. The latents of resolution r: all records, and the condition rows that have latent_trimask_<r>."""
        have = options if r == 1024 else options[options._rowid.isin(tri.to_table(
            columns=[], with_row_id=True, filter=f"latent_trimask_{r} IS NOT NULL")["_rowid"].to_numpy())]
        per_method = have.method.value_counts().reindex(METHODS, fill_value=0)
        assert (per_method > 0).all(), f"trimask methods without {r} latents in rank {rank}'s shard: {per_method.to_dict()}"
        return dict(latents=read_latent_blobs(ds, f"latent_rgb_{r}", rows._rowid.tolist()),
                    cond_latents=read_latent_blobs(tri, f"latent_trimask_{r}", have._rowid.tolist()),
                    cond_owner=torch.tensor(have.page_id.map(record_of).to_numpy()),
                    cond_method=torch.tensor(have.method.map(method_of).to_numpy()))

    return dict(pool(1024), lowres={r: pool(r) for r in resolutions if r != 1024},
                captions=rows.subject.astype(str).tolist(), page_ids=rows.page_id.tolist())


def method_uniform_draws(cond_owner, cond_method, count, batch_size, gen):
    """
    Near-pure function (advances gen). count // batch_size batches of (record indices, condition rows): each example
    draws a method uniformly from trimask.METHODS, then one of that method's rows uniformly (so a method with fewer
    rows is not under-represented); its record is the row's owner.

    Args:
        cond_owner, cond_method (torch.Tensor): (M,) record index and METHODS index of each condition row
        count (int): examples to draw (rounded down to whole batches)
        batch_size (int): per rank
        gen (torch.Generator): CPU generator

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: (records (B,), condition rows (B,)) int64

    Examples:
        >>> owner, method = torch.arange(30) % 3, torch.arange(30) % len(METHODS)
        >>> draws = method_uniform_draws(owner, method, 100, 10, torch.Generator().manual_seed(0))
        >>> len(draws), bool((owner[draws[0][1]] == draws[0][0]).all())
        (10, True)
        >>> counts = torch.bincount(method[torch.cat([c for _, c in draws])], minlength=len(METHODS))
        >>> len(counts) == len(METHODS), bool(counts.min() > 0)  # every method drawn, about equally often
        (True, True)
    """
    methods = torch.randint(len(METHODS), (count // batch_size * batch_size,), generator=gen)
    rows = torch.zeros_like(methods)
    for m in range(len(METHODS)):
        options, pick = torch.nonzero(cond_method == m).flatten(), methods == m
        rows[pick] = options[torch.randint(len(options), (int(pick.sum()),), generator=gen)]
    return [(cond_owner[r], r) for r in rows.split(batch_size)]


def epoch_draws(records, batch_size, epoch, seed, gen):
    """
    Near-pure function (advances gen for trimask records). One epoch of (record indices, condition rows) batches,
    len(captions) // batch_size of them: mattes = shuffled records (epoch_batches), condition row = record;
    trimasks = method_uniform_draws over the 1024 pool.

    Args:
        records (dict): from load_shard
        batch_size (int): per rank
        epoch, seed (int): see epoch_batches
        gen (torch.Generator): CPU generator for the trimask draws

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: (records (B,), condition rows (B,)) int64

    Examples:
        >>> [(a.tolist(), b.tolist()) for a, b in epoch_draws(dict(captions="abcd"), 2, 0, 0, None)]
        [([0, 1], [0, 1]), ([3, 2], [3, 2])]
    """
    if "cond_method" not in records:
        return [(idx, idx) for idx in epoch_batches(len(records["captions"]), batch_size, epoch, seed)]
    return method_uniform_draws(records["cond_owner"], records["cond_method"], len(records["captions"]), batch_size, gen)


def load_shard(data, rank, world, seed, cond="matte", multires=False):
    """
    Query. This rank's training records from a latent cache folder, or from the lance table when data == "lance".

    Args:
        data (str): "lance" or a folder for load_control_records
        rank, world, seed (int): see load_lance_shard
        cond (str): "matte" (latent_alpha_1024) or "trimask" (load_trimask_shard; needs data == "lance")
        multires (bool): also load the latents at every RESOLUTION_PROBS resolution (trimask only)

    Returns:
        dict: latents, cond_latents, captions, page_ids (+ cond_owner, cond_method for trimask) for this rank only
    """
    assert cond in ("matte", "trimask") and (cond == "matte" or data == "lance"), f"cond {cond!r} with data {data!r}"
    assert not multires or cond == "trimask", "multires is implemented for the trimask condition only"
    if data != "lance":
        records = load_control_records(data)
        keep = rank_shard(len(records["captions"]), rank, world, seed)
        return dict(latents=records["latents"][keep], cond_latents=records["cond_latents"][keep],
                    captions=[records["captions"][i] for i in keep], page_ids=[records["page_ids"][i] for i in keep])
    import lance
    tables = [LANCE] + [TRIMASK_LANCE] * (cond == "trimask")
    versions = [[lance.dataset(str(t)).version for t in tables] if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(versions, src=0)
    if cond == "trimask":
        return load_trimask_shard(versions[0], rank, world, seed, tuple(RESOLUTION_PROBS) if multires else (1024,))
    return load_lance_shard(LANCE, versions[0][0], rank, world, seed)


def load_base_transformer(adapter=None, device="cuda", dtype=torch.bfloat16):
    """
    Query (reads weights from disk). The frozen control base: the adapter model (full fine-tune), or the teacher.

    Args:
        adapter (str | None): diffusers transformer folder of W_adapter (float32), e.g. models/adapter/<run>/step_XXXXXX
            written by train_adapter.save_checkpoint. None = the plain teacher (tests, and the final model's base).
        device (str): torch device
        dtype (torch.dtype): base weight dtype (rounded once from float32)

    Returns:
        SanaTransformer2DModel: frozen, eval mode
    """
    path = models.localize(TEACHER) / "transformer" if adapter is None else models.localize(adapter)
    transformer = SanaTransformer2DModel.from_pretrained(path, torch_dtype=torch.float32)
    return transformer.to(device, dtype).requires_grad_(False).eval()


class ControlModel(nn.Module):
    """DDP wrapper: holds the transformer (frozen base + control LoRA + gates) and runs the two-stream forward."""

    def __init__(self, transformer):
        """Command (registers the transformer as a submodule)."""
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, cond_latents, text_embeds, text_mask, sigmas, keep):
        """Pure function (given weights). Velocity prediction, (B, 32, 32, 32); see gatecontrol.gatecontrol_forward."""
        return gc.gatecontrol_forward(self.transformer, latents, cond_latents, text_embeds, text_mask, sigmas, keep)


def control_param_groups(transformer, weight_decay):
    """
    Query. Optimizer groups: gates without weight decay (GateControl), control LoRA with weight decay.

    Args:
        transformer (SanaTransformer2DModel): with the control LoRA and gates
        weight_decay (float): for the LoRA group

    Returns:
        list[dict]: two param groups
    """
    named = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    gates = [p for n, p in named if n.startswith("gatecontrol_gates.")]
    lora = [p for n, p in named if not n.startswith("gatecontrol_gates.")]
    assert all("lora_" in n for n, p in named if not n.startswith("gatecontrol_gates.")), "unexpected trainable param"
    return [dict(params=gates, weight_decay=0.0), dict(params=lora, weight_decay=weight_decay)]


@torch.no_grad()
def write_preview(pipe, transformer, records, indices, cond_indices, path):
    """
    Command. Samples with the stock SanaPipeline (20 steps, CFG 4.5) driven by the gated model; writes a PNG grid.

    Rows: [matte | generated | reference], one per index. Noise is seeded by the row number, so previews are
    comparable across steps.

    Args:
        pipe (SanaPipeline): with vae, tokenizer, text_encoder on the GPU; its transformer is replaced temporarily
        transformer (SanaTransformer2DModel): with control LoRA + gates (gatecontrol enabled)
        records (dict): from load_control_records (raw latents)
        indices (list[int]): records to show
        cond_indices (list[int]): their rows of records["cond_latents"] (= indices for mattes)
        path (Path): output PNG
    """
    sf = pipe.vae.config.scaling_factor
    device = pipe.vae.device
    cond_latents = records["cond_latents"][cond_indices]
    cond = cond_latents.to(device, transformer.dtype) * sf
    noise = torch.stack([torch.randn(32, 32, 32, generator=torch.Generator().manual_seed(i)) for i in range(len(indices))])
    was_training = transformer.training
    transformer.eval()
    old, pipe.transformer = pipe.transformer, transformer
    out = pipe(prompt=[records["captions"][i] for i in indices], latents=noise.to(device), output_type="np",
               num_inference_steps=PREVIEW_STEPS, guidance_scale=PREVIEW_CFG, use_resolution_binning=False,
               attention_kwargs={gc.COND_KWARG: cond}).images
    pipe.transformer = old
    transformer.train(was_training)
    decode = lambda z: ((pipe.vae.decode(z.to(device, pipe.vae.dtype)).sample.float().clamp(-1, 1) + 1) / 2)
    mattes = decode(cond_latents)
    refs = decode(records["latents"][indices])
    rows = [np.concatenate([m.permute(1, 2, 0).cpu().numpy(), o, r.permute(1, 2, 0).cpu().numpy()], axis=1)
            for m, o, r in zip(mattes, out, refs)]
    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((grid * 255).round().astype(np.uint8)).resize((grid.shape[1] // 4, grid.shape[0] // 4)).save(path)


def train(run, data=str(OVERFIT_DIR), adapter=None, rank=16, batch_size=16, max_steps=10000, warmup=500,
          lr=1.0, weight_decay=0.01, max_grad_norm=1.0, cond_drop=0.1, text_drop=0.1, save_every=1000,
          preview_every=500, log_every=10, grad_checkpointing=False, compile=True, seed=0, init_from=None, start_step=0,
          cond="matte", multires=False):
    """
    Command. Trains the control LoRA + gates with DDP (torchrun) or on one GPU (plain python); writes models/control/<run>/.

    Args:
        run (str): run name, the output folder under models/control/
        data (str): "lance" (data/lance/magick_samples.lance) or a latent cache folder (load_control_records)
        adapter (str | None): W_adapter diffusers transformer folder (train_adapter checkpoint); None = plain teacher
        rank (int): control LoRA rank (alpha = rank)
        batch_size (int): per GPU
        max_steps (int): optimizer steps
        warmup (int): linear warmup steps
        lr (float): Prodigy learning rate (1.0 = Prodigy's own estimate)
        weight_decay (float): on LoRA weights only
        max_grad_norm (float): gradient clipping
        cond_drop (float): probability of dropping the matte (keys zeroed, gated term ×0)
        text_drop (float): probability of replacing the caption with the null prompt
        save_every, preview_every, log_every (int): steps (preview_every 0 = no previews)
        grad_checkpointing (bool): checkpoint each block (saves memory, ~30% slower)
        compile (bool): torch.compile the block function (gatecontrol.compile_blocks)
        seed (int): data order and noise seed (offset by rank)
        init_from (str | None): a saved control step folder to start from (same adapter and rank); the optimizer starts fresh
        start_step (int): step number of init_from, so checkpoints keep counting (e.g. 4000)
        cond (str): "matte" (latent_alpha_1024) or "trimask" (trimask_augmentations; method uniform per example, epoch_draws)
        multires (bool): each step trains at step_resolution (RESOLUTION_PROBS, m00656) on the stored latents of that size;
            a low-res step draws its batch (method uniform) from the trimasks that have latents at that size
    """
    import prodigyopt

    rank_id, world, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main = rank_id == 0
    trimask = cond == "trimask"  # `cond` is reused below for the condition latents
    out_dir = CONTROL_DIR / run
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True) if main else None

    records = load_shard(data, rank_id, world, seed, cond, multires)  # this rank only
    pipe = SanaPipeline.from_pretrained(models.localize(TEACHER), transformer=None, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    text = encode_captions(pipe, records["captions"])
    null = encode_captions(pipe, [""], null=True)[0]
    text_len = torch.tensor(max(len(t) for t in text + [null]), device=device)  # one padded length on every rank:
    if world > 1:                                                                # fixed compiled shapes (compile_blocks)
        dist.all_reduce(text_len, op=dist.ReduceOp.MAX)
    text_len = int(text_len)
    log(f"world {world}, shard {len(text)} records, {len(records['cond_latents'])} conditions, "
        f"text tokens padded to {text_len}")
    if not main:
        del pipe  # only rank 0 keeps text encoder + vae, for previews
        torch.cuda.empty_cache()

    transformer = load_base_transformer(adapter, device)
    lora_config = gc.add_control_lora(transformer, rank)
    gates_module = gc.attach_gates(transformer)
    if init_from:
        lora, gates, meta0 = gc.load_control_files(Path(init_from))
        assert int(meta0["rank"]) == rank and meta0["adapter"] == str(adapter), f"{init_from}: rank/adapter differ: {meta0}"
        result = gc.set_peft_model_state_dict(transformer, lora, adapter_name=gc.ADAPTER_NAME)
        assert not result.unexpected_keys, f"unexpected LoRA keys: {result.unexpected_keys[:5]}"
        gates_module.load_state_dict(gates, strict=True)
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # trainable LoRA + gates in float32; the frozen base stays bf16
    transformer.gatecontrol_checkpointing = grad_checkpointing
    if compile:
        gc.compile_blocks()
    gc.enable_gatecontrol(transformer)
    transformer.train()
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    log(f"trainable params {n_train / 1e6:.2f}M (LoRA rank {rank} + gates)")

    model = ControlModel(transformer)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = prodigyopt.Prodigy(control_param_groups(transformer, weight_decay), lr=lr, betas=(0.9, 0.999),
                                   decouple=True, use_bias_correction=True, safeguard_warmup=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(warmup, 1)))

    sf = AutoencoderDC.load_config(models.localize(TEACHER) / "vae")["scaling_factor"]  # 0.41407
    gen = torch.Generator().manual_seed(seed * 1000 + rank_id)
    preview_idx = preview_cond = list(range(PREVIEW_COUNT))
    drawn = torch.zeros(len(METHODS), dtype=torch.long)  # trimask methods drawn on this rank since the last log
    if trimask:  # preview one row each of PREVIEW_COUNT methods spread over METHODS
        preview_cond = [int(torch.nonzero(records["cond_method"] == m)[0]) for m in torch.linspace(0, len(METHODS) - 1, PREVIEW_COUNT).long()]
        preview_idx = records["cond_owner"][preview_cond].tolist()
        for res, pool in {1024: records, **records["lowres"]}.items():
            log(f"trimask rows per method at {res} (rank 0 shard): {dict(zip(METHODS, torch.bincount(pool['cond_method'], minlength=len(METHODS)).tolist()))}")
    res_steps, res_secs = dict.fromkeys(RESOLUTION_PROBS, 0), dict.fromkeys(RESOLUTION_PROBS, 0.0)  # since the last log
    meta = dict(run=run, adapter=str(adapter), data=str(data), cond=cond, multires=multires, objective="flow v=eps-x0, sigma timestep")
    if main and preview_every:
        write_preview(pipe, transformer, records, preview_idx, preview_cond, out_dir / "previews" / "step_000000.png")
    step, epoch, t_last, seen = start_step, 0, time.time(), 0
    torch.cuda.reset_peak_memory_stats(device)
    while step < max_steps:
        for idx, rows in epoch_draws(records, batch_size, epoch, seed, gen):
            t_step, res = time.time(), step_resolution(step, seed) if multires else 1024
            pool = records if res == 1024 else records["lowres"][res]  # a low-res step draws from its own pool
            if res != 1024:
                (idx, rows), = method_uniform_draws(pool["cond_owner"], pool["cond_method"], batch_size, batch_size, gen)
            x0 = pool["latents"][idx].to(device, torch.float32) * sf
            cond = pool["cond_latents"][rows].to(device, torch.float32) * sf
            drop_text = torch.rand(len(idx), generator=gen) < text_drop
            emb, mask = pad_embeddings([null if d else text[j] for d, j in zip(drop_text.tolist(), idx.tolist())], text_len)
            keep = torch.rand(len(idx), generator=gen) >= cond_drop
            sigmas = sample_sigmas(len(idx), gen)
            noise = torch.randn(x0.shape, generator=gen).to(device)
            xt = add_flow_noise(x0, noise, sigmas.to(device))
            target = flow_velocity_target(x0, noise).float()

            with torch.autocast("cuda", torch.bfloat16):  # PEFT casts LoRA inputs to fp32; autocast keeps saved activations bf16
                pred = model(xt, cond, emb.to(device), mask.to(device), sigmas.to(device), keep.to(device))
            loss = ((pred.float() - target) ** 2).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad],
                                                       max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step, seen = step + 1, seen + len(idx) * world
            if multires:
                torch.cuda.synchronize(device)
                res_steps[res], res_secs[res] = res_steps[res] + 1, res_secs[res] + time.time() - t_step
                if step % METHOD_LOG_EVERY == 0:
                    log("resolutions (last %d steps): %s" % (METHOD_LOG_EVERY, ", ".join(
                        f"{r}: {n} steps {res_secs[r] / max(n, 1):.2f} s/step" for r, n in res_steps.items())))
                    res_steps, res_secs = dict.fromkeys(RESOLUTION_PROBS, 0), dict.fromkeys(RESOLUTION_PROBS, 0.0)

            if step % log_every == 0:
                dt = time.time() - t_last
                g = transformer.gatecontrol_gates
                gate_bias = torch.stack([m.bias for m in g.c_gates]).sigmoid().mean().item()
                d = optimizer.param_groups[1]["d"]
                log(f"step {step} loss {loss.item():.4f} grad {grad_norm.item():.3f} {seen / dt:.1f} img/s "
                    f"mem {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GB prodigy_d {d:.2e} "
                    f"sigmoid(c_gate bias) {gate_bias:.3f}")
                t_last, seen = time.time(), 0
            if trimask:
                drawn += torch.bincount(pool["cond_method"][rows], minlength=len(METHODS))
                if step % METHOD_LOG_EVERY == 0:
                    log(f"trimask methods drawn (rank 0, last {METHOD_LOG_EVERY} steps): {dict(zip(METHODS, drawn.tolist()))}")
                    drawn.zero_()
            if main and (step % save_every == 0 or step == max_steps):
                gc.save_control(transformer, out_dir / f"step_{step:06d}", lora_config, dict(meta, step=step))
                (out_dir / "latest.txt").write_text(f"step_{step:06d}\n")
                log(f"saved {out_dir / f'step_{step:06d}'}")
            if main and preview_every and (step % preview_every == 0 or step == max_steps):
                write_preview(pipe, transformer, records, preview_idx, preview_cond, out_dir / "previews" / f"step_{step:06d}.png")
                log(f"preview {out_dir / 'previews' / f'step_{step:06d}.png'}")
            if step >= max_steps:
                break
        epoch += 1
    if main:
        (out_dir / "train_args.json").write_text(json.dumps(dict(meta, rank=rank, batch_size=batch_size, world=world,
                                                                  max_steps=max_steps, lr=lr, warmup=warmup)) + "\n")
    if world > 1:
        dist.destroy_process_group()


def merge_final(control, out=None, adapter_scale=0.0, student=False):
    """
    Command. Writes the final model W_base + (W_control − W_adapter) = clean teacher + control LoRA delta, in float32.
    With adapter_scale λ it writes W_base + λ·(W_adapter − W_base) + control LoRA delta: λ = 0 is the final model,
    λ = 1 is the control model itself.

    The adapter stayed frozen while the control trained, so W_control − W_adapter is exactly the LoRA delta; the
    gates are new modules and are copied unchanged. Output: a float32 diffusers transformer folder (like the adapter
    checkpoints) + gates.safetensors. Load: t = load_base_transformer(out); gatecontrol.attach_gates(t).load_state_dict(
    load_file(out / "gates.safetensors")); gatecontrol.enable_gatecontrol(t).

    Args:
        control (str): a saved step folder, e.g. models/control/<run>/step_010000
        out (str | None): output folder; default <run>/final_<step folder name>, one per step (models.localize assumes folders never change)
        adapter_scale (float): λ, the fraction of the adapter delta kept; the default folder name gets suffix _adapter<λ>
        student (bool): base = the stock Sprint student instead of the teacher (Step 5 transfer); the adapter delta
            (W_adapter − W_teacher) and the LoRA keys are renamed to the student's; default folder prefix sprint_
    """
    control = Path(control)
    suffix = f"_adapter{adapter_scale:g}" if adapter_scale else ""
    out = Path(out) if out else control.parent / f"{'sprint' if student else 'final'}_{control.name}{suffix}"
    lora, gates, meta = gc.load_control_files(control)
    load = lambda d: SanaTransformer2DModel.from_pretrained(models.localize(d), subfolder="transformer", torch_dtype=torch.float32)
    teacher = load(TEACHER).state_dict()
    base = load(models.SPRINT_DIR) if student else load(TEACHER)
    state = base.state_dict()
    if adapter_scale:
        adapter = load_file(models.localize(meta["adapter"]) / "diffusion_pytorch_model.safetensors")
        assert adapter.keys() == teacher.keys(), "adapter and teacher transformers differ in keys"
        delta = gc.student_lora_keys({k: adapter[k].float() - v for k, v in teacher.items()}) if student else \
            {k: adapter[k].float() - v for k, v in teacher.items()}
        assert delta.keys() <= state.keys(), f"adapter keys missing in base: {sorted(delta.keys() - state.keys())[:3]}"
        state = {k: v + adapter_scale * delta[k] if k in delta else v for k, v in state.items()}  # student-only keys (guidance_embedder) keep their weights
    lora = gc.student_lora_keys(lora) if student else lora
    merged = gc.merge_lora_into_state_dict(state, lora, int(meta["rank"]), float(meta["alpha"]))
    base.load_state_dict(merged, strict=True)
    base.save_pretrained(out, safe_serialization=True)
    save_file(gates, out / "gates.safetensors", metadata=dict(meta, source=str(control)))
    print(f"wrote {out}: {len(merged)} tensors, {len(lora) // 2} LoRA modules merged, {len(gates)} gate tensors")


if __name__ == "__main__":
    fire.Fire(dict(train=train, merge_final=merge_final))
