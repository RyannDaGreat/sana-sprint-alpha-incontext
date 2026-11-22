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
LoRA trainer of the H3 Magic project (manifest §22): one adapter at a time on MiniMax H3, on 6 GPUs (torchrun).

Adapters stack (m00912): each one is trained on H3 with every earlier adapter BAKED into the weights (W += scale·B·A, fp32
math, then bf16), and only its own fresh LoRA trains.
    pexels    --mode video  --data pexels                                   (on stock H3)
    imagegen  --mode image  --data pexels  --bake pexels                    (7 frames per clip at 0..6 s, m00917)
    graymagic --mode image  --data magick  --bake pexels,imagegen           (MAGICK on 0.5 gray, 512²)

Target and layout are MinimaxH3's verified ones (that dump's src/train_lora.py): H3's t = 1 is CLEAN,
x_t = t·x_0 + (1 − t)·noise, target v = x_0 − noise; the packed sequence comes from diffusers'
MiniMaxH3PrepareLayoutStep.build_packed_sequence and MiniMaxH3SetTimestepsStep.build_row_timesteps, so training and
inference use the same layout. Audio rows are structurally required: they carry the audio VAE latent of silence, are
noised like at inference, and get loss weight 0 (video-only loss; audio is excluded from the project, m00912).

Weights come from node-local /models/h3magic/MiniMax-H3 (only the transformer is loaded; latents and text embeddings
are precomputed BlueJay columns, agents/.scratchpad/h3train/CONTRACT.md). The audio VAE is loaded once to encode
silence, then freed.

Tasting (m00912 "like a chef"): loss per step to log.csv, a fixed validation loss every --val_every steps, the adapter
every --save_every steps, a one-line STATUS file, and a STOP file that makes the run save and exit.

Run (6 GPUs):   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run --script h3magic/train.py launch pexels --mode video --data pexels
Probe memory:   CUDA_VISIBLE_DEVICES=0 uv run --script h3magic/train.py probe --mode video
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the shipped index points at the hub: never a silent 200 GB download
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")  # 6 ranks share 96 cores with the CPU clip cutters: OpenMP spin otherwise

import einx
import fire
import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/models/h3magic/MiniMax-H3")  # node-local copy (system-wide /models convention)
LANCE_DIR = ROOT / "data/lance"
ADAPTER_ROOT = ROOT / "models/h3magic"            # durable adapters (gitignored)
LOCAL_ADAPTER_ROOT = Path("/models/h3magic/adapters")  # node-local mirror, read by inference
STATUS_DIR = ROOT / "agents/.scratchpad/h3train"

DTYPE = torch.bfloat16
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0  # MiniMaxH3SetTimestepsStep: one base grid, two shifts
KEYFRAME_NOISE_AUG = 0.999            # condition-row t; unused by t2va, passed for layout parity
AUDIO_CONDITION_TIMESTEP = 1.0
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
AUDIO_SAMPLING_RATE = 32000
AUDIO_LATENTS_PER_SECOND = 40
FPS = 24
VIDEO_FRAMES = 141  # 17·8 + 5: the longest legal H3 length inside the first 6 s (5.875 s)
IMAGE_AUDIO_LATENTS = 2  # round(1 frame / 24 fps · 40 latents/s): the soundtrack of a one-frame "video"
LATENT_SIDE = 32         # 512 / 16
LATENT_CHANNELS = 24
TEXT_DIM = 5120
TARGET_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
EXCLUDED_SUBSTRING = "adaln"  # MinimaxH3: the AdaLN branch carries the guidance distillation; never adapt it
ADAM_BETAS, ADAM_EPS = (0.9, 0.99), 1e-8
WARMUP_STEPS = 20
VAL_BASES = (0.25, 0.5, 0.75)  # unshifted grid points of the fixed validation noise levels
PROBE_SIGMAS = (0.2, 0.5, 0.8)  # video σ of the target-sign probe (MinimaxH3 probe)
EMA_DECAY = 0.98               # loss smoothing for STATUS
PREFETCH = 4                   # batches read ahead by the loader thread
COLUMNS = dict(video="h3_latent_video", frames="h3_latent_frames", image="h3_latent_512", embed="h3_text_embed")
TABLES = dict(pexels="h3_pexels_clips", magick="h3_magick_images")
KEYS = dict(pexels="pexels_id", magick="page_id")


# ---------------------------------------------------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------------------------------------------------

def shift_sigma(base, shift):
    """
    Pure function. H3's sigma shift of a base grid point: shift·u / (1 + (shift − 1)·u).

    Examples:
        >>> shift_sigma(0.0, 12.0), shift_sigma(1.0, 12.0), round(shift_sigma(0.5, 12.0), 6), round(shift_sigma(0.5, 3.0), 6)
        (0.0, 1.0, 0.923077, 0.75)
    """
    return shift * base / (1 + (shift - 1) * base)


def paired_sigmas(base):
    """
    Pure function. (video sigma, audio sigma) of one base grid point, paired as inference pairs them.

    Examples:
        >>> v, a = paired_sigmas(0.5)
        >>> round(v, 6), round(a, 6)
        (0.923077, 0.75)
    """
    return shift_sigma(base, VIDEO_SHIFT), shift_sigma(base, AUDIO_SHIFT)


def noise_sample(clean, noise, t):
    """
    Pure function. H3's forward process x_t = t·x_0 + (1 − t)·noise (t = 1 is CLEAN).

    Examples:
        >>> noise_sample(torch.ones(2), torch.zeros(2), 0.25).tolist()
        [0.25, 0.25]
    """
    return t * clean + (1.0 - t) * noise


def flow_target(clean, noise):
    """
    Pure function. H3's data-ward velocity x_0 − noise (the negative of the usual flow-match target).

    Examples:
        >>> x0, eps, t = torch.randn(4), torch.randn(4), 0.3
        >>> bool(torch.allclose(noise_sample(x0, eps, t) + (1 - t) * flow_target(x0, eps), x0, atol=1e-5))
        True
    """
    return clean - noise


def audio_latents_for(num_frames):
    """
    Pure function. Audio latents per channel covering num_frames video frames (diffusers audio_latent_num_frames).

    Examples:
        >>> audio_latents_for(141), audio_latents_for(124)
        (235, 207)
    """
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def lora_delta(lora_a, lora_b, alpha):
    """
    Pure function. The weight delta of one LoRA layer: (alpha / r) · B · A, in float32.

    Args:
        lora_a (Tensor): (r, in) down projection
        lora_b (Tensor): (out, r) up projection
        alpha (float): LoRA alpha

    Returns:
        Tensor: (out, in) float32

    Examples:
        >>> lora_delta(torch.ones(2, 3), torch.ones(4, 2), alpha=2).shape, float(lora_delta(torch.ones(2, 3), torch.ones(4, 2), 2)[0, 0])
        (torch.Size([4, 3]), 2.0)
    """
    rank = lora_a.shape[0]
    return (alpha / rank) * (lora_b.float() @ lora_a.float())


def group_by_length(lengths, batch_size, seed):
    """
    Pure function. Shuffled batches of indices whose items share one length (the packed layout of a batch is shared,
    so every item in it needs the same number of text rows). Ragged leftovers form smaller batches.

    Args:
        lengths (list[int]): text-token count per item
        batch_size (int): max items per batch
        seed (int): shuffle seed

    Returns:
        list[list[int]]

    Examples:
        >>> lengths = [5, 7, 5, 5, 7]
        >>> batches = group_by_length(lengths, 2, 0)
        >>> sorted(len(b) for b in batches), all(len({lengths[i] for i in b}) == 1 for b in batches)
        ([1, 2, 2], True)
        >>> sorted(i for b in batches for i in b)
        [0, 1, 2, 3, 4]
    """
    rng = np.random.default_rng(seed)
    buckets = {}
    for i in rng.permutation(len(lengths)).tolist():
        buckets.setdefault(lengths[i], []).append(i)
    batches = [b[k:k + batch_size] for b in buckets.values() for k in range(0, len(b), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches)).tolist()]


def ema(previous, value, decay=EMA_DECAY):
    """
    Pure function. Exponential moving average step (None starts it at value).

    Examples:
        >>> ema(None, 2.0), ema(1.0, 2.0, decay=0.5)
        (2.0, 1.5)
    """
    return value if previous is None else decay * previous + (1 - decay) * value


# ---------------------------------------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------------------------------------

class Layout:
    """The packed [text | audio | video] sequence of one (text length, latent shape), shared by a batch, on a device."""

    def __init__(self, text_token_tags, latent_frames, num_audio_latents, device):
        """
        Command. Builds the layout tensors on device with diffusers' own inference builders.

        Args:
            text_token_tags (Tensor): (N,) int64 modality tag of every text row
            latent_frames (int): LT, 42 for a 141-frame clip, 1 for an image
            num_audio_latents (int): audio latents per channel (235 for a clip, IMAGE_AUDIO_LATENTS for an image)
            device (torch.device)
        """
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

        (position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio
         ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags.cpu(), latent_frames, LATENT_SIDE, LATENT_SIDE, num_audio_latents, PATCH_SIZE, AUDIO_CHANNELS,
            AUDIO_TAG, VIDEO_TAG, keyframe_anchors=())
        assert cond_video == 0 and cond_audio == 0, f"t2va layout has conditioning rows: {cond_video}, {cond_audio}"
        self.tensors = dict(position_ids=position_ids.to(device), token_tags=token_tags.to(device),
                            video_indices=video_indices.to(device), audio_indices=audio_indices.to(device),
                            text_indices=text_indices.to(device))
        self.num_text = int(text_indices.numel())
        self.num_video_rows, self.num_audio_rows = int(video_indices.numel()), int(audio_indices.numel())

    def row_timesteps(self, video_t, audio_t):
        """Query. The transformer's (timestep, timestep_indices) for one noise level, from diffusers' builder."""
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            self.tensors["video_indices"].cpu(), self.tensors["audio_indices"].cpu(), 0, 0, self.num_text, video_t,
            audio_t, max(video_t, KEYFRAME_NOISE_AUG), AUDIO_CONDITION_TIMESTEP)
        device = self.tensors["video_indices"].device
        return unique.to(device), inverse.to(device)


def video_rows(latents):
    """
    Pure function. Patchified video rows of a batch of latents, in the inference packer's row order.

    Args:
        latents (Tensor): (B, LC, LT, LH, LW), e.g. (1, 24, 42, 32, 32) or (7, 24, 1, 32, 32)

    Returns:
        Tensor: (B, LT·LH/2·LW/2, LC·4) float32, e.g. (1, 10752, 96)

    Examples:
        >>> # video_rows(torch.zeros(1, 24, 42, 32, 32)).shape == (1, 10752, 96)  (needs diffusers)
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    return einx.rearrange("(B R) C -> B R C", patchify_video_latents(latents.float(), PATCH_SIZE), B=latents.shape[0])  # it flattens the batch


def audio_rows(latents):
    """
    Pure function. Channel-major audio rows: (C, LCa, A) -> (C·A, LCa), channel 0's latents first.

    Examples:
        >>> audio_rows(torch.arange(12.).reshape(2, 3, 2)).tolist()
        [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [6.0, 8.0, 10.0], [7.0, 9.0, 11.0]]
    """
    return einx.rearrange("C LC A -> (C A) LC", latents.float())


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------

def lora_target_modules(model):
    """
    Pure function. Sorted names of the linear layers the LoRA attaches to (TARGET_SUFFIXES, never adaln).

    Examples:
        >>> import torch.nn as nn
        >>> m = nn.Module(); m.attn = nn.Module(); m.attn.to_q = nn.Linear(2, 2); m.adaln = nn.Module(); m.adaln.attn = nn.Module()
        >>> m.adaln.attn.to_q = nn.Linear(2, 2)
        >>> lora_target_modules(m)
        ['attn.to_q']
    """
    return sorted(n for n, _ in model.named_modules() if n.endswith(TARGET_SUFFIXES) and EXCLUDED_SUBSTRING not in n)


def adapter_file(name):
    """Query. The newest saved weights of adapter `name`: its `final` folder, else its highest step folder."""
    folder = ADAPTER_ROOT / name
    final = folder / "final" / "pytorch_lora_weights.safetensors"
    if final.exists():
        return final
    steps = sorted(folder.glob("step_*/pytorch_lora_weights.safetensors"))
    if not steps:
        raise FileNotFoundError(f"adapter {name}: no final/ or step_*/ weights under {folder}")
    return steps[-1]


def bake(transformer, path, scale=1.0):
    """
    Command. Merges a saved LoRA into the transformer's weights in place: W ← bf16(W + scale·(alpha/r)·B·A), fp32 math.

    Args:
        transformer (nn.Module): H3 transformer, no peft adapter attached yet
        path (Path): pytorch_lora_weights.safetensors with a meta.json beside it (rank, alpha)
        scale (float): adapter weight in the sum (1 = bake fully)

    Returns:
        int: layers merged
    """
    from safetensors.torch import load_file

    meta = json.loads((Path(path).parent / "meta.json").read_text())
    state = load_file(str(path))
    modules = dict(transformer.named_modules())
    merged = 0
    for key in [k for k in state if ".lora_A." in k]:
        name = key.split(".lora_A.")[0].removeprefix("transformer.")
        layer = modules[name]
        delta = lora_delta(state[key].to(layer.weight.device), state[key.replace(".lora_A.", ".lora_B.")].to(layer.weight.device),
                           meta["alpha"])
        layer.weight.data = (layer.weight.data.float() + scale * delta).to(layer.weight.dtype)
        merged += 1
    assert merged == meta["layers"], f"{path}: merged {merged} layers, meta.json says {meta['layers']}"
    return merged


def build_transformer(device, bakes, rank, alpha):
    """
    Command. Loads the H3 transformer straight onto `device` (no full CPU copy: 6 ranks × 66 GB would hit the RAM
    guard), bakes earlier adapters, freezes it, enables gradient checkpointing and attaches a fresh LoRA.

    Args:
        device (torch.device)
        bakes (list[str]): adapter names to bake in, in order (each at weight 1)
        rank (int), alpha (int): the new LoRA

    Returns:
        nn.Module: only the LoRA parameters (float32) train
    """
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    started = time.time()
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR / "transformer", torch_dtype=DTYPE,
                                                              device_map={"": str(device)})
    transformer.requires_grad_(False)
    for name in bakes:
        print(f"bake {name}: {bake(transformer, adapter_file(name))} layers from {adapter_file(name)}", flush=True)
    transformer.enable_gradient_checkpointing()
    with torch.device(device):  # build the LoRA layers on the GPU: peft's CPU init + copy took > 7 min under CPU load
        transformer.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                                           target_modules=lora_target_modules(transformer)))
    cast_training_params(transformer, dtype=torch.float32)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"transformer on {device} in {time.time() - started:.0f} s, {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB, "
          f"LoRA r{rank} a{alpha}: {trainable / 1e6:.1f}M trainable", flush=True)
    return transformer


def silent_audio(device, num_latents):
    """
    Query. Normalized audio-VAE latent of silence: (AUDIO_CHANNELS, 32, num_latents) float32 on CPU.
    Loads the audio VAE (0.6 GB) once, encodes zeros of num_latents · 800 samples (posterior mode), frees it.
    """
    from diffusers import AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(MODEL_DIR / "audio_vae", torch_dtype=torch.float32).to(device).eval()
    samples = num_latents * (AUDIO_SAMPLING_RATE // AUDIO_LATENTS_PER_SECOND)
    with torch.no_grad():
        latents = vae.encode(torch.zeros(AUDIO_CHANNELS, 1, samples, device=device), return_dict=False)[0].mode().float().cpu()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
    del vae
    torch.cuda.empty_cache()
    assert latents.shape[-1] == num_latents, f"audio VAE gave {latents.shape[-1]} latents for {samples} samples"
    return (latents - mean) / std


def save_adapter(transformer, folder, meta):
    """
    Command. Writes pytorch_lora_weights.safetensors (peft keys, loadable by MiniMaxH3ModularPipeline.load_lora_weights)
    and meta.json to folder, and copies both to the node-local mirror.
    """
    from diffusers import MiniMaxH3ModularPipeline
    from peft.utils import get_peft_model_state_dict

    state = get_peft_model_state_dict(transformer)
    assert not [k for k in state if ".lora.down." in k or ".lora.up." in k], "legacy LoRA keys: the H3 loader drops them"
    folder.mkdir(parents=True, exist_ok=True)
    MiniMaxH3ModularPipeline.save_lora_weights(save_directory=str(folder), transformer_lora_layers=state)
    (folder / "meta.json").write_text(json.dumps({**meta, "layers": sum(".lora_A." in k for k in state)}, indent=1))
    local = LOCAL_ADAPTER_ROOT / folder.relative_to(ADAPTER_ROOT)
    local.mkdir(parents=True, exist_ok=True)
    for name in ("pytorch_lora_weights.safetensors", "meta.json"):
        (local / name).write_bytes((folder / name).read_bytes())


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------

def read_safetensors_blobs(dataset, column, row_ids):
    """Query. Decodes safetensors blobs of a lance blob column, in the order of row_ids, as dicts of tensors."""
    import safetensors.torch

    return [safetensors.torch.load(blob.read()) for blob in dataset.take_blobs(column, ids=row_ids)]


class Records:
    """
    One rank's training items of one table: row ids, text lengths, and lazy blob reads. An item is one row (video, MAGICK
    image) or one (row, frame) pair (Pexels frames, m00917).
    """

    def __init__(self, data, mode, split, rank, world, embed_column, latent_column, lance_dir=LANCE_DIR):
        """
        Query. Reads the row list of TABLES[data] (rows with latent + embedding + the split) and this rank's share of it.

        Args:
            data (str): pexels | magick
            mode (str): video | image (image on pexels = the 7 frames of each clip)
            split (str): train | val
            rank (int), world (int): this rank's share of the rows (all of them when world = 1)
            embed_column (str), latent_column (str): blob columns
            lance_dir (str | Path): the BlueJay lance folder
        """
        import lance

        self.dataset = lance.dataset(str(Path(lance_dir) / f"{TABLES[data]}.lance"))
        self.embed_column, self.latent_column = embed_column, latent_column
        table = self.dataset.to_table(columns=[KEYS[data]], with_row_id=True,
                                      filter=f"{latent_column} IS NOT NULL AND {embed_column} IS NOT NULL AND split = '{split}'"
                                      ).to_pandas().sort_values(KEYS[data]).reset_index(drop=True)
        assert len(table), f"{TABLES[data]}: no {split} rows with {latent_column} and {embed_column}"
        order = np.random.default_rng(0).permutation(len(table))[rank::world]
        self.keys = table[KEYS[data]].to_numpy()[order].tolist()
        self.row_ids = table["_rowid"].to_numpy()[order].tolist()
        self.frames = 7 if (data == "pexels" and mode == "image") else 0
        print(f"records {TABLES[data]} {split} rank {rank}/{world}: {len(self.row_ids)} rows"
              + (f" x {self.frames} frames" if self.frames else ""), flush=True)

    def text_lengths(self):
        """Query. Text-token count of every row (reads every embedding blob's header once)."""
        return [int(e["text_token_tags"].numel()) for e in read_safetensors_blobs(self.dataset, self.embed_column, self.row_ids)]

    def batch(self, rows):
        """
        Query. One batch: latents (B, LC, LT, LH, LW), prompt_embeds (B, N, 5120), text_token_tags (N,), keys.
        A pexels image batch is ONE row's 7 frames (they share the caption), so `rows` has one entry there.
        """
        ids = [self.row_ids[i] for i in rows]
        embeds = read_safetensors_blobs(self.dataset, self.embed_column, ids)
        latents = [b["latent"] for b in read_safetensors_blobs(self.dataset, self.latent_column, ids)]
        if self.frames:
            assert len(rows) == 1, "a pexels frame batch is one clip's frames"
            latents = latents[0]                      # (7, LC, 1, LH, LW)
            embeds = embeds * latents.shape[0]
        else:
            latents = torch.stack(latents)  # (B, LC, LT, LH, LW): video (1, 24, 42, 32, 32), MAGICK (B, 24, 1, 32, 32)
        tags = embeds[0]["text_token_tags"]
        assert all(torch.equal(e["text_token_tags"], tags) for e in embeds), "a batch must share its text layout"
        prompt = torch.cat([e["prompt_embeds"] for e in embeds])  # (B, N, 5120)
        return dict(latents=latents, prompt_embeds=prompt, text_token_tags=tags, keys=[self.keys[i] for i in rows])


def batch_plan(records, mode, data, batch_size, seed):
    """Query. This rank's list of row-index batches for one pass: video 1 row; pexels frames 1 clip; magick by text length."""
    if mode == "video" or records.frames:
        order = np.random.default_rng(seed).permutation(len(records.row_ids)).tolist()
        return [[i] for i in order]
    return group_by_length(records.text_lengths(), batch_size, seed)


def prefetch(records, plan):
    """Command. Starts a daemon thread that reads the batches of plan ahead into a queue; yields them in order."""
    out = queue.Queue(PREFETCH)

    def work():
        for rows in plan:
            out.put(records.batch(rows))
        out.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := out.get()) is not None:
        yield item


# ---------------------------------------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------------------------------------

def forward_loss(transformer, batch, audio, base, generator, device, layouts):
    """
    Near-pure function (advances generator; runs the model). Video flow loss of one batch at one base noise level.

    Args:
        transformer (nn.Module)
        batch (dict): latents (B, LC, LT, LH, LW), prompt_embeds (B, N, 5120), text_token_tags (N,)
        audio (Tensor): (AUDIO_CHANNELS, 32, A) silent audio latent for this mode
        base (float): unshifted grid point in [0, 1]; sigma = shift(base)
        generator (torch.Generator): on device
        device (torch.device)
        layouts (dict): cache of Layout by (N, LT, A)

    Returns:
        Tensor: scalar video MSE (float32)
    """
    latents = batch["latents"].to(device)
    lt, a = latents.shape[2], audio.shape[-1]
    key = (int(batch["text_token_tags"].numel()), lt, a)
    if key not in layouts:
        layouts[key] = Layout(batch["text_token_tags"], lt, a, device)
    layout = layouts[key]
    video_sigma, audio_sigma = paired_sigmas(base)
    video_t, audio_t = 1.0 - video_sigma, 1.0 - audio_sigma
    x0 = video_rows(latents)                                                         # (B, rows, 96)
    a0 = einx.rearrange("R C -> B R C", audio_rows(audio).to(device), B=x0.shape[0])  # (B, C·A, 32)
    noise = torch.randn(x0.shape, generator=generator, device=device)
    audio_noise = torch.randn(a0.shape, generator=generator, device=device)
    timestep, timestep_indices = layout.row_timesteps(video_t, audio_t)
    video_pred, _ = transformer(hidden_states=noise_sample(x0, noise, video_t).to(DTYPE),
                                audio_hidden_states=noise_sample(a0, audio_noise, audio_t).to(DTYPE),
                                encoder_hidden_states=batch["prompt_embeds"].to(device, DTYPE), timestep=timestep,
                                timestep_indices=timestep_indices, return_dict=False, **layout.tensors)
    return torch.nn.functional.mse_loss(video_pred.float(), flow_target(x0, noise))


def all_reduce_grads(params, world):
    """Command. Averages the gradients of params over all ranks (one flat all-reduce)."""
    if world == 1:
        return
    grads = [p.grad for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()


def validate(transformer, records, audio, device, layouts, world):
    """
    Query (runs the model without gradients). Mean validation loss over this rank's val items at VAL_BASES with fixed
    noise, averaged over ranks. Deterministic, so successive values are comparable.
    """
    total, count = torch.zeros((), device=device), torch.zeros((), device=device)
    with torch.no_grad():
        for i in range(len(records.row_ids)):
            batch = records.batch([i])
            for k, base in enumerate(VAL_BASES):
                generator = torch.Generator(device=device).manual_seed(1000 * i + k)
                total += forward_loss(transformer, batch, audio, base, generator, device, layouts)
                count += 1
    if world > 1:
        dist.all_reduce(total)
        dist.all_reduce(count)
    return float(total / count.clamp(min=1))


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def write_status(name, text):
    """Command. One-line STATUS file of adapter `name` (and the shared STATUS.txt) for the coordinator."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%I:%M:%S %p')} {name}: {text}\n"
    (STATUS_DIR / f"STATUS_{name}.txt").write_text(line)
    (STATUS_DIR / "STATUS.txt").write_text(line)


def train(name, mode="video", data="pexels", bake_list="", rank=16, alpha=16, lr=2e-4, weight_decay=0.01, grad_clip=1.0,
          batch_size=16, max_steps=100000, save_every=100, val_every=100, seed=0, embed_column=COLUMNS["embed"],
          latent_column=None, lance_dir=str(LANCE_DIR)):
    """
    Command. Trains adapter `name` (run under torchrun; one process per GPU). Saves to models/h3magic/<name>/step_XXXXX/
    and models/h3magic/<name>/final/, logs log.csv, writes STATUS, and stops when STOP_<name> appears or at max_steps.

    Args:
        name (str): adapter name (pexels | imagegen | graymagic)
        mode (str): video | image
        data (str): pexels | magick
        bake_list (str): comma-separated adapters to bake in first, e.g. "pexels,imagegen"
        rank, alpha (int): LoRA
        lr, weight_decay, grad_clip (float): AdamW
        batch_size (int): MAGICK images per batch per GPU, at most (one text length per batch). Pexels frames: one clip's 7
            frames (3.6 s/step, 66.7 GiB); video: 1 clip (14.2 s/step, 74.5 GiB), measured on A100 80 GB
        max_steps, save_every, val_every, seed (int)
        embed_column (str): text embedding blob column
        latent_column (str | None): default by mode/data from COLUMNS
        lance_dir (str): BlueJay lance folder (a toy folder for tests)
    """
    dist.init_process_group("nccl")
    rank_id, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    latent_column = latent_column or COLUMNS["video" if mode == "video" else ("frames" if data == "pexels" else "image")]
    bakes = list(bake_list) if isinstance(bake_list, (list, tuple)) else [b for b in str(bake_list).split(",") if b]  # Fire turns "a,b" into a tuple
    out = ADAPTER_ROOT / name
    stop_file = STATUS_DIR / f"STOP_{name}"
    meta = dict(name=name, mode=mode, data=data, baked=bakes, rank=rank, alpha=alpha, lr=lr, world=world,
                video_frames=VIDEO_FRAMES if mode == "video" else 1)

    audio = silent_audio(device, audio_latents_for(VIDEO_FRAMES) if mode == "video" else IMAGE_AUDIO_LATENTS)
    # Stagger the 66 GB loads two ranks at a time: page cache + per-rank staging stays far from the RAM guard's 75%
    for turn in range(0, world, 2):
        if rank_id // 2 == turn // 2:
            transformer = build_transformer(device, bakes, rank, alpha)
        dist.barrier()
    transformer.train()
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS))
    train_records = Records(data, mode, "train", rank_id, world, embed_column, latent_column, lance_dir)
    val_records = Records(data, mode, "val", rank_id, world, embed_column, latent_column, lance_dir)
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + rank_id)
    draws = np.random.default_rng(seed)  # same base noise level on every rank per step
    layouts, loss_ema, val_loss = {}, None, None
    if rank_id == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        log = open(out / "log.csv", "a")
        log.write("step,epoch,base,video_sigma,loss,loss_ema,val_loss,grad_norm,seconds,time\n")
    step, epoch, started = 0, 0, time.time()
    while step < max_steps and not stop_file.exists():
        for batch in prefetch(train_records, batch_plan(train_records, mode, data, batch_size, seed + epoch)):
            base = float(draws.random())
            loss = forward_loss(transformer, batch, audio, base, generator, device, layouts)
            loss.backward()
            all_reduce_grads(params, world)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            value = torch.tensor(float(loss.detach()), device=device)
            dist.all_reduce(value)
            value = float(value) / world
            loss_ema = ema(loss_ema, value)
            step += 1
            if step % val_every == 0 or step == 1:
                transformer.eval()
                val_loss = validate(transformer, val_records, audio, device, layouts, world)
                transformer.train()
            if rank_id == 0:
                seconds = (time.time() - started) / step
                log.write(f"{step},{epoch},{base:.4f},{paired_sigmas(base)[0]:.4f},{value:.5f},{loss_ema:.5f},"
                          f"{'' if val_loss is None else f'{val_loss:.5f}'},{grad_norm:.4f},{seconds:.2f},{time.time():.0f}\n")
                log.flush()
                print(f"{name} step {step} ep {epoch} sigma {paired_sigmas(base)[0]:.3f} loss {value:.4f} ema {loss_ema:.4f} "
                      f"val {val_loss} |g| {grad_norm:.3f} {seconds:.1f} s/step "
                      f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB", flush=True)
                write_status(name, f"step {step} epoch {epoch} loss_ema {loss_ema:.5f} val {val_loss} {seconds:.1f} s/step")
                if step % save_every == 0:
                    save_adapter(transformer, out / f"step_{step:05d}", {**meta, "step": step, "val_loss": val_loss})
            stop = torch.tensor(int(stop_file.exists() or step >= max_steps), device=device)
            dist.all_reduce(stop)
            if int(stop):
                break
        epoch += 1
    if rank_id == 0:
        save_adapter(transformer, out / "final", {**meta, "step": step, "val_loss": val_loss})
        write_status(name, f"FINISHED at step {step}, val {val_loss}, saved {out / 'final'}")
        print(f"{name}: saved final adapter at step {step} to {out / 'final'}", flush=True)
    dist.destroy_process_group()


def low_priority_jobs(text, gpus):
    """
    Pure function. (gpu, pid) of the low-priority jobs listed in agents/.scratchpad/fastdec_h3/GPUS.txt (GPU policy:
    lines "GPU <n> pid <pid> low-priority") on the wanted GPUs, minus the ones a "killed GPU <n> pid <pid>" line records.

    Examples:
        >>> low_priority_jobs("GPU 5 pid 42 low-priority\\nGPU 7 pid 43 low-priority\\nGPU 3 pid 9 low-priority\\nkilled GPU 3 pid 9 by x", [3, 5])
        [(5, 42)]
    """
    import re

    killed = set(re.findall(r"killed GPU (\d+) pid (\d+)", text))
    return [(int(g), int(p)) for g, p in re.findall(r"^GPU (\d+) pid (\d+) low-priority", text, re.M)
            if int(g) in gpus and (g, p) not in killed]


def preempt(gpus, name):
    """
    Command. GPU policy (user m00937): SIGTERM each low-priority job listed on our GPUs, SIGKILL it after 30 s, and log
    each kill in the list file. Kills only listed pids.
    """
    import signal

    listing = ROOT / "agents/.scratchpad/fastdec_h3/GPUS.txt"
    if not listing.exists():
        return
    for gpu, pid in low_priority_jobs(listing.read_text(), gpus):
        if not Path(f"/proc/{pid}").exists():
            continue
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(1)
        if Path(f"/proc/{pid}").exists():
            os.kill(pid, signal.SIGKILL)
        with open(listing, "a") as f:
            f.write(f"killed GPU {gpu} pid {pid} by H3Train/{name} {time.strftime('%I:%M %p')} ET\n")
        print(f"preempted low-priority pid {pid} on GPU {gpu}", flush=True)


def launch(name, gpus="0,1,2,3,4,5", **kwargs):
    """
    Command. Claims gpus in agents/.scratchpad/h3train/GPUS.txt, preempts low-priority jobs listed on them (GPU policy),
    then runs `train` under torchrun on gpus (default 0–5: GPUs 6–7 stay for inference, m00912), tee'd to
    agents/.scratchpad/logs/h3train_<name>.log.
    """
    gpus = ",".join(map(str, gpus)) if isinstance(gpus, (list, tuple)) else str(gpus)  # Fire turns "0,1" into a tuple
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "GPUS.txt").write_text(f"H3Train {name}: GPUs {gpus} from {time.strftime('%I:%M %p')} ET\n")
    preempt([int(g) for g in gpus.split(",")], name)
    args = [f"--{k}={','.join(map(str, v)) if isinstance(v, (list, tuple)) else v}" for k, v in kwargs.items()]
    log = ROOT / f"agents/.scratchpad/logs/h3train_{name}.log"
    command = (f"CUDA_VISIBLE_DEVICES={gpus} {sys.executable} -m torch.distributed.run --standalone "
               f"--nproc_per_node={len(gpus.split(','))} {__file__} train {name} {' '.join(args)} 2>&1 | tee -a {log}")
    print(command, flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True)


def probe(mode="video", steps=4, batch=7, rank=16, text_tokens=300):
    """
    Command. Measures s/step and peak GPU memory of one GPU at the exact training shapes with random latents and random
    text embeddings (no tables needed). video: (1, 24, 42, 32, 32); image: (batch, 24, 1, 32, 32).

    Args:
        mode (str): video | image
        steps (int): timed steps after one warm-up step
        batch (int): images per step in image mode
        rank (int): LoRA rank
        text_tokens (int): text rows per sample (an H3 caption is a few hundred tokens)
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    audio = silent_audio(device, audio_latents_for(VIDEO_FRAMES) if mode == "video" else IMAGE_AUDIO_LATENTS)
    transformer = build_transformer(device, [], rank, rank)
    transformer.train()
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    b, lt = (1, 42) if mode == "video" else (batch, 1)
    fake = dict(latents=torch.randn(b, LATENT_CHANNELS, lt, LATENT_SIDE, LATENT_SIDE),
                prompt_embeds=torch.randn(b, text_tokens, TEXT_DIM) * 0.02,
                text_token_tags=torch.full((text_tokens,), TEXT_TAG, dtype=torch.int64))
    generator, layouts = torch.Generator(device=device).manual_seed(0), {}
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(steps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        loss = forward_loss(transformer, fake, audio, 0.5, generator, device, layouts)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"probe {mode} step {i}: {times[-1]:.2f} s, loss {float(loss):.4f}, peak "
              f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    rows = layouts[next(iter(layouts))]
    result = dict(mode=mode, batch=b, rank=rank, text_tokens=text_tokens, video_rows=rows.num_video_rows,
                  audio_rows=rows.num_audio_rows, seconds_per_step=float(np.mean(times[1:])),
                  peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                  reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30, gpu=torch.cuda.get_device_name(device))
    print(json.dumps(result), flush=True)
    return result


def sign_probe(data="pexels", latent_column=COLUMNS["video"], embed_column=COLUMNS["embed"], split="train", lance_dir=str(LANCE_DIR)):
    """
    Command. Checks the training target on a REAL latent against the frozen base model: at three noise levels the base
    prediction must fit x_0 − noise better than its negation and better than zero (MinimaxH3's probe). Raises if not.
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    mode = "video" if latent_column == COLUMNS["video"] else "image"
    audio = silent_audio(device, audio_latents_for(VIDEO_FRAMES) if mode == "video" else IMAGE_AUDIO_LATENTS)
    from diffusers import MiniMaxH3Transformer3DModel

    transformer = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR / "transformer", torch_dtype=DTYPE,
                                                              device_map={"": str(device)}).eval()
    records = Records(data, mode, split, 0, 1, embed_column, latent_column, lance_dir)
    batch = records.batch([0])
    layouts, verdicts = {}, []
    for base in [sigma / (VIDEO_SHIFT - (VIDEO_SHIFT - 1) * sigma) for sigma in PROBE_SIGMAS]:  # unshifted, so video σ = 0.2, 0.5, 0.8
        latents = batch["latents"].to(device)
        x0 = video_rows(latents)
        generator = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(x0.shape, generator=generator, device=device)
        key = (int(batch["text_token_tags"].numel()), latents.shape[2], audio.shape[-1])
        layouts.setdefault(key, Layout(batch["text_token_tags"], latents.shape[2], audio.shape[-1], device))
        layout = layouts[key]
        video_sigma, audio_sigma = paired_sigmas(base)
        a0 = einx.rearrange("R C -> B R C", audio_rows(audio).to(device), B=x0.shape[0])
        timestep, timestep_indices = layout.row_timesteps(1 - video_sigma, 1 - audio_sigma)
        with torch.no_grad():
            pred, _ = transformer(hidden_states=noise_sample(x0, noise, 1 - video_sigma).to(DTYPE),
                                  audio_hidden_states=noise_sample(a0, torch.randn_like(a0), 1 - audio_sigma).to(DTYPE),
                                  encoder_hidden_states=batch["prompt_embeds"].to(device, DTYPE), timestep=timestep,
                                  timestep_indices=timestep_indices, return_dict=False, **layout.tensors)
        target = flow_target(x0, noise)
        right, wrong = float(((pred.float() - target) ** 2).mean()), float(((pred.float() + target) ** 2).mean())
        zero = float((target ** 2).mean())
        verdicts.append(right < wrong and right < zero)
        print(f"sigma {video_sigma:.3f}: mse(+target) {right:.4f}  mse(-target) {wrong:.4f}  mse(0) {zero:.4f}", flush=True)
    assert all(verdicts), "the base model does not predict x_0 - noise: the target or the latent normalization is wrong"
    print("target sign and latent normalization CONFIRMED on a real latent", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(train=train, launch=launch, probe=probe, sign_probe=sign_probe))
