"""
MiniMax H3 engine of the H3 Magic project (manifest §22): localization, the VAE (video clips and single images),
the text encoder, and generation of 141-frame videos or single images with a weighted sum of LoRA adapters.

The weights load from the node-local copy LOCAL_DIR (fast disk, /models/ convention). When it is missing, localize()
copies the durable copy in the repo (DURABLE_DIR) there first. Never the Hugging Face Hub at load time: HF_HUB_OFFLINE.

Geometry: 24 fps, 512 x 512, VIDEO_FRAMES = 141 (5.875 s, the longest 17n + 5 length inside the first 6 s).
Latents are normalized exactly like MinimaxH3's train_lora.encode_video_target (the training target recipe):
ImageNet-normalized pixels -> vae.encode (float32, untiled) -> posterior.sample(seed) -> (z - latents_mean) / latents_std.
    video 141 frames -> (24, 42, 32, 32) (LC, LT, LH, LW);  one image -> (24, 1, 32, 32)

Images (frames = 1) need three changes to the released code, which only accepts 5-15 s (allow_single_frame()):
a 1-frame request is not rounded to 17n + 5; it has 1 latent frame (the VAE encodes a single frame with its spatial
encoder alone, as it does for fl2va keyframes; measured: this latent is within 0.075 mean abs of a clip's first latent);
and the decoder, which needs >= 7 latent frames, decodes the latent repeated 7 times and keeps pixel frame 0.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # A stale index would otherwise download ~200 GB from the Hub without a word

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = Path("/models/h3magic/MiniMax-H3")  # Node-local fast copy: the one sanctioned absolute path (/models/)
DURABLE_DIR = ROOT / "models/h3magic/MiniMax-H3"  # Durable copy in the repo (gitignored)
FPS = 24
VIDEO_FRAMES = 141  # 17 * 8 + 5: 5.875 s
SIZE = 512
DTYPE = torch.bfloat16
ENCODE_SEED = 20261122  # Seed of every sampled VAE posterior, so a clip encodes the same twice
IMAGE_DECODE_REPEAT = 7  # Fewest latent frames the released decoder takes; frame 0 of 7 repeats: 36.4 dB round trip (frame 0 of a real clip: 38.2 dB)


def repoint_index(model_dir):
    """
    Command. Points every component of model_dir/modular_model_index.json at model_dir itself (H3 ships them pointing
    at the Hub id). Writes a new file and renames it over the old one, so a hard-linked index elsewhere is not changed.

    Returns:
        int: Number of component specs repointed
    """
    index = Path(model_dir) / "modular_model_index.json"
    spec = json.loads(index.read_text())
    entries = [v[2] for v in spec.values() if isinstance(v, list) and len(v) == 3 and isinstance(v[2], dict)
               and "pretrained_model_name_or_path" in v[2]]
    for entry in entries:
        entry["pretrained_model_name_or_path"] = str(model_dir)
    temp = index.with_suffix(".tmp")
    temp.write_text(json.dumps(spec, indent=2))
    temp.rename(index)
    return len(entries)


def localize():
    """
    Command. Returns LOCAL_DIR with a repointed index. When it is missing, copies DURABLE_DIR there first (temp dir +
    atomic rename, so 8 workers starting at once are safe). Raises when neither copy exists.

    Returns:
        Path
    """
    if not (LOCAL_DIR / "model_index.json").exists():
        if not (DURABLE_DIR / "model_index.json").exists():
            raise FileNotFoundError(f"No H3 weights at {LOCAL_DIR} or {DURABLE_DIR}: hf download MiniMaxAI/MiniMax-H3 --local-dir {LOCAL_DIR}")
        LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=LOCAL_DIR.parent, prefix=".copy_"))
        print(f"h3: copying {DURABLE_DIR} -> {LOCAL_DIR} (slow network disk)", flush=True)
        subprocess.run(["rsync", "-a", "--info=progress2", "--exclude", "transformer_ref", f"{DURABLE_DIR}/", f"{temp}/"], check=True)
        try:
            temp.rename(LOCAL_DIR)
        except OSError:  # Another worker renamed its copy first: use that one
            shutil.rmtree(temp)
    if any(v[2].get("pretrained_model_name_or_path") != str(LOCAL_DIR)
           for v in json.loads((LOCAL_DIR / "modular_model_index.json").read_text()).values()
           if isinstance(v, list) and len(v) == 3 and isinstance(v[2], dict) and "pretrained_model_name_or_path" in v[2]):
        print(f"h3: repointed {repoint_index(LOCAL_DIR)} component specs to {LOCAL_DIR}", flush=True)
    return LOCAL_DIR


def require_components(pipe):
    """Query. Raises unless every component the pipeline declared loaded (load_components only warns). Returns pipe."""
    if pipe.null_component_names:
        raise RuntimeError(f"{type(pipe).__name__} did not load {pipe.null_component_names} from {LOCAL_DIR}")
    return pipe


def allow_single_frame():
    """
    Command. Patches diffusers' H3 layout step so that num_frames = 1 is a legal request with 1 latent frame (an image).
    Every other frame count behaves exactly as released. Idempotent.
    """
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3 import before_denoise, modular_pipeline
    if getattr(before_denoise, "_h3magic_single_frame", False):
        return
    align, latent_frames = before_denoise.align_num_frames, before_denoise.video_latent_num_frames
    before_denoise.align_num_frames = lambda n, f, l: 1 if n == 1 else align(n, f, l)
    before_denoise.video_latent_num_frames = lambda n, f, l: 1 if n == 1 else latent_frames(n, f, l)
    modular_pipeline.MiniMaxH3ModularPipeline.min_duration = property(lambda self: 1 / FPS)
    decode_video = AutoencoderKLMiniMaxH3._decode

    def decode_single_frame(self, z):
        """Query. The released _decode needs >= 7 latent frames; one latent frame is repeated 7x and the first pixel frame kept."""
        if z.shape[2] != 1:
            return decode_video(self, z)
        return decode_video(self, z.repeat(1, 1, IMAGE_DECODE_REPEAT, 1, 1))[:, :, :1]

    AutoencoderKLMiniMaxH3._decode = decode_single_frame
    before_denoise._h3magic_single_frame = True


def load_pipeline(device="cuda"):
    """
    Command. The t2va pipeline without its text encoder (it takes prompt_embeds), every component on `device`.
    ~77 GB of an 80 GB GPU (transformer 66 GB + VAEs).

    Returns:
        MiniMaxH3ModularPipeline
    """
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    allow_single_frame()
    blocks = MiniMaxH3Blocks().get_workflow("t2va")
    blocks.sub_blocks.pop("text_encoder")
    pipe = blocks.init_pipeline(str(localize()))
    pipe.load_components(dtype=DTYPE)
    require_components(pipe)
    for component in pipe.components.values():
        if hasattr(component, "to"):
            component.to(device)
    return pipe


def load_text_encoder(device="cuda"):
    """Command. The t2va text-encoder sub-pipeline (Qwen3-VL-32B, ~67 GB) on `device`. Returns it."""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    block = MiniMaxH3Blocks().get_workflow("t2va").sub_blocks.pop("text_encoder")
    pipe = block.init_pipeline(str(localize()))
    pipe.load_components(dtype=DTYPE)
    require_components(pipe)
    for component in pipe.components.values():
        if hasattr(component, "to"):
            component.to(device)
    return pipe


def load_vae(device="cuda"):
    """Command. The H3 video VAE alone (~10 GB), float32 for training targets (as encode_video_target), on `device`."""
    from diffusers import AutoencoderKLMiniMaxH3
    allow_single_frame()
    return AutoencoderKLMiniMaxH3.from_pretrained(str(localize()), subfolder="vae", torch_dtype=torch.float32).to(device).eval()


@torch.no_grad()
def embed(text_encoder, prompt):
    """
    Query (runs the GPU model). H3's text conditioning of one prompt: hidden state 50 of Qwen3-VL.

    Returns:
        dict: prompt_embeds (1, N, 5120) bfloat16 CPU, text_token_tags (N,) int64 CPU
    """
    out = text_encoder(prompt=prompt, output=["prompt_embeds", "text_token_tags"])
    return {"prompt_embeds": out["prompt_embeds"].to(DTYPE).cpu().contiguous(), "text_token_tags": out["text_token_tags"].cpu().contiguous()}


def latent_stats(vae):
    """Query. The VAE's per-channel latents_mean and latents_std as (1, LC, 1, 1, 1) float32 CPU tensors."""
    shape = (1, -1, 1, 1, 1)
    return torch.tensor(vae.config.latents_mean).view(shape), torch.tensor(vae.config.latents_std).view(shape)


@torch.no_grad()
def encode(vae, frames, seed=ENCODE_SEED):
    """
    Query (runs the GPU model). Normalized H3 latents of a clip or of single images, the training-target recipe.

    Args:
        vae (AutoencoderKLMiniMaxH3): From load_vae
        frames (np.ndarray): uint8 RGB. (T, H, W, 3) one clip, T = 17n + 5 (141) or 1; or (B, 1, H, W, 3) B single images
        seed (int): Posterior sampling seed

    Returns:
        torch.Tensor: float32 CPU. (LC, LT, LH, LW): a clip of 141 frames at 512² -> (24, 42, 32, 32), 1 frame -> (24, 1, 32, 32);
        for (B, 1, H, W, 3) input: (B, LC, 1, LH, LW)
    """
    import einx
    vae.use_tiling = False  # Encode untiled: 5.1 s per 141-frame clip vs 14.7 s tiled; 36.3 vs 38.1 dB after a tiled decode (A100)
    batched = frames.ndim == 5
    pixels = torch.from_numpy(np.ascontiguousarray(frames if batched else frames[None])).to(vae.device)
    pixels = einx.rearrange("B T H W C -> B C T H W", pixels).float().div(255)
    pixels = (pixels - torch.tensor((0.485, 0.456, 0.406), device=vae.device).view(1, 3, 1, 1, 1)) / torch.tensor((0.229, 0.224, 0.225), device=vae.device).view(1, 3, 1, 1, 1)
    posterior = vae.encode(pixels, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator(device=vae.device).manual_seed(seed)).float().cpu()
    mean, std = latent_stats(vae)
    latents = (latents - mean) / std
    return latents if batched else latents[0]


@torch.no_grad()
def decode(vae, latents):
    """
    Query (runs the GPU model). Pixels of normalized latents (the inverse of encode, up to the VAE's loss).

    Args:
        latents (torch.Tensor): (LC, LT, LH, LW) normalized, e.g. (24, 42, 32, 32)

    Returns:
        np.ndarray: uint8 (T, H, W, 3), e.g. (141, 512, 512, 3); one latent frame -> (1, 512, 512, 3)
    """
    import einx
    vae.use_tiling = True  # The decoder needs its 256-pixel tiles: untiled it gives 26.5 dB, tiled 38.1 dB (A100, 512²)
    mean, std = latent_stats(vae)
    z = (latents[None].float() * std + mean).to(vae.device, vae.dtype)
    video = vae.decode(z, return_dict=False)[0].float()  # (1, 3, T, H, W), ImageNet-normalized
    video = video * torch.tensor((0.229, 0.224, 0.225), device=video.device).view(1, 3, 1, 1, 1) + torch.tensor((0.485, 0.456, 0.406), device=video.device).view(1, 3, 1, 1, 1)
    return einx.rearrange("1 C T H W -> T H W C", (video.clamp(0, 1) * 255).round().byte()).cpu().numpy()


def psnr(a, b):
    """
    Pure function. PSNR in dB of two uint8 arrays of one shape.

    Examples:
        >>> round(psnr(np.zeros((2, 2), np.uint8), np.full((2, 2), 255, np.uint8)), 1)
        0.0
        >>> round(psnr(np.full((4,), 100, np.uint8), np.full((4,), 110, np.uint8)), 2)
        28.13
    """
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255 ** 2 / mse))


def set_adapters(pipe, adapters):
    """
    Command, untested (no adapter exists yet). Makes the transformer W_H3 + sum(weight * delta) over `adapters`, loading each LoRA file once per process
    (peft named adapters; switching weights never reloads the base). An empty dict = stock H3.

    Args:
        pipe (MiniMaxH3ModularPipeline): From load_pipeline
        adapters (dict[str, tuple[str, float]]): name -> (adapter folder or file, weight), e.g.
            {"pexels": ("models/h3magic/pexels/step_000500", 1.0), "graymagic": (..., 1.0)}
    """
    loaded = set(getattr(pipe, "_h3magic_loaded", ()))
    for name, (path, _) in adapters.items():
        if name not in loaded:
            pipe.load_lora_weights(str(ROOT / path), adapter_name=name)
            loaded.add(name)
    pipe._h3magic_loaded = loaded
    active = {name: weight for name, (_, weight) in adapters.items() if weight != 0}
    if active:
        pipe.transformer.enable_lora() if hasattr(pipe.transformer, "enable_lora") else None
        pipe.set_adapters(list(active), list(active.values()))
    elif loaded:
        pipe.transformer.disable_lora()


@torch.no_grad()
def generate(pipe, text, seed, frames=VIDEO_FRAMES, steps=30, size=SIZE):
    """
    Command (runs the GPU model). One t2va generation from a precomputed text embedding; the audio is discarded.

    Args:
        text (dict): From embed: prompt_embeds (1, N, 5120), text_token_tags (N,)
        seed (int): Noise seed
        frames (int): 141 for a 5.875 s video, 1 for an image
        steps (int): num_inference_steps

    Returns:
        np.ndarray: uint8 (frames, size, size, 3)
    """
    device = pipe._execution_device
    result = pipe(prompt_embeds=text["prompt_embeds"].to(device), text_token_tags=text["text_token_tags"],
                  num_frames=frames, height=size, width=size, num_inference_steps=steps,
                  generator=torch.Generator().manual_seed(seed), output=["videos"], output_type="np")
    video = result["videos"][0] if isinstance(result, dict) else result[0]
    return to_uint8_frames(video)


def to_uint8_frames(video):
    """
    Pure function. The pipeline's decoded video (PIL list, float array in [0, 1] or uint8) as uint8 (T, H, W, 3).

    Examples:
        >>> to_uint8_frames(np.ones((2, 4, 4, 3), np.float32)).dtype.name, to_uint8_frames(np.ones((2, 4, 4, 3), np.float32)).max()
        ('uint8', np.uint8(255))
    """
    if isinstance(video, list):
        return np.stack([np.asarray(frame.convert("RGB")) for frame in video])
    array = video.cpu().numpy() if hasattr(video, "cpu") else np.asarray(video)
    return array if array.dtype == np.uint8 else (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
