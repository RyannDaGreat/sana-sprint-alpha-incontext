# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "torch", "einx", "numpy", "lpips", "imageio-ffmpeg"]
# ///
"""
Writes children/<clip_id>/{clip.mp4, fields.json}: the fd_renders rows of one candidate. NEEDS A GPU (run it through Bluejay on an idle one).
Decodes the 20 held-out val clips (the journal page's 6 clips are among them) with the candidate in fp16 autocast, exactly as score.py does,
writes H.264 CRF 12, 24 fps, yuv420p mp4, and the clip's PSNR, SSIM and LPIPS (alex) against the full decode. SSIM and LPIPS use every
FD_SCORE_STRIDE-th frame (default 4), as the scorer does. The candidate is loaded with fastdec_ga/score.py load_candidate.
"""
import json
import subprocess
import sys
from pathlib import Path

import fire
from fd_lib import FD_GA, FD_ROOT

FPS, CRF = 24, 12  # as make_videos.py wrote the journal's mp4s
DEFAULT_STRIDE = 4


def write_mp4(frames, path):
    """
    Command. Writes (T, H, W, 3) uint8 RGB frames as an H.264 yuv420p mp4 (CRF 12, 24 fps, fast start) at path. Runs ffmpeg.

    Examples:
        >>> import numpy as np, tempfile
        >>> p = Path(tempfile.mkdtemp()) / "t.mp4"; write_mp4(np.zeros((4, 16, 16, 3), np.uint8), p); p.stat().st_size > 0
        True
    """
    import imageio_ffmpeg
    import numpy as np
    t, h, w, _ = frames.shape
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
               "-i", "-", "-c:v", "libx264", "-crf", str(CRF), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=True)


def main(sample, device="cuda", stride=DEFAULT_STRIDE):
    """
    Command. Renders the 20 val clips with the method in SAMPLE/input.json (method_id, model_dir) and writes the children.

    Args:
        sample (str): The $SAMPLE folder
        device (str): torch device; cuda in practice (fp16 autocast is off on cpu, so a cpu run is a slow check)
        stride (int): SSIM and LPIPS use every stride-th frame
    """
    import einx
    import lpips
    import numpy as np
    import torch
    sys.path.insert(0, str(FD_GA))
    import score  # fastdec_ga/score.py: load_candidate, compare, to_frames, DATA

    sample = Path(sample)
    job = json.loads((sample / "input.json").read_text())
    method_id, model_dir = job["method_id"], FD_ROOT / job["model_dir"]
    mod, model = score.load_candidate(model_dir, device)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device)
    for path in sorted(score.DATA.glob("val_*.npz")):
        d = np.load(path)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            cand = mod.decode(model, torch.from_numpy(d["latent"]).float().to(device)).float().clamp(0, 1)
        u8 = einx.rearrange("f c h w -> f h w c", (cand * 255).round().byte()).cpu().numpy()  # (141, 512, 512, 3) RGB
        vs_full = score.compare(score.to_frames(u8, device), score.to_frames(d["teacher"], device), lp, stride)
        clip_id = path.stem.removeprefix("val_")
        child = sample / "children" / clip_id
        child.mkdir(parents=True)
        write_mp4(u8, child / "clip.mp4")
        (child / "fields.json").write_text(json.dumps({
            "method_id": method_id, "clip_id": clip_id, "n_frames": len(u8), "fps": float(FPS),
            "psnr_vs_full": vs_full["psnr"], "ssim_vs_full": vs_full["ssim"], "lpips_vs_full": vs_full["lpips"]}))
        print(f"{method_id} {clip_id}: psnr {vs_full['psnr']:.2f} ssim {vs_full['ssim']:.3f} lpips {vs_full['lpips']:.3f}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
