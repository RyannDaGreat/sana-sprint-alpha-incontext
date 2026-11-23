"""
Synthetic moving cut-outs for the video split readout of Alpha Video v2 (manifest §23, item 4). MAGICK has only still
RGBA images, but the split adapter drifts in video (§22: later frames' white/black backgrounds fade toward gray, alpha
0.2-0.3 left in the background). A still RGBA moved along a smooth random path (pan, zoom, small rotation) gives a video
whose composites on gray, white and black are exact, so the split heads can be taught at every latent frame.

All functions are numpy/cv2 only.
"""
import math

import numpy as np

LEVELS = (0.5, 1.0, 0.0)  # gray, white, black: h3magic/split.py SPLIT_BACKGROUNDS order
SCALE_RANGE = (0.55, 0.95)  # the RGBA's side as a fraction of the output side, at the path's start and end
SHIFT_MAX = 0.18  # largest center offset from the middle, fraction of the output side
ANGLE_MAX = 8.0  # largest rotation, degrees


def ease(u):
    """
    Pure function. Cosine ease-in-out of u in [0, 1] (smooth start and stop).

    Examples:
        >>> [round(float(ease(u)), 3) for u in (0.0, 0.25, 0.5, 1.0)]
        [0.0, 0.146, 0.5, 1.0]
    """
    return 0.5 - 0.5 * np.cos(np.pi * np.asarray(u, dtype=np.float64))


def path_params(rng):
    """
    Near-pure function (advances rng). One random motion path: start/end scale, center and angle.

    Returns:
        dict: scale (2,), center (2, 2) as (start, end) x (x, y) in output-side fractions around 0.5, angle (2,) degrees

    Examples:
        >>> p = path_params(np.random.default_rng(0)); sorted(p), p["center"].shape
        (['angle', 'center', 'scale'], (2, 2))
    """
    return dict(scale=rng.uniform(*SCALE_RANGE, size=2), center=0.5 + rng.uniform(-SHIFT_MAX, SHIFT_MAX, size=(2, 2)),
                angle=rng.uniform(-ANGLE_MAX, ANGLE_MAX, size=2))


def affine_path(params, frames, source, size):
    """
    Pure function. Per-frame 2x3 affine matrices mapping source pixels (source x source RGBA) to output pixels (size x
    size), easing from the path's start to its end.

    Returns:
        np.ndarray: (frames, 2, 3) float64

    Examples:
        >>> p = dict(scale=np.array([1.0, 1.0]), center=np.array([[0.5, 0.5], [0.5, 0.5]]), angle=np.array([0.0, 0.0]))
        >>> m = affine_path(p, 3, 1024, 768); m.shape, np.round(m[0] @ [512, 512, 1], 6).tolist()  # the source center lands mid-frame
        ((3, 2, 3), [384.0, 384.0])
    """
    u = ease(np.linspace(0, 1, frames))[:, None]
    scale = (params["scale"][0] + (params["scale"][1] - params["scale"][0]) * u[:, 0]) * size / source
    center = (params["center"][0] + (params["center"][1] - params["center"][0]) * u) * size
    angle = np.deg2rad(params["angle"][0] + (params["angle"][1] - params["angle"][0]) * u[:, 0])
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    half = source / 2
    tx = center[:, 0] - (cos * half - sin * half)
    ty = center[:, 1] - (sin * half + cos * half)
    return np.stack([np.stack([cos, -sin, tx], -1), np.stack([sin, cos, ty], -1)], 1)


def warp_rgba(rgba, matrix, size):
    """
    Pure function. Warps a straight-alpha RGBA image by matrix into a size x size frame (premultiplied bilinear, so no
    dark fringes; area-averaged first when shrinking by 2x or more). Outside the image is fully clear.

    Args:
        rgba (np.ndarray): (S, S, 4) uint8 straight alpha
        matrix (np.ndarray): (2, 3) source -> output pixels
        size (int)

    Returns:
        np.ndarray: (size, size, 4) float32 straight alpha in [0, 1]

    Examples:
        >>> rgba = np.zeros((4, 4, 4), np.uint8); rgba[..., 0] = 255; rgba[..., 3] = 255
        >>> out = warp_rgba(rgba, np.array([[1.0, 0, 2], [0, 1.0, 2]]), 8)  # opaque red square shifted to (2..6)
        >>> out[4, 4].tolist(), out[0, 0, 3].item()
        ([1.0, 0.0, 0.0, 1.0], 0.0)
    """
    import cv2
    f = rgba.astype(np.float32) / 255
    premult = np.concatenate([f[..., :3] * f[..., 3:], f[..., 3:]], -1)
    out = cv2.warpAffine(premult, matrix.astype(np.float64), (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = out[..., 3:]
    rgb = np.where(alpha > 1e-6, out[..., :3] / np.maximum(alpha, 1e-6), 0)
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)], -1)


def moving_composites(rgba, matrices, size, levels=LEVELS):
    """
    Pure function. The moving cut-out composited on each background level, 8 bit.

    Args:
        rgba (np.ndarray): (S, S, 4) uint8 straight alpha
        matrices (np.ndarray): (T, 2, 3) from affine_path
        size (int)
        levels (tuple[float]): background levels, K of them

    Returns:
        np.ndarray: (K, T, size, size, 3) uint8

    Examples:
        >>> rgba = np.zeros((4, 4, 4), np.uint8); rgba[..., 1] = 255; rgba[..., 3] = 255
        >>> out = moving_composites(rgba, np.array([[[1.0, 0, 2], [0, 1.0, 2]]]), 8)
        >>> out.shape, out[:, 0, 4, 4].tolist(), out[:, 0, 0, 0, 0].tolist()  # green subject; gray, white, black corner
        ((3, 1, 8, 8, 3), [[0, 255, 0], [0, 255, 0], [0, 255, 0]], [128, 255, 0])
    """
    frames = [warp_rgba(rgba, m, size) for m in matrices]
    out = np.empty((len(levels), len(frames), size, size, 3), np.uint8)
    for t, f in enumerate(frames):
        for k, level in enumerate(levels):
            out[k, t] = np.rint((f[..., :3] * f[..., 3:] + level * (1 - f[..., 3:])) * 255)
    return out
