"""
Trimask method op_chain: a randomly chosen, randomly shuffled chain of primitive operations applied literally one after the
other to a running soft mask, then cut into a trimask by two thresholds.

Primitives (each a small pure function on an (H, W) float32 mask in [0, 1]): dilate, erode, open, close (grayscale morphology),
blur, warp (smooth noise displacement), threshold (smooth step), quantize (few gray levels), noise (additive smooth noise, clipped).
The chain starts from the matte. 2-6 distinct primitives are drawn, shuffled, and run in that order, each with its own random
parameters and seed (so the recorded params replay the trimask exactly). The final mask m becomes:
    sure white (1) where m >= hi and the matte is fully opaque, sure black (0) where m <= lo and the matte is fully transparent,
    gray (0.5) elsewhere.
Sure pixels come only from the exact sure regions of the matte (restrict_to_matte at the end), so the result is consistent with the
matte (manifest §13 correctness). The unknown area is wherever the chain disagrees with the matte.

Speed: the chain runs at half resolution (WORK_PX on the longer side, never upscaled) and m is resized back for the final cut.
All sizes are fractions of the longer image side, so the method does not assume a size.
A primitive that would leave less than MIN_KEEP_FRAC of the matte's opaque (or transparent) pixels sure is not applied and is
recorded as such (an eroded-away or dilated-everywhere chain would be a blank, all-gray trimask).

Sources:
    - Yu et al., "Mask Guided Matting via Progressive Refinement Network" (MGMatting), CVPR 2021, arXiv 2012.06722, Sec. 3.1:
      guidance masks from the ground-truth alpha by random binarization, then dilation and/or erosion in random order, kernels 1..30.
    - Liu et al., "Towards Enhancing Fine-grained Details for Image Matting", WACV 2021, arXiv 2101.09095: chains of random
      erosion/dilation steps, each acting on the result of the previous one, to imitate noisy user trimaps.
    - Hendrycks et al., "AugMix", ICLR 2020, arXiv 1912.02781: operations chosen at random and composed in sequence.
"""

import cv2
import numpy as np

from sana_alpha.trimask import OPAQUE, TRANSPARENT, restrict_to_matte

UNKNOWN = 0.5
WORK_PX = 512  # The chain runs with the longer side at most this long
OP_COUNT_RANGE = (2, 6)  # Inclusive: number of distinct primitives in a chain
HI_RANGE, LO_RANGE = (0.6, 0.95), (0.05, 0.4)  # Cut levels: white needs m >= hi, black needs m <= lo
MIN_KEEP_FRAC = 0.05  # A primitive is rejected if it leaves less sure white than this fraction of the opaque pixels, or the same for black
SEED_BOUND = 2 ** 63
# Parameter ranges. Sizes are fractions of the longer image side (0.01 = 10 px on 1024), drawn log-uniformly
RADIUS_FRAC = (0.003, 0.05)  # dilate, erode
OPEN_CLOSE_RADIUS_FRAC = (0.003, 0.035)
SIGMA_FRAC = (0.003, 0.03)  # blur
WARP_AMPLITUDE_FRAC = (0.004, 0.04)  # warp: peak displacement
WARP_CELL_FRAC = (0.05, 0.3)  # warp: feature size of the displacement field
NOISE_AMOUNT = (0.15, 0.6)  # noise: peak added value
NOISE_CELL_FRAC = (0.01, 0.12)  # noise: feature size
THRESHOLD_RANGE, THRESHOLD_SOFTNESS = (0.2, 0.8), (0.02, 0.25)  # threshold: level and width of the smooth step
QUANTIZE_LEVELS = (2, 3, 4, 5)
OP_WEIGHTS = {"dilate": 1.0, "erode": 1.0, "open": 1.0, "close": 1.0, "blur": 1.5, "warp": 1.0,
              "threshold": 1.0, "quantize": 0.7, "noise": 1.0}  # Chance of being in the chain, relative


# ---------------------------------------------------------------- primitives: (H, W) float32 in [0, 1] -> same


def grow(mask, radius_px, op):
    """
    Pure function. Grayscale dilation (op = cv2.dilate) or erosion (cv2.erode) by an approximate disk of radius_px: radius_px steps of
    a 3x3 cross, then square, alternating (an octagon, fast at any radius). The border copies itself.

    Args:
        mask (np.ndarray): (H, W) float32 in [0, 1]
        radius_px (int): >= 0
        op (callable): cv2.dilate or cv2.erode

    Returns:
        np.ndarray: (H, W) float32

    Examples:
        >>> m = np.zeros((1, 9), np.float32); m[0, 4] = 1
        >>> grow(m, 2, cv2.dilate).tolist()
        [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]
        >>> grow(1 - grow(m, 2, cv2.dilate), 1, cv2.dilate).tolist()  # Dilating the hole shrinks the 5-px object to 3
        [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        >>> grow(grow(m, 2, cv2.dilate), 1, cv2.erode).tolist()
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    """
    cross, square = (cv2.getStructuringElement(s, (3, 3)) for s in (cv2.MORPH_CROSS, cv2.MORPH_RECT))
    for step in range(radius_px):
        mask = op(mask, cross if step % 2 == 0 else square, borderType=cv2.BORDER_REPLICATE)
    return mask


def smooth_noise(shape, cell_px, seed):
    """
    Near-pure function (a fresh rng from seed: deterministic). Smooth random field: coarse gaussian noise, bicubic-resized, scaled so max |value| = 1.

    Args:
        shape (tuple): (H, W)
        cell_px (float): Feature size in pixels, >= 1
        seed (int)

    Returns:
        np.ndarray: (H, W) float32 in [-1, 1]

    Examples:
        >>> f = smooth_noise((64, 96), 16, 0); f.shape, float(abs(f).max())
        ((64, 96), 1.0)
        >>> bool((f == smooth_noise((64, 96), 16, 0)).all()), bool((f == smooth_noise((64, 96), 16, 1)).all())
        (True, False)
        >>> float(abs(np.diff(f, axis=1)).max()) < 0.2  # Smooth: neighbors are close
        True
    """
    height, width = shape
    coarse_shape = (max(2, round(height / cell_px) + 1), max(2, round(width / cell_px) + 1))
    coarse = np.random.default_rng(seed).standard_normal(coarse_shape).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return (field / np.abs(field).max()).astype(np.float32)


def warp(mask, amplitude_px, cell_px, seed):
    """
    Near-pure function (deterministic in seed). Moves every pixel by a smooth random displacement of up to amplitude_px (x and y independent).

    Args:
        mask (np.ndarray): (H, W) float32 in [0, 1]
        amplitude_px (float): Peak displacement in pixels
        cell_px (float): Feature size of the displacement field
        seed (int)

    Returns:
        np.ndarray: (H, W) float32 in [0, 1]

    Examples:
        >>> m = np.zeros((64, 64), np.float32); m[16:48, 16:48] = 1
        >>> w = warp(m, 4, 24, 0)
        >>> bool(0 < abs(w - m).sum() < 400), float(w.min()) >= 0, float(w.max()) <= 1
        (True, True, True)
        >>> bool((warp(m, 0, 24, 0) == m).all())  # No displacement
        True
    """
    height, width = mask.shape
    dx, dy = (amplitude_px * smooth_noise(mask.shape, cell_px, seed + k) for k in range(2))
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(mask, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def threshold(mask, level, softness):
    """
    Pure function. Smooth step: 0 below level - softness / 2, 1 above level + softness / 2, linear between.

    Examples:
        >>> threshold(np.array([[0, 0.2, 0.5, 0.8, 1]], np.float32), 0.5, 0.4).tolist()
        [[0.0, 0.0, 0.5, 1.0, 1.0]]
        >>> threshold(np.array([[0.2, 0.9]], np.float32), 0.5, 0.02).tolist()  # Nearly a hard cut
        [[0.0, 1.0]]
    """
    return np.clip((mask - level) / softness + 0.5, 0, 1).astype(np.float32)


def quantize(mask, levels):
    """
    Pure function. Rounds to `levels` evenly spaced gray levels (2: binary at 0.5).

    Examples:
        >>> quantize(np.array([[0.1, 0.4, 0.6, 0.9]], np.float32), 2).tolist()
        [[0.0, 0.0, 1.0, 1.0]]
        >>> [round(v, 2) for v in quantize(np.array([[0.1, 0.4, 0.6, 0.9]], np.float32), 4)[0].tolist()]
        [0.0, 0.33, 0.67, 1.0]
    """
    return (np.round(mask * (levels - 1)) / (levels - 1)).astype(np.float32)


def add_noise(mask, amount, cell_px, seed):
    """
    Near-pure function (deterministic in seed). Adds smooth noise of peak `amount`, then clips to [0, 1].

    Examples:
        >>> m = np.full((32, 32), 0.5, np.float32)
        >>> n = add_noise(m, 0.3, 8, 0)
        >>> bool(0.19 < n.min() < 0.51), bool(n.max() <= 0.8 + 1e-6), bool(n.min() >= 0 and n.max() <= 1)
        (True, True, True)
        >>> float(add_noise(np.ones((8, 8), np.float32), 0.5, 4, 0).max())  # Clipped
        1.0
    """
    return np.clip(mask + amount * smooth_noise(mask.shape, cell_px, seed), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- chain


def apply_op(mask, op):
    """
    Pure function (deterministic: noise ops carry their seed). Runs one op record on mask. Sizes in the record are fractions of
    the longer side of mask; a radius or sigma below one pixel rounds up to one pixel.

    Args:
        mask (np.ndarray): (H, W) float32 in [0, 1]
        op (dict): "op" (one of OP_WEIGHTS) plus its parameters, see sample_op

    Returns:
        np.ndarray: (H, W) float32 in [0, 1]

    Examples:
        >>> m = np.zeros((100, 100), np.float32); m[40:60, 40:60] = 1
        >>> int(apply_op(m, {"op": "dilate", "radius_frac": 0.05}).sum())  # 5 px all around: about 30 x 30 (octagon corners cut)
        876
        >>> int(apply_op(m, {"op": "erode", "radius_frac": 0.05}).sum())  # 10 x 10 left
        100
        >>> bool(apply_op(m, {"op": "open", "radius_frac": 0.2}).max() == 0)  # Opening removes an object smaller than the element
        True
        >>> apply_op(m, {"op": "quantize", "levels": 2}).shape
        (100, 100)
    """
    name, side = op["op"], max(mask.shape)
    px = lambda key: max(1, round(op[key] * side))
    if name == "dilate":
        return grow(mask, px("radius_frac"), cv2.dilate)
    if name == "erode":
        return grow(mask, px("radius_frac"), cv2.erode)
    if name == "open":
        return grow(grow(mask, px("radius_frac"), cv2.erode), px("radius_frac"), cv2.dilate)
    if name == "close":
        return grow(grow(mask, px("radius_frac"), cv2.dilate), px("radius_frac"), cv2.erode)
    if name == "blur":
        return cv2.GaussianBlur(mask, (0, 0), op["sigma_frac"] * side, borderType=cv2.BORDER_REPLICATE)
    if name == "warp":
        return warp(mask, op["amplitude_frac"] * side, max(1, op["cell_frac"] * side), op["seed"])
    if name == "threshold":
        return threshold(mask, op["level"], op["softness"])
    if name == "quantize":
        return quantize(mask, op["levels"])
    if name == "noise":
        return add_noise(mask, op["amount"], max(1, op["cell_frac"] * side), op["seed"])
    raise ValueError(f"Unknown op {name!r}. Known: {tuple(OP_WEIGHTS)}")


def cut(alpha, mask, lo, hi):
    """
    Pure function. The trimask of a chain result: 1 where mask >= hi and alpha is fully opaque, 0 where mask <= lo and alpha is
    fully transparent, 0.5 elsewhere.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        mask (np.ndarray): (H, W) float32 in [0, 1], the chain's result at alpha's size
        lo (float), hi (float): Cut levels, lo < hi

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> alpha = np.array([[1, 1, 1, 0, 0, 0]], np.float32)
        >>> mask = np.array([[1, 0.7, 0.2, 0.9, 0.3, 0]], np.float32)
        >>> cut(alpha, mask, 0.2, 0.8).tolist()  # Opaque with m=0.7 and transparent with m=0.9 are unknown; transparent m=0.3 too
        [[1.0, 0.5, 0.5, 0.5, 0.5, 0.0]]
    """
    white = (mask >= hi) & (alpha >= OPAQUE - 1e-6)
    black = (mask <= lo) & (alpha <= TRANSPARENT + 1e-6)
    return np.where(white, 1, np.where(black, 0, UNKNOWN)).astype(np.float32)


def keeps_enough(alpha, trimask):
    """
    Pure function. True when trimask keeps at least MIN_KEEP_FRAC of the opaque pixels of alpha as sure white and of its transparent ones as sure black.

    Examples:
        >>> alpha = np.array([[1] * 20 + [0] * 20], np.float32)
        >>> ok = np.full((1, 40), 0.5, np.float32); ok[0, 0] = 1; ok[0, 39] = 0
        >>> keeps_enough(alpha, ok), keeps_enough(alpha, np.full((1, 40), 0.5, np.float32))
        (True, False)
    """
    opaque, transparent = alpha >= OPAQUE - 1e-6, alpha <= TRANSPARENT + 1e-6
    return bool((trimask == 1).sum() >= MIN_KEEP_FRAC * opaque.sum() and (trimask == 0).sum() >= MIN_KEEP_FRAC * transparent.sum())


def work_size(shape):
    """
    Pure function. (width, height) for cv2.resize: the shape scaled so the longer side is at most WORK_PX (never upscaled).

    Examples:
        >>> work_size((1024, 1024)), work_size((768, 1024)), work_size((200, 300))
        ((512, 512), (512, 384), (300, 200))
    """
    scale = min(1, WORK_PX / max(shape))
    return max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale))


# ---------------------------------------------------------------- method interface


def sample_op(rng, name):
    """
    Near-pure function (advances rng). Random parameters of one primitive, as a JSON-serializable record. Sizes are fractions of the longer side.

    Examples:
        >>> sample_op(np.random.default_rng(0), "dilate")
        {'op': 'dilate', 'radius_frac': 0.018}
        >>> sorted(sample_op(np.random.default_rng(0), "warp"))
        ['amplitude_frac', 'cell_frac', 'op', 'seed']
    """
    log_uniform = lambda low_high: round(float(np.exp(rng.uniform(*np.log(low_high)))), 4)
    if name in ("dilate", "erode"):
        return {"op": name, "radius_frac": log_uniform(RADIUS_FRAC)}
    if name in ("open", "close"):
        return {"op": name, "radius_frac": log_uniform(OPEN_CLOSE_RADIUS_FRAC)}
    if name == "blur":
        return {"op": name, "sigma_frac": log_uniform(SIGMA_FRAC)}
    if name == "warp":
        return {"op": name, "amplitude_frac": log_uniform(WARP_AMPLITUDE_FRAC), "cell_frac": log_uniform(WARP_CELL_FRAC),
                "seed": int(rng.integers(SEED_BOUND))}
    if name == "threshold":
        return {"op": name, "level": round(float(rng.uniform(*THRESHOLD_RANGE)), 2),
                "softness": round(float(rng.uniform(*THRESHOLD_SOFTNESS)), 2)}
    if name == "quantize":
        return {"op": name, "levels": int(rng.choice(QUANTIZE_LEVELS))}
    if name == "noise":
        return {"op": name, "amount": round(float(rng.uniform(*NOISE_AMOUNT)), 2), "cell_frac": log_uniform(NOISE_CELL_FRAC),
                "seed": int(rng.integers(SEED_BOUND))}
    raise ValueError(f"Unknown op {name!r}. Known: {tuple(OP_WEIGHTS)}")


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws 2-6 distinct primitives (weighted by OP_WEIGHTS), shuffles them, gives each random
    parameters, and draws the cut levels.

    Returns:
        dict: ops (list of op records, in application order), lo, hi (float). JSON-serializable.

    Examples:
        >>> p = sample_params(np.random.default_rng(0))
        >>> sorted(p), [o["op"] for o in p["ops"]], p["lo"] < p["hi"]
        (['hi', 'lo', 'ops'], ['noise', 'threshold', 'open', 'dilate', 'warp', 'blur'], True)
        >>> sample_params(np.random.default_rng(0)) == p
        True
        >>> {len(sample_params(np.random.default_rng(i))["ops"]) for i in range(100)}
        {2, 3, 4, 5, 6}
    """
    names = list(OP_WEIGHTS)
    weights = np.array(list(OP_WEIGHTS.values()))
    count = int(rng.integers(OP_COUNT_RANGE[0], OP_COUNT_RANGE[1] + 1))
    chosen = rng.choice(len(names), size=count, replace=False, p=weights / weights.sum())
    shuffled = rng.permutation(chosen)
    return {"ops": [sample_op(rng, names[i]) for i in shuffled],
            "lo": round(float(rng.uniform(*LO_RANGE)), 2), "hi": round(float(rng.uniform(*HI_RANGE)), 2)}


def make_trimask(alpha, rng, ops, lo, hi):
    """
    Near-pure function (rng is unused: noise ops carry their seeds). Runs ops one after the other on the matte (at working
    resolution), cutting the result with lo/hi (see cut) and restrict_to_matte. After each op the cut is checked at working
    resolution (keeps_enough): an op that would leave too little sure area is not applied and is recorded.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        ops (list[dict]): Op records in application order (see sample_params)
        lo (float), hi (float): Cut levels

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params: ops as records, each with applied (bool),
        reason (only when not applied) and gray_frac (fraction of the working-resolution image gray after that op), plus lo, hi and applied (op names, in order)

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> ops = [{"op": "dilate", "radius_frac": 0.04}, {"op": "blur", "sigma_frac": 0.02}]
        >>> trimask, used = make_trimask(alpha, None, ops, 0.2, 0.8)
        >>> sorted(np.unique(trimask).tolist()), used["applied"]
        ([0.0, 0.5, 1.0], ['dilate', 'blur'])
        >>> trimask[64, 4].item(), trimask[64, 28].item(), trimask[64, 64].item()  # Far outside, in the dilated rim, inside
        (0.0, 0.5, 1.0)
        >>> _, used = make_trimask(alpha, None, [{"op": "erode", "radius_frac": 0.4}], 0.2, 0.8)  # Erodes the object away
        >>> used["applied"], used["ops"][0]["reason"]
        ([], 'left too little sure area')
    """
    height, width = alpha.shape
    work_alpha = cv2.resize(alpha, work_size(alpha.shape), interpolation=cv2.INTER_AREA)
    mask, records = work_alpha, []
    gray_frac = float((cut(work_alpha, mask, lo, hi) == UNKNOWN).mean())
    for op in ops:
        candidate = apply_op(mask, op)
        candidate_trimask = cut(work_alpha, candidate, lo, hi)
        if keeps_enough(work_alpha, candidate_trimask):
            mask, gray_frac, record = candidate, float((candidate_trimask == UNKNOWN).mean()), {**op, "applied": True}
        else:
            record = {**op, "applied": False, "reason": "left too little sure area"}
        records.append({**record, "gray_frac": round(gray_frac, 4)})
    final = cut(alpha, cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR), lo, hi)
    used = {"ops": records, "lo": lo, "hi": hi, "applied": [r["op"] for r in records if r["applied"]]}
    return restrict_to_matte(alpha, final), used
