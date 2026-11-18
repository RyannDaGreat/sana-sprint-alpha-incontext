"""
Trimask method bbox_unknown: coarse box input. Black outside a padded, jittered bounding box of the object, gray everywhere
inside the box, white only on a strongly eroded core of the object (or nowhere).

This imitates box-prompted segmentation, where the user only drags a rectangle. Sources for the idea:
- Flexible Interactive Guided Image Matting (Progressive Bounding Box Deformation turns trimaps into box masks during
  training; random jitter on the box): https://ieeexplore.ieee.org/document/10141564/
- Matte Anything (box -> SAM mask -> pseudo-trimap -> matting): https://arxiv.org/abs/2306.04121
- SGAM (random trimaps by eroding fore- and background by random radii): https://arxiv.org/abs/2103.17020
Box padding is one-sided (the box only grows) so that no object pixel is ever labelled "surely background".
The core is a distance-transform level set (scale-free: it is the part of the object farther than core_frac of the
deepest point from the edge), which is a strong erosion for any object size.
"""

import cv2
import numpy as np
from sana_alpha.trimask import finalize

THRESHOLD = 0.5  # Alpha above this is object
PAD_MAX_FRAC_RANGE = (0.01, 0.3)  # Per-image maximum padding, as a fraction of the box side; drawn log-uniformly
CORE_PROBABILITY = 0.75  # Chance that the box carries a white core at all; otherwise the whole box is unknown
CORE_FRAC_RANGE = (0.5, 0.9)  # Core = pixels deeper than this fraction of the object's deepest point (distance transform)
SIDES = ("top", "bottom", "left", "right")


def mask_bbox(binary):
    """
    Pure function. Tight bounding box of the True pixels, or None for an empty mask.

    Args:
        binary (np.ndarray): (H, W) bool

    Returns:
        tuple[int, int, int, int] | None: (y0, y1, x0, x1), end-exclusive, in (row, col) order

    Examples:
        >>> binary = np.zeros((8, 8), bool); binary[2:5, 3:7] = True
        >>> mask_bbox(binary)
        (2, 5, 3, 7)
        >>> mask_bbox(np.zeros((8, 8), bool)) is None
        True
    """
    rows, cols = np.flatnonzero(binary.any(axis=1)), np.flatnonzero(binary.any(axis=0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def pad_box(box, pads, shape):
    """
    Pure function. Grows a box by per-side pixel pads, clipped to the image.

    Args:
        box (tuple[int, int, int, int]): (y0, y1, x0, x1), end-exclusive
        pads (dict): {"top", "bottom", "left", "right"} -> int pixels >= 0
        shape (tuple[int, int]): (H, W) of the image

    Returns:
        tuple[int, int, int, int]: (y0, y1, x0, x1), end-exclusive

    Examples:
        >>> pad_box((10, 20, 30, 50), {"top": 3, "bottom": 5, "left": 0, "right": 100}, (64, 64))
        (7, 25, 30, 64)
    """
    y0, y1, x0, x1 = box
    return (max(0, y0 - pads["top"]), min(shape[0], y1 + pads["bottom"]),
            max(0, x0 - pads["left"]), min(shape[1], x1 + pads["right"]))


def core_mask(binary, core_frac):
    """
    Pure function. The deep core of an object: pixels whose distance to the background is at least core_frac of the maximum
    distance. core_frac = 0 means no core.

    Args:
        binary (np.ndarray): (H, W) bool, True = object
        core_frac (float): In [0, 1]; 0 disables the core

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> binary = np.zeros((9, 9), bool); binary[1:8, 1:8] = True
        >>> int(core_mask(binary, 0.7).sum()), int(core_mask(binary, 0.0).sum())
        (9, 0)
    """
    if core_frac <= 0 or not binary.any():
        return np.zeros_like(binary)
    depth = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)  # (H, W) float32 pixels
    return binary & (depth >= core_frac * depth.max())


def box_trimask(shape, box, core):
    """
    Pure function. 0 outside the box, 0.5 inside it, 1 on the core.

    Args:
        shape (tuple[int, int]): (H, W)
        box (tuple[int, int, int, int] | None): (y0, y1, x0, x1) end-exclusive; None gives an all-black trimask
        core (np.ndarray): (H, W) bool, a subset of the box

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> core = np.zeros((4, 6), bool); core[2, 3] = True
        >>> box_trimask((4, 6), (1, 4, 1, 5), core).tolist()
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5, 0.5, 0.0], [0.0, 0.5, 0.5, 1.0, 0.5, 0.0], [0.0, 0.5, 0.5, 0.5, 0.5, 0.0]]
    """
    trimask = np.zeros(shape, np.float32)
    if box is not None:
        y0, y1, x0, x1 = box
        trimask[y0:y1, x0:x1] = 0.5
    trimask[core] = 1
    return trimask


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the box jitter and the core strength.

    pad_<side>: fraction of the tight box's height (top, bottom) or width (left, right) added on that side, each uniform in
    [0, pad_max] with pad_max log-uniform in PAD_MAX_FRAC_RANGE, so boxes range from tight to very loose and are lopsided.
    core_frac: 0 (no white at all) with probability 1 - CORE_PROBABILITY, else uniform in CORE_FRAC_RANGE.

    Examples:
        >>> sorted(sample_params(np.random.default_rng(0)))
        ['core_frac', 'pad_bottom', 'pad_left', 'pad_right', 'pad_top', 'threshold']
    """
    pad_max = float(np.exp(rng.uniform(*np.log(PAD_MAX_FRAC_RANGE))))
    pads = {f"pad_{side}": float(rng.uniform(0, pad_max)) for side in SIDES}
    has_core = rng.random() < CORE_PROBABILITY
    core_frac = float(rng.uniform(*CORE_FRAC_RANGE)) if has_core else 0.0
    return {**pads, "core_frac": core_frac, "threshold": THRESHOLD}


def make_trimask(alpha, rng, pad_top, pad_bottom, pad_left, pad_right, core_frac, threshold):
    """
    Pure function (rng is unused: all randomness is in sample_params). Black outside the padded box of the binarized alpha,
    gray inside, white on the deep core. An alpha with no object gives an all-black trimask (there is nothing to box).

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        pad_top, pad_bottom (float): Padding as a fraction of the tight box height
        pad_left, pad_right (float): Padding as a fraction of the tight box width
        core_frac (float): 0 for no core, else the depth fraction of the white core (see core_mask)
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus
            box = [y0, y1, x0, x1] (end-exclusive, after padding and clipping; null if empty)

    Examples:
        >>> alpha = np.zeros((100, 100), np.float32); alpha[30:70, 40:60] = 1
        >>> trimask, params = make_trimask(alpha, None, 0.1, 0.1, 0.5, 0.0, 0.8, 0.5)
        >>> params["box"]
        [26, 74, 30, 60]
        >>> sorted(np.unique(trimask).tolist()), float(trimask[0, 0]), float(trimask[28, 35]), float(trimask[50, 50])
        ([0.0, 0.5, 1.0], 0.0, 0.5, 1.0)
        >>> float(make_trimask(alpha, None, 0, 0, 0, 0, 0, 0.5)[0][30:70, 40:60].min())
        0.5
        >>> float(make_trimask(np.zeros((8, 8), np.float32), None, 0, 0, 0, 0, 0.5, 0.5)[0].max())
        0.0
    """
    binary = alpha > threshold
    tight = mask_bbox(binary)
    box = None
    if tight is not None:
        height, width = tight[1] - tight[0], tight[3] - tight[2]
        pads = {"top": pad_top * height, "bottom": pad_bottom * height, "left": pad_left * width, "right": pad_right * width}
        box = pad_box(tight, {side: int(round(px)) for side, px in pads.items()}, alpha.shape)
    params = {"pad_top": pad_top, "pad_bottom": pad_bottom, "pad_left": pad_left, "pad_right": pad_right,
              "core_frac": core_frac, "threshold": threshold, "box": None if box is None else list(box)}
    return finalize(alpha, box_trimask(alpha.shape, box, core_mask(binary, core_frac)), rng), params
