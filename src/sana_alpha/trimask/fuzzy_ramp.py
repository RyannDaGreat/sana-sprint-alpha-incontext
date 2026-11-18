"""
fuzzy_ramp: the one continuous trimask. The value is a smooth function of the signed distance to the matte edge, so it
goes 1 -> 0.5 -> 0 across a wide band instead of three flat levels. 0.5 sits on the edge (up to a small random shift).

The band width is random and can differ inside and outside the object, as a rough hand-made fuzzy mask does.

Prior art (manifest §13): matting papers soften the trimap in this way.
  - SGHM, Chen et al. 2021, https://arxiv.org/abs/2103.17020: "random soft segmentation" = random erosion/dilation of
    the mask by 1-59 px followed by a random Gaussian blur, used as the coarse input.
  - FBA Matting, Forte and Pitie 2020, https://arxiv.org/abs/2003.07711: trimaps from random 3-25 px erosion/dilation,
    and the network sees smooth distance-transform encodings of them, not only the hard 3 levels.
  - Progressive Trimap Deformation, Cheng et al. 2021, https://arxiv.org/abs/2110.10898: training trimaps are
    deliberately made less exact (down to scribbles), so the model must cope with vague guidance.
Here the distance transform of the matte gives the signed distance; one profile maps it to [0, 1].
"""

import cv2
import numpy as np
from sana_alpha.trimask import finalize

PROFILES = ("tanh", "linear", "smoothstep")
THRESHOLD = 0.5  # Alpha above this is object when the edge is located
WIDTH_RANGE_PX = (4.0, 100.0)  # log-uniform band half-width: the distance at which the ramp is (nearly) 1 or 0
LOG2_RATIO_MAX = 1.5  # inside/outside width ratio is 2^U(-1.5, 1.5), so up to about 2.8x
SYMMETRIC_PROBABILITY = 0.3  # share of masks whose inside and outside widths are equal
SHIFT_PROBABILITY = 0.5  # share of masks whose 0.5 level is moved off the edge
SHIFT_MAX_FRACTION = 0.3  # the shift is at most this fraction of the band width
TANH_SCALE = 2.0  # tanh(2 d / w) reaches 0.98 at d = w, so all profiles are nearly saturated at the band edge


def signed_distance(mask):
    """
    Pure function. Signed distance to the edge of a binary mask: positive inside, negative outside, in pixels.
    The edge lies halfway between the last inside pixel (+0.5) and the first outside pixel (-0.5).

    Args:
        mask (np.ndarray): (H, W) bool, True = object

    Returns:
        np.ndarray: (H, W) float32, same sign convention (+-inf everywhere when the mask is empty or full)

    Examples:
        >>> mask = np.zeros((1, 6), bool); mask[0, 2:4] = True
        >>> signed_distance(mask).tolist()
        [[-1.5, -0.5, 0.5, 0.5, -0.5, -1.5]]
        >>> signed_distance(np.zeros((1, 2), bool)).tolist()
        [[-inf, -inf]]
    """
    if mask.all() or not mask.any():  # No edge exists: every pixel is infinitely deep
        return np.full(mask.shape, np.inf if mask.all() else -np.inf, np.float32)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) - 0.5
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) - 0.5
    return np.where(mask, inside, -outside).astype(np.float32)


def ramp(distance, width_in, width_out, profile, shift=0.0):
    """
    Pure function. Maps signed distance to a trimask value: 0.5 at distance == shift, about 1 deep inside, about 0 far outside.

    Args:
        distance (np.ndarray): (H, W) float32 pixels, positive inside the object
        width_in (float): Pixels inside the edge at which the ramp saturates to 1
        width_out (float): Pixels outside the edge at which the ramp saturates to 0
        profile (str): One of PROFILES
        shift (float): Pixels the 0.5 level moves inward (positive) or outward (negative)

    Returns:
        np.ndarray: (H, W) float32 in [0, 1]

    Examples:
        >>> d = np.array([[-20, -10, 0, 10, 20]], np.float32)
        >>> ramp(d, 20, 20, "linear").tolist()
        [[0.0, 0.25, 0.5, 0.75, 1.0]]
        >>> ramp(d, 10, 20, "linear").tolist()  # Inside saturates twice as fast
        [[0.0, 0.25, 0.5, 1.0, 1.0]]
        >>> [round(float(v), 3) for v in ramp(d, 10, 10, "tanh")[0]]
        [0.0, 0.018, 0.5, 0.982, 1.0]
        >>> ramp(d, 10, 10, "smoothstep").tolist()
        [[0.0, 0.0, 0.5, 1.0, 1.0]]
        >>> ramp(d, 20, 20, "linear", shift=10).tolist()  # 0.5 moved 10 px inward
        [[0.0, 0.0, 0.25, 0.5, 0.75]]
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown ramp profile {profile!r}. Known: {PROFILES}")
    shifted = distance - shift
    t = shifted / np.where(shifted > 0, width_in, width_out)  # -1 .. 1 across the band, beyond it outside
    if profile == "tanh":
        value = 0.5 + 0.5 * np.tanh(TANH_SCALE * t)
    elif profile == "linear":
        value = 0.5 + 0.5 * t
    else:
        u = np.clip(0.5 + 0.5 * t, 0, 1)
        value = u * u * (3 - 2 * u)
    return np.clip(value, 0, 1).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the ramp hyperparameters.

    Returns:
        dict: profile (str), width_px (float, log-uniform 4..100), log2_ratio (float, 0 for symmetric ramps; else the
        log2 of width_in / width_out), shift_px (float, 0 or up to 30% of width_px either way), threshold (float, 0.5)

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params), params["profile"] in PROFILES, 4 <= params["width_px"] <= 100
        (['log2_ratio', 'profile', 'shift_px', 'threshold', 'width_px'], True, True)
    """
    width = float(np.exp(rng.uniform(*np.log(WIDTH_RANGE_PX))))
    asymmetric = rng.random() >= SYMMETRIC_PROBABILITY
    shifted = rng.random() < SHIFT_PROBABILITY
    return {
        "profile": str(rng.choice(PROFILES)),
        "width_px": width,
        "log2_ratio": float(rng.uniform(-LOG2_RATIO_MAX, LOG2_RATIO_MAX)) if asymmetric else 0.0,
        "shift_px": float(rng.uniform(-SHIFT_MAX_FRACTION, SHIFT_MAX_FRACTION) * width) if shifted else 0.0,
        "threshold": THRESHOLD,
    }


def make_trimask(alpha, rng, profile, width_px, log2_ratio, shift_px, threshold):
    """
    Near-pure function (advances rng, only in finalize; otherwise the ramp is deterministic given its params). Continuous trimask from a matte.

    The inside width is width_px * 2^(log2_ratio / 2) and the outside width width_px * 2^(-log2_ratio / 2), so their
    geometric mean stays width_px.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Only finalize uses it (draws its dilation radius)
        profile, width_px, log2_ratio, shift_px: See sample_params and ramp
        threshold (float): Alpha above this is object when the edge is located

    Returns:
        tuple[np.ndarray, dict]: trimask (H, W) float32 in [0, 1] (1 object, 0 background, 0.5 on the edge), and the
        input params plus width_in_px and width_out_px (floats)

    Examples:
        >>> yy, xx = np.mgrid[:64, :64]
        >>> alpha = (np.hypot(yy - 32, xx - 32) < 16).astype(np.float32)  # Disk, radius 16
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), "linear", 8.0, 0.0, 0.0, 0.5)
        >>> trimask.shape, trimask.dtype, float(trimask[32, 32]), float(trimask[0, 0])
        ((64, 64), dtype('float32'), 1.0, 0.0)
        >>> [round(float(v), 2) for v in trimask[32, 32 + 8:32 + 25:4]]  # Radius 8..24 along a row: 1 -> 0.5 -> 0
        [0.97, 0.72, 0.47, 0.22, 0.0]
        >>> sorted(used)
        ['log2_ratio', 'profile', 'shift_px', 'threshold', 'width_in_px', 'width_out_px', 'width_px']
        >>> used = make_trimask(alpha, np.random.default_rng(0), "linear", 8.0, 2.0, 0.0, 0.5)[1]  # Inside twice as wide as outside
        >>> used["width_in_px"], used["width_out_px"]
        (16.0, 4.0)
    """
    width_in = width_px * 2 ** (log2_ratio / 2)
    width_out = width_px * 2 ** (-log2_ratio / 2)
    distance = signed_distance(alpha > threshold)
    trimask = ramp(distance, width_in, width_out, profile, shift_px)
    used = {"profile": profile, "width_px": width_px, "log2_ratio": log2_ratio, "shift_px": shift_px,
            "threshold": threshold, "width_in_px": width_in, "width_out_px": width_out}
    return finalize(alpha, trimask, rng), used
