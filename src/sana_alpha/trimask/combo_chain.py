"""
Trimask method combo_chain: one base method's trimask, then a second, different "local" method that turns additional
sure pixels of the base gray. Compound inputs such as "a coarse lasso with some holes" or "a pixel grid with a ragged rim".

The base (polygon_simplify, grid_coarsen, morph_band, convex_hull, blur_quantize, bbox_unknown) makes the coarse trimask.
The local method (unknown_patches, noisy_band) is run on the same matte; wherever it says "unknown" and the
base says "sure", the result is unknown. The base's unknown pixels stay unknown and no gray pixel ever becomes sure, so the
result is consistent with the matte whenever both parts are (manifest §13 correctness rule). Both methods are called through the
registry (trimask.method_module) with params drawn from their own sample_params, each on its own seeded rng, so the recorded
params reproduce the trimask exactly.

Not used as local method: scribbles. It turns 70-90% of the base's sure pixels gray and often leaves no white at all,
which the sheet showed as a useless all-gray input; the base methods already cover the sparse-input end.

Sources (the chaining recipe follows these: a deterministic trimap, then an extra random perturbation of it):
    - Liu et al., "Towards Enhancing Fine-grained Details for Image Matting", arXiv 2101.09095: after the erosion/dilation trimap,
      n extra random morphological steps change the unknown region to imitate noisy user-drawn trimaps.
    - Li and Lu, "Natural Image Matting via Guided Contextual Attention", AAAI 2020, arXiv 2001.04069: random erosion/dilation
      trimaps, then further random deformation of the unknown region as augmentation.
    - Yu et al., "Free-Form Image Inpainting with Gated Convolution", ICCV 2019, arXiv 1806.03589: random free-form holes on top
      of an otherwise exact input (the unknown_patches local method).
"""

import numpy as np

from sana_alpha.trimask import method_module, restrict_to_matte

UNKNOWN = 0.5
SEED_MAX = 2**31

# Choice weights. Cost matters: the whole method must stay under 0.3 s on a 1024² matte (one core). One call of noisy_band takes
# about 0.2 s and a base about 0.03-0.15 s, so it only follows the cheapest bases.
BASE_WEIGHTS = {"polygon_simplify": 3, "grid_coarsen": 3, "morph_band": 2, "convex_hull": 1, "blur_quantize": 2, "bbox_unknown": 2}
LOCAL_WEIGHTS = {"unknown_patches": 3, "noisy_band": 1.5}
CHEAP_BASES = ("blur_quantize", "bbox_unknown")  # Under 0.1 s: the only bases that may be followed by a slow local
SLOW_LOCALS = ("noisy_band",)


def allowed_locals(base_method):
    """
    Pure function. The local methods that may follow a base method: never the same method, and a slow local only after a cheap base.

    Examples:
        >>> allowed_locals("polygon_simplify")
        ('unknown_patches',)
        >>> allowed_locals("bbox_unknown")
        ('unknown_patches', 'noisy_band')
        >>> allowed_locals("blur_quantize")
        ('unknown_patches', 'noisy_band')
    """
    return tuple(m for m in LOCAL_WEIGHTS if m != base_method and (m not in SLOW_LOCALS or base_method in CHEAP_BASES))


def choose_weighted(rng, weights):
    """
    Near-pure function (advances rng). One key of weights, with probability proportional to its value.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> picks = [choose_weighted(rng, {"a": 3, "b": 1}) for _ in range(1000)]
        >>> picks.count("a") / 1000  # About 3 / 4
        0.733
    """
    keys = list(weights)
    probabilities = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=probabilities / probabilities.sum()))]


def chain(base, local):
    """
    Pure function. Keeps the base trimask, but makes every pixel gray where the base is sure (0 or 1) and the local one is gray (0.5).
    Gray never becomes sure: the result differs from the base only on base-sure pixels.

    Args:
        base (np.ndarray): (H, W) float32 in {0, 0.5, 1}
        local (np.ndarray): (H, W) float32 in {0, 0.5, 1}, a trimask of the same matte

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> base = np.array([[0, 0, 0.5, 1, 1, 1]], np.float32)
        >>> local = np.array([[0, 0.5, 1, 0.5, 1, 0.5]], np.float32)
        >>> chain(base, local).tolist()  # base gray (col 2) stays gray even where local is sure
        [[0.0, 0.5, 0.5, 0.5, 1.0, 0.5]]
        >>> bool((chain(base, local) == base).all()), bool((chain(base, base) == base).all())
        (False, True)
    """
    return np.where((base != UNKNOWN) & (local == UNKNOWN), UNKNOWN, base).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Picks a base method (weighted, favoring the coarse ones) and a different local method
    allowed after it, draws each one's params from its own sample_params with its own seed (so make_trimask reproduces them).

    Returns:
        dict: base_method, local_method (str), base_seed, local_seed (int), base_params, local_params (dict, JSON-serializable)

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params)
        ['base_method', 'base_params', 'base_seed', 'local_method', 'local_params', 'local_seed']
        >>> params["base_method"] != params["local_method"]
        True
    """
    base_method = choose_weighted(rng, BASE_WEIGHTS)
    local_method = choose_weighted(rng, {m: LOCAL_WEIGHTS[m] for m in allowed_locals(base_method)})
    base_seed, local_seed = (int(rng.integers(SEED_MAX)) for _ in range(2))
    return {"base_method": base_method, "local_method": local_method, "base_seed": base_seed, "local_seed": local_seed,
            "base_params": method_module(base_method).sample_params(np.random.default_rng(base_seed)),
            "local_params": method_module(local_method).sample_params(np.random.default_rng(local_seed))}


def make_trimask(alpha, rng, base_method, local_method, base_seed, local_seed, base_params, local_params):
    """
    Near-pure function (imports the two method modules; rng is unused: all randomness is in base_seed and local_seed).
    Runs the base method on alpha, runs the local method on alpha, and chains them (see chain).

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        base_method (str), local_method (str): Tags of two different methods of trimask.METHODS that give {0, 0.5, 1} trimasks
        base_seed (int), local_seed (int): Seed the rng handed to each method
        base_params (dict), local_params (dict): Keyword arguments of each method's make_trimask

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus base_used, local_used (the params
        each method reports back) and added_gray_frac (fraction of the image the local method turned gray)

    Examples:
        >>> alpha = np.zeros((64, 64), np.float32); alpha[16:48, 16:48] = 1
        >>> spec = dict(base_method="morph_band", local_method="unknown_patches", base_seed=1, local_seed=2,
        ...             base_params=dict(erode_px=3, dilate_px=3),
        ...             local_params=dict(n_edge=0, n_free=2, radius_min_frac=0.08, radius_max_frac=0.12, irregularity=0.2,
        ...                               aspect_max=1.0, band_px=0, threshold=0.5))
        >>> trimask, params = make_trimask(alpha, None, **spec)
        >>> sorted(np.unique(trimask).tolist()), params["added_gray_frac"] > 0
        ([0.0, 0.5, 1.0], True)
        >>> base, _ = method_module("morph_band").make_trimask(alpha, None, erode_px=3, dilate_px=3)
        >>> bool(((trimask == base) | (trimask == 0.5)).all())  # Only ever sure -> gray
        True
        >>> trimask2, _ = make_trimask(alpha, None, **spec)
        >>> bool((trimask == trimask2).all())  # Deterministic in the params
        True
    """
    base, base_used = method_module(base_method).make_trimask(alpha, np.random.default_rng(base_seed), **base_params)
    local, local_used = method_module(local_method).make_trimask(alpha, np.random.default_rng(local_seed), **local_params)
    result = chain(base, local)
    used = {"base_method": base_method, "local_method": local_method, "base_seed": base_seed, "local_seed": local_seed,
            "base_params": base_params, "local_params": local_params, "base_used": base_used, "local_used": local_used,
            "added_gray_frac": round(float((result != base).mean()), 4)}
    return restrict_to_matte(alpha, result), used
