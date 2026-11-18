"""
Trimask method combo_union: runs 2-3 distinct other methods (random, from the METHODS registry) on the same matte and
merges their trimasks. Unknown wherever ANY of them is unknown; sure-white only where ALL are sure-white; sure-black only
where ALL are sure-black. Every input is consistent with the matte (each ends with restrict_to_matte), so the merge is
too: white means all inputs saw opaque pixels, black means all saw transparent ones. restrict_to_matte runs once more at the end.

Why: a user-made trimap is rarely one clean recipe; a rough outline plus a few spray-painted unknown patches plus a
brush stroke is typical. Taking the union of unknown regions stacks the failure modes of several recipes. Sources for
combining trimap perturbations (none builds a union of whole recipes; this extends them):
    - Liu et al., "Towards Enhancing Fine-grained Details for Image Matting", WACV 2021, arXiv 2101.09095: chains of
      random erosion and dilation steps applied to the unknown region to simulate noisy user trimaps.
    - Yu et al., "Mask Guided Matting via Progressive Refinement Network", CVPR 2021, arXiv 2012.06722: random
      perturbation operations on the guidance mask during training, for robustness to coarse guidance.
Merge rule = a three-valued AND on "sure" and an OR on "unknown".
"""

import numpy as np
from sana_alpha import trimask as registry
from sana_alpha.trimask import restrict_to_matte

N_METHODS = (2, 3)  # Inclusive range of other methods merged
EXCLUDED = ("fuzzy_ramp",)  # Continuous values: the merge rule is for 3-level trimasks. Names starting with "combo_" are excluded too.
SEED_BOUND = 2 ** 63  # Per-method rng seeds are drawn below this, so that each sub-method replays exactly


def merge_pool():
    """Query. The method tags this method may merge: the registry METHODS minus fuzzy_ramp, every combo_* tag and RGB_METHODS (generate() passes no rgb to combo_union)."""
    return tuple(m for m in registry.METHODS if m not in EXCLUDED + registry.RGB_METHODS and not m.startswith("combo_"))


def merge_trimasks(trimasks):
    """
    Pure function. Unknown (0.5) wherever any input is not exactly the same sure value; 1 only where all are 1; 0 only where all are 0.

    Args:
        trimasks (list[np.ndarray]): k arrays of (H, W) float32 in {0, 0.5, 1}

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> a = np.array([[1, 1, 0, 0.5, 0]], np.float32)
        >>> b = np.array([[1, 0.5, 0, 0, 1]], np.float32)
        >>> merge_trimasks([a, b]).tolist()  # white only 1&1, black only 0&0, everything else unknown
        [[1.0, 0.5, 0.0, 0.5, 0.5]]
        >>> c = np.array([[0.5, 1, 0, 1, 0]], np.float32)
        >>> merge_trimasks([a, b, c]).tolist()
        [[0.5, 0.5, 0.0, 0.5, 0.5]]
    """
    stack = np.stack(trimasks)
    return np.where((stack == 1).all(0), 1, np.where((stack == 0).all(0), 0, 0.5)).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng; reads the registry and imports the chosen method modules). Draws 2 or 3 distinct
    methods uniformly from merge_pool(), then each one's own params with its own seeded rng.

    Returns:
        dict: methods (list[str]), seeds (list[int], rng seed of each method), method_params (list[dict], each the
        sample_params of its method). All JSON-serializable.

    Examples:
        >>> p = sample_params(np.random.default_rng(0))
        >>> len(p["methods"]) in N_METHODS and len(set(p["methods"])) == len(p["methods"])
        True
        >>> len(p["seeds"]) == len(p["method_params"]) == len(p["methods"])
        True
        >>> sample_params(np.random.default_rng(0)) == p  # Deterministic
        True
    """
    pool = merge_pool()
    k = int(rng.integers(N_METHODS[0], N_METHODS[1] + 1))
    methods = [str(m) for m in rng.choice(pool, size=k, replace=False)]
    seeds = [int(rng.integers(SEED_BOUND)) for _ in methods]
    method_params = [registry.method_module(m).sample_params(np.random.default_rng(s)) for m, s in zip(methods, seeds)]
    return {"methods": methods, "seeds": seeds, "method_params": method_params}


def make_trimask(alpha, rng, methods, seeds, method_params):
    """
    Near-pure function (imports the method modules; rng is unused: each method gets its own seeded rng from seeds).
    Runs each method on the matte and merges the results with merge_trimasks.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        methods (list[str]): Tags of the merged methods, at least 2, distinct, none fuzzy_ramp or combo_*
        seeds (list[int]): One rng seed per method
        method_params (list[dict]): One params dict per method (the keywords of its make_trimask)

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params: methods, seeds, and
        method_params as returned by each method (inputs plus its derived values)

    Examples:
        >>> alpha = np.zeros((64, 64), np.float32); alpha[16:48, 16:48] = 1
        >>> p = sample_params(np.random.default_rng(3))
        >>> trimask, used = make_trimask(alpha, None, **p)
        >>> trimask.shape, sorted(np.unique(trimask).tolist()) <= [0.0, 0.5, 1.0], used["methods"] == p["methods"]
        ((64, 64), True, True)
        >>> a = registry.method_module(p["methods"][0]).make_trimask(alpha, np.random.default_rng(p["seeds"][0]), **p["method_params"][0])[0]
        >>> bool((trimask[a == 0.5] == 0.5).all())  # The union keeps every unknown pixel of each input
        True
    """
    if not 2 <= len(methods) == len(seeds) == len(method_params) or len(set(methods)) != len(methods):
        raise ValueError(f"Need at least 2 distinct methods with one seed and one params each, got {methods}, {seeds}, {len(method_params)}")
    bad = [m for m in methods if m not in merge_pool()]
    if bad:
        raise ValueError(f"combo_union cannot merge {bad}")
    results = [registry.method_module(m).make_trimask(alpha, np.random.default_rng(s), **p)
               for m, s, p in zip(methods, seeds, method_params)]
    merged = restrict_to_matte(alpha, merge_trimasks([trimask for trimask, _ in results]))
    return merged, {"methods": list(methods), "seeds": list(seeds), "method_params": [used for _, used in results]}
