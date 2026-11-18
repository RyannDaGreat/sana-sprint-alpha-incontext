"""
Trimask method random_stack: 2-4 distinct other methods applied one after another in a random order, each step overlaying its
own unknown region onto the running trimask. Every step may be skipped (random per-step probability).

Differs from combo_union (one unordered merge of whole-matte runs) and combo_chain (exactly two methods, the second one only
punches holes into the sure regions of the first): here a step may run on a VIEW of the running trimask instead of the matte,
so the result depends on the order.
    alpha: the method runs on the matte itself (order-independent, like combo_union).
    white: the method runs on the binary mask "running == 1", the object as it is still known. A later morph_band erodes the
           holes made by an earlier unknown_patches; bbox_unknown boxes only what is left white; partial_view hides part of it.
    outer: the method runs on "running != 0", the object plus everything unknown. Its band grows the unknown rim outward.
The step's trimask T is overlaid with merge_trimasks (combo_union): the result is unknown wherever either is unknown, and sure
only where both agree. The running trimask starts as the exact one (sure everywhere except soft-alpha pixels) and a pixel never
turns from gray to sure, so the result is consistent with the matte (manifest §13 correctness); restrict_to_matte runs at the end.

Cost: the steps cost the sum of their methods, so the drawn methods must fit COST_BUDGET_S (CPU seconds measured per method on
1024² mattes): 2-4 steps, fewer than drawn when the expensive ones would exceed the budget (noisy_band, skeleton_strokes only
come with cheap partners).

Sources (random order, random depth, per-op skipping: the standard recipe of augmentation pipelines; the operations here are trimap perturbations):
    - Hendrycks et al., "AugMix", ICLR 2020, arXiv 1912.02781: chains of 1-3 randomly chosen operations, composed in sequence.
    - Cubuk et al., "RandAugment", CVPRW 2020, arXiv 1909.13719: N randomly chosen operations applied one after another.
    - Liu et al., "Towards Enhancing Fine-grained Details for Image Matting", WACV 2021, arXiv 2101.09095: chains of random
      erosion/dilation steps, each acting on the result of the previous one, to imitate noisy user trimaps.
"""

import numpy as np
from sana_alpha import trimask as registry
from sana_alpha.trimask import restrict_to_matte
from sana_alpha.trimask.combo_union import merge_trimasks

STEP_COUNTS = (2, 3, 4)
STEP_COUNT_P = (0.35, 0.40, 0.25)
EXCLUDED = ("fuzzy_ramp", "bilateral_blur", "random_stack")  # fuzzy_ramp: continuous values, the overlay is for 3-level trimasks. bilateral_blur needs rgb. Tags starting with "combo_" are excluded too.
# CPU seconds of one make_trimask on a 1024² matte, one core (measured for this module; the sum over the steps must fit the budget)
COST_S = {"morph_band": 0.11, "noisy_band": 0.23, "blur_quantize": 0.08, "scribbles": 0.17, "bbox_unknown": 0.03,
          "convex_hull": 0.11, "grid_coarsen": 0.13, "polygon_simplify": 0.16, "unknown_patches": 0.05,
          "skeleton_strokes": 0.19, "partial_view": 0.06}
COST_BUDGET_S = 0.28
VIEWS = ("alpha", "white", "outer")
VIEW_P = (0.3, 0.45, 0.25)
P_NO_SKIPS = 0.5  # Chance that no step can be skipped; otherwise each step is skipped with a probability drawn from SKIP_P_RANGE
SKIP_P_RANGE = (0.1, 0.4)
MIN_KEEP_FRAC = 0.05  # A step is rejected if it leaves less than this fraction of the exact trimask's white pixels, or of its black ones
SEED_BOUND = 2 ** 63  # Per-step rng seeds are drawn below this, so that each step replays exactly


def step_pool():
    """Query. The method tags a step may use: the registry METHODS minus EXCLUDED and every combo_* tag."""
    return tuple(m for m in registry.METHODS if m not in EXCLUDED and not m.startswith("combo_"))


def draw_methods(rng, count, pool):
    """
    Near-pure function (advances rng). Up to count distinct methods of pool, in application order (the draw order is the random
    order), each uniform among those that still fit COST_BUDGET_S together with the ones already drawn. Raises KeyError for a
    pool method without a COST_S entry.

    Examples:
        >>> pool = ("bbox_unknown", "noisy_band", "scribbles", "unknown_patches")
        >>> draw_methods(np.random.default_rng(0), 4, pool)  # All four cost 0.48 s: the budget cuts the draw short
        ['unknown_patches', 'noisy_band']
        >>> all(len(draw_methods(np.random.default_rng(s), 4, pool)) >= 2 for s in range(200))
        True
    """
    chosen, spent = [], 0.0
    for _ in range(count):
        fits = [m for m in pool if m not in chosen and spent + COST_S[m] <= COST_BUDGET_S]
        if not fits:
            break
        chosen.append(fits[int(rng.integers(len(fits)))])
        spent += COST_S[chosen[-1]]
    return chosen


def step_matte(alpha, running, view):
    """
    Pure function. The matte a step's method runs on: the matte itself, the sure-white part of the running trimask, or everything not sure-black.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        running (np.ndarray): (H, W) float32 in {0, 0.5, 1}, the trimask so far
        view (str): One of VIEWS

    Returns:
        np.ndarray: (H, W) float32; binary {0, 1} for "white" and "outer"

    Examples:
        >>> alpha = np.array([[0, 0.3, 1, 1]], np.float32)
        >>> running = np.array([[0, 0.5, 0.5, 1]], np.float32)
        >>> step_matte(alpha, running, "alpha").tolist()
        [[0.0, 0.30000001192092896, 1.0, 1.0]]
        >>> step_matte(alpha, running, "white").tolist()  # The gray hole at column 2 is background now
        [[0.0, 0.0, 0.0, 1.0]]
        >>> step_matte(alpha, running, "outer").tolist()  # Gray counts as maybe-object
        [[0.0, 1.0, 1.0, 1.0]]
    """
    if view == "alpha":
        return alpha
    if view == "white":
        return (running == 1).astype(np.float32)
    if view == "outer":
        return (running != 0).astype(np.float32)
    raise ValueError(f"Unknown view {view!r}. Known: {VIEWS}")


def keeps_enough(candidate, exact):
    """
    Pure function. True when candidate still has at least MIN_KEEP_FRAC of the white pixels and of the black pixels of exact.
    Stops stacked steps from ending in a blank, all-gray trimask (a stack of bbox_unknown and partial_view often does).

    Args:
        candidate (np.ndarray): (H, W) float32 in {0, 0.5, 1}, a trimask after a step
        exact (np.ndarray): (H, W) float32 in {0, 0.5, 1}, the exact trimask of the matte

    Returns:
        bool

    Examples:
        >>> exact = np.array([[1, 1, 1, 1, 0, 0, 0, 0]] * 5, np.float32)  # 20 white, 20 black pixels: at least 1 of each must stay
        >>> keeps_enough(np.full((5, 8), 0.5, np.float32), exact)
        False
        >>> candidate = np.full((5, 8), 0.5, np.float32); candidate[0, 0] = 1; candidate[0, 7] = 0
        >>> keeps_enough(candidate, exact)
        True
        >>> candidate[0, 0] = 0.5  # No white left
        >>> keeps_enough(candidate, exact)
        False
    """
    return all((candidate == v).sum() >= MIN_KEEP_FRAC * (exact == v).sum() for v in (0, 1))


def sample_params(rng):
    """
    Near-pure function (advances rng; reads the registry and imports the chosen method modules). Draws the method list (see draw_methods),
    and per step a view, a skip flag and the method's own params with its own seeded rng. Skip: 50% of the trimasks skip nothing,
    the others skip each step with one probability drawn from SKIP_P_RANGE; at least one step always stays active.

    Returns:
        dict: steps (list of dict): in application order, each with method (str), seed (int), view (str), skip (bool), params (dict).
        JSON-serializable.

    Examples:
        >>> p = sample_params(np.random.default_rng(0))
        >>> [s["method"] for s in p["steps"]], [s["view"] for s in p["steps"]], [s["skip"] for s in p["steps"]]
        (['convex_hull', 'scribbles'], ['alpha', 'alpha'], [False, False])
        >>> sample_params(np.random.default_rng(0)) == p  # Deterministic
        True
        >>> all(not all(s["skip"] for s in sample_params(np.random.default_rng(i))["steps"]) for i in range(200))
        True
    """
    count = int(rng.choice(STEP_COUNTS, p=STEP_COUNT_P))
    methods = draw_methods(rng, count, step_pool())
    views = [str(v) for v in rng.choice(VIEWS, size=len(methods), p=VIEW_P)]
    skip_p = 0.0 if rng.random() < P_NO_SKIPS else float(rng.uniform(*SKIP_P_RANGE))
    skips = [bool(rng.random() < skip_p) for _ in methods]
    if all(skips):
        skips[int(rng.integers(len(methods)))] = False
    seeds = [int(rng.integers(SEED_BOUND)) for _ in methods]
    return {"steps": [{"method": m, "seed": s, "view": v, "skip": k,
                       "params": registry.method_module(m).sample_params(np.random.default_rng(s))}
                      for m, s, v, k in zip(methods, seeds, views, skips)]}


def apply_step(alpha, running, step):
    """
    Near-pure function (imports the method module). Runs one step: the method on its view of the running trimask, overlaid with merge_trimasks.
    A skipped step, or a view with no object or no background left (the running trimask has lost all its white or all its black), changes nothing and says why.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        running (np.ndarray): (H, W) float32 in {0, 0.5, 1}
        step (dict): One element of sample_params()["steps"]

    Returns:
        tuple[np.ndarray, dict]: the new running trimask (H, W) float32 in {0, 0.5, 1}, and the step record: the step plus
        applied (bool), reason (str, only when not applied), used (the params the method reports back, when applied)

    Examples:
        >>> alpha = np.zeros((64, 64), np.float32); alpha[16:48, 16:48] = 1
        >>> running = restrict_to_matte(alpha, alpha)
        >>> step = {"method": "morph_band", "seed": 0, "view": "white", "skip": False, "params": {"erode_px": 4, "dilate_px": 4}}
        >>> new, record = apply_step(alpha, running, step)
        >>> record["applied"], sorted(np.unique(new).tolist()), bool((new == 0.5).sum() > (running == 0.5).sum())
        (True, [0.0, 0.5, 1.0], True)
        >>> apply_step(alpha, running, {**step, "skip": True})[1]["reason"]
        'skipped'
        >>> apply_step(alpha, np.full((64, 64), 0.5, np.float32), step)[1]["reason"]  # Nothing is sure white any more
        'view has no object or no background'
    """
    if step["skip"]:
        return running, {**step, "applied": False, "reason": "skipped"}
    matte = step_matte(alpha, running, step["view"])
    binary = matte > 0.5
    if binary.all() or not binary.any():
        return running, {**step, "applied": False, "reason": "view has no object or no background"}
    local, used = registry.method_module(step["method"]).make_trimask(matte, np.random.default_rng(step["seed"]), **step["params"])
    return merge_trimasks([running, local]), {**step, "applied": True, "used": used}


def make_trimask(alpha, rng, steps):
    """
    Near-pure function (imports the method modules; rng is unused: each step has its own seed). Starts from the exact trimask of the matte
    (sure white on opaque, sure black on transparent, soft pixels gray), then applies the steps in order with apply_step.
    A step that would leave less than MIN_KEEP_FRAC of the exact white or black (keeps_enough) is rejected and recorded as not applied.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        steps (list[dict]): Ordered steps (see sample_params), method tags distinct, none in EXCLUDED or combo_*

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params: steps as records (apply_step), each with
        gray_frac (fraction of the image gray after that step), plus applied (list of "method:view" of the applied steps, in order)

    Examples:
        >>> alpha = np.zeros((64, 64), np.float32); alpha[16:48, 16:48] = 1
        >>> p = sample_params(np.random.default_rng(3))
        >>> trimask, used = make_trimask(alpha, None, **p)
        >>> set(np.unique(trimask).tolist()) <= {0.0, 0.5, 1.0}, [s["method"] for s in used["steps"]] == [s["method"] for s in p["steps"]]
        (True, True)
        >>> used["steps"][-1]["gray_frac"] == float((trimask == 0.5).mean()).__round__(4)
        True
        >>> bool((make_trimask(alpha, None, **p)[0] == trimask).all())  # Deterministic in the params
        True
        >>> make_trimask(alpha, None, [p["steps"][0], p["steps"][0]])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: random_stack needs distinct methods, none of ... or combo_*, got [...]
    """
    methods = [s["method"] for s in steps]
    if len(set(methods)) != len(methods) or any(m in EXCLUDED or m.startswith("combo_") for m in methods):
        raise ValueError(f"random_stack needs distinct methods, none of {EXCLUDED} or combo_*, got {methods}")
    exact = restrict_to_matte(alpha, (alpha > 0.5).astype(np.float32))
    running, records = exact, []
    for step in steps:
        candidate, record = apply_step(alpha, running, step)
        if record["applied"] and not keeps_enough(candidate, exact):
            record = {**step, "applied": False, "reason": "left too little sure area"}
        else:
            running = candidate
        records.append({**record, "gray_frac": round(float((running == 0.5).mean()), 4)})
    applied = [f"{r['method']}:{r['view']}" for r in records if r["applied"]]
    return restrict_to_matte(alpha, running), {"steps": records, "applied": applied}
