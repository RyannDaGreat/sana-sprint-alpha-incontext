"""
Trimask augmentations (manifest §13): fourteen methods that turn a MAGICK matte into a trimask, the second model's condition.

A trimask is (H, W) float32 in [0, 1]: 1 = surely object, 0 = surely background, 0.5 = unknown (the model decides).
fuzzy_ramp may use values between these; every other method uses only {0, 0.5, 1}.

METHOD INTERFACE. Each method is a module src/sana_alpha/trimask/<method>.py with two functions:
    sample_params(rng) -> dict
        Near-pure function (advances rng). Draws the method's hyperparameters. JSON-serializable values only.
    make_trimask(alpha, rng, **params) -> (trimask, params)
        Near-pure function (advances rng; methods without randomness ignore it). Every key of sample_params is a keyword.
        alpha: (H, W) float32 in [0, 1], white = object (magick_samples.alpha / 255). H = W = 1024 in MAGICK.
        Returns trimask (H, W) float32 in [0, 1] and the hyperparameters actually used (the input params, plus any
        derived values worth recording, e.g. a radius in pixels): a JSON-serializable dict.
Methods must not read files or global state. generate() below is the only caller; it seeds rng per (page_id, method, variant).

CORRECTNESS (user, m00647: "No incorrect trimasks"). A trimask is consistent with its matte (check_consistent) when
white (1) is only where the matte is fully opaque, black (0) only where it is fully transparent, every soft-alpha pixel is
strictly between 0 and 1, and (for fuzzy_ramp's continuous values) opaque pixels are >= 0.5 and transparent ones <= 0.5.
Each method builds this by construction: it ends with restrict_to_matte, which only turns sure pixels gray where the matte
does not allow them. generate() raises on any violation; it never repairs one.
"""

import hashlib
import importlib
import json

import numpy as np

METHODS = ("morph_band", "noisy_band", "blur_quantize", "scribbles", "bbox_unknown",
           "convex_hull", "grid_coarsen", "polygon_simplify", "unknown_patches", "fuzzy_ramp",
           "combo_union", "combo_chain", "skeleton_strokes", "partial_view",  # m00648
           "bilateral_blur", "random_stack")  # m00649
RGB_METHODS = ("bilateral_blur",)  # Their make_trimask also takes rgb: (H, W, 3) float32 in [0, 1], the image on gray


def seed(page_id, method, variant):
    """
    Pure function. The rng seed of one trimask: the first 8 bytes of sha256("page_id/method/variant").

    Args:
        page_id (str): magick_samples.page_id
        method (str): A tag of METHODS
        variant (int): 0, 1, ... for several trimasks of one image and method

    Returns:
        int

    Examples:
        >>> seed("a", "morph_band", 0) == seed("a", "morph_band", 0) != seed("a", "morph_band", 1)
        True
    """
    return int.from_bytes(hashlib.sha256(f"{page_id}/{method}/{variant}".encode()).digest()[:8], "big")


def method_module(method):
    """Query. Imports the module of one method tag (reads the source files). Raises for a tag not in METHODS."""
    if method not in METHODS:
        raise ValueError(f"Unknown trimask method {method!r}. Known: {METHODS}")
    return importlib.import_module(f"{__name__}.{method}")


OPAQUE = 1 - 1 / 255  # alpha >= this is fully opaque: 8-bit 254 and 255
TRANSPARENT = 1 / 255  # alpha <= this is fully transparent: 8-bit 0 and 1
HALF_STEP = 0.5 / 255  # The 8-bit PNG stores 0.5 as 128 = 0.502: the 0.5 bounds allow half a step


def consistency_violations(alpha, trimask):
    """
    Pure function. Counts the pixels of trimask that break the correctness rule of the module docstring, by kind.

    Args:
        alpha (np.ndarray): (H, W) float in [0, 1], white = object
        trimask (np.ndarray): (H, W) float in [0, 1], e.g. a stored PNG / 255

    Returns:
        dict: {"white_not_opaque", "black_not_transparent", "opaque_below_half", "transparent_above_half"}: int counts

    Examples:
        >>> alpha = np.array([[0, 0.3, 1]], np.float32)
        >>> consistency_violations(alpha, np.array([[0, 0.5, 1]], np.float32))
        {'white_not_opaque': 0, 'black_not_transparent': 0, 'opaque_below_half': 0, 'transparent_above_half': 0}
        >>> consistency_violations(alpha, np.array([[0.5, 1, 0]], np.float32))  # soft pixel white, opaque pixel black
        {'white_not_opaque': 1, 'black_not_transparent': 1, 'opaque_below_half': 1, 'transparent_above_half': 0}
    """
    opaque, transparent = alpha >= OPAQUE - 1e-6, alpha <= TRANSPARENT + 1e-6
    return {"white_not_opaque": int(((trimask >= 1) & ~opaque).sum()),
            "black_not_transparent": int(((trimask <= 0) & ~transparent).sum()),
            "opaque_below_half": int(((trimask < 0.5 - HALF_STEP) & opaque).sum()),
            "transparent_above_half": int(((trimask > 0.5 + HALF_STEP) & transparent).sum())}


def check_consistent(alpha, trimask):
    """
    Pure function. Raises ValueError, with the counts, when trimask breaks the correctness rule (consistency_violations). Returns trimask.

    Examples:
        >>> alpha = np.array([[0, 0.3, 1]], np.float32)
        >>> check_consistent(alpha, np.array([[0, 0.5, 1]], np.float32)).tolist()
        [[0.0, 0.5, 1.0]]
        >>> check_consistent(alpha, np.array([[0, 1, 1]], np.float32))
        Traceback (most recent call last):
        ValueError: Trimask is not consistent with its matte: {'white_not_opaque': 1, 'black_not_transparent': 0, 'opaque_below_half': 0, 'transparent_above_half': 0}
    """
    violations = consistency_violations(alpha, trimask)
    if any(violations.values()):
        raise ValueError(f"Trimask is not consistent with its matte: {violations}")
    return trimask


def restrict_to_matte(alpha, trimask):
    """
    Pure function. The last step of every method: keeps a sure pixel only where the matte allows it, else makes it gray (0.5).
    Never turns gray into sure. Soft-alpha pixels are kept strictly inside (0, 1) even after 8-bit storage.

    Args:
        alpha (np.ndarray): (H, W) float in [0, 1], white = object
        trimask (np.ndarray): (H, W) float32 in [0, 1]

    Returns:
        np.ndarray: (H, W) float32 in [0, 1] that passes check_consistent

    Examples:
        >>> alpha = np.array([[0, 0.3, 0.3, 1, 1]], np.float32)
        >>> restrict_to_matte(alpha, np.array([[0, 0, 1, 0, 0.75]], np.float32)).tolist()
        [[0.0, 0.5, 0.5, 0.5, 0.75]]
        >>> to_uint8(restrict_to_matte(alpha, np.array([[0.9, 0.001, 0.999, 1, 1]], np.float32))).tolist()  # fuzzy values, stored
        [[128, 1, 254, 255, 255]]
    """
    opaque, transparent = alpha >= OPAQUE - 1e-6, alpha <= TRANSPARENT + 1e-6
    soft = ~opaque & ~transparent
    out = np.where((trimask >= 1) & ~opaque, 0.5, trimask)
    out = np.where((out <= 0) & ~transparent, 0.5, out)
    out = np.where(opaque, np.maximum(out, 0.5), np.where(transparent, np.minimum(out, 0.5), out))
    return np.where(soft, np.clip(out, 1 / 255, 254 / 255), out).astype(np.float32)


def check_trimask(trimask, alpha):
    """
    Pure function. Raises ValueError unless trimask is a float32 array of alpha's shape with values in [0, 1]. Returns trimask.

    Examples:
        >>> check_trimask(np.full((2, 2), 0.5, np.float32), np.zeros((2, 2), np.float32)).shape
        (2, 2)
    """
    if trimask.shape != alpha.shape or trimask.dtype != np.float32:
        raise ValueError(f"Trimask must be float32 {alpha.shape}, got {trimask.dtype} {trimask.shape}")
    if not (np.isfinite(trimask).all() and trimask.min() >= 0 and trimask.max() <= 1):
        raise ValueError(f"Trimask values must be in [0, 1], got [{trimask.min()}, {trimask.max()}]")
    return trimask


def generate(alpha, page_id, method, variant, rgb=None):
    """
    Near-pure function (imports the method module). Makes one trimask with hyperparameters drawn from the method's distribution.
    Deterministic in (alpha, page_id, method, variant).

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        page_id (str), method (str), variant (int): See seed()
        rgb (np.ndarray | None): (H, W, 3) float32 in [0, 1], magick_samples.rgb_on_gray / 255. Required by RGB_METHODS only

    Returns:
        tuple[np.ndarray, dict]: trimask (H, W) float32 in [0, 1], and the hyperparameters used

    Examples:
        >>> alpha = np.zeros((256, 256), np.float32); alpha[64:192, 64:192] = 1
        >>> trimask, params = generate(alpha, "p", "morph_band", 0)
        >>> trimask.shape, sorted(np.unique(trimask).tolist()), "erode_px" in params
        ((256, 256), [0.0, 0.5, 1.0], True)
        >>> generate(np.zeros((8, 8), np.float32), "p", "morph_band", 0)
        Traceback (most recent call last):
        ValueError: The matte has no object pixel (alpha > 0.5) or no background pixel: a trimask of it would be constant
    """
    if (alpha > 0.5).all() or (alpha <= 0.5).all():  # MAGICK mattes always have both: anything else is a broken input
        raise ValueError("The matte has no object pixel (alpha > 0.5) or no background pixel: a trimask of it would be constant")
    module = method_module(method)
    rng = np.random.default_rng(seed(page_id, method, variant))
    if (method in RGB_METHODS) != (rgb is not None):
        raise ValueError(f"{method}: rgb is required by {RGB_METHODS} and by no other method")
    extra = {"rgb": rgb} if method in RGB_METHODS else {}
    trimask, params = module.make_trimask(alpha, rng, **extra, **module.sample_params(rng))
    json.dumps(params)  # Raises here, not in the processor's writer, if a method returns a non-JSON value
    return check_consistent(alpha, check_trimask(trimask, alpha)), params


def to_uint8(trimask):
    """
    Pure function. Turns a trimask into the stored 8-bit PNG values: 0.5 becomes 128.

    Args:
        trimask (np.ndarray): (H, W) float32 in [0, 1]

    Returns:
        np.ndarray: (H, W) uint8

    Examples:
        >>> to_uint8(np.array([[0, 0.5, 1]], np.float32)).tolist()
        [[0, 128, 255]]
    """
    return np.round(trimask * 255).astype(np.uint8)


def coverage(trimask_uint8):
    """
    Pure function. Fractions of pixels that are white (255), black (0) and gray (anything else, unknown) in a stored trimask.

    Args:
        trimask_uint8 (np.ndarray): (H, W) uint8

    Returns:
        dict: {"white": float, "gray": float, "black": float}, summing to 1

    Examples:
        >>> coverage(np.array([[0, 128, 255, 255]], np.uint8))
        {'white': 0.5, 'gray': 0.25, 'black': 0.25}
    """
    white, black = float((trimask_uint8 == 255).mean()), float((trimask_uint8 == 0).mean())
    return {"white": white, "gray": 1 - white - black, "black": black}
