"""
Trimask augmentations (manifest §13): ten methods that turn a MAGICK matte into a trimask, the second model's condition.

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
"""

import hashlib
import importlib
import json

import numpy as np

METHODS = ("morph_band", "noisy_band", "blur_quantize", "scribbles", "bbox_unknown",
           "convex_hull", "grid_coarsen", "polygon_simplify", "unknown_patches", "fuzzy_ramp")


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


def generate(alpha, page_id, method, variant):
    """
    Near-pure function (imports the method module). Makes one trimask with hyperparameters drawn from the method's distribution.
    Deterministic in (alpha, page_id, method, variant).

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        page_id (str), method (str), variant (int): See seed()

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
    trimask, params = module.make_trimask(alpha, rng, **module.sample_params(rng))
    json.dumps(params)  # Raises here, not in the processor's writer, if a method returns a non-JSON value
    return check_trimask(trimask, alpha), params


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
