"""
Trimask augmentations (manifest §13): twenty-one methods that turn a MAGICK matte into a trimask, the second model's condition.

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
Each method builds this by construction: it ends with finalize, which turns sure pixels gray, coarsely (m00652: no
pixel-level detail of the matte may get into a trimask): sure only inside the coarse allowed regions, the forced-gray set
grown and smoothed, tiny sure islands gray. generate() raises on any violation; it never repairs one.
Two methods skip finalize (m00718): original (the matte itself; consistent by definition) and threshold (the matte at
alpha >= 0.5), the one method in CONSISTENCY_EXEMPT: the user asked for a solid outline, which marks soft edges as sure.
"""

import hashlib
import importlib
import json

import cv2
import numpy as np

METHODS = ("morph_band", "noisy_band", "blur_quantize", "scribbles", "bbox_unknown",
           "convex_hull", "grid_coarsen", "polygon_simplify", "unknown_patches", "fuzzy_ramp",
           "combo_union", "combo_chain", "skeleton_strokes", "partial_view",  # m00648
           "bilateral_blur", "random_stack",  # m00649
           "perlin_warp", "op_chain", "glass_ripple",  # m00651
           "original", "threshold")  # m00718: the matte itself and the matte thresholded (no finalize)
RGB_METHODS = ("bilateral_blur",)  # Their make_trimask also takes rgb: (H, W, 3) float32 in [0, 1], the image on gray
CONSISTENCY_EXEMPT = ("threshold",)  # m00718: the user asked for a solid outline; it marks soft edges as sure object


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


OPEN_PX = 8  # Sure regions are openings of the opaque / transparent regions by this disk: no strand thinner than 16 px is sure
MARGIN_PX = 3  # ... then eroded by this disk, so sure pixels keep a gap to every soft-alpha pixel
FORCE_DILATE_PX = (6, 20)  # Sure pixels that must turn gray are first grown by a disk of a random radius in this range
MIN_ISLAND_PX = 32 * 32  # A sure region smaller than this (scaled to a 1024² image) becomes gray


def disk(radius):
    """
    Pure function. A filled disk structuring element.

    Args:
        radius (int): Pixels, >= 0

    Returns:
        np.ndarray: (2 radius + 1, 2 radius + 1) uint8 in {0, 1}

    Examples:
        >>> disk(1).tolist()
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def coarse_allowed(region, open_px=OPEN_PX, margin_px=MARGIN_PX):
    """
    Pure function. The coarse part of a region where a sure value is allowed: its opening by a disk of open_px (thin strands
    and specks go), eroded by a disk of margin_px. A subset of region, so sure pixels there stay correct.

    Args:
        region (np.ndarray): (H, W) bool, e.g. the fully opaque pixels

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> region = np.zeros((40, 40), bool); region[5:35, 5:35] = True; region[0:40, 19] = True  # a square and a 1 px strand
        >>> allowed = coarse_allowed(region)
        >>> bool(allowed[20, 20]), bool(allowed[1, 19]), bool((allowed & ~region).any())  # core kept, strand gone, a subset
        (True, False, False)
    """
    opened = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, disk(open_px), borderType=cv2.BORDER_REPLICATE)
    return cv2.erode(opened, disk(margin_px), borderType=cv2.BORDER_REPLICATE).astype(bool)


def remove_small_islands(mask, min_px):
    """
    Pure function. Drops the 8-connected components of mask smaller than min_px pixels.

    Args:
        mask (np.ndarray): (H, W) bool
        min_px (int): Smallest component kept

    Returns:
        np.ndarray: (H, W) bool, a subset of mask

    Examples:
        >>> mask = np.zeros((10, 10), bool); mask[0:4, 0:4] = True; mask[8, 8] = True
        >>> int(remove_small_islands(mask, 4).sum())
        16
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False  # Label 0 is the background of mask
    return keep[labels]


def finalize(alpha, trimask, rng):
    """
    Near-pure function (advances rng). The last step of every method: makes trimask correct (check_consistent) by turning sure
    pixels gray, coarsely, never gray into sure. The pixels that break the rule, or are sure outside coarse_allowed, are grown
    by a disk of a random FORCE_DILATE_PX radius, smoothed (blur + threshold), and made 0.5; then sure regions smaller than
    MIN_ISLAND_PX become 0.5. So no hair strand, speck or exact silhouette of the matte gets into the trimask.

    Args:
        alpha (np.ndarray): (H, W) float in [0, 1], white = object
        trimask (np.ndarray): (H, W) float32 in [0, 1] (3-level, or continuous for fuzzy_ramp)
        rng (np.random.Generator): Draws the dilation radius

    Returns:
        np.ndarray: (H, W) float32 in [0, 1] that passes check_consistent after 8-bit storage

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1; alpha[64, 0:32] = 1  # square + 1 px opaque strand
        >>> trimask = np.where(alpha > 0.5, 1, 0).astype(np.float32)  # carved exactly from the matte
        >>> out = finalize(alpha, trimask, np.random.default_rng(0))
        >>> bool(out[64, 64] == 1), bool(out[64, 10] == 0.5), bool(out[5, 5] == 0)  # core sure, strand area gray, far bg sure
        (True, True, True)
        >>> check_consistent(alpha, to_uint8(out) / 255).shape
        (128, 128)
    """
    opaque, transparent = alpha >= OPAQUE - 1e-6, alpha <= TRANSPARENT + 1e-6
    white, black = trimask >= 1 - HALF_STEP, trimask <= HALF_STEP  # As the 8-bit PNG will store them
    broken = ((white & ~coarse_allowed(opaque)) | (black & ~coarse_allowed(transparent))
              | (opaque & (trimask < 0.5)) | (transparent & (trimask > 0.5)))
    radius = int(rng.integers(FORCE_DILATE_PX[0], FORCE_DILATE_PX[1] + 1))
    grown = cv2.dilate(broken.astype(np.uint8), disk(radius), borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    smooth = cv2.GaussianBlur(grown, (0, 0), radius / 3) > 0.5
    out = np.where(smooth | broken, 0.5, trimask).astype(np.float32)  # broken is inside smooth already; the union makes it certain
    min_px = max(1, round(MIN_ISLAND_PX * alpha.size / 1024 ** 2))
    for sure in (out >= 1 - HALF_STEP, out <= HALF_STEP):
        out = np.where(sure & ~remove_small_islands(sure, min_px), 0.5, out).astype(np.float32)
    return out


def detail_score(trimask):
    """
    Pure function. How much pixel-level detail a trimask has: its sure regions' boundary length per sqrt(area), and their
    component counts. A few smooth shapes score ~4-20; strands and specks score far higher.

    Args:
        trimask (np.ndarray): (H, W) float in [0, 1], e.g. a stored PNG / 255

    Returns:
        dict: {"edge_ratio": float, "white_components": int, "black_components": int}

    Examples:
        >>> square = np.full((100, 100), 0.5, np.float32); square[25:75, 25:75] = 1
        >>> detail_score(square)
        {'edge_ratio': 3.92, 'white_components': 1, 'black_components': 0}
        >>> speckle = square.copy(); speckle[::4, ::4] = 0
        >>> detail_score(speckle)["black_components"]
        625
    """
    out = {}
    edge, area = 0, 0
    for name, sure in (("white", trimask >= 1 - HALF_STEP), ("black", trimask <= HALF_STEP)):
        eroded = cv2.erode(sure.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE).astype(bool)
        edge += int((sure & ~eroded).sum())
        area += int(sure.sum())
        out[f"{name}_components"] = cv2.connectedComponents(sure.astype(np.uint8), connectivity=8)[0] - 1
    return {"edge_ratio": round(edge / max(area, 1) ** 0.5, 2), **out}


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
    check_trimask(trimask, alpha)
    return (trimask if method in CONSISTENCY_EXEMPT else check_consistent(alpha, trimask)), params


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
