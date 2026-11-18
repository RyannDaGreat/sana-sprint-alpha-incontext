"""
Trimask method unknown_patches: the exact binary matte, with random gray blobs painted over it, on the silhouette edge and
away from it, so the model must invent parts of the silhouette (inpainting-like). Optionally a thin gray band along the edge.

Blobs are rotated ellipses whose outline is bent by a few low-frequency radial harmonics (smooth random shapes). "Edge" blobs
are centered on random silhouette pixels (they hide the boundary); "free" blobs are centered anywhere in the padded object
bounding box (they punch holes in the object or cover empty background next to it).

Sources (how such masks are made):
    - Yu et al., "Free-Form Image Inpainting with Gated Convolution", ICCV 2019, arXiv 1806.03589: random free-form masks drawn
      on the fly during training; masks must be diverse, controllable and look like real user input.
    - Zhang et al., "Deep Inception Generative Network for Cognitive Image Inpainting", arXiv 1812.01458: training holes made of
      random rectangles, circles and ellipses with random size, rotation and position.
    - Xu et al., "Deep Image Matting", CVPR 2017, arXiv 1703.03872: trimaps for training are made by randomly dilating the
      ground-truth alpha (the optional edge band here is the same idea).
"""

import cv2
import einx
import numpy as np

from .morph_band import band_trimask, disk
from sana_alpha.trimask import finalize

N_EDGE = (0, 12)  # Edge blobs: inclusive range of the count
N_FREE = (0, 8)  # Free blobs: inclusive range of the count
RADIUS_MIN_FRAC = (0.015, 0.05)  # Smallest blob radius, as a fraction of min(H, W); log-uniform
RADIUS_MAX_OVER_MIN = (2.0, 8.0)  # Largest blob radius divided by the smallest; log-uniform
IRREGULARITY = (0.05, 0.45)  # Total amplitude of the radial harmonics, as a fraction of the radius (< 1 keeps the blob star-shaped)
ASPECT_MAX = (1.0, 2.5)  # Ellipse axis ratio is drawn log-uniform from [1, aspect_max]
BAND_PROBABILITY = 0.4  # Chance that a thin gray band is also drawn along the silhouette
BAND_PX = (1, 8)  # Band radius (it extends this far on both sides of the edge), in pixels of a 1024² matte
THRESHOLD = 0.5  # Alpha above this is object
FREE_PAD_FRAC = 0.1  # Free blob centers may lie this fraction of min(H, W) outside the object's bounding box
HARMONICS = (2, 3, 4)  # Radial harmonic orders (order 1 would only shift the blob)
OUTLINE_VERTICES = 64  # Polygon vertices per blob outline


def log_uniform(rng, low, high):
    """
    Near-pure function (advances rng). A float drawn log-uniformly from [low, high].

    Examples:
        >>> 1.0 <= log_uniform(np.random.default_rng(0), 1.0, 100.0) <= 100.0
        True
    """
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def blob_outline(center_xy, radius, aspect, angle, harmonics, vertices=OUTLINE_VERTICES):
    """
    Pure function. Polygon outline of one smooth random blob: an ellipse (long axis `radius * sqrt(aspect)`, short axis
    `radius / sqrt(aspect)`, rotated by angle) with its radius scaled by 1 + sum_k amplitude_k cos(order_k t + phase_k).

    Args:
        center_xy (tuple[float, float]): (x, y) in pixels
        radius (float): Geometric mean of the two ellipse semi-axes, pixels
        aspect (float): Long axis / short axis, >= 1
        angle (float): Rotation of the long axis, radians
        harmonics (np.ndarray): (K, 3) float, rows (order, amplitude, phase)
        vertices (int): Number of outline points

    Returns:
        np.ndarray: (vertices, 2) float32, columns (x, y)

    Examples:
        >>> circle = blob_outline((10, 20), 5, 1, 0, np.zeros((0, 3)), vertices=8)
        >>> circle.shape, np.allclose(np.hypot(circle[:, 0] - 10, circle[:, 1] - 20), 5)
        ((8, 2), True)
        >>> long = blob_outline((0, 0), 5, 4, 0, np.zeros((0, 3)), vertices=4)
        >>> np.round(long, 3).tolist()  # Semi-axes 10 (x) and 2.5 (y)
        [[10.0, 0.0], [0.0, 2.5], [-10.0, 0.0], [-0.0, -2.5]]
        >>> bent = blob_outline((0, 0), 5, 1, 0, np.array([[2, 0.3, 0.0]]), vertices=4)
        >>> np.round(np.hypot(*bent.T), 3).tolist()  # Radius 5 * (1 + 0.3 cos(2t)) at t = 0, 90, 180, 270 degrees
        [6.5, 3.5, 6.5, 3.5]
    """
    t = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    orders, amplitudes, phases = harmonics.reshape(-1, 3).T  # (K,) each; K = 0 gives a plain ellipse
    scale = 1 + (amplitudes[:, None] * np.cos(orders[:, None] * t + phases[:, None])).sum(axis=0)  # (vertices,)
    semi_x, semi_y = radius * np.sqrt(aspect), radius / np.sqrt(aspect)
    local = np.stack([semi_x * scale * np.cos(t), semi_y * scale * np.sin(t)], axis=1)  # (vertices, 2) ellipse frame
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # Rotates the ellipse frame by angle
    return (einx.dot("v e, e i -> v i", local, rotation) + np.asarray(center_xy)).astype(np.float32)


def draw_blobs(shape, outlines):
    """
    Pure function. Union of filled polygons (anti-aliasing is not needed: the trimask is 3-level).

    Args:
        shape (tuple[int, int]): (H, W)
        outlines (list[np.ndarray]): Each (V, 2) float, columns (x, y)

    Returns:
        np.ndarray: (H, W) bool, True inside any outline

    Examples:
        >>> square = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], np.float32)
        >>> draw_blobs((6, 6), [square]).astype(int).tolist()
        [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]]
        >>> bool(draw_blobs((3, 3), []).any())
        False
        >>> int(draw_blobs((6, 6), [square, square + 1]).sum())  # Overlap is a union (16 + 16 - 9), not an even-odd hole
        23
    """
    mask = np.zeros(shape, np.uint8)
    for outline in outlines:  # One call per polygon: a joint call fills overlaps with even-odd parity, leaving holes
        cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 1)
    return mask.astype(bool)


def edge_pixels(binary):
    """
    Pure function. Inner boundary pixels of a binary matte (object pixels with a background 4-neighbor).

    Args:
        binary (np.ndarray): (H, W) bool

    Returns:
        np.ndarray: (N, 2) int, columns (x, y); N = 0 for an empty or full matte

    Examples:
        >>> binary = np.zeros((5, 5), bool); binary[1:4, 1:4] = True
        >>> sorted(edge_pixels(binary).tolist())
        [[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 3]]
    """
    inside = cv2.erode(binary.astype(np.uint8), disk(1), borderType=cv2.BORDER_REPLICATE).astype(bool)
    ys, xs = np.nonzero(binary & ~inside)
    return np.stack([xs, ys], axis=1)


def padded_bbox(binary, pad):
    """
    Pure function. Bounding box of the object, grown by pad pixels and clipped to the image; the whole image if the matte is empty.

    Args:
        binary (np.ndarray): (H, W) bool
        pad (int): Pixels

    Returns:
        tuple[int, int, int, int]: (x_min, y_min, x_max, y_max), max exclusive

    Examples:
        >>> binary = np.zeros((10, 20), bool); binary[4:6, 8:12] = True
        >>> padded_bbox(binary, 2)
        (6, 2, 14, 8)
        >>> padded_bbox(np.zeros((10, 20), bool), 2)
        (0, 0, 20, 10)
    """
    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return 0, 0, width, height
    return int(max(xs.min() - pad, 0)), int(max(ys.min() - pad, 0)), int(min(xs.max() + 1 + pad, width)), int(min(ys.max() + 1 + pad, height))


def band_mask(binary, band_px):
    """
    Pure function. Pixels within band_px of the silhouette edge, on both sides (dilation minus erosion); none for band_px = 0.

    Args:
        binary (np.ndarray): (H, W) bool
        band_px (int): Radius in pixels, >= 0

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> binary = np.zeros((1, 9), bool); binary[0, 3:6] = True
        >>> band_mask(binary, 1).astype(int).tolist()
        [[0, 0, 1, 1, 0, 1, 1, 0, 0]]
        >>> bool(band_mask(binary, 0).any())
        False
    """
    if band_px == 0:
        return np.zeros_like(binary)
    as_uint8 = binary.astype(np.uint8)
    border = cv2.BORDER_REPLICATE
    return cv2.dilate(as_uint8, disk(band_px), borderType=border).astype(bool) & ~cv2.erode(as_uint8, disk(band_px), borderType=border).astype(bool)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the blob counts, size range, shape irregularity and the optional band.

    Returns:
        dict: n_edge, n_free (ints; at least one blob in total unless a band is drawn), radius_min_frac, radius_max_frac (floats,
        fractions of min(H, W)), irregularity, aspect_max (floats), band_px (int, 0 = no band), threshold (float)

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params)
        ['aspect_max', 'band_px', 'irregularity', 'n_edge', 'n_free', 'radius_max_frac', 'radius_min_frac', 'threshold']
        >>> params["n_edge"] + params["n_free"] >= 1 and params["radius_min_frac"] < params["radius_max_frac"]
        True
    """
    n_edge, n_free = (int(rng.integers(low, high + 1)) for low, high in (N_EDGE, N_FREE))
    if n_edge + n_free == 0:
        n_edge = 1
    radius_min = log_uniform(rng, *RADIUS_MIN_FRAC)
    return {
        "n_edge": n_edge,
        "n_free": n_free,
        "radius_min_frac": radius_min,
        "radius_max_frac": radius_min * log_uniform(rng, *RADIUS_MAX_OVER_MIN),
        "irregularity": float(rng.uniform(*IRREGULARITY)),
        "aspect_max": float(rng.uniform(*ASPECT_MAX)),
        "band_px": int(rng.integers(BAND_PX[0], BAND_PX[1] + 1)) if rng.random() < BAND_PROBABILITY else 0,
        "threshold": THRESHOLD,
    }


def sample_outline(rng, center_xy, radius_range_px, irregularity, aspect_max):
    """
    Near-pure function (advances rng). One random blob outline at center_xy: radius log-uniform in radius_range_px, aspect log-uniform
    in [1, aspect_max], random rotation, and radial harmonics whose amplitudes sum to at most irregularity.

    Args:
        rng (np.random.Generator)
        center_xy (tuple[float, float]): (x, y) pixels
        radius_range_px (tuple[float, float]): Pixels
        irregularity (float): Total harmonic amplitude, < 1
        aspect_max (float): >= 1

    Returns:
        np.ndarray: (OUTLINE_VERTICES, 2) float32, columns (x, y)

    Examples:
        >>> outline = sample_outline(np.random.default_rng(0), (100, 100), (10, 20), 0.3, 2.0)
        >>> outline.shape
        (64, 2)
        >>> bool((np.hypot(outline[:, 0] - 100, outline[:, 1] - 100) < 20 * np.sqrt(2) * 1.3 + 1e-3).all())
        True
    """
    radius = log_uniform(rng, *radius_range_px)
    aspect = log_uniform(rng, 1.0, max(aspect_max, 1.0))
    weights = rng.dirichlet(np.ones(len(HARMONICS))) * irregularity * rng.random()  # Sum <= irregularity
    harmonics = np.stack([HARMONICS, weights, rng.uniform(0, 2 * np.pi, len(HARMONICS))], axis=1)
    return blob_outline(center_xy, radius, aspect, rng.uniform(0, np.pi), harmonics)


def make_trimask(alpha, rng, n_edge, n_free, radius_min_frac, radius_max_frac, irregularity, aspect_max, band_px, threshold):
    """
    Near-pure function (advances rng). Binarizes alpha, paints n_edge blobs centered on random silhouette pixels and n_free blobs
    centered uniformly in the padded object bounding box, adds the optional edge band. Painted pixels become unknown (0.5); the rest
    keeps the exact binary matte (1 object, 0 background). If the matte has no edge (empty or full), edge blobs are centered like
    free blobs.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator)
        n_edge (int), n_free (int): Blob counts
        radius_min_frac (float), radius_max_frac (float): Blob radius range, fractions of min(H, W)
        irregularity (float): Total radial harmonic amplitude, fraction of the radius
        aspect_max (float): Largest ellipse axis ratio
        band_px (int): Radius of the extra gray band along the edge, pixels; 0 = none
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> params = dict(n_edge=3, n_free=2, radius_min_frac=0.05, radius_max_frac=0.15, irregularity=0.3, aspect_max=1.5, band_px=0, threshold=0.5)
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), **params)
        >>> trimask.shape, sorted(np.unique(trimask).tolist()), used == params
        ((128, 128), [0.0, 0.5, 1.0], True)
        >>> bool(((trimask == 1) <= (alpha == 1)).all() and ((trimask == 0) <= (alpha == 0)).all())  # Gray never flips a sure pixel
        True
        >>> no_blobs = dict(params, n_edge=0, n_free=0)
        >>> bool((make_trimask(alpha, np.random.default_rng(0), **no_blobs)[0] == alpha).all())  # Nothing painted: the exact matte
        True
    """
    binary = alpha > threshold
    height, width = binary.shape
    short_side = min(height, width)
    radius_range_px = (radius_min_frac * short_side, radius_max_frac * short_side)
    edge = edge_pixels(binary)
    x_min, y_min, x_max, y_max = padded_bbox(binary, round(FREE_PAD_FRAC * short_side))

    def free_center():
        return rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)

    edge_centers = [tuple(edge[rng.integers(len(edge))]) if len(edge) else free_center() for _ in range(n_edge)]
    free_centers = [free_center() for _ in range(n_free)]
    outlines = [sample_outline(rng, c, radius_range_px, irregularity, aspect_max) for c in edge_centers + free_centers]
    unknown = draw_blobs(binary.shape, outlines) | band_mask(binary, band_px)
    params = dict(n_edge=n_edge, n_free=n_free, radius_min_frac=radius_min_frac, radius_max_frac=radius_max_frac,
                  irregularity=irregularity, aspect_max=aspect_max, band_px=band_px, threshold=threshold)
    return finalize(alpha, band_trimask(binary & ~unknown, binary | unknown), rng), params
