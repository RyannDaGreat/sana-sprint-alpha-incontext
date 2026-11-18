"""
Trimask method grid_coarsen: coarse blocky "pixel-art" user mask. The matte is averaged over a random (optionally shifted
and rotated) square grid; cells fully inside the object become 1, fully outside 0, cells straddling the edge 0.5.

Sources (how coarse / imperfect guidance is simulated from a ground-truth matte):
  - Yu et al., "Mask Guided Matting via Progressive Refinement Network", CVPR 2021, arXiv:2012.06722 (coarse masks made by degrading the matte)
  - Semantic-guided Automatic Natural Image Matting, arXiv:2103.17020 (random erosion/dilation/blur to simulate imperfect segmentations)
  - Towards Enhancing Fine-grained Details for Image Matting, arXiv:2101.09095 (training trimaps from GT alpha are unrealistically clean; randomize)
"""

import einx
import numpy as np

CELL_PX = (16, 128)  # Cell edge, log-uniform, in pixels of a 1024² matte
ROTATED_PROBABILITY = 0.4  # Else the grid is axis-aligned (like a low-resolution mask)
MAX_ANGLE_DEG = 45  # Rotated grids use an angle in ±this (the grid is 90° symmetric, so this covers all)
LOW_RANGE = (0.02, 0.15)  # Cell mean <= low is surely background (0 would make any stray hair pixel an unknown cell)
HIGH_RANGE = (0.85, 0.98)  # Cell mean >= high is surely object


def cell_ids(shape, cell_px, offset, angle_deg):
    """
    Pure function. Index of the grid cell containing each pixel, for a square grid of edge cell_px that is rotated
    by angle_deg about the image center and shifted by offset (x, y) pixels.

    Args:
        shape (tuple[int, int]): (H, W)
        cell_px (float): Cell edge in pixels
        offset (tuple[float, float]): (x, y) grid shift in pixels, in the rotated frame
        angle_deg (float): Grid rotation in degrees

    Returns:
        np.ndarray: (H, W) int64 in [0, n_cells)

    Examples:
        >>> cell_ids((4, 6), 2, (0, 0), 0).tolist()
        [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5], [3, 3, 4, 4, 5, 5]]
        >>> cell_ids((2, 4), 2, (1, 0), 0).tolist()  # Shifted by one pixel: the first column is its own cell
        [[0, 1, 1, 2], [0, 1, 1, 2]]
    """
    height, width = shape
    theta = np.deg2rad(angle_deg)
    x = np.arange(width, dtype=np.float64) + 0.5 - width / 2  # Pixel centers relative to the image center
    y = np.arange(height, dtype=np.float64) + 0.5 - height / 2
    u = einx.add("w, h -> h w", np.cos(theta) * x, np.sin(theta) * y) + width / 2  # Rotated-frame x
    v = einx.add("w, h -> h w", -np.sin(theta) * x, np.cos(theta) * y) + height / 2  # Rotated-frame y
    ix = np.floor((u + offset[0]) / cell_px).astype(np.int64)
    iy = np.floor((v + offset[1]) / cell_px).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    return iy * (ix.max() + 1) + ix


def cell_means(values, ids):
    """
    Pure function. Mean of values over each cell, mapped back to every pixel (nearest upsampling of the coarse grid).

    Args:
        values (np.ndarray): (H, W) float
        ids (np.ndarray): (H, W) int cell index from cell_ids

    Returns:
        np.ndarray: (H, W) float64, the mean of the pixel's cell

    Examples:
        >>> values = np.array([[0, 1, 1, 1]], np.float32)
        >>> cell_means(values, np.array([[0, 0, 1, 1]])).tolist()
        [[0.5, 0.5, 1.0, 1.0]]
    """
    sums = np.bincount(ids.ravel(), weights=values.ravel())
    counts = np.bincount(ids.ravel())
    means = sums / np.maximum(counts, 1)
    return means[ids]


def sample_params(rng):
    """
    Near-pure function (advances rng). cell_px log-uniform in CELL_PX; random grid offset in [0, cell_px)²;
    angle 0 with probability 1 - ROTATED_PROBABILITY, else uniform in ±MAX_ANGLE_DEG; low / high thresholds from LOW_RANGE / HIGH_RANGE.

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params)
        ['angle_deg', 'cell_px', 'high', 'low', 'offset_x', 'offset_y']
        >>> CELL_PX[0] <= params['cell_px'] <= CELL_PX[1]
        True
    """
    cell_px = int(round(np.exp(rng.uniform(*np.log(CELL_PX)))))
    angle_deg = float(rng.uniform(-MAX_ANGLE_DEG, MAX_ANGLE_DEG)) if rng.random() < ROTATED_PROBABILITY else 0.0
    return {"cell_px": cell_px, "offset_x": float(rng.uniform(0, cell_px)), "offset_y": float(rng.uniform(0, cell_px)),
            "angle_deg": angle_deg, "low": float(rng.uniform(*LOW_RANGE)), "high": float(rng.uniform(*HIGH_RANGE))}


def make_trimask(alpha, rng, cell_px, offset_x, offset_y, angle_deg, low, high):
    """
    Pure function (rng is unused: deterministic given its params). Averages alpha over the grid, then per cell:
    mean >= high -> 1 (object), mean <= low -> 0 (background), else 0.5 (unknown). Every pixel of a cell gets its cell's value.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        cell_px (int): Cell edge in pixels
        offset_x (float), offset_y (float): Grid shift in pixels
        angle_deg (float): Grid rotation in degrees
        low (float), high (float): Cell-mean thresholds, low < high

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((8, 8), np.float32); alpha[:, 3:] = 1  # Edge at x = 3, inside the cell x in [2, 4)
        >>> trimask, params = make_trimask(alpha, None, 2, 0, 0, 0, 0.05, 0.95)
        >>> trimask[0].tolist()
        [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
        >>> sorted(params)
        ['angle_deg', 'cell_px', 'high', 'low', 'offset_x', 'offset_y']
    """
    means = cell_means(alpha, cell_ids(alpha.shape, cell_px, (offset_x, offset_y), angle_deg))
    trimask = np.where(means >= high, 1, np.where(means <= low, 0, 0.5)).astype(np.float32)
    params = {"cell_px": cell_px, "offset_x": offset_x, "offset_y": offset_y, "angle_deg": angle_deg, "low": low, "high": high}
    return trimask, params
