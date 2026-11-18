"""
Trimask method noisy_band: the unknown band hugs the matte boundary, but its width varies smoothly along the boundary,
like an unevenly hand-drawn trimap. Width field = base width x exp(sigma x smooth noise), measured against the signed distance to the matte.

Related practice: trimaps for training matting nets come from a binarized alpha plus random erosion / dilation,
with a random kernel size (GCA Matting, arXiv:2001.04069: 5-29 px; Mask Guided Matting, arXiv:2012.06722: random
threshold, random erode/dilate order, kernel 1-30 px; arXiv:2101.09095: extra random morphology on the unknown region
to imitate noisy trimaps). Those widths are constant along an object; here the width itself is a random smooth field.
"""

import cv2
import numpy as np
from sana_alpha.trimask import finalize

BASE_PX = (4, 40)  # Base band half-width, log-uniform, drawn separately for the inside and the outside; pixels of a 1024² matte
NOISE_CELLS = (2, 10)  # Coarse noise grid size per image side; fewer cells = slower variation along the boundary
SIGMA = (0.35, 1.0)  # Log-width standard deviation: width varies by a factor of about exp(+-2 sigma)
DETAIL_CELLS_FACTOR = 4  # The fine octave has this many times more cells than the coarse one
DETAIL_WEIGHT = 0.35  # Fine octave amplitude relative to the coarse one
MIN_WIDTH_PX = 1.0  # The band never vanishes: it is at least this wide on each side
MAX_WIDTH_PX = 120.0  # Cap on the width field, so a high-noise peak does not swallow the whole object
THRESHOLD = 0.5  # Alpha above this is object before the distance transform
NOISE_SEED_MAX = 2**31


def signed_distance(binary):
    """
    Pure function. Signed Euclidean distance to the boundary of a binary mask, in pixels: positive inside, negative outside.

    Args:
        binary (np.ndarray): (H, W) bool, True = object

    Returns:
        np.ndarray: (H, W) float32; a pixel next to the boundary is +-1

    Examples:
        >>> signed_distance(np.array([[False, False, True, True, True]])).tolist()
        [[-2.0, -1.0, 1.0, 2.0, 3.0]]
    """
    inside = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return inside - outside


def smooth_noise(rng, shape, cells):
    """
    Near-pure function (advances rng). Low-frequency random field: a (cells, cells) gaussian grid, cubic-upsampled, standardized.

    Args:
        rng (np.random.Generator)
        shape (tuple[int, int]): (H, W) of the output
        cells (int): Grid cells per side, >= 2

    Returns:
        np.ndarray: (H, W) float32 with mean 0 and std 1

    Examples:
        >>> field = smooth_noise(np.random.default_rng(0), (64, 96), 4)
        >>> field.shape, abs(round(float(field.mean()), 3)), round(float(field.std()), 3)
        ((64, 96), 0.0, 1.0)
        >>> bool(np.abs(np.diff(field, axis=1)).max() < 0.3)  # Smooth: neighbouring pixels are close
        True
    """
    grid = rng.standard_normal((cells, cells)).astype(np.float32)
    field = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (field - field.mean()) / field.std()


def width_field(noise_seed, shape, base_px, sigma, cells):
    """
    Near-pure function (seeds its own rng from noise_seed, so it is deterministic). Band width in pixels at every pixel:
    base_px x exp(sigma x noise), noise = coarse octave + a weaker fine octave, clipped to [MIN_WIDTH_PX, MAX_WIDTH_PX].

    Args:
        noise_seed (int): Seeds the noise
        shape (tuple[int, int]): (H, W)
        base_px (float): Width where the noise is 0
        sigma (float): Log-width standard deviation
        cells (int): Coarse octave grid cells per side

    Returns:
        np.ndarray: (H, W) float32 pixels

    Examples:
        >>> w = width_field(0, (128, 128), base_px=10, sigma=0.7, cells=3)
        >>> w.shape, bool(w.min() >= MIN_WIDTH_PX), bool(w.max() <= MAX_WIDTH_PX), bool(w.max() > 2 * w.min())
        ((128, 128), True, True, True)
    """
    rng = np.random.default_rng(noise_seed)
    noise = smooth_noise(rng, shape, cells) + DETAIL_WEIGHT * smooth_noise(rng, shape, cells * DETAIL_CELLS_FACTOR)
    noise /= noise.std()
    return np.clip(base_px * np.exp(sigma * noise), MIN_WIDTH_PX, MAX_WIDTH_PX).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the inside and outside base widths (log-uniform over BASE_PX, independent),
    the noise strength sigma, the coarse noise grid size and the noise seed (so the params alone reproduce the trimask).

    Examples:
        >>> sorted(sample_params(np.random.default_rng(0)))
        ['base_in_px', 'base_out_px', 'cells', 'noise_seed', 'sigma', 'threshold']
    """
    base_in_px, base_out_px = np.exp(rng.uniform(np.log(BASE_PX[0]), np.log(BASE_PX[1]), size=2))
    return {
        "base_in_px": round(float(base_in_px), 2),
        "base_out_px": round(float(base_out_px), 2),
        "sigma": round(float(rng.uniform(*SIGMA)), 3),
        "cells": int(rng.integers(NOISE_CELLS[0], NOISE_CELLS[1] + 1)),
        "noise_seed": int(rng.integers(NOISE_SEED_MAX)),
        "threshold": THRESHOLD,
    }


def make_trimask(alpha, rng, base_in_px, base_out_px, sigma, cells, noise_seed, threshold):
    """
    Near-pure function (advances rng, only in finalize; otherwise all randomness is in noise_seed). Binarizes alpha, takes its signed distance sd (positive
    inside), draws two independent smooth width fields w_in, w_out and sets 1 where sd >= w_in, 0 where sd <= -w_out, 0.5 between.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Only finalize uses it (draws its dilation radius)
        base_in_px, base_out_px (float): Base band width inside / outside the boundary, pixels
        sigma (float): Log-width noise strength
        cells (int): Coarse noise grid cells per side
        noise_seed (int): Seeds both width fields (the outside field uses noise_seed + 1)
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> params = dict(base_in_px=8, base_out_px=8, sigma=0.6, cells=3, noise_seed=1, threshold=0.5)
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), **params)
        >>> sorted(np.unique(trimask).tolist()), used == params
        ([0.0, 0.5, 1.0], True)
        >>> float(trimask[64, 64]), float(trimask[0, 0]), float(trimask[64, 32])  # Object core, far background, on the boundary: unknown
        (1.0, 0.0, 0.5)
        >>> widths = (trimask[64, :64] == 0.5).sum(), (trimask[:64, 64] == 0.5).sum()  # Band width differs along the boundary
        >>> bool(widths[0] != widths[1])
        True
    """
    shape = alpha.shape
    sd = signed_distance(alpha > threshold)
    w_in = width_field(noise_seed, shape, base_in_px, sigma, cells)
    w_out = width_field(noise_seed + 1, shape, base_out_px, sigma, cells)
    trimask = np.where(sd >= w_in, 1, np.where(sd <= -w_out, 0, 0.5)).astype(np.float32)
    params = {"base_in_px": base_in_px, "base_out_px": base_out_px, "sigma": sigma, "cells": cells,
              "noise_seed": noise_seed, "threshold": threshold}
    return finalize(alpha, trimask, rng), params
