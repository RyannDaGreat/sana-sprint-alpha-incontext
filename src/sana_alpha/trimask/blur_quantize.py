"""
Trimask method blur_quantize: Gaussian-blur the matte, then cut it into 3 levels with two thresholds.

Blurred values below `low` become background, above `high` become object, in between unknown. The unknown region
is a smooth, rounded halo around the edge whose width grows with sigma (about (z_high - z_low) sigma for a straight
edge, z = inverse normal CDF of the threshold). Unequal sigma_x and sigma_y give an anisotropic (smeared) halo.

Sources (none blurs-then-thresholds the matte to get a trimap; these are the closest practice):
    SGM, "Semantic-guided Automatic Natural Image Matting" (arXiv 2103.17020): random erosion/dilation then random Gaussian
        blur to make soft coarse masks for matting training.
    Liu et al., "Towards Enhancing Fine-grained Details for Image Matting" (WACV 2021, arXiv 2101.09095): random
        morphology on the unknown band to imitate noisy user trimaps.
    DPT, "Disentangled Pre-training for Image Matting" (WACV 2024, arXiv 2304.00784): trimap from a threshold on alpha.
"""

import cv2
import numpy as np
from sana_alpha.trimask import finalize

SIGMA_PX = (2.0, 32.0)  # Isotropic sigma is log-uniform in this range, in pixels of a 1024² matte
ANISOTROPIC_PROB = 0.3  # Chance that the two axes get different sigmas
MAX_ASPECT = 4.0  # An anisotropic draw makes one axis up to this many times wider (log-uniform from 1)
LOW_RANGE = (0.05, 0.45)  # Blurred values below low are background
HIGH_RANGE = (0.55, 0.95)  # Blurred values above high are object
UNKNOWN = 0.5  # Trimask value of unknown pixels


def blur(alpha, sigma_x, sigma_y):
    """
    Pure function. Gaussian blur with separate horizontal and vertical sigma; image borders are replicated.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1]
        sigma_x (float), sigma_y (float): Standard deviations along W and along H, in pixels, > 0

    Returns:
        np.ndarray: (H, W) float32 in [0, 1]

    Examples:
        >>> a = np.zeros((1, 9), np.float32); a[0, 4] = 1
        >>> b = blur(a, 1.0, 1.0)
        >>> b.shape, bool(b[0, 4] == b.max()), bool(abs(b[0, 3] - b[0, 5]) < 1e-6), round(float(b.sum()), 3)
        ((1, 9), True, True, 1.0)
    """
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REPLICATE)


def quantize3(values, low, high):
    """
    Pure function. 0 below low, 1 above high, 0.5 in between (both bounds inclusive in the unknown level).

    Args:
        values (np.ndarray): (H, W) float32 in [0, 1]
        low (float), high (float): Thresholds, low <= high

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> quantize3(np.array([[0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]], np.float32), 0.2, 0.8).tolist()
        [[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]]
    """
    return np.where(values > high, 1, np.where(values < low, 0, UNKNOWN)).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws sigma_x, sigma_y (log-uniform; unequal with probability ANISOTROPIC_PROB),
    low, high (uniform in LOW_RANGE and HIGH_RANGE).

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {'sigma_x': 12.379, 'sigma_y': 11.695, 'low': 0.375, 'high': 0.915}
    """
    sigma = float(np.exp(rng.uniform(*np.log(SIGMA_PX))))
    aspect = float(np.exp(rng.uniform(0, np.log(MAX_ASPECT)))) if rng.random() < ANISOTROPIC_PROB else 1.0
    sigma_x, sigma_y = (sigma * aspect, sigma) if rng.random() < 0.5 else (sigma, sigma * aspect)
    low, high = (float(rng.uniform(*r)) for r in (LOW_RANGE, HIGH_RANGE))
    return {"sigma_x": round(sigma_x, 3), "sigma_y": round(sigma_y, 3), "low": round(low, 3), "high": round(high, 3)}


def make_trimask(alpha, rng, sigma_x, sigma_y, low, high):
    """
    Pure function (rng is unused: the method is deterministic given its params). Blurs alpha, then quantizes it to 3 levels.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        sigma_x (float), sigma_y (float): Blur standard deviations along W and H, in pixels
        low (float), high (float): Background / object thresholds on the blurred matte

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((1, 41), np.float32); alpha[0, 20:] = 1  # a vertical edge between columns 19 and 20
        >>> trimask, params = make_trimask(alpha, None, sigma_x=3.0, sigma_y=3.0, low=0.2, high=0.8)
        >>> trimask[0, [0, 14, 16, 19, 20, 23, 25, 40]].tolist()
        [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        >>> params
        {'sigma_x': 3.0, 'sigma_y': 3.0, 'low': 0.2, 'high': 0.8}
    """
    trimask = quantize3(blur(alpha, sigma_x, sigma_y), low, high)
    return finalize(alpha, trimask, rng), {"sigma_x": sigma_x, "sigma_y": sigma_y, "low": low, "high": high}
