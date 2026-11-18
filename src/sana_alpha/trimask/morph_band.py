"""Trimask method morph_band: the unknown band lies between the eroded and the dilated binary matte. Reference method of the contract in __init__.py."""

import cv2
import numpy as np

RADIUS_PX = (2, 48)  # Erode and dilate radii are drawn independently from this range, in pixels of a 1024² matte
THRESHOLD = 0.5  # Alpha above this is object before the morphology


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


def band_trimask(sure_object, maybe_object):
    """
    Pure function. 1 where surely object, 0 outside maybe_object, 0.5 between.

    Args:
        sure_object (np.ndarray): (H, W) bool
        maybe_object (np.ndarray): (H, W) bool, a superset of sure_object

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> band_trimask(np.array([True, False, False]), np.array([True, True, False])).tolist()
        [1.0, 0.5, 0.0]
    """
    return np.where(sure_object, 1, np.where(maybe_object, 0.5, 0)).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws erode_px and dilate_px uniformly from RADIUS_PX, independently.

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {'erode_px': 41, 'dilate_px': 31, 'threshold': 0.5}
    """
    erode_px, dilate_px = (int(r) for r in rng.integers(RADIUS_PX[0], RADIUS_PX[1] + 1, size=2))
    return {"erode_px": erode_px, "dilate_px": dilate_px, "threshold": THRESHOLD}


def make_trimask(alpha, rng, erode_px, dilate_px, threshold):
    """
    Pure function (rng is unused: the method is deterministic given its params). Binarizes alpha, then
    object = eroded by a disk of erode_px, background = outside the dilation by a disk of dilate_px, unknown = between.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        erode_px (int), dilate_px (int): Disk radii in pixels
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((1, 9), np.float32); alpha[0, 3:6] = 1
        >>> trimask, params = make_trimask(alpha, None, erode_px=1, dilate_px=2, threshold=0.5)
        >>> trimask.tolist()
        [[0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]]
        >>> params
        {'erode_px': 1, 'dilate_px': 2, 'threshold': 0.5}
    """
    binary = (alpha > threshold).astype(np.uint8)
    sure = cv2.erode(binary, disk(erode_px), borderType=cv2.BORDER_REPLICATE).astype(bool)
    maybe = cv2.dilate(binary, disk(dilate_px), borderType=cv2.BORDER_REPLICATE).astype(bool)
    return band_trimask(sure, maybe), {"erode_px": erode_px, "dilate_px": dilate_px, "threshold": threshold}
