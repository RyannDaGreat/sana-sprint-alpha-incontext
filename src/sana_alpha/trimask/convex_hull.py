"""
Trimask method convex_hull: unknown = (optionally dilated) convex hull of the object minus an eroded core. Mimics a "roughly this blob" input.

Black outside the hull, gray inside the hull but outside the core, white = core. The hull is either one hull of the whole object
("global") or one hull per connected component ("per_component"), which keeps the gaps between separate parts black.

Sources (no paper builds trimaps from hulls for augmentation; these are the closest):
  - Li and Lu, "Natural Image Matting via Guided Contextual Attention" (GCA-Matting), 2020, arXiv:2001.04069:
    training trimaps come from random erosion of the alpha (1 to 29 px); here erosion makes the core and the hull replaces dilation.
  - Yu et al., "Towards Enhancing Fine-grained Details for Image Matting", 2021, arXiv:2101.09095: coarse trimaps with large unknown regions.
  - "A Hierarchical Image Matting Model for Blood Vessel Segmentation in Fundus Images", 2017, arXiv:1701.00892: convex hull and solidity of a region as trimap features.
"""

import cv2
import numpy as np
from sana_alpha.trimask import restrict_to_matte

CORE_ERODE_PX = (2, 48)  # Core erosion radius range, in pixels of a 1024² matte
HULL_DILATE_PX = (4, 40)  # Hull dilation radius range when dilation is used
HULL_DILATE_PROBABILITY = 0.6  # Else the hull is used as is
PER_COMPONENT_PROBABILITY = 0.4  # Else one global hull
MIN_COMPONENT_AREA_FRACTION = 1e-4  # Components smaller than this fraction of the image get no hull (speckle noise)
THRESHOLD = 0.5  # Alpha above this is object
MODES = ("global", "per_component")


def grow(mask, radius_px):
    """
    Pure function. Dilates a binary mask by a disk of radius_px via the Euclidean distance transform (fast for any radius).

    Args:
        mask (np.ndarray): (H, W) bool
        radius_px (float): >= 0

    Returns:
        np.ndarray: (H, W) bool, a superset of mask

    Examples:
        >>> grow(np.array([[False, False, True, False, False]]), 1).tolist()
        [[False, True, True, True, False]]
    """
    if radius_px <= 0:
        return mask
    distance_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_to_mask <= radius_px


def shrink(mask, radius_px):
    """
    Pure function. Erodes a binary mask by a disk of radius_px; the image border does not erode the mask.

    Args:
        mask (np.ndarray): (H, W) bool
        radius_px (float): >= 0

    Returns:
        np.ndarray: (H, W) bool, a subset of mask

    Examples:
        >>> shrink(np.array([[True] * 5 + [False]]), 2).tolist()
        [[True, True, True, False, False, False]]
    """
    if radius_px <= 0:
        return mask
    distance_to_background = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_to_background > radius_px


def hull_mask(binary, per_component, min_area_px):
    """
    Pure function. Filled convex hull of the object.

    Args:
        binary (np.ndarray): (H, W) bool, True = object
        per_component (bool): True = one hull per external contour (union of hulls); False = one hull of all object pixels
        min_area_px (float): per_component only: contours with a smaller area get no hull

    Returns:
        np.ndarray: (H, W) bool, True inside the hull(s). All False if there is no object.

    Examples:
        >>> binary = np.zeros((12, 12), bool); binary[1:4, 1:4] = True; binary[8:11, 8:11] = True
        >>> int(hull_mask(binary, per_component=False, min_area_px=0).sum()) > int(hull_mask(binary, per_component=True, min_area_px=0).sum())
        True
        >>> int(hull_mask(binary, per_component=True, min_area_px=0).sum())
        18
    """
    hull = np.zeros(binary.shape, np.uint8)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not per_component:
        contours = [np.concatenate(contours)] if contours else []
    else:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    for contour in contours:
        cv2.fillPoly(hull, [cv2.convexHull(contour)], 1)
    return hull.astype(bool)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the hull mode, hull dilation and core erosion.

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {'mode': 'global', 'hull_dilate_px': 15, 'erode_px': 3, 'threshold': 0.5}
    """
    mode = MODES[int(rng.random() < PER_COMPONENT_PROBABILITY)]
    hull_dilate_px = int(rng.integers(HULL_DILATE_PX[0], HULL_DILATE_PX[1] + 1)) if rng.random() < HULL_DILATE_PROBABILITY else 0
    erode_px = int(rng.integers(CORE_ERODE_PX[0], CORE_ERODE_PX[1] + 1))
    return {"mode": mode, "hull_dilate_px": hull_dilate_px, "erode_px": erode_px, "threshold": THRESHOLD}


def make_trimask(alpha, rng, mode, hull_dilate_px, erode_px, threshold):
    """
    Pure function (rng is unused: the method is deterministic given its params). Binarizes alpha; black outside the hull
    dilated by hull_dilate_px, white = the object eroded by erode_px (clipped to the dilated hull), gray in between.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        mode (str): "global" (one hull) or "per_component" (union of one hull per connected part)
        hull_dilate_px (int): Hull dilation radius in pixels, >= 0
        erode_px (int): Core erosion radius in pixels, >= 0
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((9, 9), np.float32); alpha[2:7, 1:4] = 1; alpha[2:7, 5:8] = 1
        >>> trimask, _ = make_trimask(alpha, None, "global", 0, 1, 0.5)
        >>> trimask[4].tolist()  # The gap between the two bars is inside the global hull, so unknown
        [0.0, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0]
        >>> trimask, _ = make_trimask(alpha, None, "per_component", 0, 1, 0.5)
        >>> trimask[4].tolist()
        [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
        >>> float(make_trimask(np.zeros((4, 4), np.float32), None, "global", 3, 3, 0.5)[0].max())
        0.0
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Known: {MODES}")
    binary = alpha > threshold
    min_area_px = MIN_COMPONENT_AREA_FRACTION * binary.size
    hull = grow(hull_mask(binary, mode == "per_component", min_area_px), hull_dilate_px)
    core = shrink(binary, erode_px) & hull
    trimask = np.where(core, 1, np.where(hull, 0.5, 0)).astype(np.float32)
    return restrict_to_matte(alpha, trimask), {"mode": mode, "hull_dilate_px": hull_dilate_px, "erode_px": erode_px, "threshold": threshold}
