"""
Trimask method morph_band: the unknown band lies between the eroded and the dilated binary matte. Reference method of the contract in __init__.py.

Object = matte eroded by a structuring element, background = outside the matte dilated by another one, unknown = between.
The two elements are independent: radii differ (often much) between inside and outside, and either side may be a
rotated ellipse instead of a disk. This is the training-trimap recipe of deep matting, with a wider hyperparameter spread:
  - Li and Lu, "Natural Image Matting via Guided Contextual Attention" (GCA Matting, AAAI 2020), arXiv:2001.04069:
    dilation and erosion of the alpha by a random number of pixels in 5..29 (on 512² crops).
  - Cheng et al., "Deep Image Matting with Flexible Guidance Input", arXiv:2110.10898: trimaps by dilating and eroding the
    ground-truth alpha, then deforming them toward scribbles.
  - "Towards Enhancing Fine-grained Details for Image Matting", arXiv:2101.09095: random kernels 1..30 for dilation and
    1..10 for erosion (different ranges inside and outside), plus chained random erode/dilate steps for a noisier band.
"""

import cv2
import numpy as np
from sana_alpha.trimask import finalize

# All pixel values are for a 1024² matte
DISK_RADIUS_PX = (2, 80)  # Disk radius range, drawn log-uniformly per side (eroded objects of ~100 px inscribed radius survive)
ELLIPSE_RADIUS_PX = (2, 48)  # Ellipse semi-major range; capped lower than disks because cv2 kernel cost grows with kernel area
ELLIPSE_ASPECT = (1.5, 4.0)  # Ellipse semi-major / semi-minor, drawn log-uniformly
THRESHOLD_RANGE = (0.35, 0.65)  # Alpha above this is object before the morphology
LOPSIDED_RADIUS_PX = (1, 4)  # The thin side of a lopsided band
P_ELLIPSE = 0.3  # Chance that a trimask uses ellipses (on both sides) instead of disks
P_SYMMETRIC = 0.3  # Chance that erode and dilate radii are equal
P_LOPSIDED = 0.15  # Chance that one side is a thin sliver and the other is drawn normally (else: independent radii)


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


def ellipse_kernel(radius, aspect, angle_deg):
    """
    Pure function. A filled ellipse structuring element, point-symmetric by construction: semi-major radius,
    semi-minor max(1, radius / aspect), major axis rotated clockwise on screen (image coordinates, y down).

    Args:
        radius (int): Semi-major axis in pixels, >= 1
        aspect (float): Semi-major / semi-minor, >= 1
        angle_deg (float): Angle of the major axis from the x axis (pointing right), degrees, clockwise on screen

    Returns:
        np.ndarray: (2 radius + 1, 2 radius + 1) uint8 in {0, 1}

    Examples:
        >>> ellipse_kernel(4, 2.0, 0).tolist()[3:6]  # Wide: major axis along x (rows 3..5 of 9)
        [[0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]]
        >>> ellipse_kernel(4, 2.0, 90).T.tolist() == ellipse_kernel(4, 2.0, 0).tolist()  # 90 degrees is the transpose
        True
        >>> k = ellipse_kernel(6, 3.0, 45); (k == k[::-1, ::-1]).all()  # Point-symmetric, so dilation does not shift the mask
        np.True_
    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    theta = np.deg2rad(angle_deg)
    along = x * np.cos(theta) + y * np.sin(theta)
    across = -x * np.sin(theta) + y * np.cos(theta)
    semi_minor = max(1.0, radius / aspect)
    return ((along / radius) ** 2 + (across / semi_minor) ** 2 <= 1).astype(np.uint8)


def dilate(mask, radius, aspect=1.0, angle_deg=0.0):
    """
    Pure function. Binary dilation by a disk (aspect 1: exact Euclidean distance transform, fast at any radius)
    or by a rotated ellipse (cv2 kernel from ellipse_kernel). The image border does not erode: pixels outside the image copy the border.

    Args:
        mask (np.ndarray): (H, W) bool
        radius (int): Pixels (semi-major axis for an ellipse), >= 1
        aspect (float): 1 for a disk, else the ellipse semi-major / semi-minor
        angle_deg (float): Ellipse rotation, see ellipse_kernel (ignored for a disk)

    Returns:
        np.ndarray: (H, W) bool, a superset of mask

    Examples:
        >>> mask = np.zeros((1, 9), bool); mask[0, 4] = True
        >>> dilate(mask, 2).astype(int).tolist()
        [[0, 0, 1, 1, 1, 1, 1, 0, 0]]
        >>> dilate(mask, 2, aspect=2.0, angle_deg=90).astype(int).tolist()  # Tall ellipse: only 1 px sideways on this row
        [[0, 0, 0, 1, 1, 1, 0, 0, 0]]
        >>> bool(dilate(np.zeros((4, 4), bool), 3).any())  # Nothing to grow
        False
    """
    if aspect == 1:
        distance_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return distance_to_mask <= radius
    return cv2.dilate(mask.astype(np.uint8), ellipse_kernel(radius, aspect, angle_deg), borderType=cv2.BORDER_REPLICATE).astype(bool)


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


def log_uniform_int(rng, low, high):
    """
    Near-pure function (advances rng). An integer from [low, high], uniform in log space, so small radii are as likely as large ones per octave.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> radii = [log_uniform_int(rng, 2, 80) for _ in range(1000)]
        >>> min(radii), max(radii), int(np.median(radii))  # Median near the geometric mean sqrt(2 * 80) = 12.6
        (2, 80, 14)
    """
    return int(round(np.exp(rng.uniform(np.log(low), np.log(high)))))


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the radii, the structuring-element shapes and the binarization threshold.
    Band layout: 30% symmetric (erode = dilate), 55% independent radii, 15% lopsided (one side 1..4 px, the other normal).
    Shape: 70% disks, 30% ellipses (both sides, independent aspect 1.5..4 and angle); a disk has aspect 1 and angle 0.

    Returns:
        dict: erode_px, dilate_px (int), erode_aspect, dilate_aspect (float), erode_angle_deg, dilate_angle_deg (float), threshold (float)

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {'erode_px': 5, 'dilate_px': 5, 'erode_aspect': 1.0, 'dilate_aspect': 1.0, 'erode_angle_deg': 0.0, 'dilate_angle_deg': 0.0, 'threshold': 0.59}
    """
    ellipse = rng.random() < P_ELLIPSE
    radius_range = ELLIPSE_RADIUS_PX if ellipse else DISK_RADIUS_PX
    erode_px, dilate_px = (log_uniform_int(rng, *radius_range) for _ in range(2))
    layout = rng.random()
    if layout < P_SYMMETRIC:
        dilate_px = erode_px
    elif layout < P_SYMMETRIC + P_LOPSIDED:
        thin_px = int(rng.integers(LOPSIDED_RADIUS_PX[0], LOPSIDED_RADIUS_PX[1] + 1))
        if rng.random() < 0.5:
            erode_px = thin_px
        else:
            dilate_px = thin_px
    aspects = [round(float(np.exp(rng.uniform(*np.log(ELLIPSE_ASPECT)))), 2) if ellipse else 1.0 for _ in range(2)]
    angles = [round(float(rng.uniform(0, 180)), 1) if ellipse else 0.0 for _ in range(2)]
    return {"erode_px": erode_px, "dilate_px": dilate_px,
            "erode_aspect": aspects[0], "dilate_aspect": aspects[1],
            "erode_angle_deg": angles[0], "dilate_angle_deg": angles[1],
            "threshold": round(float(rng.uniform(*THRESHOLD_RANGE)), 2)}


def make_trimask(alpha, rng, erode_px, dilate_px, erode_aspect=1.0, dilate_aspect=1.0, erode_angle_deg=0.0, dilate_angle_deg=0.0, threshold=0.5):
    """
    Near-pure function (advances rng, only in finalize; otherwise the method is deterministic given its params). Binarizes alpha, then
    object = eroded by the erode element, background = outside the dilation by the dilate element, unknown = between.
    Erosion is dilation of the complement, so both sides share dilate(). An erosion larger than the object leaves no white.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Only finalize uses it (draws its dilation radius)
        erode_px (int), dilate_px (int): Element radii in pixels (semi-major for an ellipse)
        erode_aspect (float), dilate_aspect (float): 1 for a disk, else semi-major / semi-minor
        erode_angle_deg (float), dilate_angle_deg (float): Ellipse rotation, see ellipse_kernel
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> trimask, params = make_trimask(alpha, np.random.default_rng(0), erode_px=8, dilate_px=12)
        >>> trimask[64, ::4].tolist()  # Edges at x = 32 and 96: black below x = 20, unknown 20..39, white 40..87, unknown, black
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> params["threshold"]
        0.5
        >>> disk_alpha = np.zeros((101, 101), np.float32); disk_alpha[np.hypot(*np.mgrid[-50:51, -50:51]) <= 30] = 1
        >>> trimask, _ = make_trimask(disk_alpha, np.random.default_rng(0), erode_px=5, dilate_px=15)  # A ring of unknown, 20 px wide
        >>> trimask[50, 4:51:5].tolist()  # Along a row, edge at x=20: black below x=5, unknown 5..25, white from 26
        [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> trimask, _ = make_trimask(disk_alpha, np.random.default_rng(0), erode_px=20, dilate_px=10, erode_aspect=3.0, erode_angle_deg=0)  # Wide ellipse
        >>> sorted(np.unique(trimask).tolist())
        [0.0, 0.5, 1.0]
        >>> trimask[50, 35].item(), trimask[35, 50].item()  # Wide element: erodes 20 px along x (unknown), only ~7 along y (still sure)
        (0.5, 1.0)
    """
    binary = alpha > threshold
    sure = ~dilate(~binary, erode_px, erode_aspect, erode_angle_deg)
    maybe = dilate(binary, dilate_px, dilate_aspect, dilate_angle_deg)
    used = {"erode_px": erode_px, "dilate_px": dilate_px, "erode_aspect": erode_aspect, "dilate_aspect": dilate_aspect,
            "erode_angle_deg": erode_angle_deg, "dilate_angle_deg": dilate_angle_deg, "threshold": threshold}
    return finalize(alpha, band_trimask(sure, maybe), rng), used
