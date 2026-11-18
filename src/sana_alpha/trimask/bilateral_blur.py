"""
Trimask method bilateral_blur: an edge-aware unknown region. The matte is blurred, then re-sharpened along the structure of the
object's RGB image by a guided filter, then cut into 3 levels with two thresholds.

Where the RGB image has an edge, the filtered matte stays sharp (thin unknown band); where it is flat (e.g. object color close
to the gray background, or an interior texture-free area), the matte stays blurry (wide unknown band). The unknown region therefore
bleeds along image structure instead of growing isotropically as in blur_quantize.

INTERFACE EXTENSION: this method needs the RGB image, which the contract does not pass. make_trimask takes an extra keyword
`rgb` ((H, W, 3) float32 in [0, 1], RGB order, the object on 0.5 gray = magick_samples.rgb_on_gray / 255). The caller MUST pass it
(trimask_augment.py: rgb=rgb_on_gray); with rgb=None the method raises, it never falls back to a plain blur.

Filter choice: cv2.ximgproc (guidedFilter, jointBilateralFilter) is not in opencv-python-headless, and cv2.bilateralFilter is
not joint (it cannot take a separate guide). So the guided filter of He et al. is implemented here with cv2.boxFilter, in its
"fast" form (low-resolution coefficients, upsampled), with a color or a luminance guide.

Sources:
    He, Sun and Tang, "Guided Image Filtering" (ECCV 2010 / TPAMI 2013): the filter; refining a coarse matte with it is one of its
        applications.
    He and Sun, "Fast Guided Filter" (arXiv 1505.00996): the subsampled version used here.
    Tomasi and Manduchi, "Bilateral filtering for gray and color images" (ICCV 1998) and Kopf et al., "Joint Bilateral Upsampling"
        (SIGGRAPH 2007): the edge-aware weighting by a guide image. Matting papers (e.g. Levin et al., "A Closed-Form Solution to
        Natural Image Matting", TPAMI 2008, arXiv 1504.01819) use the same fact: alpha is locally an affine function of the colors.
"""

import cv2
import einx
import numpy as np
from sana_alpha.trimask import finalize

# All pixel values are for a 1024² matte
RADIUS_PX = (4, 48)  # Guided filter window radius, log-uniform. Larger: the unknown band follows structure over a longer reach
EPS = (1e-4, 1e-1)  # Guided filter regularizer, log-uniform. On [0, 1] colors: small keeps every edge sharp, large flattens to a box blur
PRE_BLUR_SIGMA_PX = (3.0, 24.0)  # The matte is Gaussian-blurred by this first (log-uniform): the guided filter then re-sharpens it
LOW_RANGE = (0.05, 0.45)  # Filtered values below low are background
HIGH_RANGE = (0.55, 0.95)  # Filtered values above high are object
P_GRAY_GUIDE = 0.25  # Chance that the guide is the luminance of the RGB image instead of its three channels
MAX_SUBSAMPLE = 4  # Fast guided filter: coefficients are computed at 1 / this resolution
LUMA_WEIGHTS = (0.299, 0.587, 0.114)  # Rec. 601 luma of (R, G, B)
UNKNOWN = 0.5  # Trimask value of unknown pixels


def box_mean(x, radius):
    """
    Pure function. Mean over a (2 radius + 1)² window, borders replicated. Works on (H, W) and (H, W, K) arrays.

    Args:
        x (np.ndarray): (H, W) or (H, W, K) float32
        radius (int): Pixels, >= 0

    Returns:
        np.ndarray: same shape and dtype as x

    Examples:
        >>> x = np.zeros((1, 7), np.float32); x[0, 3] = 3
        >>> box_mean(x, 1).tolist()
        [[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]]
    """
    return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REPLICATE).reshape(x.shape)  # cv2 drops a K=1 axis


def resize_hw(x, height, width, interpolation):
    """
    Pure function. cv2.resize that always keeps the channel axis of an (H, W, K) array (cv2 drops it when K == 1).

    Examples:
        >>> resize_hw(np.ones((8, 8, 1), np.float32), 4, 2, cv2.INTER_AREA).shape
        (4, 2, 1)
    """
    out = cv2.resize(x, (width, height), interpolation=interpolation)
    return out.reshape(height, width, -1)


def guided_filter(guide, p, radius, eps, subsample):
    """
    Pure function. Fast guided filter (He et al.): q = a . guide + b with a, b fitted per window so that q follows p but has the
    edges of the guide. Coefficients are computed at 1 / subsample resolution, smoothed, and bilinearly upsampled.

    Args:
        guide (np.ndarray): (H, W, K) float32, K = 1 (luminance) or 3 (RGB), values in [0, 1]
        p (np.ndarray): (H, W) float32, the image to filter (here: the blurred matte)
        radius (int): Window radius in full-resolution pixels, >= 1
        eps (float): Regularizer on the guide variance (guide units squared), > 0
        subsample (int): >= 1; 1 is the exact filter

    Returns:
        np.ndarray: (H, W) float32, not clipped (can leave [0, 1] slightly)

    Examples:
        >>> guide = np.zeros((9, 41, 1), np.float32); guide[:, 20:] = 1  # a step edge at column 20
        >>> ramp = np.tile(np.linspace(0, 1, 41, dtype=np.float32), (9, 1))  # p knows nothing about the edge
        >>> q = guided_filter(guide, ramp, radius=6, eps=1e-4, subsample=1)
        >>> q.shape, round(float(q[4, 19]), 2), round(float(q[4, 20]), 2)  # p changes 0.025 per pixel; q jumps 0.17 at the guide edge
        ((9, 41), 0.4, 0.57)
        >>> flat = guided_filter(np.full((9, 41, 1), 0.5, np.float32), ramp, radius=6, eps=1e-4, subsample=1)  # flat guide: two box blurs
        >>> bool(np.allclose(flat, box_mean(box_mean(ramp, 6), 6), atol=1e-4))
        True
    """
    height, width, channels = guide.shape
    low_h, low_w = -(-height // subsample), -(-width // subsample)
    small_guide = resize_hw(guide, low_h, low_w, cv2.INTER_AREA).astype(np.float64)
    small_p = resize_hw(p[..., None], low_h, low_w, cv2.INTER_AREA)[..., 0].astype(np.float64)
    r = max(1, radius // subsample)
    mean_guide, mean_p = box_mean(small_guide, r), box_mean(small_p, r)
    guide_p = einx.multiply("h w k, h w -> h w k", small_guide, small_p)
    guide_guide = einx.multiply("h w k, h w l -> h w k l", small_guide, small_guide)
    cov_guide_p = box_mean(guide_p, r) - einx.multiply("h w k, h w -> h w k", mean_guide, mean_p)
    var_guide = box_mean(guide_guide.reshape(low_h, low_w, -1), r).reshape(low_h, low_w, channels, channels)
    var_guide = var_guide - einx.multiply("h w k, h w l -> h w k l", mean_guide, mean_guide) + eps * np.eye(channels)
    a = np.linalg.solve(var_guide, cov_guide_p[..., None])[..., 0]  # (low_h, low_w, K)
    b = mean_p - einx.dot("h w k, h w k -> h w", a, mean_guide)
    a_up = resize_hw(box_mean(a, r).astype(np.float32), height, width, cv2.INTER_LINEAR)
    b_up = resize_hw(box_mean(b, r).astype(np.float32)[..., None], height, width, cv2.INTER_LINEAR)[..., 0]
    return einx.dot("h w k, h w k -> h w", a_up, guide) + b_up


def make_guide(rgb, gray_guide):
    """
    Pure function. The guided filter's guide image: the three RGB channels, or their luminance.

    Args:
        rgb (np.ndarray): (H, W, 3) float32 in [0, 1], RGB order
        gray_guide (bool): True for luminance

    Returns:
        np.ndarray: (H, W, 3) or (H, W, 1) float32

    Examples:
        >>> rgb = np.zeros((2, 2, 3), np.float32); rgb[..., 1] = 1  # pure green
        >>> make_guide(rgb, False).shape, make_guide(rgb, True).shape, float(make_guide(rgb, True)[0, 0, 0])
        ((2, 2, 3), (2, 2, 1), 0.5870000123977661)
    """
    return einx.dot("h w c, c -> h w 1", rgb, np.array(LUMA_WEIGHTS, np.float32)) if gray_guide else rgb


def check_rgb(rgb, alpha):
    """
    Pure function. Returns rgb as a contiguous float32 array, raising ValueError unless it is an (H, W, 3) float array in [0, 1]
    with the shape of alpha. None is an error: this method has no RGB-free fallback.

    Examples:
        >>> check_rgb(np.zeros((2, 3, 3), np.float32), np.zeros((2, 3), np.float32)).shape
        (2, 3, 3)
        >>> check_rgb(None, np.zeros((2, 3), np.float32))
        Traceback (most recent call last):
        ValueError: bilateral_blur needs the RGB image (rgb_on_gray as (H, W, 3) float32 in [0, 1]) as the rgb keyword; got None
        >>> check_rgb(np.zeros((2, 2, 3), np.float32), np.zeros((2, 3), np.float32))
        Traceback (most recent call last):
        ValueError: rgb must be (2, 3, 3) to match alpha, got (2, 2, 3)
    """
    if rgb is None:
        raise ValueError("bilateral_blur needs the RGB image (rgb_on_gray as (H, W, 3) float32 in [0, 1]) as the rgb keyword; got None")
    if rgb.shape != (*alpha.shape, 3):
        raise ValueError(f"rgb must be {(*alpha.shape, 3)} to match alpha, got {rgb.shape}")
    if not (np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1):
        raise ValueError(f"rgb values must be in [0, 1] (not 0..255), got [{rgb.min()}, {rgb.max()}]")
    return np.ascontiguousarray(rgb, dtype=np.float32)


def quantize3(values, low, high):
    """
    Pure function. 0 below low, 1 above high, 0.5 in between (both bounds inclusive in the unknown level).

    Args:
        values (np.ndarray): (H, W) float32
        low (float), high (float): Thresholds, low <= high

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> quantize3(np.array([[-0.1, 0.2, 0.5, 0.8, 1.1]], np.float32), 0.2, 0.8).tolist()
        [[0.0, 0.5, 0.5, 0.5, 1.0]]
    """
    return np.where(values > high, 1, np.where(values < low, 0, UNKNOWN)).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the filter radius, regularizer eps, pre-blur sigma (all log-uniform), the guide kind
    (75% RGB, 25% luminance) and the thresholds low, high (uniform in LOW_RANGE and HIGH_RANGE).

    Returns:
        dict: radius_px (int), eps, pre_blur_sigma_px, low, high (float), gray_guide (bool)

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {'radius_px': 19, 'eps': 0.000645, 'pre_blur_sigma_px': 3.27, 'gray_guide': True, 'low': 0.375, 'high': 0.915}
    """
    log_uniform = lambda bounds: float(np.exp(rng.uniform(*np.log(bounds))))
    return {"radius_px": int(round(log_uniform(RADIUS_PX))), "eps": round(log_uniform(EPS), 6),
            "pre_blur_sigma_px": round(log_uniform(PRE_BLUR_SIGMA_PX), 2), "gray_guide": bool(rng.random() < P_GRAY_GUIDE),
            "low": round(float(rng.uniform(*LOW_RANGE)), 3), "high": round(float(rng.uniform(*HIGH_RANGE)), 3)}


def make_trimask(alpha, rng, radius_px, eps, pre_blur_sigma_px, gray_guide, low, high, rgb=None):
    """
    Near-pure function (advances rng, only in finalize; otherwise the method is deterministic given its params). Gaussian-blurs alpha, filters it with a guided
    filter whose guide is the RGB image, then quantizes to 3 levels. Raises ValueError when rgb is None or malformed.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Only finalize uses it (draws its dilation radius)
        radius_px (int): Guided filter window radius in pixels
        eps (float): Guided filter regularizer
        pre_blur_sigma_px (float): Gaussian sigma of the matte blur before filtering, pixels
        gray_guide (bool): Guide is the luminance of rgb (True) or its three channels
        low (float), high (float): Background / object thresholds on the filtered matte
        rgb (np.ndarray): (H, W, 3) float32 in [0, 1], RGB order: the object on 0.5 gray (magick_samples.rgb_on_gray / 255). Required.

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus subsample (the fast-filter factor used)

    Examples:
        >>> alpha = np.zeros((64, 128), np.float32); alpha[:, 64:] = 1  # a vertical matte edge between columns 63 and 64
        >>> rgb = np.full((64, 128, 3), 0.5, np.float32); rgb[:, 64:] = 0.9  # an image edge exactly there
        >>> kw = dict(radius_px=12, eps=1e-3, pre_blur_sigma_px=8.0, gray_guide=False, low=0.2, high=0.8)
        >>> trimask, params = make_trimask(alpha, np.random.default_rng(0), rgb=rgb, **kw)
        >>> trimask[32, 56:72].tolist()  # image edge at the matte edge: the unknown band is 8 px wide, centered on it
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
        >>> flat_trimask, _ = make_trimask(alpha, np.random.default_rng(0), rgb=np.full((64, 128, 3), 0.5, np.float32), **kw)  # no image structure: 3x wider
        >>> int((trimask[32] == 0.5).sum()), int((flat_trimask[32] == 0.5).sum())
        (8, 24)
        >>> params["subsample"]
        4
        >>> make_trimask(alpha, np.random.default_rng(0), **kw)
        Traceback (most recent call last):
        ValueError: bilateral_blur needs the RGB image (rgb_on_gray as (H, W, 3) float32 in [0, 1]) as the rgb keyword; got None
    """
    rgb = check_rgb(rgb, alpha)
    subsample = min(MAX_SUBSAMPLE, radius_px)  # radius_px >= 1; keeps the low-res window radius >= 1
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=pre_blur_sigma_px, borderType=cv2.BORDER_REPLICATE)
    filtered = guided_filter(make_guide(rgb, gray_guide), blurred, radius_px, eps, subsample)
    used = {"radius_px": radius_px, "eps": eps, "pre_blur_sigma_px": pre_blur_sigma_px, "gray_guide": gray_guide,
            "low": low, "high": high, "subsample": subsample}
    return finalize(alpha, quantize3(filtered, low, high), rng), used
