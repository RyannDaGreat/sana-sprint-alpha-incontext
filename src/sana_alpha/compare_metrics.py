"""
Metrics of the SANA vs FIBO2 compare reports (manifest §15). All images are float in [0, 1]; trimask 1 = object,
0 = background, 0.5 = unknown; matte = the true alpha.
"""

import numpy as np

from sana_alpha.metrics import EXTERIOR_ALPHA, INTERIOR_ALPHA, foreground_mask, image_metrics, mask_iou, trimask_metrics

BOOTSTRAP_DRAWS = 2000  # resamples per interval: the 2.5% / 97.5% quantiles are stable to ~0.001 at this count
CI_LEVEL = 0.95


def gray_band(trimask):
    """
    Pure function. The trimask's unknown region: neither surely object nor surely background.

    Args:
        trimask (np.ndarray): (H, W) float in [0, 1]

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> gray_band(np.array([[1.0, 0.5, 0.0, 0.3]])).tolist()
        [[False, True, False, True]]
    """
    return (trimask > EXTERIOR_ALPHA) & (trimask < 1 - EXTERIOR_ALPHA)


def band_iou(predicted, trimask, matte):
    """
    Pure function. IoU of a predicted object mask against the true matte, counted only inside the trimask's gray
    band, the part the model has to decide. NaN when the band is empty (a trimask with no unknown pixels).

    Args:
        predicted (np.ndarray): (H, W) bool, the predicted object
        trimask (np.ndarray): (H, W) float in [0, 1]
        matte (np.ndarray): (H, W) float in [0, 1], the true alpha

    Returns:
        float

    Examples:
        >>> trimask = np.array([[1.0, 0.5, 0.5, 0.5, 0.0]])
        >>> matte = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
        >>> band_iou(np.array([[True, True, False, False, False]]), trimask, matte)  # 1 of the band's 2 object pixels
        0.5
        >>> band_iou(np.array([[True, True, True, True, False]]), trimask, matte)  # both, plus one false pixel
        0.6666666666666666
    """
    band = gray_band(trimask)
    return float("nan") if not band.any() else mask_iou(predicted[band], matte[band] > INTERIOR_ALPHA)


def item_metrics(rgb, trimask, matte):
    """
    Pure function. Every compare metric of one generated image.

    Args:
        rgb (np.ndarray): (H, W, 3) float RGB, the generated image
        trimask (np.ndarray): (H, W) float, its condition
        matte (np.ndarray): (H, W) float, the true alpha of the item

    Returns:
        dict: band_iou, white_miss, black_hit, iou, exterior_gray_error, band_iou_bg, band_iou_fg (the two trivial
        baselines: the band filled as background or as object)

    Examples:
        >>> matte = np.zeros((64, 64)); matte[16:48, 16:48] = 1
        >>> trimask = np.zeros((64, 64)); trimask[12:52, 12:52] = 0.5; trimask[20:44, 20:44] = 1
        >>> rgb = np.full((64, 64, 3), 0.5); rgb[16:48, 16:48] = 0.9
        >>> {k: round(v, 3) for k, v in item_metrics(rgb, trimask, matte).items()}
        {'band_iou': 1.0, 'white_miss': 0.0, 'black_hit': 0.0, 'iou': 1.0, 'exterior_gray_error': 0.0, 'band_iou_bg': 0.0, 'band_iou_fg': 0.438}
    """
    predicted = foreground_mask(rgb)
    image = image_metrics(rgb, matte)
    return dict(band_iou=band_iou(predicted, trimask, matte), **trimask_metrics(rgb, trimask),
                iou=image["iou"], exterior_gray_error=image["exterior_gray_error"],
                band_iou_bg=band_iou(trimask > 1 - EXTERIOR_ALPHA, trimask, matte),
                band_iou_fg=band_iou(trimask > EXTERIOR_ALPHA, trimask, matte))


def paired_bootstrap_ci(a, b=None, draws=BOOTSTRAP_DRAWS, level=CI_LEVEL, seed=0):
    """
    Near-pure function (seeded RNG: same inputs, same output). Mean and percentile bootstrap interval of per-item
    values `a`, or of the per-item difference a − b when `b` is given (paired: item k of a and b is the same item).
    NaN items are dropped (from both, when paired).

    Args:
        a (array-like): per-item values
        b (array-like | None): per-item values of the other model, same item order
        draws (int): bootstrap resamples
        level (float): interval coverage
        seed (int): RNG seed

    Returns:
        tuple[float, float, float]: (mean, low, high)

    Examples:
        >>> paired_bootstrap_ci([0.5, 0.5, 0.5])
        (0.5, 0.5, 0.5)
        >>> mean, low, high = paired_bootstrap_ci(np.linspace(0, 1, 101))
        >>> round(mean, 3), low < mean < high, round(high - low, 2)
        (0.5, True, 0.11)
        >>> paired_bootstrap_ci([0.9, 0.8, np.nan], [0.7, 0.6, 0.1])  # NaN item dropped from both
        (0.2, 0.2, 0.2)
    """
    a = np.asarray(a, float)
    values = a if b is None else a - np.asarray(b, float)
    values = values[~np.isnan(values)]
    means = np.random.default_rng(seed).choice(values, (draws, len(values))).mean(axis=1)
    tail = (1 - level) / 2
    return tuple(float(round(v, 12)) for v in (values.mean(), *np.quantile(means, [tail, 1 - tail])))
