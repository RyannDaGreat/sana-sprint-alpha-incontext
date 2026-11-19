"""
Trimask method original: the MAGICK matte itself, continuous, as the condition (m00718: "we need ... the original mask").

No finalize: the matte is consistent with itself by definition (white only where it is 255, black only where it is 0, soft
pixels strictly between), so check_consistent passes. This is the plain "here is my exact mask" task.
"""

import numpy as np


def sample_params(rng):
    """
    Pure function (draws nothing; rng is part of the method interface). No hyperparameters.

    Examples:
        >>> sample_params(np.random.default_rng(0))
        {}
    """
    return {}


def make_trimask(alpha, rng):
    """
    Pure function (rng unused). The matte as the trimask.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 equal to alpha, and {}

    Examples:
        >>> alpha = np.array([[0, 0.3, 1]], np.float32)
        >>> make_trimask(alpha, None)[0].tolist()
        [[0.0, 0.30000001192092896, 1.0]]
    """
    return alpha.astype(np.float32), {}
