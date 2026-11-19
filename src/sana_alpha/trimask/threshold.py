"""
Trimask method threshold: the MAGICK matte thresholded to a solid black-and-white mask (m00718: "the original mask
thresholded ... doing a solid outline is a pretty common task").

No finalize, and the only method exempt from check_consistent (trimask.CONSISTENCY_EXEMPT): at soft edges (hair, glass) it
says "surely object" where the matte is only partly opaque. The user asked for exactly this input: a hard silhouette, as a
person would paint it.
"""

import numpy as np

THRESHOLD = 0.5  # Alpha at or above this is object: the usual binarization of a matte


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
    Pure function (rng unused). 1 where alpha >= THRESHOLD, else 0.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 in {0, 1}, and {}

    Examples:
        >>> make_trimask(np.array([[0, 0.3, 0.5, 0.9, 1]], np.float32), None)[0].tolist()
        [[0.0, 0.0, 1.0, 1.0, 1.0]]
    """
    return (alpha >= THRESHOLD).astype(np.float32), {}
