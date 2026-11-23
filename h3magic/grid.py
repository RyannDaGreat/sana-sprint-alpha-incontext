"""
Split v3-grid (manifest §23, m01003, m01005, m01006): H3 draws ONE 768² frame of 2x2 panels of 384 - top left the subject
on 0.5 gray, top right on white, bottom left on black, bottom right its alpha matte in grayscale (white = opaque). The three
background panels decontaminate the colour; the 4th panel is the alpha (the mask). The alpha the three backgrounds imply
(least squares) is a free consistency check: their disagreement |Δα| is reported.

Pure numpy functions.
"""
import numpy as np

BACKGROUNDS = (0.5, 1.0, 0.0)  # top left, top right, bottom left
ALPHA_EPS = 0.02  # below this alpha the colour is undefined (fully clear)
EDGE = (0.02, 0.98)  # the edge band: pixels whose panel alpha is strictly inside this range


def panels(frames):
    """
    Pure function. The four panels of grid frames.

    Args:
        frames (np.ndarray): (T, 2P, 2P, 3) uint8, e.g. (141, 768, 768, 3)

    Returns:
        tuple[np.ndarray, np.ndarray]: composites (3, T, P, P, 3) uint8 in BACKGROUNDS order (gray, white, black), and
        the alpha panel (T, P, P) float32 in [0, 1] (mean of its RGB)

    Examples:
        >>> f = np.zeros((1, 4, 4, 3), np.uint8); f[0, :2, :2] = 128; f[0, :2, 2:] = 255; f[0, 2:, 2:] = 255
        >>> c, a = panels(f)
        >>> c.shape, c[:, 0, 0, 0, 0].tolist(), a.tolist()
        ((3, 1, 2, 2, 3), [128, 255, 0], [[[1.0, 1.0], [1.0, 1.0]]])
    """
    p = frames.shape[1] // 2
    composites = np.stack([frames[:, :p, :p], frames[:, :p, p:], frames[:, p:, :p]])
    alpha = frames[:, p:, p:].astype(np.float32).mean(-1) / 255
    return composites, alpha


def decontaminate(composites, alpha, backgrounds=BACKGROUNDS, eps=ALPHA_EPS):
    """
    Pure function. The foreground colour given the alpha: per background k, F_k = (C_k − (1 − α)·b_k) / α, the per-pixel
    MEDIAN over the three backgrounds (one inconsistent panel cannot wash the colour out; coordinator 9:45 AM), clipped to
    [0, 1]; 0 where α < eps.

    Args:
        composites (np.ndarray): (K, ..., 3) float in [0, 1], C_k = α·F + (1 − α)·b_k
        alpha (np.ndarray): (...) float in [0, 1]

    Returns:
        np.ndarray: (..., 3) float32

    Examples:
        >>> F, a = np.array([0.8, 0.2, 0.1]), 0.6
        >>> C = np.stack([a * F + (1 - a) * b for b in BACKGROUNDS])  # exact composites
        >>> np.round(decontaminate(C, np.array(a)).astype(float), 6).tolist()
        [0.8, 0.2, 0.1]
    """
    b = np.asarray(backgrounds, np.float32).reshape((-1,) + (1,) * alpha.ndim + (1,))
    a = alpha[None, ..., None]
    fg = (composites - (1 - a) * b) / np.maximum(a, eps)
    return np.where(alpha[..., None] >= eps, np.clip(np.median(fg, 0), 0, 1), 0).astype(np.float32)


def panel_spread(composites, alpha, backgrounds=BACKGROUNDS, eps=ALPHA_EPS):
    """
    Pure function. How much the three panels disagree on the colour: per pixel, the largest spread (max − min over the
    backgrounds, mean over RGB) of the foreground estimates F_k, averaged over the subject (alpha > 0.5). 0 = consistent.

    Args:
        composites (np.ndarray): (K, ..., 3) float in [0, 1]
        alpha (np.ndarray): (...) float in [0, 1]

    Returns:
        float (nan without subject pixels)

    Examples:
        >>> F, a = np.array([0.8, 0.2, 0.1]), np.ones((1,))
        >>> C = np.stack([a[..., None] * F + (1 - a[..., None]) * b for b in BACKGROUNDS])
        >>> panel_spread(C, a), round(panel_spread(C + np.array([0.0, 0.3, 0.0]).reshape(3, 1, 1), a), 3)
        (0.0, 0.267)
    """
    b = np.asarray(backgrounds, np.float32).reshape((-1,) + (1,) * alpha.ndim + (1,))
    a = alpha[None, ..., None]
    fg = np.clip((composites - (1 - a) * b) / np.maximum(a, eps), 0, 1)
    spread = (fg.max(0) - fg.min(0)).mean(-1)
    mask = alpha > 0.5
    return float(spread[mask].mean()) if mask.any() else float("nan")


def disagreement(alpha_panel, alpha_ls, edge=EDGE):
    """
    Pure function. How much the 4th-panel alpha and the least-squares alpha disagree: |Δα| statistics over all pixels and
    over the edge band (panel alpha strictly inside `edge`).

    Args:
        alpha_panel, alpha_ls (np.ndarray): same shape, float in [0, 1]

    Returns:
        dict: mean, median, p95 (all pixels), edge_mean, edge_median, edge_p95 (nan if no edge pixel), edge_share

    Examples:
        >>> a = np.array([0.0, 0.5, 1.0, 1.0]); b = np.array([0.1, 0.3, 1.0, 0.9])
        >>> d = disagreement(a, b); round(d["mean"], 3), round(d["edge_mean"], 3), d["edge_share"]
        (0.1, 0.2, 0.25)
    """
    d = np.abs(alpha_panel.astype(np.float64) - alpha_ls)
    band = (alpha_panel > edge[0]) & (alpha_panel < edge[1])
    e = d[band]
    stat = lambda x, f: float(f(x)) if x.size else float("nan")
    return dict(mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)),
                edge_mean=stat(e, np.mean), edge_median=stat(e, np.median), edge_p95=stat(e, lambda x: np.percentile(x, 95)),
                edge_share=float(band.mean()))
