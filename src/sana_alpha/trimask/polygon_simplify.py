"""
Trimask method polygon_simplify: a coarse polygon traced around the matte, like a quick lasso, with an unknown band around it.

Steps: binarize alpha, find all contours (objects, holes, islands in holes), simplify each with Douglas-Peucker at a random
tolerance, optionally jitter the vertices (a shaky hand), fill the polygons by nesting depth (even = object, odd = hole),
then object = polygon eroded by erode_px, background = outside the polygon dilated by dilate_px, unknown = between.
Because the polygon is only an approximation, the band is scaled with the tolerance so the true edge mostly falls inside it.

Sources:
    Douglas & Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its
    caricature", The Canadian Cartographer 10(2), 1973 (the simplification, cv2.approxPolyDP).
    Suzuki & Abe, "Topological structural analysis of digitized binary images by border following", CVGIP 30(1), 1985
    (the contour extraction and nesting hierarchy, cv2.findContours).
    Rough hand-made trimaps for matting: interactive trimap tools (e.g. "A novel interactive image matting framework",
    and rough-trimap refinement, "Robust trimap generation based on manifold ranking", Information Sciences 2020)
    treat a user-drawn outline plus a uniform unknown band as the standard input.
"""

import cv2
import einx
import numpy as np
from sana_alpha.trimask import restrict_to_matte

THRESHOLD = 0.5  # Alpha above this is object
EPS_FRAC = (0.003, 0.03)  # DP tolerance as a fraction of max(H, W), log-uniform: 3 to 31 px on a 1024² matte
JITTER_PROB = 0.5  # Probability that vertices get jitter
JITTER_RATIO = (0.1, 0.5)  # Jitter sigma as a fraction of the tolerance
BAND_RATIO = (0.5, 2.0)  # Erode and dilate radii as multiples of the tolerance, drawn independently
BAND_MIN_PX = 2  # Smallest band radius, so even a tight polygon leaves a visible unknown band
MIN_SIMPLIFY_AREA_FRAC = 1e-4  # Contours smaller than this fraction of H*W are filled as they are, not simplified


def contour_depths(hierarchy):
    """
    Pure function. Nesting depth of each contour from a cv2.RETR_TREE hierarchy (0 = outermost).

    Args:
        hierarchy (np.ndarray): (N, 4) int, rows (next, previous, first_child, parent), parent -1 for roots

    Returns:
        list[int]: N depths

    Examples:
        >>> contour_depths(np.array([[-1, -1, 1, -1], [-1, -1, 2, 0], [-1, -1, -1, 1]]))
        [0, 1, 2]
        >>> contour_depths(np.array([[1, -1, -1, -1], [-1, 0, -1, -1]]))
        [0, 0]
    """
    parents = hierarchy[:, 3].tolist()
    depths = [None] * len(parents)
    for start in range(len(parents)):
        node, chain = start, []
        while depths[node] is None and parents[node] != -1:
            chain.append(node)
            node = parents[node]
        depths[node] = depths[node] if depths[node] is not None else 0
        for step, member in enumerate(reversed(chain), start=1):
            depths[member] = depths[node] + step
    return depths


def simplify_contour(points, eps, jitter, rng, min_area):
    """
    Near-pure function (advances rng only if jitter > 0). Douglas-Peucker simplification of a closed contour, then vertex jitter.
    Contours with area < min_area are returned unchanged.

    Args:
        points (np.ndarray): (N, 2) int32, (x, y) pixel coordinates of a closed contour
        eps (float): DP tolerance in pixels
        jitter (float): Gaussian sigma of the vertex jitter in pixels, 0 for none
        rng (np.random.Generator)
        min_area (float): Pixels

    Returns:
        np.ndarray: (M, 2) int32 (x, y), M <= N for jitter 0

    Examples:
        >>> square = np.array([[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]], np.int32)
        >>> simplify_contour(square, 1.0, 0, None, 1).tolist()
        [[0, 0], [10, 0], [10, 10], [0, 10]]
        >>> simplify_contour(square, 1.0, 0, None, 1000).shape  # Too small to simplify
        (8, 2)
    """
    if cv2.contourArea(points) < min_area:
        return points
    simple = einx.rearrange("n 1 xy -> n xy", cv2.approxPolyDP(points, eps, True))
    if jitter > 0:
        simple = np.round(simple + rng.normal(0, jitter, simple.shape)).astype(np.int32)
    return simple


def fill_polygons(shape, polygons, depths):
    """
    Pure function. Rasterizes nested polygons: even depth is filled with 1 (object), odd depth with 0 (hole), shallow first.

    Args:
        shape (tuple[int, int]): (H, W)
        polygons (list[np.ndarray]): each (M, 2) int32 (x, y)
        depths (list[int]): nesting depth of each polygon

    Returns:
        np.ndarray: (H, W) uint8 in {0, 1}

    Examples:
        >>> outer = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.int32)
        >>> hole = np.array([[3, 3], [5, 3], [5, 5], [3, 5]], np.int32)
        >>> fill_polygons((9, 9), [outer, hole], [0, 1])[:, 4].tolist()  # The hole polygon's boundary pixels are filled too
        [1, 1, 1, 0, 0, 0, 1, 1, 1]
    """
    canvas = np.zeros(shape, np.uint8)
    for depth in range(max(depths, default=-1) + 1):
        group = [polygon for polygon, d in zip(polygons, depths) if d == depth]
        cv2.fillPoly(canvas, group, 1 - depth % 2)
    return canvas


def band_trimask(polygon, erode_px, dilate_px):
    """
    Pure function. 1 where the polygon is deeper than erode_px inside, 0 where it is farther than dilate_px outside, 0.5 between.
    Uses Euclidean distance transforms; the image border does not count as outside.

    Args:
        polygon (np.ndarray): (H, W) uint8 in {0, 1}
        erode_px (float), dilate_px (float): Band radii in pixels

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> polygon = np.zeros((1, 9), np.uint8); polygon[0, 3:6] = 1
        >>> band_trimask(polygon, 1, 2).tolist()
        [[0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]]
    """
    inside = cv2.distanceTransform(polygon, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - polygon, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(inside > erode_px, 1, np.where(outside > dilate_px, 0, 0.5)).astype(np.float32)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the DP tolerance (log-uniform), the jitter, and the band radii as multiples of the tolerance.

    Returns:
        dict: eps_frac (tolerance / max(H, W)), jitter_ratio (jitter sigma / tolerance, 0 = none),
        erode_ratio and dilate_ratio (band radius / tolerance), threshold

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params)
        ['dilate_ratio', 'eps_frac', 'erode_ratio', 'jitter_ratio', 'threshold']
        >>> EPS_FRAC[0] <= params['eps_frac'] <= EPS_FRAC[1]
        True
    """
    eps_frac = float(np.exp(rng.uniform(*np.log(EPS_FRAC))))
    jitter_ratio = float(rng.uniform(*JITTER_RATIO)) if rng.random() < JITTER_PROB else 0.0
    erode_ratio, dilate_ratio = (float(r) for r in rng.uniform(*BAND_RATIO, size=2))
    return {"eps_frac": eps_frac, "jitter_ratio": jitter_ratio, "erode_ratio": erode_ratio,
            "dilate_ratio": dilate_ratio, "threshold": THRESHOLD}


def make_trimask(alpha, rng, eps_frac, jitter_ratio, erode_ratio, dilate_ratio, threshold):
    """
    Near-pure function (advances rng when jitter_ratio > 0). Traces and simplifies the outline of alpha > threshold, fills it,
    and returns object = polygon eroded, background = outside the polygon dilated, unknown = between.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Used for vertex jitter
        eps_frac (float): DP tolerance as a fraction of max(H, W)
        jitter_ratio (float): Vertex jitter sigma as a fraction of the tolerance
        erode_ratio (float), dilate_ratio (float): Band radii as multiples of the tolerance (at least BAND_MIN_PX)
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus the derived pixel values
        eps_px, jitter_px, erode_px, dilate_px

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> trimask, params = make_trimask(alpha, np.random.default_rng(0), 0.02, 0.0, 1.0, 1.0, 0.5)
        >>> sorted(np.unique(trimask).tolist()), trimask[64, 64].item(), trimask[0, 0].item(), trimask[64, 32].item()
        ([0.0, 0.5, 1.0], 1.0, 0.0, 0.5)
        >>> params['eps_px'], params['erode_px'], params['dilate_px']
        (2.56, 2.56, 2.56)
    """
    eps_px = eps_frac * max(alpha.shape)
    jitter_px = jitter_ratio * eps_px
    erode_px, dilate_px = (max(BAND_MIN_PX, ratio * eps_px) for ratio in (erode_ratio, dilate_ratio))
    contours, hierarchy = cv2.findContours((alpha > threshold).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    min_area = MIN_SIMPLIFY_AREA_FRAC * alpha.size
    polygons = [simplify_contour(contour, eps_px, jitter_px, rng, min_area) for contour in contours]
    depths = contour_depths(hierarchy[0]) if len(contours) else []
    polygon = fill_polygons(alpha.shape, polygons, depths)
    used = {"eps_frac": eps_frac, "jitter_ratio": jitter_ratio, "erode_ratio": erode_ratio, "dilate_ratio": dilate_ratio,
            "threshold": threshold, "eps_px": float(eps_px), "jitter_px": float(jitter_px),
            "erode_px": float(erode_px), "dilate_px": float(dilate_px)}
    return restrict_to_matte(alpha, band_trimask(polygon, erode_px, dilate_px)), used
