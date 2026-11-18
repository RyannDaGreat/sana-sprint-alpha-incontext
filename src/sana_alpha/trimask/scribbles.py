"""
Trimask method scribbles: everything is unknown (gray) except a few smooth, thick user-like strokes, white inside the
object and black outside it, each kept a margin away from the matte edge (distance transform). Simulates the
scribble inputs of interactive segmentation and scribble-based matting.

Sources (the simulation recipe follows these, adapted to a known matte):
    - Wong et al., "ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image", ECCV 2024,
      arXiv 2312.07381 (line and contour scribbles drawn inside the mask, then deformed to vary shape and thickness).
    - Xu et al., "Deep Interactive Object Selection", CVPR 2016, arXiv 1603.04042 (positive/negative user inputs
      sampled at least d_margin pixels away from the object boundary).
    - Chen et al., "ScribbleSeg: Scribble-based Interactive Image Segmentation", arXiv 2303.11320 (Bezier-curve
      scribbles inside the mask, thickness 3-7 px, stroke count with geometric decay).
"""

import cv2
import numpy as np
from sana_alpha.trimask import restrict_to_matte

THRESHOLD = 0.5  # Alpha above this is object
MARGIN_PX = (2, 30)  # Gap between a stroke's edge and the matte edge, in pixels of a 1024² matte
STROKES = (1, 6)  # Strokes per class (white and black), inclusive
WIDTH_LO_PX = (6, 24)  # Narrowest stroke width, log-uniform
WIDTH_HI_RATIO = (1.5, 4.0)  # Widest stroke = narrowest * this, log-uniform
LENGTH_PX = (200, 900)  # Typical stroke length
TURN_RAD = (0.02, 0.2)  # Std of the per-step heading change of a random-walk stroke (smoothed by momentum)
BG_REACH_PX = (60, 500)  # With probability BG_NEAR_P black strokes stay within this distance of the object
BG_NEAR_P = 0.5  # Probability that black strokes are restricted to the neighborhood of the object
BEZIER_P = 0.5  # Probability that a stroke is a Bezier curve instead of a random walk
MOMENTUM = 0.8  # Heading-change smoothing of random walks: turn = MOMENTUM * previous turn + noise
STEP_RATIO = 0.75  # Walk step = this * stroke half-width + STEP_MIN_PX, so that round dabs overlap
STEP_MIN_PX = 8
PATH_ATTEMPTS = 8  # Tries (each with a shorter stroke) to place one stroke before it degrades to a dot
BEZIER_POINTS = 96  # Samples along a Bezier curve
EDGE_SLACK_PX = 1.5  # Extra clearance for rasterization of a thick line
MAX_STROKE_FRACTION = 0.8  # Half-width never exceeds this fraction of the room between margin and the deepest point


def distance_to_other_class(object_mask):
    """
    Pure function. Euclidean distance of every pixel to the nearest pixel of the opposite class (0 on the opposite class).
    Where the opposite class is absent the matte has no edge, so the distance is max(H, W) everywhere (strokes anywhere).

    Args:
        object_mask (np.ndarray): (H, W) bool, True = this class

    Returns:
        np.ndarray: (H, W) float32, in pixels

    Examples:
        >>> mask = np.zeros((1, 7), bool); mask[0, 2:6] = True
        >>> distance_to_other_class(mask).round(1).tolist()
        [[0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0]]
        >>> distance_to_other_class(np.ones((3, 5), bool)).tolist()[0]
        [5.0, 5.0, 5.0, 5.0, 5.0]
    """
    if object_mask.all():
        return np.full(object_mask.shape, max(object_mask.shape), np.float32)
    if not object_mask.any():
        return np.zeros(object_mask.shape, np.float32)
    padded = np.pad(object_mask, 1, constant_values=True)  # The image border is not an edge of the matte
    return cv2.distanceTransform(padded.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def sample_log_uniform(rng, low, high):
    """
    Near-pure function (advances rng). Log-uniform float in [low, high].

    Examples:
        >>> 4 <= sample_log_uniform(np.random.default_rng(0), 4, 16) <= 16
        True
    """
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def bezier_points(control_points, count):
    """
    Pure function. Samples a cubic Bezier curve.

    Args:
        control_points (np.ndarray): (4, 2) float, (x, y) pixels
        count (int): Number of samples

    Returns:
        np.ndarray: (count, 2) float64, (x, y) pixels, from the first to the last control point

    Examples:
        >>> bezier_points(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]), 3).tolist()
        [[0.0, 0.0], [5.0, 7.5], [10.0, 0.0]]
    """
    t = np.linspace(0, 1, count)[:, None]
    p0, p1, p2, p3 = np.asarray(control_points, np.float64)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def inside_region(points, region):
    """
    Pure function. Which (x, y) points fall on a True pixel of region (points outside the image count as outside).

    Args:
        points (np.ndarray): (N, 2) float, (x, y) pixels
        region (np.ndarray): (H, W) bool

    Returns:
        np.ndarray: (N,) bool

    Examples:
        >>> region = np.zeros((4, 6), bool); region[1, 4] = True
        >>> inside_region(np.array([[4.2, 1.1], [1.0, 1.0], [-3.0, 1.0], [4.0, 9.0]]), region).tolist()
        [True, False, False, False]
    """
    x, y = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)
    height, width = region.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return valid & region[np.where(valid, y, 0), np.where(valid, x, 0)]


def leading_run(points, region):
    """
    Pure function. The longest prefix of points that all lie inside region.

    Examples:
        >>> region = np.zeros((1, 5), bool); region[0, :3] = True
        >>> leading_run(np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [1, 0]]), region).tolist()
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    """
    inside = inside_region(points, region)
    return points[: len(inside) if inside.all() else int(np.argmin(inside))]


def random_walk_path(start, region, rng, length_px, step_px, turn_rad):
    """
    Near-pure function (advances rng). A smooth random walk from start that never leaves region: the heading changes by
    a momentum-smoothed Gaussian turn each step; a step that would leave region tries other headings, and the walk
    ends if none works.

    Args:
        start (np.ndarray): (2,) float, (x, y) pixels, inside region
        region (np.ndarray): (H, W) bool
        rng (np.random.Generator)
        length_px (float): Target path length
        step_px (float): Length of one step
        turn_rad (float): Std of the heading change per step, radians

    Returns:
        np.ndarray: (N, 2) float64, (x, y) pixels, N >= 1, every point inside region

    Examples:
        >>> region = np.zeros((100, 100), bool); region[10:90, 10:90] = True
        >>> path = random_walk_path(np.array([50.0, 50.0]), region, np.random.default_rng(0), 300, 5, 0.3)
        >>> path.shape, bool(inside_region(path, region).all())
        ((61, 2), True)
    """
    heading, turn = rng.uniform(0, 2 * np.pi), 0.0
    path = [np.asarray(start, np.float64)]
    for _ in range(int(length_px / step_px)):
        turn = MOMENTUM * turn + rng.normal(0, turn_rad)
        for headings in (heading + turn, *rng.uniform(0, 2 * np.pi, 8)):  # Smooth step first, then random escapes
            candidate = path[-1] + step_px * np.array([np.cos(headings), np.sin(headings)])
            if inside_region(candidate[None], region)[0]:
                heading = headings
                path.append(candidate)
                break
        else:
            break
    return np.array(path)


def bezier_path(start, region, rng, length_px):
    """
    Near-pure function (advances rng). A cubic Bezier curve from start to a random point length_px away, with random
    bulging control points, cut at the first sample that leaves region.

    Args:
        start (np.ndarray): (2,) float, (x, y) pixels, inside region
        region (np.ndarray): (H, W) bool
        rng (np.random.Generator)
        length_px (float): Chord length of the curve

    Returns:
        np.ndarray: (N, 2) float64, (x, y) pixels, N >= 1, every point inside region

    Examples:
        >>> region = np.zeros((100, 100), bool); region[10:90, 10:90] = True
        >>> path = bezier_path(np.array([50.0, 50.0]), region, np.random.default_rng(1), 60)
        >>> bool(inside_region(path, region).all()), bool((path[0] == [50, 50]).all())
        (True, True)
    """
    angle = rng.uniform(0, 2 * np.pi)
    chord = length_px * np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-chord[1], chord[0]])
    start = np.asarray(start, np.float64)
    controls = np.array([start, start + chord / 3 + rng.uniform(-0.6, 0.6) * normal,
                         start + 2 * chord / 3 + rng.uniform(-0.6, 0.6) * normal, start + chord])
    return leading_run(bezier_points(controls, BEZIER_POINTS), region)


def draw_path(canvas, path, half_width):
    """
    Command. Draws a round-capped thick polyline into canvas in place (1 where drawn). A single point becomes a dot.

    Args:
        canvas (np.ndarray): (H, W) uint8, modified
        path (np.ndarray): (N, 2) float, (x, y) pixels
        half_width (float): Pixels

    Examples:
        >>> canvas = np.zeros((9, 9), np.uint8)
        >>> draw_path(canvas, np.array([[2.0, 4.0], [6.0, 4.0]]), 1.0)
        >>> int(canvas[4, 2]), int(canvas[4, 6]), int(canvas[0].sum()), int(canvas[8].sum())
        (1, 1, 0, 0)
    """
    points = np.rint(path).astype(int)
    radius = max(1, int(round(half_width)))
    for a, b in zip(points[:-1], points[1:]):
        cv2.line(canvas, tuple(a), tuple(b), 1, 2 * radius + 1)
    for point in points:
        cv2.circle(canvas, tuple(point), radius, 1, -1)


def draw_class_strokes(distance, rng, count, margin_px, width_range, length_px, turn_rad, reach_px):
    """
    Near-pure function (advances rng). Draws count strokes of one class. Each stroke is a random walk or a Bezier curve
    whose centerline stays where distance >= margin + half-width (and <= reach_px if given), so its edge keeps
    margin_px from the matte edge. Widths are log-uniform in width_range, capped to fit the shape. A stroke that cannot
    be placed (no admissible start) is skipped.

    Args:
        distance (np.ndarray): (H, W) float32, distance to the other class from distance_to_other_class
        rng (np.random.Generator)
        count (int): Strokes to attempt
        margin_px (float): Required gap between stroke edge and matte edge
        width_range (tuple[float, float]): Min and max stroke width, pixels
        length_px (float): Typical stroke length (each stroke draws 0.5-1.5 times this)
        turn_rad (float): Random-walk turn std, radians
        reach_px (float or None): Max distance of stroke centers from the matte edge, None = unlimited

    Returns:
        tuple[np.ndarray, int, float]: (H, W) uint8 canvas (1 on strokes), strokes drawn, margin actually used (pixels)

    Examples:
        >>> yy, xx = np.mgrid[:200, :200]
        >>> disk = (xx - 100) ** 2 + (yy - 100) ** 2 < 80 ** 2
        >>> canvas, drawn, margin = draw_class_strokes(distance_to_other_class(disk), np.random.default_rng(0),
        ...                                            3, 10, (6, 12), 150, 0.3, None)
        >>> drawn, margin, int(canvas.sum()) > 0, bool(disk[canvas > 0].all())
        (3, 10.0, True, True)
        >>> int(distance_to_other_class(disk)[canvas > 0].min()) >= 10
        True
    """
    canvas = np.zeros(distance.shape, np.uint8)
    deepest = float(distance.max())
    margin = min(float(margin_px), deepest / 2)  # A thin object cannot afford the full margin
    drawn = 0
    for _ in range(count if deepest > 2 * EDGE_SLACK_PX else 0):
        half_width = min(sample_log_uniform(rng, *width_range) / 2, MAX_STROKE_FRACTION * (deepest - margin))
        low = margin + half_width + EDGE_SLACK_PX
        region = distance >= low
        if reach_px is not None:
            region &= distance <= max(margin + reach_px, low + 1)
        starts = np.argwhere(region)
        if len(starts) == 0:
            continue
        length = length_px * rng.uniform(0.5, 1.5)
        use_bezier = rng.random() < BEZIER_P
        for attempt in range(PATH_ATTEMPTS):
            y, x = starts[rng.integers(len(starts))]
            reduced = length / 2 ** attempt
            path = (bezier_path((x, y), region, rng, reduced) if use_bezier else
                    random_walk_path((x, y), region, rng, reduced, STEP_RATIO * half_width + STEP_MIN_PX, turn_rad))
            if len(path) > 1:
                break
        draw_path(canvas, path, half_width)  # After PATH_ATTEMPTS the path may be one point: a dot, like a user click
        drawn += 1
    return canvas * (distance >= margin), drawn, margin


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the stroke counts, margin, width range, stroke length, curviness and
    whether black strokes hug the object.

    Examples:
        >>> sorted(sample_params(np.random.default_rng(0)))
        ['bg_reach_px', 'length_px', 'margin_px', 'n_bg', 'n_fg', 'threshold', 'turn_rad', 'width_hi_px', 'width_lo_px']
    """
    width_lo = sample_log_uniform(rng, *WIDTH_LO_PX)
    return {"n_fg": int(rng.integers(STROKES[0], STROKES[1] + 1)), "n_bg": int(rng.integers(STROKES[0], STROKES[1] + 1)),
            "margin_px": float(rng.uniform(*MARGIN_PX)), "width_lo_px": width_lo,
            "width_hi_px": width_lo * sample_log_uniform(rng, *WIDTH_HI_RATIO),
            "length_px": float(rng.uniform(*LENGTH_PX)), "turn_rad": sample_log_uniform(rng, *TURN_RAD),
            "bg_reach_px": float(rng.uniform(*BG_REACH_PX)) if rng.random() < BG_NEAR_P else None,
            "threshold": THRESHOLD}


def make_trimask(alpha, rng, n_fg, n_bg, margin_px, width_lo_px, width_hi_px, length_px, turn_rad, bg_reach_px, threshold):
    """
    Near-pure function (advances rng). Binarizes alpha, draws n_fg white strokes inside the object and n_bg black
    strokes in the background, each at least margin_px from the matte edge; all other pixels are 0.5 (unknown).
    Lengths and widths are in pixels of a 1024² matte. On an all-black or all-white alpha strokes go anywhere
    (no edge to keep away from), and a thin object gets a smaller margin; the returned params report the margins used.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator)
        n_fg (int), n_bg (int): Strokes to attempt inside / outside the object
        margin_px (float): Gap between stroke edge and matte edge
        width_lo_px (float), width_hi_px (float): Stroke width range
        length_px (float): Typical stroke length
        turn_rad (float): Random-walk turn std, radians
        bg_reach_px (float or None): Black strokes stay this near the object, None = anywhere
        threshold (float): Alpha above this is object

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus n_fg_drawn, n_bg_drawn,
        margin_fg_px, margin_bg_px (the margins actually used)

    Examples:
        >>> yy, xx = np.mgrid[:256, :256]
        >>> alpha = (((xx - 128) ** 2 + (yy - 128) ** 2) < 70 ** 2).astype(np.float32)
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(3), 3, 3, 12, 6, 14, 150, 0.3, None, 0.5)
        >>> sorted(np.unique(trimask).tolist()), (used["n_fg_drawn"], used["n_bg_drawn"])
        ([0.0, 0.5, 1.0], (3, 3))
        >>> bool((alpha[trimask == 1] == 1).all() and (alpha[trimask == 0] == 0).all())
        True
        >>> trimask, used = make_trimask(np.zeros((64, 64), np.float32), np.random.default_rng(3), 3, 2, 5, 4, 8, 40, 0.3, None, 0.5)
        >>> used["n_fg_drawn"], used["n_bg_drawn"], float(trimask.max())
        (0, 2, 0.5)
    """
    object_mask = alpha > threshold
    widths = (width_lo_px, width_hi_px)
    fg, n_fg_drawn, margin_fg = draw_class_strokes(distance_to_other_class(object_mask), rng, n_fg, margin_px, widths,
                                                   length_px, turn_rad, None)
    reach_px = bg_reach_px if object_mask.any() else None  # With no object there is nothing to hug
    bg, n_bg_drawn, margin_bg = draw_class_strokes(distance_to_other_class(~object_mask), rng, n_bg, margin_px, widths,
                                                   length_px, turn_rad, reach_px)
    trimask = np.full(alpha.shape, 0.5, np.float32)
    trimask[bg > 0] = 0
    trimask[fg > 0] = 1
    used = {"n_fg": n_fg, "n_bg": n_bg, "margin_px": margin_px, "width_lo_px": width_lo_px, "width_hi_px": width_hi_px,
            "length_px": length_px, "turn_rad": turn_rad, "bg_reach_px": bg_reach_px, "threshold": threshold,
            "n_fg_drawn": n_fg_drawn, "n_bg_drawn": n_bg_drawn, "margin_fg_px": margin_fg, "margin_bg_px": margin_bg}
    return restrict_to_matte(alpha, trimask), used
