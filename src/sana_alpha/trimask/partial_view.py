"""
Trimask method partial_view: the exact trimask (sure-white on fully opaque, sure-black on fully transparent, soft pixels gray)
except one region of the canvas is entirely unknown, so the model must complete the hidden part of the object (outpainting-like).
The region covers 20-60% of the canvas and is one of four kinds: a half-plane at a random angle and offset, a wedge (circular
sector) with a random apex, a rectangle, or a stripe (band between two parallel lines). Optionally a thin gray band lies along
the boundary of the hidden region.

Every region is a level set of a scalar field over the canvas, cut at the quantile that gives the drawn coverage exactly.
Region anchors (wedge apex, rectangle center, stripe center line) are fractions of the object's bounding box, and a drawn
region that hides too little or too much of the object is redrawn, so the hidden part really contains some of the object.

Sources (how such masks are made):
    - Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", CVPR 2022, arXiv 2201.09865:
      benchmark masks "Half" (only the left half is given) and "Expand" (only a center crop is given, i.e. outpainting).
    - Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (LaMa), WACV 2022, arXiv 2109.07161:
      aggressive training masks, thick strokes plus rectangles of arbitrary aspect ratio, covering a large share of the image.
    - Pathak et al., "Context Encoders: Feature Learning by Inpainting", CVPR 2016, arXiv 1604.07379: large central and
      random block masks as training holes.
"""

import cv2
import einx
import numpy as np

from .morph_band import disk
from .unknown_patches import padded_bbox
from sana_alpha.trimask import finalize

KINDS = ("halfplane", "wedge", "rect", "stripe")
KIND_PROBABILITIES = (0.30, 0.25, 0.25, 0.20)
COVERAGE = (0.2, 0.6)  # Fraction of the canvas that is hidden
RECT_ASPECT = (0.25, 4.0)  # Rectangle extent along its first axis / along its second axis, log-uniform
P_AXIS_ALIGNED = 0.5  # Chance that a rectangle is not rotated
BAND_PROBABILITY = 0.4  # Chance that a thin gray band is also drawn along the boundary of the hidden region
BAND_PX = (2, 10)  # Band radius (it extends this far on both sides of the boundary), in pixels of a 1024² matte
THRESHOLD = 0.5  # Alpha above this is object, for measuring how much of the object is hidden
MIN_HIDDEN_OBJECT = 0.1  # A redrawn region must hide at least this fraction of the object ...
MAX_HIDDEN_OBJECT = 0.9  # ... and at most this fraction
THRESHOLD_SAMPLES = 256  # Grid size (per side) on which hidden_region measures the quantile that cuts the region
PREVIEW_STRIDE = 4  # Redraw candidates are judged on every 4th pixel only
MAX_REDRAWS = 8  # Extra draws of angle and anchor when the hidden object fraction is outside the range; the best one is kept


def canvas_axes(shape, stride=1):
    """
    Pure function. Pixel-center coordinates of every stride-th column and row, divided by max(H, W) so that rotations stay
    isotropic on a non-square canvas.

    Args:
        shape (tuple[int, int]): (H, W)
        stride (int): Keep every stride-th row and column

    Returns:
        tuple[np.ndarray, np.ndarray]: xs (ceil(W / stride),) and ys (ceil(H / stride),) float32; x to the right, y down

    Examples:
        >>> xs, ys = canvas_axes((2, 4))
        >>> xs.tolist(), ys.tolist()  # Pixel centers / 4
        ([0.125, 0.375, 0.625, 0.875], [0.125, 0.375])
        >>> canvas_axes((2, 4), stride=2)[0].tolist()
        [0.125, 0.625]
    """
    height, width = shape
    scale = max(height, width)
    return ((np.arange(width, dtype=np.float32)[::stride] + 0.5) / scale, (np.arange(height, dtype=np.float32)[::stride] + 0.5) / scale)


def unit_vector(angle_deg):
    """
    Pure function. (cos, sin) of an angle given in degrees, clockwise on screen (y down).

    Examples:
        >>> np.round(unit_vector(90), 6).tolist()  # Points down the image
        [0.0, 1.0]
    """
    return np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))], np.float32)


def project(xs, ys, direction, origin_xy=(0.0, 0.0)):
    """
    Pure function. Coordinate along a direction of every pixel of the grid, measured from origin_xy: (x - ox) dx + (y - oy) dy.

    Args:
        xs (np.ndarray): (W,) float32 x coordinates
        ys (np.ndarray): (H,) float32 y coordinates
        direction (np.ndarray): (2,) float32 (dx, dy), see unit_vector
        origin_xy (tuple[float, float]): (x, y)

    Returns:
        np.ndarray: (H, W) float32

    Examples:
        >>> xs, ys = np.array([0.0, 1.0], np.float32), np.array([0.0, 2.0, 4.0], np.float32)
        >>> project(xs, ys, np.array([0, 1], np.float32)).tolist()  # Along y: the row coordinate, any column
        [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]
        >>> project(xs, ys, np.array([1, 0], np.float32), (1.0, 0.0)).tolist()[0]  # Along x from x = 1
        [-1.0, 0.0]
    """
    return einx.add("w, h -> h w", (xs - origin_xy[0]) * direction[0], (ys - origin_xy[1]) * direction[1])


def region_field(xs, ys, kind, angle_deg, anchor_xy, aspect):
    """
    Pure function. Scalar field whose upper level sets are the hidden regions of one kind: larger = deeper inside the region.

    halfplane: signed distance along the direction angle_deg (anchor and aspect unused).
    wedge: minus the angle between (p - anchor) and the direction angle_deg, in radians (aspect unused).
    rect: minus max(|u| / sqrt(aspect), |v| * sqrt(aspect)) with (u, v) the coordinates from anchor along angle_deg and
        its perpendicular, so a level set is a rectangle with u-extent / v-extent = aspect.
    stripe: minus the distance to the line through anchor with direction angle_deg (aspect unused).

    Args:
        xs (np.ndarray): (W,) float32 x coordinates, see canvas_axes
        ys (np.ndarray): (H,) float32 y coordinates
        kind (str): One of KINDS
        angle_deg (float): Direction angle, degrees clockwise on screen from +x
        anchor_xy (tuple[float, float]): (x, y) in canvas coordinates
        aspect (float): Rectangle aspect, > 0

    Returns:
        np.ndarray: (H, W) float32

    Examples:
        >>> xs = ys = np.array([0.25, 0.75], np.float32)
        >>> region_field(xs, ys, "halfplane", 0, (0, 0), 1).round(2).tolist()  # Grows to the right
        [[0.25, 0.75], [0.25, 0.75]]
        >>> region_field(xs, ys, "stripe", 0, (0.5, 0.25), 1).round(2).tolist()  # Line along x at y = 0.25: top row on it
        [[-0.0, -0.0], [-0.5, -0.5]]
        >>> region_field(xs, ys, "rect", 0, (0.75, 0.75), 4).round(2).tolist()  # Wide: u counts 4x less than v, so the bottom-left pixel is nearest
        [[-1.0, -1.0], [-0.25, -0.0]]
        >>> region_field(xs, ys, "wedge", 0, (0, 0.25), 1).astype(float).round(2).tolist()  # Axis along x through y = 0.25: minus the angle in radians
        [[-0.0, -0.0], [-1.11, -0.59]]
    """
    if kind == "halfplane":
        return project(xs, ys, unit_vector(angle_deg))
    u = project(xs, ys, unit_vector(angle_deg), anchor_xy)
    v = project(xs, ys, unit_vector(angle_deg + 90), anchor_xy)
    if kind == "wedge":
        return -np.abs(np.arctan2(v, u))
    if kind == "rect":
        return -np.maximum(np.abs(u) / np.sqrt(aspect), np.abs(v) * np.sqrt(aspect))
    if kind == "stripe":
        return -np.abs(v)
    raise ValueError(f"Unknown region kind {kind!r}. Known: {KINDS}")


def hidden_region(shape, kind, coverage, angle_deg, anchor_xy, aspect=1.0, stride=1):
    """
    Pure function. The hidden region: the pixels of region_field above the cut that makes it cover `coverage` of the canvas.
    The cut is the quantile of the field on a grid of about THRESHOLD_SAMPLES pixels per side, so coverage is exact on small
    canvases and within about a percent on 1024², and the quantile never sorts the full canvas.

    Args:
        shape (tuple[int, int]): (H, W)
        kind (str): One of KINDS
        coverage (float): Fraction of the canvas, in (0, 1)
        angle_deg (float), anchor_xy (tuple[float, float]), aspect (float): See region_field
        stride (int): Return the region on every stride-th pixel only, shape (ceil(H / stride), ceil(W / stride)), same cut

    Returns:
        np.ndarray: (H, W) bool, or the strided shape

    Examples:
        >>> hidden = hidden_region((100, 100), "halfplane", 0.3, 0, (0, 0))
        >>> round(float(hidden.mean()), 2), hidden[:, 70:].all(), hidden[:, :69].any()  # The right 30% columns
        (0.3, np.True_, np.False_)
        >>> hidden = hidden_region((100, 200), "wedge", 0.25, 0, (0.25, 0.25))  # Wedge opening right from the canvas middle
        >>> round(float(hidden.mean()), 2), bool(hidden[50, 199]), bool(hidden[50, 5]), bool(hidden[0, 100])
        (0.25, True, False, False)
        >>> hidden = hidden_region((100, 100), "rect", 0.25, 0, (0.5, 0.5), aspect=1.0)  # A centered 50x50 square
        >>> round(float(hidden.mean()), 2), bool(hidden[50, 50]), bool(hidden[50, 20]), np.nonzero(hidden[50])[0][[0, -1]].tolist()
        (0.25, True, False, [25, 74])
        >>> hidden = hidden_region((100, 100), "stripe", 0.4, 90, (0.5, 0.5))  # A band along y at x = 0.5 (angle 90: the line points down)
        >>> round(float(hidden.mean()), 2), hidden[:, 50].all() or hidden[50].all()
        (0.4, np.True_)
        >>> hidden_region((1000, 1000), "halfplane", 0.4, 30, (0.5, 0.5), stride=10).shape  # The coarse preview used for redraws
        (100, 100)
    """
    threshold_stride = max(1, min(shape) // THRESHOLD_SAMPLES)
    cut = np.quantile(region_field(*canvas_axes(shape, threshold_stride), kind, angle_deg, anchor_xy, aspect), 1 - coverage)
    return region_field(*canvas_axes(shape, stride), kind, angle_deg, anchor_xy, aspect) > cut


def boundary_band(hidden, band_px):
    """
    Pure function. Pixels within band_px of the boundary of a region, on both sides (morphological gradient by a disk); none for band_px = 0.

    Args:
        hidden (np.ndarray): (H, W) bool
        band_px (int): Radius in pixels, >= 0

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> hidden = np.zeros((1, 9), bool); hidden[0, 5:] = True
        >>> boundary_band(hidden, 1).astype(int).tolist()
        [[0, 0, 0, 0, 1, 1, 0, 0, 0]]
        >>> bool(boundary_band(hidden, 0).any())
        False
    """
    if band_px == 0:
        return np.zeros_like(hidden)
    return cv2.morphologyEx(hidden.astype(np.uint8), cv2.MORPH_GRADIENT, disk(band_px), borderType=cv2.BORDER_REPLICATE).astype(bool)


def hidden_object_fraction(hidden, binary):
    """
    Pure function. Fraction of the object pixels (binary) inside the hidden region; 0 for an empty object.

    Examples:
        >>> hidden_object_fraction(np.array([[True, True, False, False]]), np.array([[False, True, True, True]]))
        0.3333333333333333
    """
    return float((hidden & binary).sum() / max(binary.sum(), 1))


def draw_region_params(rng, kind):
    """
    Near-pure function (advances rng). The random geometry of one hidden region: direction angle, anchor as fractions of the object
    bounding box, and for a rectangle its aspect and rotation.

    Args:
        rng (np.random.Generator)
        kind (str): One of KINDS

    Returns:
        dict: angle_deg (float), anchor_frac_x, anchor_frac_y (floats in [0, 1]), aspect (float; 1.0 unless kind is rect)

    Examples:
        >>> sorted(draw_region_params(np.random.default_rng(0), "rect"))
        ['anchor_frac_x', 'anchor_frac_y', 'angle_deg', 'aspect']
        >>> draw_region_params(np.random.default_rng(0), "wedge")["aspect"]
        1.0
    """
    angle = 0.0 if kind == "rect" and rng.random() < P_AXIS_ALIGNED else float(rng.uniform(0, 360))
    aspect = float(np.exp(rng.uniform(*np.log(RECT_ASPECT)))) if kind == "rect" else 1.0
    anchor = rng.uniform(0, 1, 2)
    return {"angle_deg": round(angle, 1), "anchor_frac_x": round(float(anchor[0]), 3),
            "anchor_frac_y": round(float(anchor[1]), 3), "aspect": round(aspect, 2)}


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the region kind (30% halfplane, 25% wedge, 25% rect, 20% stripe), its coverage
    (uniform 0.2..0.6), geometry (draw_region_params) and the optional boundary band (40%, 2..10 px).

    Returns:
        dict: kind (str), coverage (float), angle_deg, anchor_frac_x, anchor_frac_y, aspect (floats), band_px (int, 0 = none), threshold (float)

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params)
        ['anchor_frac_x', 'anchor_frac_y', 'angle_deg', 'aspect', 'band_px', 'coverage', 'kind', 'threshold']
        >>> params["kind"] in KINDS and 0.2 <= params["coverage"] <= 0.6
        True
    """
    kind = str(rng.choice(KINDS, p=KIND_PROBABILITIES))
    band_px = int(rng.integers(BAND_PX[0], BAND_PX[1] + 1)) if rng.random() < BAND_PROBABILITY else 0
    return {"kind": kind, "coverage": round(float(rng.uniform(*COVERAGE)), 3), **draw_region_params(rng, kind),
            "band_px": band_px, "threshold": THRESHOLD}


def make_trimask(alpha, rng, kind, coverage, angle_deg, anchor_frac_x, anchor_frac_y, aspect, band_px, threshold):
    """
    Near-pure function (advances rng, only when the first region hides too little or too much of the object). Starts from the exact
    binary matte (finalize then makes soft alpha 0.5: 1 on fully opaque, 0 on fully transparent) and turns the hidden region, plus the optional band
    along its boundary, gray. Gray never becomes sure, so the result is consistent with the matte by construction.

    The anchor fractions place the anchor inside the object's bounding box. If the region hides less than MIN_HIDDEN_OBJECT or more
    than MAX_HIDDEN_OBJECT of the object (alpha > threshold), angle and anchor are redrawn up to MAX_REDRAWS times, and the draw
    closest to the allowed range is kept. The returned params hold the geometry actually used, plus redraws and hidden_object_fraction (measured on every 4th pixel).

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator)
        kind (str), coverage (float), angle_deg (float), anchor_frac_x (float), anchor_frac_y (float), aspect (float): See hidden_region
        band_px (int): Radius of the gray band along the hidden region's boundary, pixels; 0 = none
        threshold (float): Alpha above this is object, for the hidden object fraction

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask, and the params used

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> params = dict(kind="halfplane", coverage=0.4, angle_deg=0, anchor_frac_x=0.5, anchor_frac_y=0.5, aspect=1.0, band_px=0, threshold=0.5)
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), **params)
        >>> sorted(np.unique(trimask).tolist()), used["redraws"], used["angle_deg"]  # Right 40% of the canvas is gray: x >= 77
        ([0.0, 0.5, 1.0], 0, 0)
        >>> trimask[64, [76, 77]].tolist(), trimask[64, 100].item(), trimask[5, 5].item()
        ([1.0, 0.5], 0.5, 0.0)
        >>> exact = np.where(alpha > 0.5, 1, 0)
        >>> bool(((trimask == 1) <= (exact == 1)).all() and ((trimask == 0) <= (exact == 0)).all())  # Gray never flips a sure pixel
        True
        >>> far = dict(params, angle_deg=180)  # Hides the left 40% = columns 0..51: the object is 20/64 hidden, inside the range
        >>> make_trimask(alpha, np.random.default_rng(0), **far)[1]["redraws"]
        0
        >>> corner = np.zeros((128, 128), np.float32); corner[:20, :20] = 1  # Small object: this half-plane misses it
        >>> used = make_trimask(corner, np.random.default_rng(0), **dict(params, angle_deg=0, coverage=0.2))[1]
        >>> used["redraws"] > 0 or used["hidden_object_fraction"] >= MIN_HIDDEN_OBJECT
        True
        >>> soft = alpha.copy(); soft[60:70, 60:70] = 0.5  # Soft pixels stay gray even outside the hidden region
        >>> make_trimask(soft, np.random.default_rng(0), **dict(params, coverage=0.2, angle_deg=180))[0][65, 65].item()
        0.5
    """
    binary = alpha > threshold
    preview = binary[::PREVIEW_STRIDE, ::PREVIEW_STRIDE]  # Candidates are judged on this small copy; only the winner is drawn at full size
    scale = max(binary.shape)
    x_min, y_min, x_max, y_max = (PREVIEW_STRIDE * v for v in padded_bbox(preview, 0))
    bbox_origin, bbox_size = np.array([x_min, y_min]), np.array([x_max - x_min, y_max - y_min])

    def region(geometry, stride):
        anchor_xy = (bbox_origin + np.array([geometry["anchor_frac_x"], geometry["anchor_frac_y"]]) * bbox_size) / scale
        return hidden_region(binary.shape, kind, coverage, geometry["angle_deg"], anchor_xy, geometry["aspect"], stride)

    def distance_to_range(fraction):
        return max(MIN_HIDDEN_OBJECT - fraction, fraction - MAX_HIDDEN_OBJECT, 0)

    geometry = dict(angle_deg=angle_deg, anchor_frac_x=anchor_frac_x, anchor_frac_y=anchor_frac_y, aspect=aspect)
    fraction = hidden_object_fraction(region(geometry, PREVIEW_STRIDE), preview)
    best, redraws = (distance_to_range(fraction), geometry, fraction), 0
    while best[0] > 0 and redraws < MAX_REDRAWS:
        redraws += 1
        geometry = draw_region_params(rng, kind)
        geometry["aspect"] = aspect  # The rectangle shape stays as drawn; only its position and direction are redrawn
        fraction = hidden_object_fraction(region(geometry, PREVIEW_STRIDE), preview)
        if distance_to_range(fraction) < best[0]:
            best = (distance_to_range(fraction), geometry, fraction)
    _, geometry, fraction = best
    hidden = region(geometry, 1)
    unknown = hidden | boundary_band(hidden, band_px)
    trimask = np.where(unknown, np.float32(0.5), binary.astype(np.float32))  # Exact binary matte, gray where unknown
    used = dict(kind=kind, coverage=coverage, **geometry, band_px=band_px, threshold=threshold,
                redraws=redraws, hidden_object_fraction=round(fraction, 4))
    return finalize(alpha, trimask, rng), used  # Soft alpha becomes gray here, coarsely; sure pixels stay only where the matte allows
