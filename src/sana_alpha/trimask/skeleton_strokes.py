"""
Trimask method skeleton_strokes: a "stick figure" of the object. Sure-white (1) only on thick strokes along the medial axis
of the fully-opaque region, sure-black (0) only far outside the object, everything else unknown (0.5).

Recipe: core = fully-opaque pixels at least `margin_px` from any non-opaque pixel; skeleton = skeletonize(core); strokes =
skeleton dilated by a random radius (optionally with smoothly varying width), warped by a smooth random displacement
field, broken by random dropout discs, then clipped to the core (so a stroke never touches the matte edge).
Black = transparent pixels farther than `far_px` from the nearest non-transparent pixel. By construction white is only
on opaque pixels and black only on transparent ones.

Sources (the skeleton-scribble simulation recipe, adapted from segmentation to a known matte):
    - Chen et al., "ScribbleSeg: Scribble-based Interactive Image Segmentation", arXiv 2303.11320 (axial / medial-axis
      scribble meta-simulator; stroke thickness 3-7 px).
    - Wong et al., "ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image", ECCV 2024,
      arXiv 2312.07381 (centerline scribbles: thin to a 1-px skeleton, break it with a random mask, warp it with a random
      deformation field, multiply by the mask to remove parts that left it).
    - Cheng et al., "Deep Image Matting with Flexible Guidance Input", arXiv 2110.10898 (matting scribbles clipped to the
      trimap's sure regions).
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from sana_alpha.trimask import OPAQUE, TRANSPARENT, finalize

# All pixel values are for a 1024² matte
MARGIN_PX = (2, 30)  # Gap between a stroke and the matte edge, log-uniform
MAX_MARGIN_RATIO = 0.5  # The margin never exceeds this fraction of the deepest point of the object (thin objects keep a core)
SKELETON_DOWNSCALE = 4  # The skeleton is computed at 1 / this resolution (skeletonize is too slow at 1024²)
RADIUS_PX = (1.5, 24)  # Stroke half-width, log-uniform
P_WOBBLE, WOBBLE = 0.6, (0.1, 0.6)  # Chance of a varying stroke width, and its log-width std range
P_WARP, WARP_PX = 0.5, (2, 14)  # Chance of a smooth displacement of the strokes, and its amplitude range
P_DROPOUT, DROPOUTS, DROPOUT_RADIUS_PX = 0.5, (1, 8), (15, 120)  # Chance of breaking strokes with discs: count, radius (log-uniform)
FAR_PX = (15, 400)  # Black only farther than this from the object, log-uniform
NOISE_CELLS = 6  # Smooth noise is a random grid of this many cells per side, upsampled bicubically
NOISE_CLIP = 2.0  # Smooth noise is clipped to this many std
WARP_NOISE_STD = 1.0  # Std of the smooth displacement field in units of the amplitude


def sample_log_uniform(rng, low, high):
    """
    Near-pure function (advances rng). Log-uniform float in [low, high].

    Examples:
        >>> 4 <= sample_log_uniform(np.random.default_rng(0), 4, 16) <= 16
        True
    """
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def smooth_noise(rng, shape):
    """
    Near-pure function (advances rng). Low-frequency noise: a NOISE_CELLS² normal grid upsampled bicubically, clipped to ±NOISE_CLIP.

    Args:
        rng (np.random.Generator)
        shape (tuple[int, int]): (H, W)

    Returns:
        np.ndarray: (H, W) float32, roughly unit std, smooth over ~H / NOISE_CELLS pixels

    Examples:
        >>> noise = smooth_noise(np.random.default_rng(0), (256, 192))
        >>> noise.shape, noise.dtype, bool(abs(noise).max() <= NOISE_CLIP)
        ((256, 192), dtype('float32'), True)
        >>> float(abs(np.diff(noise, axis=1)).max()) < 0.1  # Neighbouring pixels are nearly equal
        True
    """
    grid = rng.normal(size=(NOISE_CELLS, NOISE_CELLS)).astype(np.float32)
    return np.clip(cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC), -NOISE_CLIP, NOISE_CLIP)


def distance_to_other_class(mask):
    """
    Pure function. Approximate Euclidean distance (cv2 5x5 mask, error under 2%, 7x faster than the exact one) of every pixel
    to the nearest pixel of the opposite class (0 on the opposite class). The image border is not an edge.

    Args:
        mask (np.ndarray): (H, W) bool, True = this class

    Returns:
        np.ndarray: (H, W) float32, in pixels; huge where the opposite class is absent

    Examples:
        >>> mask = np.zeros((1, 9), bool); mask[0, 2:8] = True
        >>> distance_to_other_class(mask).round(1).tolist()
        [[0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0]]
        >>> bool(distance_to_other_class(np.ones((4, 4), bool)).min() > 1000)
        True
    """
    padded = np.pad(mask, 1, constant_values=True)  # The image border is not an edge of the matte
    return cv2.distanceTransform(padded.view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)[1:-1, 1:-1]


def coarse_skeleton(core):
    """
    Pure function. Skeleton of a mask computed at 1 / SKELETON_DOWNSCALE resolution (skeletonize costs ~0.5 s at 1024², ~0.03 s at 256²).

    Args:
        core (np.ndarray): (H, W) bool

    Returns:
        np.ndarray: (H // SKELETON_DOWNSCALE, W // SKELETON_DOWNSCALE) bool, 1-px curves of the mask at the coarse resolution

    Examples:
        >>> core = np.zeros((40, 80), bool); core[8:32, 8:72] = True  # A 24 x 64 rectangle
        >>> skeleton = coarse_skeleton(core); skeleton.shape
        (10, 20)
        >>> int(skeleton[4, 10]), int(skeleton[2, 10])  # The medial axis runs along the middle row; none near the edge
        (1, 0)
    """
    height, width = core.shape
    coarse = cv2.resize(core.astype(np.float32), (width // SKELETON_DOWNSCALE, height // SKELETON_DOWNSCALE), interpolation=cv2.INTER_AREA)
    return skeletonize(coarse >= 0.5)


def thick_strokes(skeleton, shape, radius_px, scale=None):
    """
    Pure function. Strokes of half-width radius_px around a coarse skeleton: the exact Euclidean distance to the skeleton,
    computed at coarse resolution, is upsampled bilinearly (smooth edges) and cut at the radius. Optional per-pixel radius scale.

    Args:
        skeleton (np.ndarray): (h, w) bool, 1-px curves at 1 / SKELETON_DOWNSCALE of the output resolution
        shape (tuple[int, int]): (H, W) of the output
        radius_px (float): Half-width in output pixels
        scale (np.ndarray | None): (H, W) float > 0, multiplies the radius at each pixel (width wobble); None = 1

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> skeleton = np.zeros((5, 9), bool); skeleton[2, 1:8] = True  # A horizontal line, at 1/4 resolution of a (20, 36) image
        >>> strokes = thick_strokes(skeleton, (20, 36), 1.5)
        >>> strokes[:, 18].astype(int).tolist()  # Column through the middle: 4 px (the radius reaches 1.5 px either side of y=9.5)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> int(thick_strokes(skeleton, (20, 36), 6)[:, 18].sum())  # Radius 6 is about 12 px wide
        12
    """
    distance_to_skeleton = cv2.distanceTransform((~skeleton).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance = SKELETON_DOWNSCALE * cv2.resize(distance_to_skeleton, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return distance <= (radius_px if scale is None else radius_px * scale)


def warp_mask(mask, dx, dy):
    """
    Pure function. Moves a binary mask by a displacement field: output(y, x) = mask(y + dy, x + dx), bilinear, cut at 0.5.

    Args:
        mask (np.ndarray): (H, W) bool
        dx (np.ndarray): (H, W) float32, pixels along x (right)
        dy (np.ndarray): (H, W) float32, pixels along y (down)

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> mask = np.zeros((1, 6), bool); mask[0, 1:3] = True
        >>> warp_mask(mask, np.full((1, 6), -2, np.float32), np.zeros((1, 6), np.float32)).astype(int).tolist()  # Shifted right by 2
        [[0, 0, 0, 1, 1, 0]]
    """
    height, width = mask.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    moved = cv2.remap(mask.astype(np.float32), x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return moved >= 0.5


def drop_discs(mask, centers_xy, radius_px):
    """
    Pure function. Clears discs from a mask (breaks strokes like a hand lifting off the screen).

    Args:
        mask (np.ndarray): (H, W) bool
        centers_xy (np.ndarray): (N, 2) int, (x, y) pixel coordinates
        radius_px (np.ndarray): (N,) int, one radius per disc

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> drop_discs(np.ones((1, 7), bool), np.array([[3, 0]]), np.array([1])).astype(int).tolist()
        [[1, 1, 0, 0, 0, 1, 1]]
    """
    holes = np.zeros(mask.shape, np.uint8)
    for (x, y), radius in zip(centers_xy, radius_px):
        cv2.circle(holes, (int(x), int(y)), int(radius), 1, -1)
    return mask & ~holes.astype(bool)


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws margin, stroke radius, width wobble, warp, dropout and the far distance of the black.
    Wobble 60%, warp 50%, dropout 50% (each 0 when off). Radii and distances are log-uniform.

    Returns:
        dict: margin_px, radius_px, wobble, warp_px (float), n_dropouts, dropout_radius_px (int), far_px (float), seed (int, for noise and dropout positions)

    Examples:
        >>> params = sample_params(np.random.default_rng(0)); sorted(params)
        ['dropout_radius_px', 'far_px', 'margin_px', 'n_dropouts', 'radius_px', 'seed', 'warp_px', 'wobble']
        >>> 2 <= params["margin_px"] <= 30 and 15 <= params["far_px"] <= 400
        True
    """
    return {"margin_px": round(sample_log_uniform(rng, *MARGIN_PX), 1),
            "radius_px": round(sample_log_uniform(rng, *RADIUS_PX), 1),
            "wobble": round(float(rng.uniform(*WOBBLE)), 2) if rng.random() < P_WOBBLE else 0.0,
            "warp_px": round(float(rng.uniform(*WARP_PX)), 1) if rng.random() < P_WARP else 0.0,
            "n_dropouts": int(rng.integers(DROPOUTS[0], DROPOUTS[1] + 1)) if rng.random() < P_DROPOUT else 0,
            "dropout_radius_px": int(round(sample_log_uniform(rng, *DROPOUT_RADIUS_PX))),
            "far_px": round(sample_log_uniform(rng, *FAR_PX), 1),
            "seed": int(rng.integers(2 ** 31))}


def make_trimask(alpha, rng, margin_px, radius_px, wobble, warp_px, n_dropouts, dropout_radius_px, far_px, seed):
    """
    Near-pure function (noise and dropout positions come from np.random.default_rng(seed), not from rng, so they are reproducible from the params).
    White = skeleton strokes inside the margin-eroded opaque region, black = beyond far_px from the object, else 0.5.
    The margin is capped at MAX_MARGIN_RATIO of the deepest point of the opaque region, so thin objects keep a skeleton.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Unused
        margin_px (float): Minimum distance of a stroke from the nearest non-opaque pixel
        radius_px (float): Stroke half-width
        wobble (float): Std of the log of the local stroke width (0 = constant width)
        warp_px (float): Amplitude of the smooth displacement of the strokes, in pixels (0 = none)
        n_dropouts (int), dropout_radius_px (int): Discs (centered on the strokes) that break the strokes up
        far_px (float): Black only where the distance to the nearest non-transparent pixel exceeds this
        seed (int): Seeds the noise fields and the dropout positions

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params (margin_px as capped)

    Examples:
        >>> alpha = np.zeros((60, 100), np.float32); alpha[10:50, 10:90] = 1  # A 40 x 80 rectangle
        >>> trimask, used = make_trimask(alpha, None, 4, 2, 0, 0, 0, 0, 12, 0)
        >>> sorted(np.unique(trimask).tolist())
        [0.0, 0.5, 1.0]
        >>> int(trimask[30, 50]), trimask[30, 50 + 20].item(), trimask[16, 50].item()  # On the stroke; inside, off the stroke
        (1, 1.0, 0.5)
        >>> trimask[30, 12].item(), trimask[0, 0].item(), trimask[30, 99].item()  # Near the edge: gray; far outside: black; near: gray
        (0.5, 0.0, 0.5)
        >>> bool((alpha[trimask == 1] == 1).all() and (alpha[trimask == 0] == 0).all())  # Correct by construction
        True
        >>> make_trimask(np.zeros((8, 8), np.float32), None, 4, 2, 0, 0, 0, 0, 12, 0)[0].max().item()  # No opaque pixel: no white (all black)
        0.0
    """
    opaque, transparent = alpha >= OPAQUE - 1e-6, alpha <= TRANSPARENT + 1e-6
    depth = distance_to_other_class(opaque)
    margin_px = min(margin_px, MAX_MARGIN_RATIO * float(depth.max()))
    core = depth > margin_px
    skeleton = coarse_skeleton(core)
    noise = np.random.default_rng(seed)
    scale = np.exp(wobble * smooth_noise(noise, alpha.shape)) if wobble > 0 else None
    strokes = thick_strokes(skeleton, alpha.shape, radius_px, scale)
    if warp_px > 0:
        strokes = warp_mask(strokes, *(warp_px * WARP_NOISE_STD * smooth_noise(noise, alpha.shape) for _ in range(2)))
    on_skeleton = np.argwhere(skeleton)[:, ::-1] * SKELETON_DOWNSCALE + SKELETON_DOWNSCALE // 2  # (N, 2) as (x, y), full-resolution pixels
    if n_dropouts > 0 and len(on_skeleton) > 0:
        centers = on_skeleton[noise.integers(len(on_skeleton), size=n_dropouts)]
        strokes = drop_discs(strokes, centers, np.maximum(1, noise.integers(dropout_radius_px // 2 + 1, dropout_radius_px + 1, size=n_dropouts)))
    far_outside = distance_to_other_class(transparent) > far_px
    trimask = np.where(strokes & core, 1, np.where(far_outside, 0, 0.5)).astype(np.float32)
    used = {"margin_px": round(margin_px, 1), "radius_px": radius_px, "wobble": wobble, "warp_px": warp_px, "n_dropouts": n_dropouts,
            "dropout_radius_px": dropout_radius_px, "far_px": far_px, "seed": seed}
    return finalize(alpha, trimask, rng), used
