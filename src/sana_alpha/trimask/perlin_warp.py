"""
Trimask method perlin_warp: "a distorted glass by Perlin noise" (user m00651). The matte is warped by a smooth fractal-noise
displacement field, which plays the user's distorted guess of the object; the unknown is where the guess and the true matte disagree.

Sure object = true matte fully opaque AND warped matte object; sure background = true matte fully transparent AND warped matte
background; everything else (disagreement, soft alpha, plus an optional thin band around it) is unknown. So sure pixels are always
exact, and the unknown band follows the displacement: wide where the warp moves the edge far, thin where it moves little.

Noise is gradient (Perlin) noise summed over octaves (fractional Brownian motion), implemented in numpy and evaluated at 1/4
resolution, then upsampled (the field is smooth, so nothing is lost). Sources:
  - Perlin, "Improving Noise" (ACM TOG 21(3), SIGGRAPH 2002): gradient noise with the quintic fade curve used here.
  - Perlin, "An Image Synthesizer" (SIGGRAPH 1985): octave sums of noise (turbulence); also the origin of using noise to displace textures.
  - Trimap-robustness training of matting models perturbs the trimap with random morphology / guidance corruption so it
    stops being a perfect function of the alpha: arXiv:2101.09095 ("Towards Enhancing Fine-grained Details for Image Matting"),
    Yu et al., "Mask Guided Matting via Progressive Refinement Network" (CVPR 2021). A smooth warp is a structured version of that.
"""

import cv2
import einx
import numpy as np
from sana_alpha.trimask import finalize

# All pixel values are for a 1024² matte
NOISE_DOWNSCALE = 4  # The noise is computed at 1/4 resolution and upsampled
BASE_CELL_PX = (60, 500)  # Lattice cell of the lowest octave, drawn log-uniformly: big cells are smooth glass, small ones are ripples
MIN_CELL_PX = 16  # No octave has a cell smaller than this (keeps the 1/4-resolution noise well sampled)
OCTAVES = (1, 4)  # Octave count range, inclusive
PERSISTENCE = (0.35, 0.6)  # Amplitude ratio between an octave and the next lower one
AMPLITUDE_PX = (4, 60)  # Peak displacement in pixels, drawn log-uniformly
P_BAND = 0.6  # Chance of an extra thin unknown band around the disagreement
BAND_PX = (1, 8)  # That band's radius
THRESHOLD_RANGE = (0.4, 0.6)  # The warped matte is object where it is above this
NOISE_SEED_RANGE = 2 ** 32  # The noise seed is a param, so make_trimask is deterministic given its params


def fade(t):
    """
    Pure function. Perlin's quintic ease curve 6t^5 - 15t^4 + 10t^3 (zero first and second derivative at 0 and 1).

    Examples:
        >>> fade(np.array([0.0, 0.5, 1.0])).tolist()
        [0.0, 0.5, 1.0]
        >>> round(float(fade(np.float64(0.25))), 4)  # Flatter than linear near the ends
        0.1035
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(shape, cell_px, rng):
    """
    Near-pure function (advances rng, which draws the random lattice gradients). 2D gradient noise: random unit gradients on a
    lattice of cell_px spacing, dot products with the offsets to the 4 surrounding lattice points, blended with fade().

    Args:
        shape (tuple[int, int]): (H, W)
        cell_px (float): Lattice spacing in pixels
        rng (np.random.Generator)

    Returns:
        np.ndarray: (H, W) float64, zero at every lattice point, roughly within [-0.7, 0.7]

    Examples:
        >>> noise = perlin_noise((33, 33), 8, np.random.default_rng(0))
        >>> noise.shape, float(noise[0, 0]), float(noise[8, 16]), float(noise[32, 32])  # Zero on the lattice
        ((33, 33), 0.0, 0.0, 0.0)
        >>> bool(np.abs(noise[3:6, 3:6]).max() > 0.05)  # Not zero between lattice points
        True
        >>> bool((perlin_noise((33, 33), 8, np.random.default_rng(0)) == noise).all())  # Deterministic in rng
        True
    """
    height, width = shape
    lattice_h, lattice_w = int(np.ceil(height / cell_px)) + 2, int(np.ceil(width / cell_px)) + 2
    angle = rng.uniform(0, 2 * np.pi, (lattice_h, lattice_w))
    gradient = einx.rearrange("c h w -> h w c", np.stack([np.cos(angle), np.sin(angle)]))  # (lattice_h, lattice_w, 2) as (gx, gy)
    y, x = np.arange(height) / cell_px, np.arange(width) / cell_px
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0

    def corner(dy, dx):
        g = gradient[(y0 + dy)[:, None], (x0 + dx)[None, :]]  # (H, W, 2)
        return g[..., 0] * (fx - dx)[None, :] + g[..., 1] * (fy - dy)[:, None]

    u, v = fade(fx)[None, :], fade(fy)[:, None]
    top = corner(0, 0) * (1 - u) + corner(0, 1) * u
    bottom = corner(1, 0) * (1 - u) + corner(1, 1) * u
    return top * (1 - v) + bottom * v


def fractal_noise(shape, base_cell_px, octaves, persistence, rng):
    """
    Near-pure function (advances rng). Fractional Brownian motion: octaves of perlin_noise, each with half the cell and
    `persistence` times the amplitude of the one before, normalized so the peak absolute value is exactly 1.
    Octaves whose cell would fall under MIN_CELL_PX / NOISE_DOWNSCALE (in the working resolution) are dropped.

    Args:
        shape (tuple[int, int]): (H, W) of the working resolution
        base_cell_px (float): Cell of the first octave, in working-resolution pixels
        octaves (int): >= 1
        persistence (float): Amplitude ratio between consecutive octaves
        rng (np.random.Generator)

    Returns:
        np.ndarray: (H, W) float64 in [-1, 1] with max |value| = 1

    Examples:
        >>> field = fractal_noise((64, 64), 32, 3, 0.5, np.random.default_rng(1))
        >>> field.shape, round(float(np.abs(field).max()), 6)
        ((64, 64), 1.0)
        >>> one = fractal_noise((64, 64), 32, 1, 0.5, np.random.default_rng(1))  # The octaves 2.. add finer detail to octave 1
        >>> bool(np.abs(np.diff(field, axis=1)).mean() > np.abs(np.diff(one, axis=1)).mean())
        True
    """
    min_cell = MIN_CELL_PX / NOISE_DOWNSCALE
    total = np.zeros(shape)
    for octave in range(octaves):
        cell = base_cell_px / 2 ** octave
        if octave > 0 and cell < min_cell:
            break
        total += persistence ** octave * perlin_noise(shape, cell, rng)
    return total / np.abs(total).max()


def displacement_field(shape, base_cell_px, octaves, persistence, amplitude_px, rng):
    """
    Near-pure function (advances rng). Smooth random displacement: independent fractal noise for x and y, scaled so the
    largest displacement of either axis is amplitude_px.

    Args:
        shape (tuple[int, int]): (H, W) of the full-resolution matte
        base_cell_px (float): Full-resolution cell of the first octave
        octaves (int), persistence (float): See fractal_noise
        amplitude_px (float): Peak displacement
        rng (np.random.Generator)

    Returns:
        np.ndarray: (2, H, W) float32, channel 0 = dx (right), channel 1 = dy (down), in pixels, max |.| = amplitude_px

    Examples:
        >>> field = displacement_field((128, 96), 64, 2, 0.5, 10.0, np.random.default_rng(2))
        >>> field.shape, field.dtype, round(float(np.abs(field).max()), 4)
        ((2, 128, 96), dtype('float32'), 10.0)
    """
    small = (int(np.ceil(shape[0] / NOISE_DOWNSCALE)), int(np.ceil(shape[1] / NOISE_DOWNSCALE)))
    raw = np.stack([fractal_noise(small, base_cell_px / NOISE_DOWNSCALE, octaves, persistence, rng) for _ in range(2)])
    field = np.stack([cv2.resize(channel.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) for channel in raw])
    return (field * amplitude_px / np.abs(field).max()).astype(np.float32)


def warp(image, field):
    """
    Pure function. Backward warp: output(y, x) = image(y + dy, x + dx), bilinear, the border replicated.

    Args:
        image (np.ndarray): (H, W) float32
        field (np.ndarray): (2, H, W) float32, (dx, dy) in pixels

    Returns:
        np.ndarray: (H, W) float32

    Examples:
        >>> image = np.zeros((1, 6), np.float32); image[0, 2] = 1
        >>> field = np.zeros((2, 1, 6), np.float32); field[0] = 1  # Every pixel reads its right neighbour: the dot moves left
        >>> warp(image, field).tolist()
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    """
    height, width = image.shape
    map_x = einx.add("w, h w -> h w", np.arange(width, dtype=np.float32), field[0])
    map_y = einx.add("h, h w -> h w", np.arange(height, dtype=np.float32), field[1])
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def disagreement_trimask(opaque, transparent, guess, band_px):
    """
    Pure function. 1 where the matte is fully opaque and the guess says object, 0 where it is fully transparent and the guess
    says background, 0.5 everywhere else (the guess and the matte disagree, or the matte is soft). The 0.5 region then grows by band_px.

    Args:
        opaque (np.ndarray): (H, W) bool, the matte is fully opaque
        transparent (np.ndarray): (H, W) bool, the matte is fully transparent
        guess (np.ndarray): (H, W) bool, the warped binary matte
        band_px (int): Radius of the extra unknown band, >= 0

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> opaque = np.array([[1, 1, 1, 0, 0, 0, 0, 0]], bool); transparent = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], bool)
        >>> guess = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], bool)  # The guess is too wide by 2 px
        >>> disagreement_trimask(opaque, transparent, guess, 0).tolist()
        [[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0]]
        >>> disagreement_trimask(opaque, transparent, guess, 1).tolist()  # One more pixel of band on each side
        [[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]]
    """
    sure_object, sure_background = opaque & guess, transparent & ~guess
    unknown = ~sure_object & ~sure_background
    if band_px > 0:
        unknown = cv2.distanceTransform((~unknown).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= band_px
    return np.where(unknown, 0.5, np.where(sure_object, 1, 0)).astype(np.float32)


def log_uniform(rng, low, high):
    """
    Near-pure function (advances rng). A float from [low, high], uniform in log space.

    Examples:
        >>> values = [log_uniform(np.random.default_rng(i), 4, 60) for i in range(200)]
        >>> bool(4 <= min(values) and max(values) <= 60)
        True
    """
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the noise and the warp.
    base_cell_px 60..500 (log-uniform), octaves 1..4, persistence 0.35..0.6, amplitude_px 4..60 (log-uniform),
    band_px 0 (40%) or 1..8, threshold 0.4..0.6, noise_seed (an int: the noise is a function of it).

    Returns:
        dict: base_cell_px (float), octaves (int), persistence (float), amplitude_px (float), band_px (int), threshold (float), noise_seed (int)

    Examples:
        >>> params = sample_params(np.random.default_rng(0))
        >>> sorted(params), params["octaves"], params["band_px"]
        (['amplitude_px', 'band_px', 'base_cell_px', 'noise_seed', 'octaves', 'persistence', 'threshold'], 3, 0)
    """
    return {"base_cell_px": round(log_uniform(rng, *BASE_CELL_PX), 1),
            "octaves": int(rng.integers(OCTAVES[0], OCTAVES[1] + 1)),
            "persistence": round(float(rng.uniform(*PERSISTENCE)), 2),
            "amplitude_px": round(log_uniform(rng, *AMPLITUDE_PX), 1),
            "band_px": int(rng.integers(BAND_PX[0], BAND_PX[1] + 1)) if rng.random() < P_BAND else 0,
            "threshold": round(float(rng.uniform(*THRESHOLD_RANGE)), 2),
            "noise_seed": int(rng.integers(NOISE_SEED_RANGE))}


def make_trimask(alpha, rng, base_cell_px, octaves, persistence, amplitude_px, band_px, threshold, noise_seed):
    """
    Near-pure function (advances rng, only in finalize; otherwise the noise comes from noise_seed). Warps the matte by a fractal-noise displacement field,
    binarizes the warp at threshold (the guess), and marks as unknown whatever disagrees with the true matte.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Only finalize uses it (draws its dilation radius)
        base_cell_px, octaves, persistence, amplitude_px: The displacement field, see displacement_field
        band_px (int): Extra unknown band radius around the disagreement
        threshold (float): The warped matte is object where above this
        noise_seed (int): Seed of the noise

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params

    Examples:
        >>> disk = np.zeros((256, 256), np.float32); disk[np.hypot(*np.mgrid[-128:128, -128:128] + 0.5) <= 80] = 1
        >>> params = dict(base_cell_px=80.0, octaves=2, persistence=0.5, amplitude_px=200.0, band_px=0, threshold=0.5, noise_seed=3)
        >>> trimask, used = make_trimask(disk, np.random.default_rng(0), **params)
        >>> sorted(np.unique(trimask).tolist()), used == params
        ([0.0, 0.5, 1.0], True)
        >>> bool((trimask[disk == 1] >= 0.5).all() and (trimask[disk == 0] <= 0.5).all())  # Never wrong, only unsure
        True
        >>> gray = (trimask == 0.5).mean(); bool(0.01 < gray < 0.6)  # A ring of unknown, not everything
        True
        >>> still, _ = make_trimask(disk, np.random.default_rng(0), **{**params, "amplitude_px": 1e-3})  # A zero warp: the guess is the matte
        >>> round(float((still == 0.5).mean()), 3), round(float(gray), 3)  # Only finalize's coarse ring is gray; the warp adds more
        (0.324, 0.495)
    """
    field = displacement_field(alpha.shape, base_cell_px, octaves, persistence, amplitude_px, np.random.default_rng(noise_seed))
    guess = warp(alpha, field) > threshold
    opaque, transparent = alpha >= 1 - 1 / 255, alpha <= 1 / 255
    used = {"base_cell_px": base_cell_px, "octaves": octaves, "persistence": persistence, "amplitude_px": amplitude_px,
            "band_px": band_px, "threshold": threshold, "noise_seed": noise_seed}
    return finalize(alpha, disagreement_trimask(opaque, transparent, guess, band_px), rng), used
