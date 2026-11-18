"""
Trimask method glass_ripple: the matte seen through refracting glass. The matte is displaced by a structured, periodic field;
the trimask is sure only where the true matte and the distorted matte agree, unknown where they disagree (plus an optional thin band).
Unlike perlin_warp's smooth noise, the fields are regular: they imitate a hand-drawn outline wobbling like a ripple or a flute.

Three field kinds (displacement in pixels of a 1024² matte, applied with cv2.remap, bilinear, border replicated):
  - sine: one or two plane waves, random angle / wavelength / amplitude, displacement along the wave vector (longitudinal),
    across it (transverse) or at any angle in between.
  - radial: concentric ripples around a random center, displacement along the radius, optionally decaying with distance (a drop in water).
  - fluted: vertical (slightly rotated) flutes of equal width; inside each flute the displacement ramps linearly like a cylindrical
    lens, with a random per-flute offset: reeded / fluted glass.

Related practice: trimaps for matting nets come from dilating / eroding a binarized alpha (GCA Matting, arXiv:2001.04069);
here the unknown region instead marks where a plausible geometric distortion of the outline changes the object, in the spirit of
elastic / sinusoidal deformation augmentation (Simard et al., "Best Practices for CNNs Applied to Visual Document Analysis", ICDAR 2003,
random displacement fields) and of the ripple / glass distortion filters of image editors.
"""

import cv2
import einx
import numpy as np
from sana_alpha.trimask import finalize

# All pixel values are for a 1024² matte; make_trimask scales them by sqrt(H W) / 1024
REFERENCE_SIZE_PX = 1024
KINDS = ("sine", "radial", "fluted")
KIND_WEIGHTS = (0.4, 0.3, 0.3)
AMPLITUDE_PX = (3, 40)  # Peak displacement, log-uniform. The unknown band along the outline is about this wide
WAVELENGTH_PX = (24, 300)  # Ripple period, log-uniform (flute width for fluted)
FLUTED_WAVELENGTH_PX = (16, 120)  # Flutes are narrower than ripples
FLUTED_TILT_DEG = 12  # Flutes are tilted by up to +-this many degrees from vertical
DECAY_PX = (150, 1200)  # Radial decay length (e^-1 distance), log-uniform
P_NO_DECAY = 0.3  # Chance that radial ripples do not decay (decay_px = 0 means no decay)
P_SECOND_WAVE = 0.35  # Chance that a sine field adds a second plane wave at another angle
SECOND_WAVE_WEIGHT = 0.7  # Second wave amplitude relative to the first
SECOND_WAVELENGTH_FACTOR = (0.6, 1.6)  # Second wavelength relative to the first
P_BAND = 0.4  # Chance of an extra band around every disagreement
BAND_PX = (1, 8)  # Extra band half-width
JITTER = (0.0, 1.0)  # Fluted: per-flute random offset, as a fraction of the amplitude
THRESHOLD = 0.5  # Alpha above this is object


def phase_ramp(height, width, angle_deg):
    """
    Pure function. Signed distance in pixels along a direction: u = x cos(angle) + y sin(angle), pixel (0, 0) is the origin.

    Args:
        height (int), width (int): Output size
        angle_deg (float): Direction from the x axis (pointing right), degrees, clockwise on screen (y points down)

    Returns:
        np.ndarray: (height, width) float32, [y, x] order

    Examples:
        >>> phase_ramp(2, 3, 0).tolist()
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        >>> np.round(phase_ramp(2, 2, 90), 3).tolist()  # Straight down: grows with y
        [[0.0, 0.0], [1.0, 1.0]]
    """
    theta = float(np.deg2rad(angle_deg))
    rows, cols = np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32)
    return einx.add("h, w -> h w", rows * np.float32(np.sin(theta)), cols * np.float32(np.cos(theta)))


def sine_field(height, width, amplitude, wavelength, angle_deg, disp_angle_deg, phase):
    """
    Pure function. A plane-wave displacement field: amplitude x sin(2 pi u / wavelength + phase), u along angle_deg,
    pointing along disp_angle_deg.

    Args:
        height (int), width (int): Output size
        amplitude (float): Peak displacement, pixels
        wavelength (float): Period along the wave vector, pixels
        angle_deg (float): Wave-vector direction (see phase_ramp)
        disp_angle_deg (float): Direction in which pixels move: equal to angle_deg is longitudinal, +90 is transverse
        phase (float): Radians

    Returns:
        np.ndarray: (2, height, width) float32, [(dx, dy)] along axis 0: dx to the right, dy down

    Examples:
        >>> field = sine_field(1, 5, amplitude=2, wavelength=4, angle_deg=0, disp_angle_deg=0, phase=0)
        >>> (np.round(field[0, 0], 3) + 0).tolist(), (np.round(field[1, 0], 3) + 0).tolist()  # x ripples: dx = 2 sin(pi x / 2), no dy
        ([0.0, 2.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        >>> sine_field(4, 4, 3, 8, 30, 120, 1.0).shape
        (2, 4, 4)
    """
    wave = np.float32(amplitude) * np.sin(np.float32(2 * np.pi / wavelength) * phase_ramp(height, width, angle_deg) + np.float32(phase))
    direction = np.deg2rad(disp_angle_deg)
    return einx.multiply("h w, d -> d h w", wave, np.array([np.cos(direction), np.sin(direction)], np.float32))


def radial_field(height, width, amplitude, wavelength, center_xy, decay, phase):
    """
    Pure function. Concentric ripples: displacement along the radius from center_xy,
    amplitude x sin(2 pi r / wavelength + phase) x exp(-r / decay).

    Args:
        height (int), width (int): Output size
        amplitude (float): Peak displacement, pixels
        wavelength (float): Ripple period along the radius, pixels
        center_xy (tuple[float, float]): Ripple center (x, y) in pixels
        decay (float): e^-1 distance in pixels; 0 = no decay
        phase (float): Radians

    Returns:
        np.ndarray: (2, height, width) float32, [(dx, dy)] along axis 0

    Examples:
        >>> field = radial_field(1, 5, amplitude=2, wavelength=4, center_xy=(0, 0), decay=0, phase=0)
        >>> (np.round(field[0, 0], 3) + 0).tolist()  # Moves right along +x, sin(pi x / 2) as for a plane wave
        [0.0, 2.0, 0.0, -2.0, 0.0]
        >>> radial_field(3, 3, 5, 10, (1, 1), 100, 0.5).shape
        (2, 3, 3)
    """
    offset_x = np.arange(width, dtype=np.float32) - np.float32(center_xy[0])
    offset_y = np.arange(height, dtype=np.float32) - np.float32(center_xy[1])
    radius = np.hypot(*np.broadcast_arrays(offset_y[:, None], offset_x[None, :]))
    wave = np.float32(amplitude) * np.sin(np.float32(2 * np.pi / wavelength) * radius + np.float32(phase))
    if decay > 0:
        wave *= np.exp(radius * np.float32(-1 / decay))
    wave /= np.maximum(radius, np.float32(1e-6))  # Radial unit vector (offset / radius) times the wave
    return np.stack([wave * offset_x[None, :], wave * offset_y[:, None]])


def fluted_field(height, width, amplitude, wavelength, tilt_deg, phase, flute_offsets):
    """
    Pure function. Fluted glass: stripes of width wavelength, perpendicular to the tilted x axis. Inside a flute the displacement
    along that axis ramps from -amplitude to +amplitude (a cylindrical lens) plus the flute's own offset.

    Args:
        height (int), width (int): Output size
        amplitude (float): Half-range of the ramp inside a flute, pixels
        wavelength (float): Flute width, pixels
        tilt_deg (float): Rotation of the flute axis from the x axis, degrees (0 = vertical stripes, displacement along x)
        phase (float): Stripe position shift, in flutes (0..1)
        flute_offsets (np.ndarray): (n,) float in [-1, 1], per-flute offset in units of amplitude, n >= the number of flutes crossed;
            flute i uses flute_offsets[i % n]

    Returns:
        np.ndarray: (2, height, width) float32, [(dx, dy)] along axis 0

    Examples:
        >>> field = fluted_field(1, 8, amplitude=4, wavelength=4, tilt_deg=0, phase=0, flute_offsets=np.zeros(2))
        >>> np.round(field[0, 0], 1).tolist()  # Two flutes of 4 px: ramp -4 .. +2, restarts at 4 px
        [-4.0, -2.0, 0.0, 2.0, -4.0, -2.0, 0.0, 2.0]
        >>> fluted_field(1, 4, 4, 4, 0, 0, np.array([1.0, -1.0]))[0, 0, 0].item()  # Flute 0 offset +1 x amplitude: -4 + 4
        0.0
    """
    flutes = phase_ramp(height, width, tilt_deg) * np.float32(1 / wavelength) + np.float32(phase)
    index = np.floor(flutes)
    ramp = 2 * (flutes - index) - 1
    wave = np.float32(amplitude) * (ramp + np.asarray(flute_offsets, np.float32)[index.astype(np.intp) % len(flute_offsets)])
    direction = np.deg2rad(tilt_deg)
    return einx.multiply("h w, d -> d h w", wave, np.array([np.cos(direction), np.sin(direction)], np.float32))


def warp(image, field):
    """
    Pure function. Backward warp: out[y, x] = image[y + dy, x + dx] (bilinear, border replicated). The content moves by -field.

    Args:
        image (np.ndarray): (H, W) float32
        field (np.ndarray): (2, H, W) float32, [(dx, dy)] along axis 0, pixels

    Returns:
        np.ndarray: (H, W) float32

    Examples:
        >>> image = np.array([[0, 0, 1, 0]], np.float32)
        >>> warp(image, np.full((2, 1, 4), [[[1]], [[0]]], np.float32)).tolist()  # Sample one pixel to the right: content moves left
        [[0.0, 1.0, 0.0, 0.0]]
    """
    height, width = image.shape
    columns, rows = np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    return cv2.remap(image, field[0] + columns[None, :], field[1] + rows[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def agreement_trimask(true_object, warped_object, band_px=0):
    """
    Pure function. 1 where both masks are object, 0 where both are background, 0.5 where they disagree (and within band_px of a disagreement).

    Args:
        true_object (np.ndarray): (H, W) bool
        warped_object (np.ndarray): (H, W) bool
        band_px (float): Extra gray margin around the disagreement, pixels (Euclidean); 0 = none

    Returns:
        np.ndarray: (H, W) float32 in {0, 0.5, 1}

    Examples:
        >>> true = np.array([[1, 1, 1, 0, 0, 0, 0, 0]], bool)
        >>> warped = np.array([[1, 1, 0, 0, 0, 0, 0, 1]], bool)
        >>> agreement_trimask(true, warped).tolist()
        [[1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5]]
        >>> agreement_trimask(true, warped, band_px=1).tolist()
        [[1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]]
    """
    disagree = true_object != warped_object
    if band_px > 0:
        disagree = cv2.distanceTransform((~disagree).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= band_px
    return np.where(disagree, 0.5, np.where(true_object, 1, 0)).astype(np.float32)


def log_uniform(rng, low, high):
    """
    Near-pure function (advances rng). A float from [low, high], uniform in log space.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> values = [log_uniform(rng, 3, 40) for _ in range(1000)]
        >>> 3 <= min(values) and max(values) <= 40 and 9 < np.median(values) < 12  # Geometric mean sqrt(3 x 40) = 11
        np.True_
    """
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_params(rng):
    """
    Near-pure function (advances rng). Draws the field kind (40% sine, 30% radial, 30% fluted) and its parameters, in pixels of a 1024² matte.
    Common keys: kind, amplitude_px (3..40), wavelength_px (24..300; fluted 16..120), band_px (0 in 60%, else 1..8).
    sine adds angle_deg, disp_angle_deg (longitudinal, transverse or free), phase, second_wave (None or a dict of the second wave).
    radial adds center_x, center_y (fractions of the width / height), decay_px (0 = none, else 150..1200), phase.
    fluted adds tilt_deg (+-12), phase (flutes), jitter (0..1, per-flute offset amplitude).

    Returns:
        dict: JSON-serializable

    Examples:
        >>> params = sample_params(np.random.default_rng(1))
        >>> params["kind"] in KINDS and 3 <= params["amplitude_px"] <= 40
        True
        >>> sorted({sample_params(np.random.default_rng(s))["kind"] for s in range(30)})
        ['fluted', 'radial', 'sine']
    """
    kind = str(rng.choice(KINDS, p=KIND_WEIGHTS))
    wavelength_range = FLUTED_WAVELENGTH_PX if kind == "fluted" else WAVELENGTH_PX
    params = {"kind": kind, "amplitude_px": round(log_uniform(rng, *AMPLITUDE_PX), 1),
              "wavelength_px": round(log_uniform(rng, *wavelength_range), 1),
              "band_px": round(float(rng.uniform(*BAND_PX)), 1) if rng.random() < P_BAND else 0.0}
    if kind == "sine":
        angle = float(rng.uniform(0, 180))
        mode = rng.choice(["longitudinal", "transverse", "free"])
        disp = angle if mode == "longitudinal" else angle + 90 if mode == "transverse" else float(rng.uniform(0, 360))
        params.update(angle_deg=round(angle, 1), disp_angle_deg=round(disp, 1), phase=round(float(rng.uniform(0, 2 * np.pi)), 2), second_wave=None)
        if rng.random() < P_SECOND_WAVE:
            second_angle = float(rng.uniform(0, 180))
            params["second_wave"] = {"wavelength_px": round(params["wavelength_px"] * float(rng.uniform(*SECOND_WAVELENGTH_FACTOR)), 1),
                                     "angle_deg": round(second_angle, 1), "disp_angle_deg": round(second_angle + float(rng.choice([0, 90])), 1),
                                     "phase": round(float(rng.uniform(0, 2 * np.pi)), 2)}
    elif kind == "radial":
        params.update(center_x=round(float(rng.uniform(0, 1)), 3), center_y=round(float(rng.uniform(0, 1)), 3),
                      decay_px=0.0 if rng.random() < P_NO_DECAY else round(log_uniform(rng, *DECAY_PX), 1),
                      phase=round(float(rng.uniform(0, 2 * np.pi)), 2))
    else:
        params.update(tilt_deg=round(float(rng.uniform(-FLUTED_TILT_DEG, FLUTED_TILT_DEG)), 1), phase=round(float(rng.uniform(0, 1)), 3),
                      jitter=round(float(rng.uniform(*JITTER)), 2))
    return params


def make_trimask(alpha, rng, kind, amplitude_px, wavelength_px, band_px=0.0, angle_deg=0.0, disp_angle_deg=0.0, phase=0.0,
                 second_wave=None, center_x=0.5, center_y=0.5, decay_px=0.0, tilt_deg=0.0, jitter=0.0):
    """
    Near-pure function (fluted draws its per-flute offsets from rng; the other kinds ignore it). Displaces the binarized matte
    by the chosen field; unknown where the true and the displaced matte disagree (plus band_px), sure elsewhere, then finalize.
    Sure white therefore needs a fully opaque matte pixel that is also object after the distortion; sure black the same for transparent.

    Args:
        alpha (np.ndarray): (H, W) float32 in [0, 1], white = object
        rng (np.random.Generator): Draws the flute offsets
        kind (str): "sine", "radial" or "fluted"
        amplitude_px, wavelength_px, band_px (float): In pixels of a 1024² matte; scaled by sqrt(H W) / 1024
        angle_deg, disp_angle_deg, phase, second_wave: sine parameters (see sample_params); phase is also used by radial and fluted
        center_x, center_y (float), decay_px (float): radial parameters; the center is a fraction of the width / height
        tilt_deg (float), jitter (float): fluted parameters

    Returns:
        tuple[np.ndarray, dict]: (H, W) float32 trimask in {0, 0.5, 1}, and the params plus scale and the pixel values actually used

    Examples:
        >>> alpha = np.zeros((128, 128), np.float32); alpha[32:96, 32:96] = 1
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), "sine", 10, 80, angle_deg=0, disp_angle_deg=0, phase=0)
        >>> sorted(np.unique(trimask).tolist()), used["scale"]
        ([0.0, 0.5, 1.0], 0.125)
        >>> trimask[64, 30:36].tolist()  # Left edge at x = 32: the ripple moves it by ~1 px (10 x 0.125): one gray pixel
        [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        >>> trimask, _ = make_trimask(alpha, np.random.default_rng(0), "radial", 400, 800, center_x=0.5, center_y=0.5)
        >>> bool((trimask == 0.5).any() and (trimask == 1).any() and (trimask == 0).any())  # Radial ripple centered in the square
        True
        >>> trimask, used = make_trimask(alpha, np.random.default_rng(0), "fluted", 80, 160, tilt_deg=0, phase=0.3, jitter=0.5)
        >>> sorted(np.unique(trimask).tolist())
        [0.0, 0.5, 1.0]
    """
    height, width = alpha.shape
    scale = float(np.sqrt(height * width) / REFERENCE_SIZE_PX)
    amplitude, wavelength, band = amplitude_px * scale, wavelength_px * scale, band_px * scale
    if kind == "sine":
        field = sine_field(height, width, amplitude, wavelength, angle_deg, disp_angle_deg, phase)
        if second_wave is not None:
            field += sine_field(height, width, amplitude * SECOND_WAVE_WEIGHT, second_wave["wavelength_px"] * scale,
                                second_wave["angle_deg"], second_wave["disp_angle_deg"], second_wave["phase"])
    elif kind == "radial":
        field = radial_field(height, width, amplitude, wavelength, (center_x * width, center_y * height), decay_px * scale, phase)
    elif kind == "fluted":
        flute_count = int(np.ceil((width + height) / wavelength)) + 2  # Upper bound on the flutes a tilted axis crosses
        flute_offsets = jitter * rng.uniform(-1, 1, flute_count)
        field = fluted_field(height, width, amplitude, wavelength, tilt_deg, phase, flute_offsets)
    else:
        raise ValueError(f"Unknown glass_ripple kind {kind!r}. Known: {KINDS}")
    true_object = alpha > THRESHOLD
    trimask = agreement_trimask(true_object, warp(true_object.astype(np.float32), field) > THRESHOLD, band)
    used = {"kind": kind, "amplitude_px": amplitude_px, "wavelength_px": wavelength_px, "band_px": band_px, "scale": scale,
            "angle_deg": angle_deg, "disp_angle_deg": disp_angle_deg, "phase": phase, "second_wave": second_wave,
            "center_x": center_x, "center_y": center_y, "decay_px": decay_px, "tilt_deg": tilt_deg, "jitter": jitter}
    return finalize(alpha, trimask, rng), used
