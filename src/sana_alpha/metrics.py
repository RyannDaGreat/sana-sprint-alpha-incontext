"""Eval metrics (gray purity, matte IoU) and captioned montages. All images are float RGB in [0,1]."""

import textwrap

import einx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRAY_LEVEL = 0.5
FOREGROUND_THRESHOLD = 0.06  # a pixel is foreground when some channel is this far from 0.5 gray (~15/255)
EXTERIOR_ALPHA = 0.02  # matte values below this count as exterior (fully transparent)
INTERIOR_ALPHA = 0.5  # matte values above this count as the object, for IoU
BORDER_PX = 64  # width of the image-edge band used as "estimated background" when there is no matte to follow
CAPTION_CHARS = 34  # caption wrap width (characters) in a 256 px montage cell; scales with the cell


def gray_deviation(rgb):
    """
    Pure function. Per-pixel distance from 50% gray: max over channels of |rgb - 0.5|.

    Args:
        rgb (np.ndarray): (H, W, 3) float RGB in [0,1]

    Returns:
        np.ndarray: (H, W) float

    Examples:
        >>> gray_deviation(np.array([[[0.5, 0.5, 0.5], [0.5, 0.9, 0.4]]])).round(2).tolist()
        [[0.0, 0.4]]
    """
    return einx.max("H W [C]", np.abs(rgb - GRAY_LEVEL))


def foreground_mask(rgb, threshold=FOREGROUND_THRESHOLD):
    """
    Pure function. Predicted object mask: pixels that are visibly not 50% gray.

    Args:
        rgb (np.ndarray): (H, W, 3) float RGB in [0,1]
        threshold (float): see FOREGROUND_THRESHOLD

    Returns:
        np.ndarray: (H, W) bool

    Examples:
        >>> foreground_mask(np.array([[[0.5, 0.52, 0.5], [0.1, 0.1, 0.1]]])).tolist()
        [[False, True]]
    """
    return gray_deviation(rgb) > threshold


def mask_iou(a, b):
    """
    Pure function. Intersection over union of two boolean masks (1.0 when both are empty).

    Args:
        a, b (np.ndarray): (H, W) bool

    Returns:
        float

    Examples:
        >>> mask_iou(np.array([True, True, False, False]), np.array([True, False, True, False]))
        0.3333333333333333
    """
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def border_band(height, width, band=BORDER_PX):
    """
    Pure function. Mask of the pixels within `band` pixels of the image edge.

    Args:
        height, width (int): image size
        band (int): band width in pixels

    Returns:
        np.ndarray: (height, width) bool

    Examples:
        >>> border_band(4, 5, 1).astype(int).tolist()
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
    """
    mask = np.ones((height, width), bool)
    mask[band:height - band, band:width - band] = False
    return mask


def image_metrics(rgb, matte):
    """
    Pure function. All eval metrics of one generated image against its eval matte.

    exterior_gray_error: mean |rgb-0.5| (mean over channels) where matte < EXTERIOR_ALPHA. Meaningful once a model
        follows the matte; for unconditioned models it mixes object and background.
    border_gray_error / border_std: mean |rgb-0.5| and per-channel std in the BORDER_PX edge band, the estimated
        background of an unconditioned model. A flat 0.5 gray background gives 0 and 0.
    iou: IoU of foreground_mask(rgb) against matte > INTERIOR_ALPHA.
    foreground_fraction: share of pixels in foreground_mask(rgb).

    Args:
        rgb (np.ndarray): (H, W, 3) float RGB in [0,1]
        matte (np.ndarray): (H, W) float in [0,1], 1 = opaque

    Returns:
        dict of floats

    Examples:
        >>> matte = np.zeros((256, 256)); matte[96:160, 96:160] = 1
        >>> rgb = np.full((256, 256, 3), 0.5); rgb[96:160, 96:160] = 0.9
        >>> {k: round(v, 3) for k, v in image_metrics(rgb, matte).items()}
        {'exterior_gray_error': 0.0, 'border_gray_error': 0.0, 'border_std': 0.0, 'iou': 1.0, 'foreground_fraction': 0.062}
    """
    deviation = np.abs(rgb - GRAY_LEVEL)
    exterior = matte < EXTERIOR_ALPHA
    border = border_band(*matte.shape)
    predicted = foreground_mask(rgb)
    return dict(
        exterior_gray_error=float(deviation[exterior].mean()),
        border_gray_error=float(deviation[border].mean()),
        border_std=float(rgb[border].std(axis=0).mean()),
        iou=mask_iou(predicted, matte > INTERIOR_ALPHA),
        foreground_fraction=float(predicted.mean()),
    )


def mean_metrics(per_image):
    """
    Pure function. Averages a list of image_metrics dicts key by key.

    Args:
        per_image (list[dict]): image_metrics outputs

    Returns:
        dict of floats

    Examples:
        >>> mean_metrics([dict(iou=0.2), dict(iou=0.4)])
        {'iou': 0.30000000000000004}
    """
    return {k: float(np.mean([m[k] for m in per_image])) for k in per_image[0]}


def captioned_cell(image, caption, size):
    """
    Pure function. Resizes an image to size x size and puts a wrapped caption under it.

    Args:
        image (np.ndarray): (H, W, 3) float RGB or (H, W) float gray, in [0,1]
        caption (str): text under the image
        size (int): output cell width

    Returns:
        np.ndarray: (size + caption height, size, 3) uint8 RGB

    Examples:
        >>> captioned_cell(np.zeros((1024, 1024, 3)), "a red apple", 128).shape[1:]
        (128, 3)
    """
    image = np.asarray(image, np.float32)
    image = image if image.ndim == 3 else einx.rearrange("H W -> H W C", image, C=3)
    picture = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    font_px = max(size // 18, 8)
    font = ImageFont.load_default(font_px)
    lines = textwrap.wrap(caption, max(size * CAPTION_CHARS // 256, 8)) or [""]
    line_px = font_px + font_px // 4
    cell = Image.new("RGB", (size, size + line_px * len(lines) + font_px // 2), "white")
    cell.paste(picture, (0, 0))
    draw = ImageDraw.Draw(cell)
    for k, line in enumerate(lines):
        draw.text((font_px // 3, size + font_px // 4 + k * line_px), line, fill="black", font=font)
    return np.asarray(cell)


def montage(rows, size=256):
    """
    Pure function. A grid of captioned cells on white. Short rows and short captions are padded with white.

    Args:
        rows (list[list[tuple]]): rows of (image, caption) pairs; image (H, W, 3) or (H, W) float in [0,1]
        size (int): cell width in pixels

    Returns:
        np.ndarray: (total_H, columns * size, 3) uint8 RGB

    Examples:
        >>> g = montage([[(np.zeros((64, 64, 3)), "a"), (np.ones((64, 64)), "b")], [(np.zeros((64, 64)), "c")]], size=32)
        >>> g.shape[1:], g.dtype
        ((64, 3), dtype('uint8'))
    """
    cells = [[captioned_cell(image, caption, size) for image, caption in row] for row in rows]
    height = max(cell.shape[0] for row in cells for cell in row)
    columns = max(len(row) for row in cells)
    white = np.full((height, size * columns, 3), 255, np.uint8)
    lines = []
    for row in cells:
        line = white.copy()
        for k, cell in enumerate(row):
            line[:cell.shape[0], k * size:(k + 1) * size] = cell
        lines.append(line)
    return np.concatenate(lines, axis=0)
