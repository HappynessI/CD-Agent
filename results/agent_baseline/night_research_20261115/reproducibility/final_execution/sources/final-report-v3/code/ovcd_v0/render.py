"""T2 zoom: square crop around a region and red-contour rendering.

crop_box / edge / render_pair follow exp6j_render.py (used by exp6k/exp6l): side = 4x the longer
bbox side, at least 64 px, clipped to the image; output 384x384 bicubic; 1-px red contour dilated
with a 3x3 max filter.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter

SIZE = 384


def crop_box(bbox, shape, scale=4, min_side=64):
    x0, y0, x1, y1 = bbox
    side = int(min(max(scale * max(x1 - x0, y1 - y0), min_side), *shape))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, shape[1] - side))
    top = int(np.clip((y0 + y1) / 2 - side / 2, 0, shape[0] - side))
    return [left, top, left + side, top + side]


def edge(mask, size=SIZE):
    enlarged = Image.fromarray(mask.astype('uint8') * 255).resize((size, size), Image.Resampling.NEAREST)
    padded = Image.new('L', (size + 4, size + 4))
    padded.paste(enlarged, (2, 2))
    inner = padded.filter(ImageFilter.MinFilter(3)).crop((2, 2, size + 2, size + 2))
    boundary = (np.asarray(enlarged) > 0) & (np.asarray(inner) == 0)
    return np.asarray(Image.fromarray(boundary.astype('uint8') * 255).filter(ImageFilter.MaxFilter(3))) > 0


def render_pair(rgb, mask, box, size=SIZE):
    l, t, r, b = box
    red = edge(mask[t:b, l:r], size)
    out = []
    for image in rgb:
        px = np.array(Image.fromarray(image[t:b, l:r]).resize((size, size), Image.Resampling.BICUBIC))
        px[red] = (255, 0, 0)
        out.append(Image.fromarray(px))
    return out


def crop_to_full(box_norm, crop):
    """0-1000 box in crop coordinates -> full-image xyxy (float)."""
    l, t, r, b = crop
    s = r - l
    x0, y0, x1, y1 = box_norm
    return [l + x0 / 1000 * s, t + y0 / 1000 * s, l + x1 / 1000 * s, t + y1 / 1000 * s]
