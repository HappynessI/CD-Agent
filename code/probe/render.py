"""Visual panels: t1 | t2 | GT | prediction overlay, cropped around one object."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from common import load_label, load_pred, load_rgb

PANEL = 224
MIN_CROP = 96


def crop_box(bbox, shape, scale=3.0, min_side=MIN_CROP):
    """Square crop centred on bbox [x0,y0,x1,y1], side max(scale*long side, min_side)."""
    x0, y0, x1, y1 = bbox
    height, width = shape
    side = int(min(max(scale * max(x1 - x0, y1 - y0), min_side), height, width))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(round(cx - side / 2), 0, width - side))
    top = int(np.clip(round(cy - side / 2), 0, height - side))
    return left, top, left + side, top + side


def overlay(rgb, pred, gt):
    out = rgb.astype(np.float32).copy()
    for mask, colour in ((pred & gt, (0, 255, 0)), (pred & ~gt, (255, 0, 0)), (~pred & gt, (0, 128, 255))):
        out[mask] = 0.45 * out[mask] + 0.55 * np.asarray(colour)
    return out.astype(np.uint8)


def _panel(array, box, bbox):
    left, top, right, bottom = box
    image = Image.fromarray(array[top:bottom, left:right]).resize((PANEL, PANEL), Image.Resampling.NEAREST)
    k = PANEL / (right - left)
    x0, y0, x1, y1 = bbox
    ImageDraw.Draw(image).rectangle([(x0 - left) * k - 2, (y0 - top) * k - 2, (x1 - left) * k + 1, (y1 - top) * k + 1],
                                    outline=(255, 255, 0), width=2)
    return image


def object_panel(dataset, detector, name, bbox, title, pred=None):
    t1, t2 = load_rgb(dataset, name, "t1"), load_rgb(dataset, name, "t2")
    gt = load_label(dataset, name)
    pred = load_pred(dataset, detector, name) if pred is None else pred
    box = crop_box(bbox, gt.shape)
    gt_rgb = np.repeat((gt * 255).astype(np.uint8)[..., None], 3, axis=2)
    panels = [_panel(a, box, bbox) for a in (t1, t2, gt_rgb, overlay(t2, pred, gt))]
    canvas = Image.new("RGB", (PANEL * 4 + 15, PANEL + 34), "white")
    for i, p in enumerate(panels):
        canvas.paste(p, (i * (PANEL + 5), 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill="black")
    draw.text((4, 17), "t1 | t2 | GT | pred on t2 (green TP, red FP, blue FN), yellow = object box", fill="gray")
    return canvas
