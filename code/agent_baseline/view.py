"""Aspect-preserving views with full-image ticks, contours and component IDs."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from render import crop_box, render_pair
from .state import ToolError, chosen_ids

BORDER = 28


def ask_images(rgb, region, bbox, context=False):
    """Close-up T1, T2 for one region; with context, also a wider T1, T2 at the same size."""
    shape = region.shape
    images = render_pair(rgb, region, crop_box(bbox, shape))
    if context:
        images = images + render_pair(rgb, region, crop_box(bbox, shape, scale=10, min_side=256))
    return images


def box_pixels(box, shape, field='box'):
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in box)):
        raise ToolError('bad_args', f'{field} must contain four finite pixel coordinates.')
    x0, y0, x1, y1 = box
    h, w = shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ToolError('bad_args', f'{field} must be an ordered box inside the full image.')
    return [int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))]


def expanded_region(region, shape):
    h, w = shape
    box = box_pixels(region, shape, 'region') if region is not None else [0, 0, w, h]
    x0, y0, x1, y1 = box
    for axis, maximum in ((0, w), (1, h)):
        lo, hi = box[axis], box[axis + 2]
        side = min(max(hi - lo, 64), maximum)
        start = max(0, min(int((lo + hi - side) / 2), maximum - side))
        box[axis], box[axis + 2] = start, start + side
    return box


def render_view(rgb, directory, number, region=None, overlay=None, overlay_handle=None,
                label_components=False, ids=None):
    shape = rgb[0].shape[:2]
    box = expanded_region(region, shape)
    x0, y0, x1, y1 = box
    scale = 768 / max(x1 - x0, y1 - y0)
    size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    if (label_components or ids is not None) and overlay is None:
        raise ToolError('bad_args', 'overlay is required for component labels or ids.')
    selected = chosen_ids(overlay.components, ids) if overlay is not None else set()
    red = None
    candidates = []
    if overlay is not None:
        # Draw one display-pixel inner contour, without filling or thickening small objects.
        crop = overlay.data[y0:y1, x0:x1]
        small = np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.NEAREST))
        red = small & ~ndi.binary_erosion(small, border_value=0)
        if label_components or ids is not None:
            for c in overlay.components:
                if c['id'] not in selected:
                    continue
                yy, xx = np.nonzero(overlay.labels[y0:y1, x0:x1] == c['id'])
                if len(xx):
                    candidates.append((c['id'], BORDER + int(xx.mean() * scale),
                                       BORDER + int(yy.mean() * scale)))
    paths, labelled = [], []
    for t, im in enumerate(rgb, 1):
        arr = np.array(Image.fromarray(im[y0:y1, x0:x1]).resize(size, Image.Resampling.BICUBIC))
        if red is not None:
            arr[red] = (255, 0, 0)
        canvas = Image.new('RGB', (size[0] + BORDER, size[1] + BORDER), 'white')
        canvas.paste(Image.fromarray(arr), (BORDER, BORDER))
        draw = ImageDraw.Draw(canvas)
        for value in np.linspace(x0, x1, 5):
            x = min(canvas.width - 1, BORDER + round((value - x0) * scale))
            draw.line((x, BORDER - 4, x, BORDER), fill='black')
            label = str(round(value))
            width = draw.textbbox((0, 0), label, font=font)[2]
            draw.text((max(0, min(x - width // 2, canvas.width - width)), 5), label, fill='black', font=font)
        for value in np.linspace(y0, y1, 5):
            y = min(canvas.height - 1, BORDER + round((value - y0) * scale))
            draw.line((BORDER - 4, y, BORDER, y), fill='black')
            draw.text((0, max(BORDER, min(y - 6, canvas.height - 14))), str(round(value)), fill='black', font=font)
        occupied, marked = [], []
        for cid, x, y in candidates:
            label = str(cid)
            tw = draw.textbbox((0, 0), label, font=font)[2] + 4
            for dx, dy in ((4, -18), (4, 4), (-tw - 4, -18), (-tw - 4, 4)):
                rect = (x + dx, y + dy, x + dx + tw, y + dy + 16)
                if rect[0] < BORDER or rect[1] < BORDER or rect[2] > canvas.width or rect[3] > canvas.height:
                    continue
                if any(rect[0] < b[2] and b[0] < rect[2] and rect[1] < b[3] and b[1] < rect[3] for b in occupied):
                    continue
                draw.rectangle(rect, fill='white')
                draw.text((rect[0] + 2, rect[1]), label, fill='red', font=font)
                occupied.append(rect)
                marked.append(cid)
                break
        labelled = marked
        path = Path(directory) / f'view_{number}_t{t}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=92)
        paths.append(str(path))
    info = dict(view_id=f'view#{number}', region=box, overlay=overlay_handle,
                display_size=list(size), image_size=[size[0] + BORDER, size[1] + BORDER],
                scale=scale, padding=[BORDER, BORDER], labelled_ids=labelled,
                unlabelled_count=len(candidates) - len(labelled))
    if info['unlabelled_count']:
        info['message'] = f'{info["unlabelled_count"]} visible components left unlabelled to avoid overlap; zoom in or pass ids.'
    return info, paths
