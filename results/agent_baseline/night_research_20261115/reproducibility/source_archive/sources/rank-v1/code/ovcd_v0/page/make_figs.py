#!/usr/bin/env python3
"""Render OVCD replay panels, portable call crops and summary charts."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
import warnings

# Keep plotting caches beside this script, including during the fake-data test.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from review_data import DATA, REPO, RUN, STATES, call_records, image_f1, load_image, read_json, write_json

COLORS = ('#3676bd', '#29a080', '#dfaa39', '#986bb5')


def configure_font(font=None):
    """Use an installed CJK font; return its name, or None for English labels."""
    candidates = [str(font)] if font else sorted(font_manager.findSystemFonts())
    for path in candidates:
        if font or any(s in Path(path).name.lower() for s in
                       ('notosanscjk', 'notoserifcjk', 'wqy', 'simhei', 'sourcehansans')):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return name
    warnings.warn('No CJK font found; chart labels use English. Supply --font for Chinese labels.')
    return None


def label(data, key, default, cjk):
    value = str(data.get(key, default))
    if not cjk and any('\u3400' <= c <= '\u9fff' for c in value):
        return str(data.get(key + '_en', default))
    return value


def save_chart(fig, ax, out):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    return out


def plot_bars(data, out, font=None):
    """data: states=[{label, f1, iou, precision, recall}], values in [0, 1]."""
    cjk = configure_font(font)
    rows = data['states']
    if len(rows) != 4:
        raise ValueError('bars requires four states: M0, A, B, C')
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = np.arange(4)
    for i, row in enumerate(rows):
        values = np.array([row[k] for k in ('f1', 'iou', 'precision', 'recall')], float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Metric values must be finite fractions in [0, 1]')
        ax.bar(x + (i - 1.5) * .19, values * 100, .19, color=COLORS[i],
               label=label(row, 'label', STATES[i], cjk))
    ax.set_xticks(x, ['$F_1^c$', '$IoU^c$', 'P', 'R'])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Score (%)')
    ax.set_title(label(data, 'title', 'Four-state metrics', cjk))
    ax.legend(ncol=4, frameon=False, loc='upper center')
    return save_chart(fig, ax, out)


def plot_budget(data, out, font=None):
    """data: series=[{label, k:[10,20,50], f1:[fractions]}]."""
    cjk = configure_font(font)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, row in enumerate(data['series']):
        k, f1 = np.array(row['k'], float), np.array(row['f1'], float)
        if not len(k) or len(k) != len(f1) or not np.isfinite(k).all() or not np.isfinite(f1).all():
            raise ValueError('Each budget series needs equally sized, finite k and f1 arrays')
        if np.any(k < 0) or np.any((f1 < 0) | (f1 > 1)):
            raise ValueError('k must be nonnegative and f1 must be in [0, 1]')
        order = np.argsort(k)
        ax.plot(k[order], 100 * f1[order], 'o-', color=COLORS[i % 4],
                label=label(row, 'label', f'Series {i + 1}', cjk))
    ax.set_xlabel('MLLM budget k')
    ax.set_ylabel('$F_1^c$ (%)')
    ax.set_title(label(data, 'title', 'Budget curve', cjk))
    ax.legend(frameon=False)
    return save_chart(fig, ax, out)


def plot_composition(data, out, font=None):
    """data: labels=[states], segments=[{label, values:[counts]}], ylabel optional."""
    cjk = configure_font(font)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = data['labels']
    x, base = np.arange(len(labels)), np.zeros(len(labels))
    for i, row in enumerate(data['segments']):
        values = np.array(row['values'], float)
        if len(values) != len(labels) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Segment values must match labels and be finite nonnegative counts')
        ax.bar(x, values, bottom=base, width=.6, color=COLORS[i % 4],
               label=label(row, 'label', f'Component {i + 1}', cjk))
        base += values
    ax.set_xticks(x, [label({'label': s}, 'label', f'State {i + 1}', cjk) for i, s in enumerate(labels)])
    ax.set_ylabel(label(data, 'ylabel', 'Pixels', cjk))
    ax.set_title(label(data, 'title', 'False positives / false negatives', cjk))
    ax.margins(y=.2)
    ax.legend(frameon=False, ncol=2)
    return save_chart(fig, ax, out)


def change_bbox(masks, padding=64):
    """Union of all adjacent-state changes, xyxy with an exclusive upper edge."""
    changed = np.logical_or.reduce([a != b for a, b in zip(masks, masks[1:])])
    y, x = np.nonzero(changed)
    if not len(x):
        return None
    h, w = changed.shape
    return [max(0, int(x.min()) - padding), max(0, int(y.min()) - padding),
            min(w, int(x.max()) + 1 + padding), min(h, int(y.max()) + 1 + padding)]


def mask_panel(mask, previous, gt):
    canvas = np.full((*mask.shape, 3), 24, dtype=np.uint8)
    canvas[mask] = (255, 255, 255)
    if previous is not None:
        canvas[mask & ~previous] = (40, 200, 95)
        canvas[previous & ~mask] = (235, 65, 65)
    # Put the pale GT contour immediately outside GT so changed pixels stay visible.
    edge = ndimage.binary_dilation(gt) & ~gt
    if previous is not None:
        edge &= mask == previous
    canvas[edge] = (170, 195, 235)
    return canvas


def render_replay(rgb, gt, masks, name, out, font=None):
    """Full view plus a common zoom when padded changes occupy <=60% of the image."""
    cjk = configure_font(font)
    if any(m.shape != gt.shape for m in masks) or any(im.shape[:2] != gt.shape for im in rgb):
        raise ValueError(f'{name}: T1, T2, GT and masks must share dimensions')
    bbox = change_bbox(masks)
    zoom = bbox is not None and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) <= .6 * gt.size
    nrows = 3 if zoom else 2
    fig, axes = plt.subplots(nrows, 4, figsize=(16, nrows * 4.4), dpi=100)
    for ax in axes.flat:
        ax.axis('off')
    for ax, im, title in zip(axes[0], [*rgb, gt], ['T1', 'T2', 'Ground truth']):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=13)
    legend = ('白：当前掩码\n绿：相对上一状态新增\n红：相对上一状态删除\n淡蓝：真值边界\nM0 没有前一状态\nF1 为整张图的分数'
              if cjk else 'White: current mask\nGreen: added vs previous\nRed: removed vs previous\nPale blue: GT boundary\nM0: no previous state\nF1: whole-image score')
    axes[0, 3].text(.04, .9, legend, va='top', fontsize=13, linespacing=1.9)
    f1 = {s: image_f1(m, gt) for s, m in zip(STATES, masks)}
    for i, (state, mask) in enumerate(zip(STATES, masks)):
        panel = mask_panel(mask, masks[i - 1] if i else None, gt)
        axes[1, i].imshow(panel, interpolation='nearest')
        axes[1, i].set_title(f'{state}  $F_1^c$ = {100 * f1[state]:.2f}%', fontsize=13)
        if zoom:
            x0, y0, x1, y1 = bbox
            axes[2, i].imshow(panel[y0:y1, x0:x1], interpolation='nearest')
            axes[2, i].set_title(f'{state} zoom  $F_1^c$ = {100 * f1[state]:.2f}%\n'
                                 f'[{x0}, {y0}, {x1}, {y1}]', fontsize=11)
    fig.suptitle(name, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out, dpi=100, facecolor='white')
    plt.close(fig)
    return f1, bbox if zoom else None


def save_thumbnail(source, target, max_bytes=30_000):
    """Preserve the sent crop; resize/compress until each JPEG is <=30 KB."""
    with Image.open(source) as source_image:
        im = source_image.convert('RGB')
    im.thumbnail((320, 320), Image.Resampling.LANCZOS)
    while True:
        for quality in (85, 75, 65, 50, 35):
            buffer = io.BytesIO()
            im.save(buffer, format='JPEG', quality=quality, optimize=True)
            if buffer.tell() <= max_bytes:
                Path(target).write_bytes(buffer.getvalue())
                return
        im = im.resize((max(1, im.width * 3 // 4), max(1, im.height * 3 // 4)), Image.Resampling.LANCZOS)


def make_example(round_id, cfg, split, name, run=RUN, data_root=DATA, figures=None, font=None):
    figures = Path(figures) if figures else REPO / 'figures/ovcd_v0'
    out = figures / f'round_{round_id}'
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    manifest_path = out / f'replay_{stem}.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (old['cfg'], old['split'], old['name']) != (cfg, split, name):
            raise ValueError(f'{manifest_path}: same output name already belongs to another config/split')
    base = Path(run) / 'agent' / cfg / split
    trajectory = read_json(base / 'traj' / f'{stem}.json')
    masks = [load_image(base / 'masks' / s / name, 'L') > 127 for s in STATES]
    source = Path(data_root) / f'{split}_original'
    rgb = [load_image(source / t / name) for t in ('A', 'B')]
    gt = load_image(source / 'label' / name, 'L') > 127
    replay = out / f'replay_{stem}.png'
    f1, zoom = render_replay(rgb, gt, masks, name, replay, font)
    calls = call_records(trajectory['steps'])
    crop_dir = out / f'calls_{stem}'
    crop_dir.mkdir(exist_ok=True)
    for index, call in enumerate(calls, 1):
        if len(call['images']) != 2:
            raise ValueError(f'{name}, step {call["step"]}: expected T1 and T2 crops')
        thumbs = []
        for t, path in enumerate(call.pop('images'), 1):
            path = Path(path)
            if not path.is_absolute():
                path = Path(run) / path
            target = crop_dir / f'{index:03d}_{call["kind"]}_t{t}.jpg'
            save_thumbnail(path, target)
            thumbs.append(target.relative_to(out).as_posix())
        call['thumbs'] = thumbs
    stats = trajectory.get('stats', {})
    if 'calls' in stats and stats['calls'] != len(calls):
        raise ValueError(f'{name}: stats.calls={stats["calls"]}, decoded {len(calls)} model calls')
    operations = [{k: s[k] for k in ('step', 'stage', 'action', 'op', 'f1_before', 'f1_after') if k in s}
                  for s in trajectory['steps'] if s.get('stage') == 'C']
    manifest = dict(name=name, cfg=cfg, split=split, replay=replay.name,
                    f1=f1, zoom_bbox=zoom, calls=calls, operations=operations, stats=stats)
    write_json(manifest_path, manifest)
    return manifest_path


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('bars', 'budget', 'composition'):
        parser = argparse.ArgumentParser(description='Summary chart from JSON; see README_batch57.md')
        parser.add_argument('kind', choices=('bars', 'budget', 'composition'))
        parser.add_argument('--json', type=Path, required=True)
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--font', type=Path)
        args = parser.parse_args()
        fn = {'bars': plot_bars, 'budget': plot_budget, 'composition': plot_composition}[args.kind]
        print(fn(read_json(args.json), args.out, args.font))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--examples', help='Comma-separated image names; default: examples selected in rounds JSON')
    parser.add_argument('--rounds-json', type=Path, default=REPO / 'results/ovcd_v0_rounds.json')
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--figures', type=Path, default=REPO / 'figures/ovcd_v0')
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    if args.examples:
        names = [n.strip() for n in args.examples.split(',') if n.strip()]
    else:
        rounds = read_json(args.rounds_json)['rounds']
        row = next(r for r in rounds if str(r['id']) == str(args.round))
        names = [e['name'] for e in row['examples'] if e['cfg'] == args.cfg and e['split'] == args.split]
    if not names:
        parser.error('No matching examples; provide --examples or add examples to rounds JSON')
    for name in dict.fromkeys(names):
        print(make_example(args.round, args.cfg, args.split, name, args.run, args.data_root, args.figures, args.font))


if __name__ == '__main__':
    main()
