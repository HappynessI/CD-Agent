"""SECOND external-example tasks. Offline CPU preparation, no models or API.

Examples use only rest.txt. T3.2: annotated T2 class component -> all incoming
changes to that class. T3.3/4: directed transition component -> all pixels of
that transition. Example components: 8-connected, 16..4096 native pixels.
Truth keeps every matching pixel, including smaller/larger components.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from render import crop_box
from common import save_mask_png, write_json
from agent_baseline.examples_t3x import INSTRUCTIONS
from .make_t3_tasks import SEED
from .probe_t13_common import RUN, CLASSES, binary_truth, inventories
from .second_labels import DATASET, check_encoding, load_labels

MIN_AREA, MAX_AREA = 16, 4096


def split_names(run):
    groups = inventories(run)
    # rest.txt is the pool the 60 development images were drawn from, so it still lists them; examples use
    # rest minus development minus report images (109 images on SECOND).
    used = set(groups['dev']) | set(groups['val'])
    rest = [n for n in (run / 'second_dev/rest.txt').read_text().split() if n not in used]
    if not rest or len(set(rest)) != len(rest):
        raise ValueError('Example inventory must be unique and disjoint from development and report images')
    if any(Path(n).name != n for n in rest):
        raise ValueError('Example inventory must contain image filenames')
    return dict(groups, rest=sorted(rest))


def components(mask, min_area=MIN_AREA, max_area=MAX_AREA):
    labels, n = ndi.label(mask, np.ones((3, 3), bool))
    areas = np.bincount(labels.ravel())
    for i, box in enumerate(ndi.find_objects(labels), 1):
        if min_area <= areas[i] <= max_area:
            yy, xx = box
            yield dict(component=i, area=int(areas[i]), bbox=[xx.start, yy.start, xx.stop, yy.stop])


def catalogue(names, loader, min_area=MIN_AREA, max_area=MAX_AREA):
    """Store component descriptors, never full rest masks or RGB arrays."""
    if not 1 <= min_area <= max_area:
        raise ValueError('Require 1 <= min_area <= max_area')
    targets, transitions, by_image = defaultdict(list), defaultdict(list), defaultdict(list)
    for name in sorted(names):
        a, b = loader(name)
        changed = binary_truth(a, b)
        for target in CLASSES:
            for item in components(b == target, min_area, max_area):
                targets[target].append(dict(item, name=name, target=target))
        for source, target in sorted({tuple(map(int, p)) for p in np.unique(np.stack((a[changed], b[changed]), axis=1), axis=0)}):
            for item in components(changed & (a == source) & (b == target), min_area, max_area):
                row = dict(item, name=name, source=source, target=target)
                transitions[(source, target)].append(row)
                by_image[name].append(row)
    return targets, transitions, by_image


def select_examples(pool, name, task_type, source, target, seed=SEED):
    targets, transitions, by_image = pool
    local_seed = seed + sum((i+1)*ord(c) for i, c in enumerate(name + task_type)) + source*101 + target*1009
    rng = np.random.default_rng(local_seed)
    if task_type == 'T3.2':
        candidates = targets[target]
    else:
        candidates = transitions[(source, target)]
        if task_type == 'T3.4':
            candidates = [r for r in candidates if any((s['source'], s['target']) != (source, target) for s in by_image[r['name']])]
    if not candidates:
        raise ValueError(f'No eligible rest example for {task_type} {source}->{target}')
    positive = dict(candidates[int(rng.integers(len(candidates)))], role='positive')
    result = [positive]
    if task_type == 'T3.4':
        negatives = [r for r in by_image[positive['name']] if (r['source'], r['target']) != (source, target)]
        result.append(dict(negatives[int(rng.integers(len(negatives)))], role='negative'))
    return result, local_seed


def tasks_for_pair(a, b, name, pool, seed=SEED, skipped=None):
    changed = binary_truth(a, b)
    rows = []
    for task_type in INSTRUCTIONS:
        pairs = ([(0, int(t)) for t in np.unique(b[changed])] if task_type == 'T3.2' else
                 [tuple(map(int, p)) for p in np.unique(np.stack((a[changed], b[changed]), axis=1), axis=0)])
        for source, target in pairs:
            truth = changed & (b == target)
            if task_type != 'T3.2':
                truth &= a == source
            try:
                examples, local_seed = select_examples(pool, name, task_type, source, target, seed)
            except ValueError:
                # Rare transitions may have no eligible example in the example pool; skip and count them.
                if skipped is None:
                    raise
                skipped.append(f'{task_type} {source}->{target}')
                continue
            transition = f'*->{target}' if task_type == 'T3.2' else f'{source}->{target}'
            key = f'{Path(name).stem}_{task_type.replace(".", "")}_{source}to{target}'
            rows.append((dict(id=key, task_type=task_type, target='any change', change_direction='any',
                instruction=INSTRUCTIONS[task_type], constraint=None, init_mask=None,
                meta=dict(transition=transition, seed=local_seed, example_provenance=examples)), truth))
    return rows


def write_example(descriptor, task_type, dataset, destination):
    a, b = load_labels(dataset, descriptor['name'])
    support = b == descriptor['target']
    if task_type != 'T3.2':
        support &= (a == descriptor['source']) & binary_truth(a, b)
    labels, _ = ndi.label(support, np.ones((3, 3), bool))
    mask = labels == descriptor['component']
    box = crop_box(descriptor['bbox'], mask.shape)  # ask close-up: 4x bbox, >=64, clipped
    x0, y0, x1, y1 = box
    destination.mkdir(parents=True, exist_ok=True)
    spec = dict(role=descriptor['role'])
    dates = ('t2',) if task_type == 'T3.2' else ('t1', 't2')
    for t in dates:
        with Image.open(dataset / ('img1' if t == 't1' else 'img2') / descriptor['name']) as im:
            crop = im.convert('RGB').crop(tuple(box))
            path = destination / f'{t}.png'
            crop.save(path)
            spec['image_' + t] = str(path.resolve())
    path = destination / 'mask.png'
    save_mask_png(path, mask[y0:y1, x0:x1])
    spec['mask'] = str(path.resolve())
    return spec, box


def generate(args):
    check_encoding(args.dataset_root)
    groups = split_names(args.run_root)
    loader = lambda name: load_labels(args.dataset_root, name)
    pool = catalogue(groups['rest'], loader, args.min_area, args.max_area)
    phases = ('dev', 'val') if args.phase == 'both' else (args.phase,)
    reports = {}
    for phase in phases:
        output = args.output_root / phase
        output.mkdir(parents=True, exist_ok=True)
        rows, counts, skipped = [], {k: Counter() for k in INSTRUCTIONS}, []
        for name in sorted(groups[phase]):
            for row, truth in tasks_for_pair(*loader(name), name, pool, args.seed, skipped):
                row.update(image_t1=str((args.dataset_root / 'img1' / name).resolve()),
                           image_t2=str((args.dataset_root / 'img2' / name).resolve()))
                row['examples'] = []
                for k, descriptor in enumerate(row['meta']['example_provenance']):
                    spec, box = write_example(descriptor, row['task_type'], args.dataset_root,
                                              output / 'examples' / row['id'] / str(k))
                    row['examples'].append(spec)
                    descriptor['crop_box'] = box
                gt = output / 'truth' / (row['id'] + '.png')
                save_mask_png(gt, truth)
                row['gt_mask'] = str(gt.resolve())
                rows.append(row)
                counts[row['task_type']][row['meta']['transition']] += 1
        manifest = output / 'tasks.jsonl'
        manifest.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
        report = dict(phase=phase, names=sorted(groups[phase]), example_names=groups['rest'], seed=args.seed,
            min_area=args.min_area, max_area=args.max_area, total=len(rows), counts=counts,
            totals_by_type={k: sum(c.values()) for k, c in counts.items()},
            skipped_no_example=dict(Counter(skipped)),
            rule='8-connected rest examples only; T3.2 incoming class changes; T3.3/4 exact directed transition; all truth pixels retained; no evaluation pixels excluded')
        write_json(output / 'inventory.json', report)
        reports[phase] = report
        print(json.dumps(dict(phase=phase, total=len(rows), counts=counts), ensure_ascii=False))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--phase', choices=['dev', 'val', 'both'], default='both')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--min-area', type=int, default=MIN_AREA)
    p.add_argument('--max-area', type=int, default=MAX_AREA)
    generate(p.parse_args())


if __name__ == '__main__':
    main()
