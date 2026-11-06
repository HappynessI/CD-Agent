"""Offline LEVIR feedback simulation. 8-connectivity matches common.components.

Labels are read here, never by the planner. T5.3 can reuse error_regions and
region_mask and replace only the point-to-instruction rendering (not implemented).
"""
import argparse
import json
import os
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from scipy import ndimage as ndi
from common import LEVIR, REPO, components, load_mask_png, save_mask_png, write_json
from agent_baseline.tasks import t52_instruction, read_tasks, levir_cache_name

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))


def farthest_point(region):
    """Euclidean distance to background, including outside the image.

    Equal distances choose the smallest y, then the smallest x (row-major order).
    Padding makes image-edge regions and all-foreground images well defined.
    """
    if not region.any():
        raise ValueError('Cannot click an empty error region')
    distance = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), region.shape)
    return [int(x), int(y)]


def region_mask(item, gt_labels, init_labels, initial, full=False):
    labels = gt_labels if item['type'] == 'missed' else init_labels
    region = labels == item['component_id']
    if item['type'] == 'missed' and not full:
        region &= ~initial
    return region


def error_regions(truth, initial):
    """Strictly less than half is an error; area ranks the actual click region.

    Region IDs retain the source component ID, prefixed by gt/given. A missed
    component's uncovered pixels remain one error even if disconnected.
    """
    truth, initial = np.asarray(truth, bool), np.asarray(initial, bool)
    if truth.shape != initial.shape or truth.ndim != 2:
        raise ValueError('Truth and given mask must have the same 2D shape')
    gl, gc = components(truth)
    il, ic = components(initial)
    errors = []
    for kind, labels, comps, other, prefix in (
            ('missed', gl, gc, initial, 'gt'),
            ('false_positive', il, ic, truth, 'given')):
        for c in comps:
            component = labels == c['id']
            if 2 * int(other[component].sum()) >= c['area']:
                continue
            region = component & ~initial if kind == 'missed' else component
            errors.append(dict(type=kind, region_id=f"{prefix}:{c['id']}",
                component_id=c['id'], component_area=c['area'], area=int(region.sum()),
                point=farthest_point(region)))
    # Deterministic ties: missed before false_positive, then source component ID.
    return sorted(errors, key=lambda e: (-e['area'], e['type'] != 'missed', e['component_id']))


def select_feedback(errors, k):
    if k not in (1, 3):
        raise ValueError('T5.2 supports k=1 or k=3')
    return sorted(errors, key=lambda e: (-e['area'], e['type'] != 'missed', e['component_id']))[:k]


def generate(dataset, init_root, run_root, output, expected_images=64):
    dataset, init_root, run_root = map(Path, (dataset, init_root, run_root))
    paths = sorted((dataset / 'val_original/A').glob('*.png'))
    if len(paths) != expected_images:
        raise ValueError(f'Expected {expected_images} validation images, found {len(paths)}')
    report = dict(schema='t52-v1', connectivity=8, split='val',
        ranking='error area descending, missed first on ties, source component ID ascending',
        point_ties='smallest y then x; padded Euclidean distance to background', images={})
    manifests = {k: [] for k in (1, 3)}
    for path in paths:
        truth = load_mask_png(dataset / 'val_original/label' / path.name)
        init_path = init_root / path.name
        initial = load_mask_png(init_path)
        errors = error_regions(truth, initial)
        gl, _ = components(truth)
        il, _ = components(initial)
        image = {}
        for k in (1, 3):
            selected = select_feedback(errors, k)
            image[f'k{k}'] = selected
            for item in selected:
                dest = run_root / f't52/regions/k{k}' / path.stem / (item['region_id'].replace(':', '_') + '.png')
                save_mask_png(dest, region_mask(item, gl, il, initial))
            manifests[k].append(dict(id=path.name,
                image_t1=str(path.resolve()),
                image_t2=str((dataset / 'val_original/B' / path.name).resolve()),
                instruction=t52_instruction(selected), task_type='T5.2', constraint=None,
                target='building', change_direction='any', init_mask=str(init_path.resolve()),
                gt_mask=str((dataset / 'val_original/label' / path.name).resolve()),
                meta=dict(k=k, feedback=selected)))
        report['images'][path.name] = image
    for k, rows in manifests.items():
        dest = run_root / f't52/tasks/T5.2_k{k}.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(output, report)
    return report


def check_caches(manifest):
    """Read real SAM3 instances and inspect stored SAM embeddings; no inference."""
    from agent_baseline.backends import read_soft
    from sam_tools import EMB
    rows = read_tasks(manifest, 'T5.2')
    if not rows:
        raise ValueError('Empty T5.2 manifest')
    for row in rows:
        name = levir_cache_name(row, 'val')
        for index in (0, 1):
            read_soft('val', name, index, .85, .5)
        path = EMB / 'val' / (Path(name).stem + '.pt')
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(images=len(rows), split='val', sam3='sam3_soft/t512s256; grid512; pixel=.85; score=.5',
                sam_embeddings=str(EMB / 'val'), cache_namespace='split/name; no image re-encoding')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=LEVIR)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--init-root', type=Path)
    p.add_argument('--output', type=Path, default=REPO / 'results/agent_baseline/data/t52_feedback.json')
    p.add_argument('--check-caches', type=Path, metavar='MANIFEST', help='Only check existing caches for this manifest')
    a = p.parse_args()
    if a.check_caches:
        print(json.dumps(check_caches(a.check_caches)))
    else:
        generate(a.dataset_root, a.init_root or a.run_root / 'init/sam3_imc_p85/val', a.run_root, a.output)


if __name__ == '__main__':
    main()
