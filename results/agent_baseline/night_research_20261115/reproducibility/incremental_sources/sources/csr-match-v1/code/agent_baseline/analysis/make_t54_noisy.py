"""Opt-in noisy old maps; task_type stays T5.4 and meta.variant identifies T5.4n.

Translate with zero padding (no wrap), then drop ceil(10% of surviving 8-connected
buildings). Integer displacement has Euclidean length in [2,4] pixels.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from common import components
from agent_baseline.fixed_routes import RUN
from agent_baseline.map_input import map_mask, render_map, check_task
from agent_baseline.tasks import read_tasks
from .make_t54_tasks import building_truth
from .io_retry import read_image, write_json, write_text, retry_io
from .io_retry import save_image

OFFSETS = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if 4 <= dx*dx + dy*dy <= 16]


def translate(old, dx, dy):
    h, w = old.shape
    result = np.zeros_like(old, dtype=bool)
    x0, x1 = max(0, -dx), min(w, w-dx)
    y0, y1 = max(0, -dy), min(h, h-dy)
    if x1 > x0 and y1 > y0:
        result[y0+dy:y1+dy, x0+dx:x1+dx] = old[y0:y1, x0:x1]
    return result


def noisy_map(old, name, seed=1016):
    rng = np.random.default_rng(np.random.SeedSequence([seed, *name.encode('utf-8')]))
    dx, dy = OFFSETS[int(rng.integers(len(OFFSETS)))]
    shifted = translate(np.asarray(old, bool), dx, dy)
    labels, comps = components(shifted)
    n = math.ceil(.1 * len(comps))
    ids = [int(x) for x in rng.choice([c['id'] for c in comps], size=n, replace=False)] if n else []
    result = shifted & ~np.isin(labels, ids)
    return result, dict(dx=dx, dy=dy, displacement=float(np.hypot(dx, dy)),
                       buildings_after_shift=len(comps), buildings_removed=n, removed_ids=sorted(ids),
                       removed_pixels=int((shifted & ~result).sum()),
                       clipped_pixels=int(np.asarray(old, bool).sum()-shifted.sum()))


def generate(manifests, output_root, seed=1016, old_map_dir=None):
    output = Path(output_root).resolve()
    inventories = {split: retry_io(lambda p=p: read_tasks(p)) for split, p in manifests.items()}
    if set(r['id'] for r in inventories['dev']) & set(r['id'] for r in inventories['val']):
        raise ValueError('Development and report images overlap')
    report = dict(seed=seed, task_name='T5.4n', task_type='T5.4',
                  truth='(slabel2 == 1) XOR noisy_old_map; WHU 0 is labelled background',
                  noise='uniform integer dx,dy with length 2..4; zero padding; then ceil(10% of 8-connected components)', splits={})
    for split, rows in inventories.items():
        generated, stats, distribution = [], [], Counter()
        for row in rows:
            check_task(row)
            name = Path(row['image_t1']).name
            old = map_mask(read_image(Path(old_map_dir) / name if old_map_dir else row['image_t1']))
            new = read_image(Path(row['image_t2']).parent.parent / 'slabel2' / name)
            noisy, stat = noisy_map(old, name, seed)
            truths = building_truth(noisy.astype('uint8'), new)
            clean = building_truth(old.astype('uint8'), new)['any']
            stat.update(name=name, clean_truth_pixels=int(clean.sum()), noisy_truth_pixels=int(truths['any'].sum()))
            stat['truth_pixel_delta'] = stat['noisy_truth_pixels']-stat['clean_truth_pixels']
            stat['truth_relative_delta'] = stat['truth_pixel_delta']/stat['clean_truth_pixels'] if stat['clean_truth_pixels'] else None
            stats.append(stat)
            distribution[f"{stat['dx']},{stat['dy']}"] += 1
            map_path = output / 'maps' / name
            save_image(map_path, render_map(noisy))
            for kind, truth in truths.items():
                save_image(output / 'gt' / kind / name, truth.astype('uint8')*255)
            generated.append(dict(row, image_t1=str(map_path), gt_mask=str(output / 'gt/any' / name),
                                  meta=dict(row.get('meta', {}), variant='T5.4n', noise=stat)))
        path = output / 'tasks' / f'T5.4n_{split}.jsonl'
        write_text(path, ''.join(json.dumps(r) + '\n' for r in generated))
        clean_area = sum(s['clean_truth_pixels'] for s in stats)
        noisy_area = sum(s['noisy_truth_pixels'] for s in stats)
        report['splits'][split] = dict(images=len(rows), manifest=str(path),
            removed_buildings=sum(s['buildings_removed'] for s in stats),
            buildings_after_shift=sum(s['buildings_after_shift'] for s in stats),
            displacement_distribution=dict(distribution), clean_truth_pixels=clean_area,
            noisy_truth_pixels=noisy_area, truth_pixel_delta=noisy_area-clean_area,
            truth_relative_delta=(noisy_area-clean_area)/clean_area if clean_area else None, per_image=stats)
    write_json(output / 'preparation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--tasks-dev', type=Path)
    p.add_argument('--tasks-val', type=Path)
    p.add_argument('--old-map-dir', type=Path, help='Optional existing WHU maps; default follows each source manifest')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    manifests = dict(dev=a.tasks_dev or a.run_root / 't54/tasks/T5.4_dev.jsonl',
                     val=a.tasks_val or a.run_root / 't54/tasks/T5.4_val.jsonl')
    for split, count in (('dev', 60), ('val', 128)):
        if len(retry_io(lambda: read_tasks(manifests[split]))) != count:
            p.error(f'{split} requires {count} frozen images')
    print(json.dumps(generate(manifests, a.output_root, a.seed, a.old_map_dir)))


if __name__ == '__main__':
    main()
