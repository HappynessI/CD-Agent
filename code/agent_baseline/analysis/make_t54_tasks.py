"""Prepare WHU old maps, map-update truth and manifests from frozen training lists.

No resampling and no empty-truth filtering. Run full preparation on a compute node.
"""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from PIL import Image
from common import save_mask_png, write_json
from agent_baseline.fixed_routes import RUN
from agent_baseline.map_input import INSTRUCTION, render_map

DATASET = Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/WHU-CD')
ENCODING = {'background': 0, 'building': 1}


def building_truth(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.ndim != 2 or old.shape != new.shape:
        raise ValueError('WHU labels must be matching 2D arrays')
    if not (np.isin(old, [0, 1]).all() and np.isin(new, [0, 1]).all()):
        raise ValueError('WHU semantic labels must use background=0 and building=1')
    old, new = old == 1, new == 1
    return dict(any=old ^ new, new=new & ~old, demolished=old & ~new)


def names_from(path):
    names = Path(path).read_text().split()
    if not names or len(names) != len(set(names)):
        raise ValueError('Image list must be nonempty and unique')
    if any(Path(n).name != n or not n.startswith('train_') or not n.endswith('.png') for n in names):
        raise ValueError('Use training-image filenames train_*.png only')
    return sorted(names)


def task_row(name, dataset, run):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    return dict(id=name, task_type='T5.4', input_type='map_image',
                image_t1=str(run / 't54/maps' / name), image_t2=str(dataset / 'img2' / name),
                instruction=INSTRUCTION, target='building', change_direction='any', scope=None,
                constraint=None, init_mask=None, gt_mask=str(run / 't54/gt/any' / name), meta={})


def generate(dataset, run, names_val=None, names_dev=None):
    dataset, run = Path(dataset), Path(run)
    encoding = json.loads((dataset / 'cls_gray.json').read_text())
    if encoding != ENCODING:
        raise ValueError(f'Unexpected WHU encoding: {encoding}')
    sources = dict(val=Path(names_val) if names_val else run / 'whu/names_val128.txt',
                   dev=Path(names_dev) if names_dev else run / 'whu/names_dev60.txt')
    inventories = {split: names_from(path) for split, path in sources.items()}
    if set(inventories['val']) & set(inventories['dev']):
        raise ValueError('Development and report lists overlap')
    report = dict(encoding=encoding, rendering='white building footprints (255), black background (0), RGB',
                  truth='(slabel1 == 1) XOR (slabel2 == 1); no image filtering', splits={})
    for split, names in inventories.items():
        rows, nonempty, mismatches, pixels = [], 0, 0, 0
        values = {part: set() for part in ('slabel1', 'slabel2', 'label_unch0chi1')}
        for name in names:
            arrays = []
            for part in values:
                with Image.open(dataset / part / name) as image:
                    data = np.asarray(image)
                values[part].update(int(x) for x in np.unique(data))
                arrays.append(data)
            old, new, change = arrays
            truths = building_truth(old, new)
            if change.shape != old.shape or not np.isin(change, [0, 1]).all():
                raise ValueError('WHU change labels must use 0/1 and match semantic labels')
            with Image.open(dataset / 'img2' / name) as image:
                if image.size != (old.shape[1], old.shape[0]):
                    raise ValueError('Map and new imagery must have the same pixel dimensions')
            mismatches += int((truths['any'] != (change == 1)).sum())
            pixels += old.size
            nonempty += bool(truths['any'].any())
            map_path = run / 't54/maps' / name
            map_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(render_map(old == 1)).save(map_path)
            for kind, mask in truths.items():
                save_mask_png(run / 't54/gt' / kind / name, mask)
            rows.append(task_row(name, dataset, run))
        task_path = run / 't54/tasks' / f'T5.4_{split}.jsonl'
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        report['splits'][split] = dict(images=len(names), nonempty_truth=nonempty, names=names,
            source=str(sources[split].resolve()), manifest=str(task_path.resolve()),
            values={part: sorted(v) for part, v in values.items()},
            xor_agreement=1 - mismatches / pixels, xor_mismatched_pixels=mismatches, pixels=pixels)
    write_json(run / 't54/preparation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--names-val', type=Path)
    p.add_argument('--names-dev', type=Path)
    a = p.parse_args()
    print(json.dumps(generate(a.dataset_root, a.run_root, a.names_val, a.names_dev)))


if __name__ == '__main__':
    main()
