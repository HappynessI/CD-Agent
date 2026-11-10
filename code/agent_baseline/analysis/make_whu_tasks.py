"""WHU-CD building change: sample report and development tiles, write truth and manifests.

Same procedure as SECOND (make_second_tasks): official training tiles only (list/test.txt is never
read for sampling), keep tiles whose building labels differ between dates, draw 128 report tiles with
seed 1016, then 60 development tiles from the remaining changed tiles with the same seed. Labels:
slabel1 = 2012, slabel2 = 2016, 0 background, 1 building. Truth: new = building only in 2016,
demolished = building only in 2012, c0 = either. Zero-shot transfer: instructions, scope card,
thresholds and tools are those of SECOND C0/C1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import save_mask_png, write_json
from agent_baseline.tasks import INSTRUCTIONS, C1_INSTRUCTIONS
from agent_baseline.analysis.second_labels import SCOPE
from agent_baseline.analysis.make_second_tasks import sample_changed

DATASET = Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/WHU-CD')
RUN = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
SUBTASKS = [('T1.1xC0', 'any'), ('T1.1xC1', 'appeared'), ('T1.1xC1', 'disappeared')]
GT = {'any': 'c0', 'appeared': 'new', 'disappeared': 'demolished'}


def load_labels(dataset, name):
    out = []
    for part in ('slabel1', 'slabel2'):
        with Image.open(Path(dataset) / part / name) as image:
            out.append(np.asarray(image))
    return out


def whu_truth(t1, t2):
    t1, t2 = np.asarray(t1), np.asarray(t2)
    if t1.ndim != 2 or t1.shape != t2.shape:
        raise ValueError('Labels must be matching 2D arrays')
    if not (np.isin(t1, (0, 1)).all() and np.isin(t2, (0, 1)).all()):
        raise ValueError('WHU labels must be 0 background / 1 building')
    new, demolished = (t2 == 1) & (t1 == 0), (t1 == 1) & (t2 == 0)
    return dict(new=new, demolished=demolished, c0=new | demolished)


def train_names(dataset):
    test = set(Path(dataset, 'list/test.txt').read_text().split())
    test |= {Path(t).name for t in test}
    return sorted(n.name for n in Path(dataset, 'img1').glob('*.png') if n.name not in test and not n.name.startswith('test_'))


def task_row(name, dataset, run, task, direction):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    return dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
        instruction=C1_INSTRUCTIONS[task][direction] if task in C1_INSTRUCTIONS else INSTRUCTIONS[task],
        task_type=task, constraint=None, target='building', change_direction=direction, scope=SCOPE,
        init_mask=None, gt_mask=str(run / 'whu/gt' / GT[direction] / name), meta={})


def generate(dataset=DATASET, run=RUN, seed=1016):
    dataset, run = Path(dataset), Path(run)
    names = train_names(dataset)
    truth = lambda n: whu_truth(*load_labels(dataset, n))
    report = sample_changed(names, truth, 128, seed)
    dev = sample_changed([n for n in names if n not in set(report)], truth, 60, seed)
    out = dict(dataset=str(dataset), train_tiles=len(names), seed=seed, splits={})
    for split, chosen, suffix in (('val', report, ''), ('dev', dev, '_dev')):
        listing = run / 'whu' / ('names_val128.txt' if split == 'val' else 'names_dev60.txt')
        listing.parent.mkdir(parents=True, exist_ok=True)
        if listing.exists() and listing.read_text().split() != chosen:
            raise ValueError(f'Existing frozen list differs: {listing}')
        listing.write_text('\n'.join(chosen) + '\n')
        agree = []
        for name in chosen:
            t = truth(name)
            for kind, m in t.items():
                save_mask_png(run / 'whu/gt' / kind / name, m)
            with Image.open(dataset / 'label_unch0chi1' / name) as image:
                official = np.asarray(image) > 0
            agree.append(float((official == t['c0']).mean()))
        (run / 'whu/tasks').mkdir(parents=True, exist_ok=True)
        for task, direction in SUBTASKS:
            with (run / 'whu/tasks' / f'{task}_{direction}{suffix}.jsonl').open('w') as f:
                for name in chosen:
                    f.write(json.dumps(task_row(name, dataset, run, task, direction)) + '\n')
        out['splits'][split] = dict(images=len(chosen), names=chosen,
            label_agreement_with_official_change=dict(min=min(agree), mean=float(np.mean(agree))))
    write_json(run / 'whu/preparation.json', out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    a = p.parse_args()
    r = generate(a.dataset_root, a.run_root)
    print(json.dumps({k: (v if k != 'splits' else {s: {x: y for x, y in d.items() if x != 'names'} for s, d in v.items()}) for k, v in r.items()}))


if __name__ == '__main__':
    main()
