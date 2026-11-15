"""Generate official WHU C0/C1 references using frozen report/development lists.
C1 uncertainty is ignored; never sample again or generate semantic XOR truth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import save_mask_png, write_json
from agent_baseline.tasks import INSTRUCTIONS, C1_INSTRUCTIONS
from agent_baseline.analysis.second_labels import SCOPE
from agent_baseline.whu_gt import truth as whu_truth, load_truth, update_task, GT_DIR, VERSION

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


def train_names(dataset):
    test = set(Path(dataset, 'list/test.txt').read_text().split())
    test |= {Path(t).name for t in test}
    return sorted(n.name for n in Path(dataset, 'img1').glob('*.png') if n.name not in test and not n.name.startswith('test_'))


def task_row(name, dataset, run, task, direction):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    return update_task(dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
        instruction=C1_INSTRUCTIONS[task][direction] if task in C1_INSTRUCTIONS else INSTRUCTIONS[task],
        task_type=task, constraint=None, target='building', change_direction=direction, scope=SCOPE,
        init_mask=None, gt_mask=None, meta={}), run)


def generate(dataset=DATASET, run=RUN, seed=1016):
    dataset, run = Path(dataset), Path(run)
    report = (run/'whu/names_val128.txt').read_text().split()
    dev = (run/'whu/names_dev60.txt').read_text().split()
    if len(report)!=len(set(report)) or len(dev)!=len(set(dev)) or set(report)&set(dev):
        raise ValueError('Frozen WHU lists must be unique and disjoint')
    allowed=set(train_names(dataset))
    if not set(report+dev)<=allowed:
        raise ValueError('Frozen WHU list includes a test or unavailable tile')
    out=dict(dataset=str(dataset),version=VERSION,selection='existing frozen lists retained',splits={})
    for split, chosen, suffix in [('val',report,''),('dev',dev,'_dev')]:
        pixels=dict(c0=0,new=0,demolished=0,ignore=0)
        for name in chosen:
            t=load_truth(dataset,name)
            for kind,m in t.items():
                save_mask_png(run/'whu'/GT_DIR/kind/name,m)
                pixels[kind]+=int(m.sum())
        dest=run/'whu/tasks';dest.mkdir(parents=True,exist_ok=True)
        for task,direction in SUBTASKS:
            p=dest/f'{task}_{direction}{suffix}.jsonl'
            p.write_text(''.join(json.dumps(task_row(n,dataset,run,task,direction))+'\n' for n in chosen))
        out['splits'][split]=dict(images=len(chosen),names=chosen,pixels=pixels)
    write_json(run/'whu'/GT_DIR/'definition.json',out)
    write_json(run/'whu/preparation.json',out)
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
