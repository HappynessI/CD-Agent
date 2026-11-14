"""SECOND: classes 1=low vegetation, 5=building; 0 is unlabelled.

Keep pixels annotated at both dates, with differing classes and either date in
{1,5}. Frozen inventories are reused without sampling or filtering empty truth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import save_mask_png, write_json
from agent_baseline.tasks import INSTRUCTIONS
from .second_labels import DATASET, check_encoding, load_labels, building_truth
from .probe_t13_common import RUN, inventories

TARGETS = ['building', 'low vegetation']
SOURCE_CLASSES = ['low vegetation', 'non-vegetated ground surface']


def t12_truth(a, b):
    building_truth(a, b)  # Validate grayscale IDs and shared annotation support.
    return (a != 0) & (b != 0) & (a != b) & (np.isin(a, [1, 5]) | np.isin(b, [1, 5]))


def task_row(name, dataset, run, kind):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    task = 'T1.2xC0' if kind == 't12' else 'T1.1xC1t'
    row = dict(id=name, image_t1=str(dataset/'img1'/name), image_t2=str(dataset/'img2'/name),
        task_type=task, instruction=INSTRUCTIONS[task], constraint=None, target='building',
        change_direction='any' if kind == 't12' else 'appeared',
        scope=None if kind == 't12' else 'building_rebuild_excluded', init_mask=None,
        gt_mask=str(run/kind/'gt'/name if kind == 't12' else run/kind/'gt/new'/name), meta={})
    if kind == 't12':
        row['targets'] = TARGETS.copy()
    else:
        row['source_classes'] = SOURCE_CLASSES.copy()
    return row


def generate(dataset, run, kind='t12'):
    from .make_c1t_tasks import conversion_truth
    dataset, run = Path(dataset), Path(run)
    encoding = check_encoding(dataset)
    lists = inventories(run)
    report = dict(encoding=encoding, targets=TARGETS if kind == 't12' else SOURCE_CLASSES,
                  zero='unlabelled unchanged; never a land-cover class', splits={})
    for phase in ('dev', 'val'):
        names = sorted(lists[phase])
        rows, nonempty = [], 0
        for name in names:
            labels = load_labels(dataset, name)
            truths = {'any': t12_truth(*labels)} if kind == 't12' else conversion_truth(*labels)
            nonempty += bool(truths['any' if kind == 't12' else 'new'].any())
            for direction, truth in truths.items():
                path = run/kind/'gt'/name if kind == 't12' else run/kind/'gt'/direction/name
                save_mask_png(path, truth)
            rows.append(task_row(name, dataset, run, kind))
        stem = 'T1.2xC0_any' if kind == 't12' else 'T1.1xC1t_appeared'
        path = run/kind/'tasks'/(stem + ('_dev' if phase == 'dev' else '') + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        report['splits'][phase] = dict(names=names, images=len(names), nonempty_truth=nonempty,
                                       manifest=str(path))
    write_json(run/kind/'preparation.json', report)
    return report


def main(kind='t12'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    a = p.parse_args()
    print(json.dumps(generate(a.dataset_root, a.run_root, kind)))


if __name__ == '__main__':
    main()
