"""SECOND T1.3 x C0: every land-cover change (both dates annotated, classes differ).

Frozen inventories are reused without sampling or filtering. target='any change'
uses scopes/any_change.json; ask_variant is the verifier question frozen on the
60 development images (analysis/t13_ask_select.py, selection.json).
"""
import argparse
import json
from pathlib import Path
from common import save_mask_png, write_json
from agent_baseline.tasks import INSTRUCTIONS
from .second_labels import DATASET, check_encoding, load_labels
from .probe_t13_common import RUN, inventories, binary_truth


def task_row(name, dataset, run, ask_variant):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    return dict(id=name, image_t1=str(dataset/'img1'/name), image_t2=str(dataset/'img2'/name),
        task_type='T1.3', instruction=INSTRUCTIONS['T1.3'], constraint=None, target='any change',
        change_direction='any', scope=None, init_mask=None, gt_mask=str(run/'t13'/'gt'/name),
        ask_variant=ask_variant, meta={})


def generate(dataset, run, ask_variant):
    dataset, run = Path(dataset), Path(run)
    encoding = check_encoding(dataset)
    lists = inventories(run)
    report = dict(encoding=encoding, ask_variant=ask_variant, splits={})
    for phase in ('dev', 'val'):
        names = sorted(lists[phase])
        rows, nonempty = [], 0
        for name in names:
            truth = binary_truth(*load_labels(dataset, name))
            nonempty += bool(truth.any())
            save_mask_png(run/'t13'/'gt'/name, truth)
            rows.append(task_row(name, dataset, run, ask_variant))
        path = run/'t13'/'tasks'/('T1.3_any' + ('_dev' if phase == 'dev' else '') + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        report['splits'][phase] = dict(images=len(names), nonempty_truth=nonempty, manifest=str(path))
    write_json(run/'t13'/'preparation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--selection', type=Path, required=True, help='t13_ask_select dev selection.json')
    a = p.parse_args()
    variant = json.loads(a.selection.read_text())['selected']['question']
    print(json.dumps(generate(a.dataset_root, a.run_root, variant)))


if __name__ == '__main__':
    main()
