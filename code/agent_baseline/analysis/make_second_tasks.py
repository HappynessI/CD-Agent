"""Compute-node preparation only: sample validation images, write GT and five manifests."""
import argparse
import json
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
from common import save_mask_png, write_json
from agent_baseline.fixed_routes import RUN
from agent_baseline.tasks import INSTRUCTIONS, C1_INSTRUCTIONS, needs_init
from agent_baseline.analysis.second_labels import (DATASET, VALIDATION, SCOPE, DIRECTION_GT,
    check_encoding, load_labels, building_truth, validation_names)

SUBTASKS = [('T1.1xC1', 'appeared'), ('T1.1xC1', 'disappeared'), ('T1.1xC0', 'any'),
            ('T5.1xC1', 'appeared'), ('T5.1xC1', 'disappeared')]


def sample_changed(names, truth_loader, count=128, seed=1016):
    candidates = [n for n in sorted(set(names)) if truth_loader(n)['c0'].any()]
    if len(candidates) < count or count < 1:
        raise ValueError(f'Need {count} changed images; found {len(candidates)}')
    chosen = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    return sorted(candidates[i] for i in chosen)


def task_row(name, dataset, run, task, direction):
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    return dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
        instruction=C1_INSTRUCTIONS[task][direction] if task in C1_INSTRUCTIONS else INSTRUCTIONS[task],
        task_type=task, constraint=None, target='building', change_direction=direction, scope=SCOPE,
        init_mask=str(run / 'init/second_sam3_imc_p85/val' / name) if needs_init(task) else None,
        gt_mask=str(run / 'second/gt' / DIRECTION_GT[direction] / name), meta={})


def generate(dataset, validation, run, names_output, count=128, seed=1016):
    dataset, run, names_output = Path(dataset), Path(run), Path(names_output)
    encoding = check_encoding(dataset)
    source_names = validation_names(validation)
    selected = sample_changed(source_names, lambda n: building_truth(*load_labels(dataset, n)), count, seed)
    if names_output.exists() and names_output.read_text().split() != selected:
        raise ValueError('Existing frozen image list differs; choose a new output explicitly')
    names_output.parent.mkdir(parents=True, exist_ok=True)
    names_output.write_text('\n'.join(selected) + '\n')
    for name in selected:
        for kind, mask in building_truth(*load_labels(dataset, name)).items():
            save_mask_png(run / 'second/gt' / kind / name, mask)
    tasks = run / 'second/tasks'
    tasks.mkdir(parents=True, exist_ok=True)
    for task, direction in SUBTASKS:
        with (tasks / f'{task}_{direction}.jsonl').open('w') as file:
            for name in selected:
                file.write(json.dumps(task_row(name, dataset, run, task, direction)) + '\n')
    write_json(run / 'second/preparation.json', dict(validation_source=str(Path(validation).resolve()),
        validation_count=len(source_names), count=count, seed=seed, encoding=encoding,
        unchanged_encoding='0: unlabelled unchanged, not a land-cover class', names=selected,
        scope=SCOPE, pixel_thr=.85, score_thr=.5, tile=512, overlap_thr=.25))
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--validation-file', type=Path, default=VALIDATION)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--names-output', type=Path, default=Path(__file__).resolve().parents[3] / 'results/agent_baseline/data/second_val128.txt')
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    print('\n'.join(generate(a.dataset_root, a.validation_file, a.run_root, a.names_output, a.count, a.seed)))


if __name__ == '__main__':
    main()
