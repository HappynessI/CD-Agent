"""Offline T5.3 language feedback from the same LEVIR errors/given masks as T5.2."""
import argparse
from collections import Counter
from pathlib import Path
import json
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
from common import LEVIR, REPO, components, load_mask_png, save_mask_png, write_json
from agent_baseline.analysis.make_t52_feedback import RUN, error_regions, region_mask
from agent_baseline.feedback import cell_at
from agent_baseline.tasks import t53_instruction


def select_feedback(errors, shape):
    """Assign each error by its farthest point; choose the largest singleton cell.

    Count every error, of either type, before selecting. Ties retain T5.2's
    missed-first/source-ID ordering. A crossing error belongs to one cell only.
    """
    located = [dict(e, where=cell_at(e['point'], shape)) for e in errors]
    counts = Counter(e['where'] for e in located)
    eligible = [e for e in located if counts[e['where']] == 1]
    eligible.sort(key=lambda e: (-e['area'], e['type'] != 'missed', e['component_id']))
    return eligible[:1]


def generate(dataset, init_root, run_root, output, expected_images=64):
    dataset, init_root, run_root = map(Path, (dataset, init_root, run_root))
    paths = sorted((dataset / 'val_original/A').glob('*.png'))
    if len(paths) != expected_images:
        raise ValueError(f'Expected {expected_images} validation images, found {len(paths)}')
    report = dict(schema='t53-v1', connectivity=8, split='val',
        cell_rule='farthest error-region point; floor(3*x/width), floor(3*y/height); right/bottom on boundaries',
        uniqueness='exactly one assigned error of either type in the cell',
        ranking='error area descending, missed first on ties, source component ID ascending', images={})
    manifest = []
    for path in paths:
        truth = load_mask_png(dataset / 'val_original/label' / path.name)
        init_path = init_root / path.name
        initial = load_mask_png(init_path)
        errors = error_regions(truth, initial)
        selected = select_feedback(errors, initial.shape)
        gl, _ = components(truth)
        il, _ = components(initial)
        for item in selected:
            dest = run_root / 't53/regions' / path.stem / (item['region_id'].replace(':', '_') + '.png')
            save_mask_png(dest, region_mask(item, gl, il, initial))
        instruction = t53_instruction(selected)
        report['images'][path.name] = dict(feedback=selected,
            errors=[dict(e, where=cell_at(e['point'], initial.shape)) for e in errors], instruction=instruction)
        manifest.append(dict(id=path.name, image_t1=str(path.resolve()),
            image_t2=str((dataset / 'val_original/B' / path.name).resolve()),
            task_type='T5.3', instruction=instruction, constraint=None,
            target='building', change_direction='any', init_mask=str(init_path.resolve()),
            gt_mask=str((dataset / 'val_original/label' / path.name).resolve()),
            meta=dict(feedback=selected)))
    dest = run_root / 't53/tasks/T5.3.jsonl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest))
    write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=LEVIR)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--init-root', type=Path)
    p.add_argument('--output', type=Path, default=REPO / 'results/agent_baseline/data/t53_feedback.json')
    a = p.parse_args()
    generate(a.dataset_root, a.init_root or a.run_root / 'init/sam3_imc_p85/val', a.run_root, a.output)


if __name__ == '__main__':
    main()
