"""Frozen T5.4 SAM3(T2 building) XOR old map; optional dev-selected area filter.

SAM3 uses tile=512, pixel=.85, score=.5, confidence=.1. Raw T2 segmentation is
saved and scored separately. No T1 photograph or new semantic label enters tools.
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
from common import components, save_mask_png, write_json
from agent_baseline.backends import SegmentBackend
from agent_baseline.fixed_routes import RUN
from agent_baseline.state import Mask
from agent_baseline.tools import Toolbox
from agent_baseline.tasks import load_pair, read_tasks
from agent_baseline.map_input import check_task
from agent_baseline.analysis.second_metrics import counts, score, mask

SETTINGS = dict(pixel_thr=.85, score_thr=.5, tile=512, conf=.1)


def filter_components(data, min_area=0):
    if type(min_area) is not int or min_area < 0:
        raise ValueError('min_area must be a nonnegative integer')
    if min_area == 0:
        return np.array(data, bool, copy=True)
    labels, comps = components(data)
    return np.isin(labels, [c['id'] for c in comps if c['area'] >= min_area])


def route(toolbox, min_area=0):
    """Use the same atomic SAM3 tool as Agent, once at T2 only."""
    old = toolbox.session.get(toolbox.session.old_map, Mask).data
    instances = toolbox.execute('segment_text', dict(time='t2', text='building', tile=512, score_thr=.5))
    if not instances.public['ok']:
        raise RuntimeError(instances.text)
    result = toolbox.execute('to_mask', dict(sets=[instances.public['inst']], set_current=False))
    if not result.public['ok']:
        raise RuntimeError(result.text)
    buildings = toolbox.session.get(result.public['mask'], Mask).data.copy()
    return filter_components(old ^ buildings, min_area), buildings


def resolve_filter(rows, min_area=None, frozen=None, dev_names=()):
    """Positive filtering is selected on dev only; val consumes the saved dev choice."""
    if frozen is not None:
        if min_area is not None:
            raise ValueError('Choose --min-area or --frozen-settings')
        policy = json.loads(Path(frozen).read_text())
        if policy.get('selected_on') != 'dev' or any(policy.get(k) != v for k, v in SETTINGS.items()):
            raise ValueError('Frozen settings must come from T5.4 development with the same SAM3 setup')
        min_area = policy['min_area']
    min_area = 0 if min_area is None else min_area
    if type(min_area) is not int or min_area < 0:
        raise ValueError('min_area must be a nonnegative integer')
    if min_area and frozen is None and any(r['id'] not in dev_names for r in rows):
        raise ValueError('Positive --min-area requires the frozen development image list')
    return min_area


def run(rows, output, worker=None, min_area=0, segment_factory=None):
    if not rows or any(r['task_type'] != 'T5.4' for r in rows):
        raise ValueError('Use a nonempty T5.4 manifest')
    output = Path(output)
    if len({Path(r['id']).stem for r in rows}) != len(rows):
        raise ValueError('Duplicate image stems')
    settings = dict(SETTINGS, min_area=min_area, ids=[r['id'] for r in rows],
                    images=[[r['image_t1'], r['image_t2']] for r in rows])
    settings_path = output / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Fixed output has different inputs/settings; use a fresh directory')
    write_json(settings_path, settings)
    per = dict(xor={}, buildings={})
    for row in rows:
        check_task(row)
        paths = [row['image_t1'], row['image_t2']]
        segment = (segment_factory(row) if segment_factory else
                   SegmentBackend('val', Path(paths[0]).name, worker, .85, image_paths=paths))
        box = Toolbox(load_pair(row), output / 'tools', segment=segment, map_input=True)
        prediction, buildings = route(box, min_area)
        stem = Path(row['id']).stem
        save_mask_png(output / 'xor' / (stem + '.png'), prediction)
        save_mask_png(output / 'buildings' / (stem + '.png'), buildings)
        # References are opened only after both predictions have been finalized.
        truth = mask(row['gt_mask'], prediction.shape)
        per['xor'][stem] = counts(prediction, truth)
        per['buildings'][stem] = counts(buildings, truth)
    report = dict(settings=settings, methods={key: dict(score=score(list(data.values())), per_image=data)
                                              for key, data in per.items()})
    write_json(output / 'counts.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--sam3-worker')
    p.add_argument('--min-area', type=int)
    p.add_argument('--dev-names', type=Path, help='Required to select positive component filtering')
    p.add_argument('--frozen-settings', type=Path)
    p.add_argument('--write-selection', type=Path, help='Save the evaluated dev setting for report use')
    a = p.parse_args()
    os.environ['AGENT_RUN'] = str(a.run_root)
    rows = read_tasks(a.tasks)
    dev = set()
    if a.dev_names:
        from agent_baseline.analysis.make_t54_tasks import names_from
        dev = set(names_from(a.dev_names))
    if a.write_selection and (not rows or not dev or any(r['id'] not in dev for r in rows)):
        p.error('--write-selection requires tasks from --dev-names')
    area = resolve_filter(rows, a.min_area, a.frozen_settings, dev)
    report = run(rows, a.output_dir, a.sam3_worker, area)
    if a.write_selection:
        write_json(a.write_selection, dict(SETTINGS, min_area=area, selected_on='dev', ids=[r['id'] for r in rows]))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
