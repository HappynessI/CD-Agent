"""N9 asset QC and frozen image splits. Run only on a Slurm node."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from agent_baseline.evaluation import load_reference
from agent_baseline.whu_gt import load_truth, KINDS
from .missed_sources import specs, rows_for
from .n8_missed_assets import ROOT, RUN, WORK, SOURCES, CLASSES, read, write
from .n9_missed_protocol import SEED, PROFILE_GRIDS, profile_for, split_names, online_names

OUT = ROOT/'CD-Agent/results/agent_baseline/n9_missed_20261116'
HISTORY = ROOT/'CD-Agent/results/agent_baseline'


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def frozen_splits():
    # Inherit actual archived query-level membership, not a newly sampled SECOND split.
    old = read(HISTORY/'night_t3_20261114/audit.json')['split']
    second = {task.rsplit('_', 1)[0]: part for task, part in old.items()}
    expected = {Path(x).stem for x in (RUN/'second_dev/names.txt').read_text().split()}
    assert set(second) == expected and Counter(second.values()) == {'tune': 30, 'confirm': 30}
    whu = split_names([Path(x).stem for x in (RUN/'whu/names_dev60.txt').read_text().split()])
    report = {Path(x).stem for x in (RUN/'whu/names_val128.txt').read_text().split()}
    assert len(whu) == 60 and not set(whu) & report
    # Existing development train images only: no new data, no report/test labels inspected.
    levir = sorted(Path(x).stem for x in (RUN/'dev2/names.txt').read_text().split())
    assert len(levir) == 200 and len(set(levir)) == 200
    random.Random(SEED).shuffle(levir)
    levir = split_names(levir[:60])
    return dict(second=second, whu=whu, levir=levir)


def known_probability_paths():
    known = {}
    for row in read(HISTORY/'n8_missed_20261115/asset_manifest.json'):
        for t in ('t1', 't2'):
            known[str(Path(row['image_'+t]).resolve())] = row['semantic'][t]
    for row in jsonl(RUN/'results/agent_baseline/data/t3_tasks_dev_0of1/tasks.jsonl'):
        for t in ('t1', 't2'):
            image = str(Path(row['image_'+t]).resolve())
            path = ROOT/'exp6_staging/batch98_t3_routes/results/route_a/semantic'/f"{row['id']}_{t}.npz"
            if image not in known and path.is_file():
                known[image] = str(path)
    return known


def inventory(a):
    start = time.monotonic()
    splits = frozen_splits()
    freeze = dict(seed=SEED, second_source=str(HISTORY/'night_t3_20261114/audit.json'),
        splits=splits, online={d: online_names(s) for d, s in splits.items()},
        online_selection='seed20261117 shuffle sorted confirm names; first10, before any N9 scoring',
        levir_source=str(RUN/'dev2/names.txt'), levir_selection='seed20261116 choose60 of frozen development200, then30/30',
        authorization='Existing development pools explicitly authorized for N9; no newly constructed task data; WHU official repair already accepted',
        exposure='All development pools have historical exposure. SECOND inherits query30/30, which differs from N8 pilot5/5; no untouched-holdout claim',
        profile_grids=PROFILE_GRIDS)
    frozen = a.out/'frozen_protocol.json'
    if frozen.exists() and read(frozen) != json.loads(json.dumps(freeze)):
        raise ValueError('Existing frozen protocol differs')
    write(frozen, freeze)  # Written before reading any GT or prediction arrays.
    known = known_probability_paths()
    manifest = []; evaluation = []; missing = []; unique = {}; qc = []; raw_whu = {}
    for spec in specs(RUN, WORK):
        dataset = spec['dataset']; group = spec['group']
        for task in rows_for(RUN, spec):
            stem = Path(task['id']).stem
            if stem not in splits[dataset]:
                continue
            row = dict(group=group, stem=stem, split=splits[dataset][stem],
                profile=profile_for(task['change_direction']), direction=task['change_direction'],
                scope=task.get('scope', spec['scope']), instruction=task.get('instruction', 'Find all changed buildings.'),
                image_t1=task['image_t1'], image_t2=task['image_t2'],
                base=str(WORK/'base'/group/(stem+'.npz')),
                sources={s: str(WORK/'candidates'/group/s/(stem+'.npz')) for s in SOURCES},
                semantic={}, features={})
            evaluation.append(dict(group=group, stem=stem, split=row['split'], task=task))
            needed = [row['base'], *row['sources'].values(), row['image_t1'], row['image_t2'], task['gt_mask']]
            if task.get('gt_ignore_mask'): needed.append(task['gt_ignore_mask'])
            absent = [p for p in needed if not Path(p).is_file()]
            missing.extend(absent)
            for t in ('t1', 't2'):
                image = str(Path(row['image_'+t]).resolve())
                sem = known.get(image, str(a.out/'semantic'/dataset/(stem+'_'+t+'.npz')))
                feature = WORK/'gpu_cache/dynamic'/dataset/stem/f'extract_features_{t}.npz'
                if not feature.is_file():
                    feature = a.out/'features'/dataset/(stem+'_'+t+'.npz')
                row['semantic'][t] = sem; row['features'][t] = str(feature)
                unique[image] = dict(image=image, semantic=sem, features=str(feature))
            if absent:
                manifest.append(row); continue
            with np.load(row['base'], allow_pickle=False) as z:
                shape = tuple(z['start'].shape)
            assert len(shape) == 2
            row['shape'] = shape
            for p in row['sources'].values():
                with np.load(p, allow_pickle=False) as z:
                    assert tuple(z['shape']) == shape and z['masks'].ndim == 2
                    assert z['masks'].shape[1] == (int(np.prod(shape))+7)//8
            gt, valid = load_reference(task['gt_mask'], task.get('gt_ignore_mask'))
            assert gt.shape == shape
            if dataset == 'whu':
                if stem not in raw_whu:
                    raw_whu[stem] = load_truth(Path(task['image_t1']).parent.parent, Path(task['image_t1']).name)
                official = raw_whu[stem]; direction = row['direction']
                assert 'gt_official_v3' in Path(task['gt_mask']).parts
                assert np.array_equal(gt, official[KINDS[direction]])
                expected_valid = np.ones(shape, bool) if direction == 'any' else ~official['ignore']
                assert np.array_equal(valid, expected_valid)
                assert np.all(valid[~official['c0']]), 'Official negatives must remain evaluable'
            qc.append(dict(group=group, stem=stem, shape=shape, official_valid_checked=dataset=='whu'))
            manifest.append(row)
    write(a.out/'asset_manifest.json', manifest)
    write(a.out/'evaluation_manifest.json', evaluation)
    report = dict(status='missing_required_source' if missing else 'complete', tasks=len(manifest),
        task_lines=dict(Counter(r['group'] for r in manifest)),
        split_lines=dict(Counter(r['group']+'/'+r['split'] for r in manifest)), images=len(unique),
        missing_required=sorted(set(missing)),
        semantic_available=sum(Path(v['semantic']).is_file() for v in unique.values()),
        semantic_missing=sum(not Path(v['semantic']).is_file() for v in unique.values()),
        dino_available=sum(Path(v['features']).is_file() for v in unique.values()),
        dino_missing=sum(not Path(v['features']).is_file() for v in unique.values()), assets=list(unique.values()),
        probability_classes=['background', *CLASSES], building_channel=5,
        elapsed_seconds=time.monotonic()-start, slurm_job=os.environ['SLURM_JOB_ID'])
    write(a.out/'asset_qc.json', dict(checked=qc, whu_physical_images=len(raw_whu),
        note='Labels used only to validate official contracts, no model scoring or selection'))
    write(a.out/'asset_inventory.json', report)
    write(a.out/'metric.json', dict(stage='asset_qc', status=report['status'], tasks=len(manifest), cost_cny=0, cumulative=True))
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets', 'missing_required')}))
    if missing: raise RuntimeError('Required source assets missing; see inventory before reconstructing')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    a.out.mkdir(parents=True, exist_ok=True); inventory(a)

if __name__ == '__main__': main()
