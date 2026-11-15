"""Freeze historical development cohorts before answer-state paired inference."""
import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--journal', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    whu = {Path(r['id']).stem: r for r in map(json.loads, (a.run/'whu/tasks/T1.1xC0_any_dev.jsonl').read_text().splitlines())}
    pools = {('LEVIR', 'T1.1xC0'): [], ('LEVIR', 'T5.1'): []}
    for line in a.journal.open():
        r = json.loads(line)
        key = (r['dataset'], r['task'])
        if key not in pools or r['split'] != 'val' or r.get('f1') is None:
            continue
        path = Path(r['session'])
        if not any(g in r['config'] for g in ('g3', 'g4', 'g5')):
            continue
        if key[0] == 'WHU' and path.name not in whu:
            continue
        if key[0] == 'LEVIR' and not path.name.startswith('val_'):
            continue
        inp = json.loads((path/'input.json').read_text())
        if key[0] == 'LEVIR':
            images = inp['source_images']
            if not all('/LEVIR-CD/val_original/' in x for x in images):
                continue
            row = dict(id=path.name+'.png', image_t1=images[0], image_t2=images[1], instruction=inp['instruction'],
                       task_type=r['task'], constraint=None, target='building', change_direction='any',
                       init_mask=inp.get('init_mask_path'), gt_mask=images[0].replace('/A/', '/label/'), meta={})
        else:
            row = whu[path.name]
        state_events = [e for e in r['events'] if any(s in json.dumps(e).lower() for s in
            ('no current', 'no_current', 'expected a inst', 'expected an inst', 'current_missing'))]
        r = dict(r, manifest=row, state_events=state_events)
        pools[key].append(r)
    selected, used = [], set()
    for key, pool in pools.items():
        for cohort in ('hard', 'simple'):
            candidates = sorted(pool, key=lambda r: (-len(r['state_events']), r['f1'], r['session'])) if cohort == 'hard' else sorted(
                [r for r in pool if r['f1'] >= .75 and not r['errors'] and r['termination'] == 'submit'],
                key=lambda r: (-r['f1'], r['calls'], r['session']))
            chosen = []
            for r in candidates:
                if cohort == 'simple' and not np.asarray(Image.open(r['manifest']['gt_mask'])).any():
                    continue
                identity = (key[0], Path(r['session']).name)
                if identity in used:
                    continue
                used.add(identity)
                chosen.append(dict(cohort=cohort, dataset=key[0], task=key[1], historical_session=r['session'],
                                   historical_f1=r['f1'], state_events=r['state_events'], manifest=r['manifest'],
                                   historical_errors=r['errors'], historical_termination=r['termination'],
                                   pilot=len(chosen) < (2 if cohort == 'hard' else 1)))
                if len(chosen) == (4 if cohort == 'hard' else 3):
                    break
            if len(chosen) < (2 if cohort == 'hard' else 1):
                raise ValueError(f'Insufficient candidates for {key}/{cohort}: {len(chosen)}')
            selected += chosen
    for r in selected:
        for field in ('image_t1', 'image_t2', 'gt_mask', 'init_mask'):
            if r['manifest'].get(field) and not Path(r['manifest'][field]).exists():
                raise FileNotFoundError(r['manifest'][field])
    (a.output/'selection.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    (a.output/'pilot.jsonl').write_text(''.join(json.dumps(r['manifest'])+'\n' for r in selected if r['pilot']))
    (a.output/'metric.json').write_text(json.dumps(dict(score=len(selected), cost_cny=0,
        frozen=len(selected), pilot=sum(r['pilot'] for r in selected))))
    print(json.dumps([dict(cohort=r['cohort'], task=r['task'], dataset=r['dataset'],
                          session=r['historical_session'], f1=r['historical_f1'], state_events=len(r['state_events']))
                      for r in selected if r['pilot']], indent=2))

if __name__ == '__main__':
    main()
