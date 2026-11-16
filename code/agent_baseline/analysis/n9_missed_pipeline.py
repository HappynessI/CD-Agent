"""N9 profile candidate request union and fresh V0/V2/V3/V4 verification."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .n8_missed_assets import read, write
from .n8_missed_prepare import load_masks
from .n8_missed_pipeline import cost_total
from .n9_missed_assets import OUT
from .n9_missed_protocol import configurations, candidate_indices, profile_for
from .night_missed_views import CONSERVATIVE, BLUE_LEGEND, render


def config_key(config):
    return '__'.join(str(config[k]) for k in ('profile', 'sources', 'semantic', 'dino', 'alpha', 'verifier', 'k'))


def configs_for(out, profile, split):
    if split == 'tune': return configurations(profile)
    return [read(out/'frozen_profiles.json')['profiles'][profile]['config']]


def prepare(a):
    states = []; requests = []
    for item in read(a.out/'fm1_manifest.json'):
        if item['split'] != a.split: continue
        profile = profile_for(item['direction'])
        rankings = {}; variants = {}
        for config in configs_for(a.out, profile, a.split):
            chosen = candidate_indices(item['candidates'], config)
            rankings[config_key(config)] = chosen
            for index in chosen: variants.setdefault(index, set()).add(config['verifier'])
        states.append(dict(group=item['group'], stem=item['stem'], profile=profile, split=a.split,
            rankings=rankings, generated_find_missed=True, net_candidates=len(item['candidates'])))
        for index in sorted(variants):
            requests.append(dict(uid=f"{item['group']}__{item['stem']}__n9_{index:04d}",
                group=item['group'], stem=item['stem'], profile=profile, split=a.split,
                direction=item['direction'], scope=item['scope'], shape=item['shape'], base=item['base'],
                masks=item['masks'], mask_index=index, candidate=item['candidates'][index],
                image_t1=item['image_t1'], image_t2=item['image_t2'], variants=sorted(variants[index])))
    write(a.out/f'states_{a.split}.json', states)
    write(a.out/f'requests_{a.split}.json', requests)
    write(a.out/'metric.json', dict(stage='requests_'+a.split, status='complete', tasks=len(states),
        unique_candidates=len(requests), request_slots=sum(len(r['variants']) for r in requests),
        cost_cny=cost_total(a.out), cumulative=True))


def make_request(row, paths, variant):
    from agent_baseline.state import Session
    from agent_baseline.scope_t13 import session_prompts
    prompts = session_prompts(Session(tuple(row['shape']), target='building',
                                     change_direction=row['direction'], scope=row['scope']))
    question = prompts['CORRESPONDENCE']
    if variant in ('V3', 'V4'):
        question = CONSERVATIVE + ' Requested direction: ' + row['direction'] + '.'
    if variant in ('V2', 'V4'): question += BLUE_LEGEND
    content = []
    for label, path in zip(('T1 close-up', 'T2 close-up', 'T1 wider view', 'T2 wider view'), paths):
        content.extend([dict(type='text', text=label), dict(type='image_url', image_url=dict(url=str(path)))])
    content.append(dict(type='text', text=question))
    return dict(messages=[dict(role='system', content=prompts['SYSTEM']), dict(role='user', content=content)],
        parameters=dict(model='qwen3.7-flash', temperature=0, max_tokens=256, enable_thinking=False,
            response_format={'type':'json_object'}, logprobs=True, top_logprobs=5, seed=1016+row['mask_index']))


def verify(a):
    from agent_baseline.planner import Planner, usage_tokens, request_cost
    from agent_baseline.view import ask_images
    from .missed_verifier_study import parse_response
    from .io_retry import read_image
    cost = cost_total(a.out); attempts = len(list((a.out/'usage').glob('*.json')))
    initial_cost = cost; initial_attempts = attempts; started = time.monotonic()
    loaded = None; completed = 0
    for row in read(a.out/f'requests_{a.split}.json'):
        for variant in row['variants']:
            dest = a.out/'answers'/variant/(row['uid']+'.json')
            if dest.exists(): completed += 1; continue
            previous = sorted((a.out/'usage').glob(row['uid']+'__'+variant+'__*.json'))
            recovered = next((read(p) for p in reversed(previous) if read(p).get('raw') and not read(p).get('error_type')), None)
            if recovered:
                write(dest, dict(uid=row['uid'], variant=variant, **parse_response(recovered['raw']),
                    usage=recovered['usage'], cost_cny=recovered['cost_cny'], recovered=True))
                completed += 1; continue
            if cost >= a.budget:
                raise RuntimeError('Actual N9 FM allocation reached; main agent can reallocate within total authorization')
            if loaded != row['masks']:
                masks = load_masks(row['masks'])
                rgb = [read_image(row['image_'+t], rgb=True) for t in ('t1', 't2')]
                with np.load(row['base'], allow_pickle=False) as z: base = z['start'].astype(bool)
                loaded = row['masks']
            style = 'blue' if variant in ('V2', 'V4') else 'original'
            viewdir = a.out/'views'/row['uid']/style; viewdir.mkdir(parents=True, exist_ok=True)
            paths = [viewdir/f'{i}.jpg' for i in range(4)]
            if not all(p.exists() for p in paths):
                mask = masks[row['mask_index']]
                images = (render(rgb, mask, base, row['candidate']['bbox'], 'V2')[0] if style == 'blue'
                          else ask_images(rgb, mask, row['candidate']['bbox'], True))
                for path, image in zip(paths, images): image.save(path, quality=92)
            folder = a.out/'inputs'/variant/row['uid']; folder.mkdir(parents=True, exist_ok=True)
            request = make_request(row, paths, variant)
            write(folder/'request.json', dict(request, uid=row['uid'], variant=variant, reuse='none; fresh N9 request'))
            parameters = dict(request['parameters']); seed = parameters.pop('seed'); last = {}
            def capture(event):
                nonlocal cost, attempts, last
                tokens = usage_tokens(event.get('usage')); charge = request_cost('qwen3.7-flash', tokens)
                body = event.get('error_body'); secret = os.environ.get('DASHSCOPE_API_KEY')
                if body and secret: body = body.replace(secret, '[credential omitted]')
                last = dict(uid=row['uid'], variant=variant, split=a.split, model='qwen3.7-flash',
                    raw=event.get('raw'), usage=event.get('usage'), tokens=tokens, cost_cny=charge,
                    http_status=event.get('http_status'), error_type=event.get('error_type'), error_body=body)
                write(a.out/'usage'/f"{row['uid']}__{variant}__{time.time_ns()}.json", last)
                cost += charge; attempts += 1
                write(a.out/'metric.json', dict(stage='verify_'+a.split, status='running', completed=completed,
                    requests=attempts, cost_cny=cost, cumulative=True))
                if event.get('error_type') and (event.get('http_status') != 429 or event.get('quota_error')):
                    raise RuntimeError('API failure recorded; no fabricated answer')
            answer = Planner(seed=seed, parameters=parameters, max_attempts=3, record=capture).complete(request['messages'], [], 1)
            write(dest, dict(uid=row['uid'], variant=variant, **parse_response(answer['raw']),
                usage=answer['usage'], cost_cny=last['cost_cny'], input_path=str(folder)))
            completed += 1
    write(a.out/f'verify_{a.split}.json', dict(status='complete', completed=completed,
        new_requests=attempts-initial_attempts, new_cost_cny=cost-initial_cost, cost_cny=cost,
        elapsed_seconds=time.monotonic()-started, slurm_job=os.environ['SLURM_JOB_ID']))
    write(a.out/'metric.json', dict(stage='verified_'+a.split, status='complete', completed=completed,
        requests=attempts, cost_cny=cost, cumulative=True))


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('prepare', 'verify'))
    p.add_argument('--split', choices=('tune', 'confirm'), required=True)
    p.add_argument('--out', type=Path, default=OUT); p.add_argument('--budget', type=float, default=20.)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    globals()[a.command](a)

if __name__ == '__main__': main()
