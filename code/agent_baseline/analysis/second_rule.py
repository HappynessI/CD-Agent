"""Frozen f37_corr deletion rule on a supplied mask directory; resumable per component.

Uses ask_eval's exact flash model, crop rendering, seed formula and API parameters.
Labels never enter verification. Different directions have separate output/ledger
namespaces. A parsed no removes a component; yes or unparsed keeps it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from PIL import Image
from common import components, save_mask_png, write_json
from agent_baseline.ask_eval import api_answer, VARIANTS, RUN
from agent_baseline.planner import RunGuard
from agent_baseline.tasks import read_tasks, load_pair
from agent_baseline.state import Session
from agent_baseline.view import ask_images


def apply_rule(mask, labels, answers):
    no = [ident for ident, answer in answers.items() if answer.get('answer') == 'no']
    return mask & ~np.isin(labels, no)


def run(rows, mask_dir, output, max_cost, concurrency=4, scope=None, direction=None):
    if not rows or concurrency < 1 or max_cost <= 0:
        raise ValueError('Require nonempty tasks and positive concurrency/cost cap')
    policies = {(direction or r['change_direction'], scope or r.get('scope')) for r in rows}
    if len(policies) != 1:
        raise ValueError('Use one subtask/direction manifest per invocation')
    actual_direction, actual_scope = next(iter(policies))
    dest = Path(output) / actual_direction
    dest.mkdir(parents=True, exist_ok=True)
    settings = dict(mask_dir=str(Path(mask_dir).resolve()), direction=actual_direction, scope=actual_scope,
                    variant='f37_corr', ids=[r['id'] for r in rows],
                    images=[[r['image_t1'], r['image_t2']] for r in rows],
                    targets=[list(r['target']) if isinstance(r['target'], (tuple, list)) else r['target'] for r in rows],
                    source_classes=[r.get('source_classes') for r in rows],
                    masks=[dict(name=Path(r['id']).stem + '.png',
                        size=(Path(mask_dir)/(Path(r['id']).stem + '.png')).stat().st_size,
                        mtime=(Path(mask_dir)/(Path(r['id']).stem + '.png')).stat().st_mtime_ns) for r in rows])
    config = dest / 'settings.json'
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError('Existing rule answers have different inputs or scope; use a fresh directory')
    write_json(config, settings)
    answers_path = dest / 'answers.jsonl'
    done = {r['uid']: r for r in map(json.loads, answers_path.read_text().splitlines())} if answers_path.exists() else {}
    guard = RunGuard(RUN / 'STOP.json', dest / 'budget' / 'ledger.txt', max_cost)
    guard.check()
    guard.charge(0)
    with ThreadPoolExecutor(max_workers=concurrency) as executor, answers_path.open('a') as file:
        for row in rows:
            stem = Path(row['id']).stem
            mask = np.asarray(Image.open(Path(mask_dir) / (stem + '.png'))) > 0
            labels, comps = components(mask)
            rgb = load_pair(row)
            session = Session(mask.shape, target=row['target'], change_direction=actual_direction, scope=actual_scope,
                              source_classes=row.get('source_classes'))
            def one(comp):
                uid = f"{stem}:{comp['id']}"
                pair = ask_images(rgb, labels == comp['id'], comp['bbox'], True)
                result = api_answer(pair, VARIANTS['f37_corr'], guard, sum(map(ord, uid)) % (2**31), session)
                return dict(uid=uid, **result)
            todo = [comp for comp in comps if f"{stem}:{comp['id']}" not in done]
            # At most concurrency requests are in flight; spending is recorded by RunGuard.
            for start in range(0, len(todo), concurrency):
                futures = [executor.submit(one, comp) for comp in todo[start:start + concurrency]]
                failure = None
                for future in futures:
                    try:
                        result = future.result()
                        done[result['uid']] = result
                        file.write(json.dumps(result) + '\n')
                        file.flush()
                    except Exception as exc:
                        failure = failure or exc
                if failure is not None:
                    raise failure
            per = {comp['id']: done[f"{stem}:{comp['id']}"] for comp in comps}
            save_mask_png(dest / 'masks' / (stem + '.png'), apply_rule(mask, labels, per))
            print(stem, len(comps), flush=True)
    report = dict(components=len(done), no=sum(r.get('answer') == 'no' for r in done.values()),
                  rebuilt=sum(r.get('type') == 'rebuilt' for r in done.values()),
                  unparsed=sum(r.get('answer') not in ('yes', 'no') for r in done.values()),
                  cost_cny=sum(r.get('cost_cny', 0) for r in done.values()))
    write_json(dest / 'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-cost-cny', type=float, required=True)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--scope')
    p.add_argument('--direction', choices=['any', 'appeared', 'disappeared'])
    a = p.parse_args()
    if a.concurrency < 1 or a.max_cost_cny <= 0:
        p.error('Concurrency and cost cap must be positive')
    print(json.dumps(run(read_tasks(a.tasks), a.mask_dir, a.output_dir, a.max_cost_cny, a.concurrency, a.scope, a.direction)))


if __name__ == '__main__':
    main()
