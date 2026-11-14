"""Two preregistered T5.3 responses: largest candidate or frozen flash verifier.

No planner and no label access. Empty candidates or entirely missing p_yes keep
the given mask. Both methods share the exact same complete-instance candidates.
"""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from common import components, load_mask_png, save_mask_png, write_json
from agent_baseline.analysis.make_t52_feedback import farthest_point, RUN
from agent_baseline.backends import read_soft
from agent_baseline.feedback import cell_at
from agent_baseline.tasks import read_tasks, levir_cache_name, load_pair
from agent_baseline.state import Session
from agent_baseline.ask_eval import api_answer, VARIANTS
from agent_baseline.planner import RunGuard
from agent_baseline.view import ask_images


def candidates(initial, feedback, t1_instances, t2_instances):
    """Membership is the candidate's farthest point cell. Keep complete masks.

    Misses: both dates, strict <50% coverage by the given mask. Identical
    outlines are deduplicated (T1 wins). False positives: given components.
    Stable ties: area descending, then T1 before T2, then source ID.
    """
    if not feedback:
        return []
    item, out, seen = feedback[0], [], set()
    labels, comps = components(initial)
    if item['type'] == 'missed':
        sources = [(time, r.get('id', n), r['idx'])
                   for time, rows in (('t1', t1_instances), ('t2', t2_instances))
                   for n, r in enumerate(rows, 1)]
    elif item['type'] == 'false_positive':
        sources = [('given', c['id'], np.flatnonzero(labels == c['id'])) for c in comps]
    else:
        raise ValueError('Unknown feedback type')
    for time, ident, idx in sources:
        mask = np.zeros_like(initial, dtype=bool)
        mask.ravel()[idx] = True
        area = int(mask.sum())
        if not area:
            continue
        coverage = float(initial[mask].mean())
        if item['type'] == 'missed' and coverage >= .5:
            continue
        if cell_at(farthest_point(mask), initial.shape) != item['where']:
            continue
        outline = np.flatnonzero(mask).tobytes()
        if outline in seen:
            continue
        seen.add(outline)
        yy, xx = np.nonzero(mask)
        out.append(dict(uid=f'{time}:{ident}', time=time, id=int(ident), area=area,
                        coverage=coverage, bbox=[int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1], mask=mask))
    return sorted(out, key=lambda c: (-c['area'], c['time'], c['id']))


def correct(initial, feedback, pool, mode='largest', verify=None):
    if mode not in ('largest', 'verifier'):
        raise ValueError('Unknown fixed response mode')
    trace = dict(candidates=[{k: v for k, v in c.items() if k != 'mask'} for c in pool], selected=None)
    result = initial.copy()
    if not feedback or not pool:
        trace['reason'] = 'no_feedback' if not feedback else 'no_candidates'
        return result, trace
    if mode == 'largest':
        selected = pool[0]
    else:
        scored = []
        for c, diagnostic in zip(pool, trace['candidates']):
            answer = verify(c)
            diagnostic['verification'] = answer
            p = answer.get('p_yes')
            if p is not None:
                if not np.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('Verifier p_yes must be in [0,1] or null')
                scored.append((c, p))
        trace['unscored'] = len(pool) - len(scored)
        if not scored:
            trace['reason'] = 'no_scored_candidates'
            return result, trace
        choose = max if feedback[0]['type'] == 'missed' else min
        selected = choose(scored, key=lambda pair: pair[1])[0]
    if feedback[0]['type'] == 'missed':
        result |= selected['mask']
    else:
        result &= ~selected['mask']
    trace.update(selected=selected['uid'], reason='corrected')
    return result, trace


def run(manifest, output, mode, max_cost=None, limit=None):
    rows = read_tasks(manifest, 'T5.3')
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError('Empty T5.3 manifest')
    if mode == 'verifier' and (max_cost is None or not np.isfinite(max_cost) or max_cost <= 0):
        raise ValueError('Verifier response requires a positive finite --max-cost-cny')
    dest = Path(output) / mode
    dest.mkdir(parents=True, exist_ok=True)
    settings = dict(mode=mode, variant=VARIANTS['f37_corr'], pixel_thr=.85, score_thr=.5,
        cell_rule='candidate farthest point', candidate_rule='two dates; given coverage < 50%; deduplicate exact outlines',
        tasks=[dict(id=r['id'], image_t1=r['image_t1'], image_t2=r['image_t2'], init_mask=r['init_mask'],
                    feedback=[{k: e[k] for k in ('where', 'type')} for e in r['meta']['feedback']]) for r in rows])
    settings_path = dest / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing fixed responses use different inputs/settings; use a fresh output directory')
    write_json(settings_path, settings)
    answers_path = dest / 'answers.jsonl'
    done = {r['uid']: r['answer'] for r in map(json.loads, answers_path.read_text().splitlines())} if answers_path.exists() else {}
    guard = RunGuard(RUN / 'STOP.json', dest / 'budget/ledger.txt', max_cost) if mode == 'verifier' else None
    if guard:
        guard.check()
    for row in rows:
        stem = Path(row['id']).stem
        if (dest / 'masks' / (stem + '.png')).exists() and (dest / 'traces' / (stem + '.json')).exists():
            continue
        name = levir_cache_name(row, 'val')
        initial = load_mask_png(row['init_mask'])
        # Only public language feedback is passed to candidate selection.
        feedback = [{k: e[k] for k in ('where', 'type')} for e in row['meta']['feedback']]
        dates = [read_soft('val', name, t, .85, .5) for t in (0, 1)] if feedback and feedback[0]['type'] == 'missed' else [[], []]
        pool = candidates(initial, feedback, *dates)
        rgb = load_pair(row) if mode == 'verifier' and pool else None
        session = Session(initial.shape)
        def verify(candidate):
            uid = f"{stem}:{candidate['uid']}"
            if uid not in done:
                guard.check()
                images = ask_images(rgb, candidate['mask'], candidate['bbox'], True)
                answer = api_answer(images, VARIANTS['f37_corr'], guard, sum(map(ord, uid)) % (2**31), session)
                with answers_path.open('a') as file:
                    file.write(json.dumps(dict(uid=uid, answer=answer)) + '\n')
                done[uid] = answer
            return done[uid]
        final, trace = correct(initial, feedback, pool, mode, verify)
        write_json(dest / 'traces' / (stem + '.json'), trace)
        save_mask_png(dest / 'masks' / (stem + '.png'), final)
        print(stem, trace['reason'], trace['selected'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='Separate largest/ and verifier/ directories are created here')
    p.add_argument('--mode', choices=['largest', 'verifier'], required=True)
    p.add_argument('--max-cost-cny', type=float)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    run(a.tasks, a.output_dir, a.mode, a.max_cost_cny, a.limit)


if __name__ == '__main__':
    main()
