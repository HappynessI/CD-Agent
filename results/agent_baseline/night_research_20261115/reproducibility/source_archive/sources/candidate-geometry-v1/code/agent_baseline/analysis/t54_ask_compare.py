"""Paired development comparison on every fixed-route component. API only in run()."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
import numpy as np
from common import components
from agent_baseline.ask_eval import api_answer, auc, VARIANTS
from agent_baseline.map_input import attach_map, check_task, t2_state_change
from agent_baseline.planner import RunGuard, RunStopped
from agent_baseline.state import Session, Mask
from agent_baseline.tasks import read_tasks
from agent_baseline.view import ask_images
from .io_retry import read_image, read_text, write_json, write_text, retry_io, save_image
from .second_metrics import counts, score
from .second_rule import apply_rule

VARIANTS_MAP = ('map_compare', 't2_state')


class ComparisonGuard(RunGuard):
    """Check the combined ledger after every response, including resumed requests."""
    def trip(self, reason, batch_only=False):
        target = self.stop_file
        write_json(target, dict(reason=reason, at=time.time()))
        raise RunStopped(reason)

    def check(self):
        retry_io(super().check)

    def total(self):
        return ledger_cost(self.directory)

    def charge(self, cost):
        super().charge(cost)
        with self.lock:
            # The existing writer swallows a transient OSError; complete its
            # current value atomically before permitting another request.
            write_text(self.own, repr(self.spent))
            total = self.total()
            if total >= self.max_cost:
                self.trip(f'budget: {total:.4f} >= {self.max_cost:.4f} CNY')



def ledger_cost(directory):
    return sum(float(read_text(p)) for p in Path(directory).glob('*.txt'))


def run(rows, mask_dir, output, max_cost, concurrency=1, answer_fn=api_answer):
    if not rows or concurrency < 1 or max_cost <= 0:
        raise ValueError('Require tasks, positive concurrency and cost cap')
    output, mask_dir = Path(output), Path(mask_dir)
    for row in rows:
        check_task(row)
    settings = dict(rows=rows, mask_dir=str(mask_dir.resolve()), variants=list(VARIANTS_MAP))
    config = output / 'settings.json'
    # Check all candidate files before making any paid request.
    for row in rows:
        path = mask_dir / (Path(row['id']).stem + '.png')
        if not path.is_file():
            raise FileNotFoundError(f'{path}: generate dev candidates with t54_fixed.py; see README_t54.md')
    settings['candidate_files'] = [dict(name=r['id'], size=retry_io(lambda r=r: (mask_dir / (Path(r['id']).stem+'.png')).stat()).st_size,
        mtime_ns=retry_io(lambda r=r: (mask_dir / (Path(r['id']).stem+'.png')).stat()).st_mtime_ns) for r in rows]
    if config.exists() and json.loads(read_text(config)) != settings:
        raise ValueError('Inputs changed; use a fresh output directory')
    write_json(config, settings)
    answer_path = output / 'answers.jsonl'
    done = {r['uid']: r for r in map(json.loads, read_text(answer_path).splitlines())} if answer_path.exists() else {}
    guard = ComparisonGuard(output / 'STOP.json', output / 'budget/ledger.txt', max_cost)
    retry_io(guard.check)
    guard.charge(0)
    totals = {v: [] for v in VARIANTS_MAP}
    labels_for_auc = {v: [] for v in VARIANTS_MAP}
    base, expected = [], 0
    try:
        with ThreadPoolExecutor(concurrency) as pool:
            for row in rows:
                stem = Path(row['id']).stem
                mask = read_image(mask_dir / (stem + '.png')) > 0
                labels, comps = components(mask)
                rgb = [read_image(row[k], rgb=True) for k in ('image_t1', 'image_t2')]
                sessions = {v: Session(mask.shape, ask_variant='t2_state' if v == 't2_state' else None) for v in VARIANTS_MAP}
                for session in sessions.values():
                    attach_map(session, rgb[0])
                def one(variant, comp):
                    retry_io(guard.check)
                    region = labels == comp['id']
                    images = ask_images(rgb, region, comp['bbox'], True)
                    uid = f"{variant}:{stem}:{comp['id']}"
                    result = answer_fn(images, dict(VARIANTS['f37_corr'], strict_stop=True), guard,
                                       sum(map(ord, f"{stem}:{comp['id']}")) % (2**31), sessions[variant])
                    if variant == 't2_state':
                        old = sessions[variant].get('map#0', Mask).data
                        result = t2_state_change(result, old, region)
                    return dict(uid=uid, variant=variant, **result)
                todo = [(v, c) for c in comps for v in VARIANTS_MAP if f"{v}:{stem}:{c['id']}" not in done]
                for start in range(0, len(todo), concurrency):
                    futures = [pool.submit(one, v, c) for v, c in todo[start:start + concurrency]]
                    failure = None
                    for future in futures:
                        try:
                            result = future.result()
                            done[result['uid']] = result
                            # Atomic checkpoint avoids partially appended JSONL after EIO.
                            write_text(answer_path, ''.join(json.dumps(r) + '\n' for r in done.values()))
                        except Exception as exc:
                            failure = failure or exc
                    if failure:
                        raise failure
                # Truth is read only after every prediction for this image is finalized.
                truth = read_image(row['gt_mask']) > 0
                if truth.shape != mask.shape:
                    raise ValueError('Truth/candidate shape mismatch')
                base.append(counts(mask, truth))
                expected += len(comps)
                for v in VARIANTS_MAP:
                    answers = {c['id']: done[f"{v}:{stem}:{c['id']}"] for c in comps}
                    pred = apply_rule(mask, labels, answers)
                    save_image(output / v / 'masks' / (stem + '.png'), pred.astype('uint8') * 255)
                    totals[v].append(counts(pred, truth))
                    for c in comps:
                        reading = answers[c['id']]
                        real = int((truth & (labels == c['id'])).sum()) / c['area'] >= .5
                        labels_for_auc[v].append((real, reading))
                print(stem, len(comps), flush=True)
    except Exception as exc:
        write_json(output / 'incomplete.json', dict(status='incomplete', error=type(exc).__name__,
                    saved_answers=len(done), cost_cny=ledger_cost(output / 'budget/ledger')))
        raise
    report = dict(images=len(rows), candidates=expected, fixed=score(base), variants={},
                  cost_cny=ledger_cost(output / 'budget/ledger'),
                  auc_label='component truth purity >= 0.5; p_change, fallback to binary answer',
                  rule='delete parsed change=no only; unparsed kept')
    for v in VARIANTS_MAP:
        usable = [(real, r) for real, r in labels_for_auc[v] if r.get('answer') in ('yes', 'no')]
        prob = lambda r: r['p_yes'] if r.get('p_yes') is not None else float(r['answer'] == 'yes')
        values = [r for _, r in labels_for_auc[v]]
        report['variants'][v] = dict(score=score(totals[v]),
            auc=auc([prob(r) for real, r in usable if real], [prob(r) for real, r in usable if not real]),
            asked=len(values), parsed=len(usable),
            yes_rate=sum(r['answer'] == 'yes' for _, r in usable) / len(usable) if usable else None,
            state_yes_rate=sum(r.get('state_answer', r.get('answer')) == 'yes' for r in values) / len(values) if values else None,
            probability_available=sum(r.get('p_yes') is not None for r in values),
            response_cost_cny=sum(r.get('cost_cny', 0) for r in values))
    write_json(output / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--names', type=Path, required=True, help='Frozen WHU names_dev60.txt (dev) or names_val128.txt (report)')
    p.add_argument('--split', choices=['dev', 'report'], default='dev', help='report: score the frozen 128 report images once (Ethan 2026-11-11 12:27)')
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--max-cost-cny', type=float, required=True, help='Total for both questions, including resumes')
    a = p.parse_args()
    rows = retry_io(lambda: read_tasks(a.tasks))
    names = read_text(a.names).split()
    expected = 128 if a.split == 'report' else 60
    if len(names) != expected or len(set(names)) != expected or len(rows) != expected or {r['id'] for r in rows} != set(names):
        p.error(f'Require exactly the frozen {expected} {a.split} images')
    print(json.dumps(run(rows, a.mask_dir, a.output_dir, a.max_cost_cny, a.concurrency)))


if __name__ == '__main__':
    main()
