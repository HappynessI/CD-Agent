"""Aggregate every recorded session, including all fallback/error sessions."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
from collections import Counter
from .tasks import INSTRUCTIONS, needs_init
import json
import os
from pathlib import Path
import numpy as np
from common import evaluate, load_gt, load_mask_png, names, read_jsonl, write_json

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
MODIFY = {'python', 'combine', 'morph', 'remove_region', 'add_region', 'refine_region'}


def route(calls):
    used = [c['tool'] for c in calls if c['ok']]
    mci = 'segment_everything' in used and 'compare_features' in used
    imc = used.count('segment_text') >= 2 and 'match_instances' in used
    return 'mixed' if mci and imc else 'mci_like' if mci else 'imc_like' if imc else 'other'


def summarize(directory, split, task, errors=False, gts=None):
    directory = Path(directory)
    finals = sorted(directory.glob('*/final.json'))
    if not finals:
        raise ValueError('No final sessions found')
    preds, initial, rows, diagnostics, references = {}, {}, [], [], {}
    routes = Counter()
    modification_delta = []
    python_sessions = python_success_sessions = python_calls = python_success = 0
    for path in finals:
        name = path.parent.name + '.png'
        row = json.loads(path.read_text())
        info_path = path.parent / 'input.json'
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if gts is not None:
            gt = gts[name]
            valid = np.ones_like(gt, bool)
        elif info.get('gt_mask_path'):
            from .evaluation import load_reference
            gt, valid = load_reference(info['gt_mask_path'], info.get('gt_ignore_mask_path'))
        else:
            gt = load_gt(split, name)
            valid = np.ones_like(gt, bool)
        references[name] = gt
        mask_path = row.get('mask_path')
        preds[name] = load_mask_png(path.parent / mask_path) if mask_path else np.zeros_like(gt)
        preds[name] &= valid
        calls = [c for step in read_jsonl(path.parent / 'steps.jsonl') for c in step.get('calls', [])]
        py = [c for c in calls if c['tool'] == 'python' and c.get('status') != 'skipped_due_to_submit']
        python_sessions += bool(py)
        python_success_sessions += any(c['ok'] for c in py)
        python_calls += len(py)
        python_success += sum(c['ok'] for c in py)
        routes[route(calls)] += 1
        edits = [c for c in calls if c['tool'] in MODIFY and c.get('status') != 'skipped_due_to_submit']
        deltas = [c['eval']['delta_f1'] for c in edits if c['ok'] and c.get('eval') and c['eval']['delta_f1'] is not None]
        modification_delta.append(sum(deltas))
        row['called_modification'] = bool(edits)
        rows.append(row)
        if needs_init(task):
            init_path = path.parent / 'masks' / 'mask_0.png'
            if row.get('initial_eval') is None or not init_path.exists():
                # Missing init must not be silently substituted with a generated error fallback.
                info_path = path.parent / 'input.json'
                info = json.loads(info_path.read_text()) if info_path.exists() else {}
                init_path = Path(info.get('init_mask_path') or RUN / 'init' / 'sam3_imc' / split / name)
            if init_path.exists():
                initial[name] = load_mask_png(init_path) & valid
        diagnostics.append(dict(image=name, submitted_by=row['submitted_by'],
            termination_reason=row.get('termination_reason'), steps=row.get('planning_rounds', 0),
            view_calls=sum(c['tool']=='view' for c in calls), ask_calls=sum(c['tool']=='ask' for c in calls),
            modification_calls=len(edits), initial_f1=(row.get('initial_eval') or {}).get('f1'),
            final_f1=(row.get('eval') or {}).get('f1'), fp=(row.get('eval') or {}).get('fp'),
            fn=(row.get('eval') or {}).get('fn'), tool_errors=[dict(tool=c['tool'], error_code=c.get('error_code'),
                raw_arguments=c.get('raw_arguments'), result_text=c.get('result_text')) for c in calls if not c['ok']]))
    metrics, per = evaluate(preds, split, references)
    mean = lambda values: float(np.mean(values))
    summary = dict(metrics, task_id=task, per_image_f1=per,
        mean_steps=mean([r.get('planning_rounds',0) for r in rows]),
        mean_tool_calls=mean([r.get('tool_calls',0) for r in rows]),
        mean_tokens={k: mean([r.get('tokens',{}).get(k,0) for r in rows]) for k in ('input','cached','output')},
        mean_seconds=mean([r.get('elapsed_seconds',0) for r in rows]),
        mean_cost_cny=mean([r.get('cost_cny',0) for r in rows]),
        failure_rate=mean([r['submitted_by'] != 'model' for r in rows]),
        submitted_by=dict(Counter(r['submitted_by'] for r in rows)),
        routes={k: routes[k] for k in ('mci_like','imc_like','mixed','other')},
        modification_rate=mean([r['called_modification'] for r in rows]),
        mean_modification_delta_f1=mean(modification_delta),
        python_sessions=python_sessions, python_success_sessions=python_success_sessions,
        python_calls=python_calls, python_success_calls=python_success,
        python_success_rate=python_success / python_calls if python_calls else None,
        python_session_success_rate=python_success_sessions / python_sessions if python_sessions else None)
    if needs_init(task) and len(initial) == len(preds):
        base, base_per = evaluate(initial, split, references)
        delta = {name: per[name]-base_per[name] for name in per}
        summary.update(initial=base, delta_f1_c=metrics['f1']-base['f1'],
            unchanged_submission_rate=mean([np.array_equal(preds[n],initial[n]) for n in preds]),
            improved=sum(v>1e-12 for v in delta.values()), worsened=sum(v< -1e-12 for v in delta.values()),
            unchanged=sum(abs(v)<=1e-12 for v in delta.values()))
    elif needs_init(task):
        summary.update(initial=None, delta_f1_c=None, unchanged_submission_rate=None,
            improved=None, worsened=None, unchanged=None,
            missing_initial_images=sorted(set(preds) - set(initial)),
            initial_comparison_status='unavailable: some initial masks could not be read')
    write_json(directory / 'summary.json', summary)
    if errors:
        write_json(directory / 'errors.json', diagnostics)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', choices=list(INSTRUCTIONS), required=True)
    parser.add_argument('--split', default='val', choices=['val','test'])
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--tasks', help='Manifest used for this run, for the expected session count.')
    args = parser.parse_args()
    found = len(list((RUN / args.config / args.task / args.split).glob('*/final.json')))
    from .tasks import read_tasks
    expected = len(read_tasks(args.tasks, args.task)) if args.tasks else len(names(args.split))
    if found != expected:
        parser.error(f'{found} of {expected} sessions have final.json; finish the run before summarizing.')
    print(json.dumps(summarize(RUN / args.config / args.task / args.split, args.split, args.task, args.errors), ensure_ascii=False))


if __name__ == '__main__':
    main()
