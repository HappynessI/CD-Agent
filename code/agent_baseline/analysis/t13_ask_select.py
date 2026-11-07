"""CPU/API only: choose an unrestricted verifier question on the 60-image dev set.

prepare reconstructs probe_t13's score>=.22 union from its existing worker cache;
it never launches a model. answers resumes per component, shares a total budget
across questions and selects by pooled pixel F1 of delete-no. val uses one frozen
question and probability threshold, with no selection or sweep.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from common import components, save_mask_png, write_json
from agent_baseline.ask_eval import api_answer, VARIANTS, auc
from agent_baseline.planner import RunGuard, request_cost
from agent_baseline.scope_t13 import ASK_CANDIDATES, ASK_SOURCE, session_prompts
from agent_baseline.state import Session
from agent_baseline.view import ask_images
from .probe_t13 import route
from .probe_t13_common import RUN, inventories, binary_truth, proposal_union
from .second_labels import DATASET, check_encoding, load_labels
from .second_metrics import counts, score, mask
from .t12_fixed import bind, save_prediction

THRESHOLD = .22
P_THRESHOLDS = [.02, .05, .1, .2, .3, .5, .7, .9]


def cached_pair(cache, config, paths):
    """Read the DynamicBackend inventory format; a miss fails without inference."""
    root = Path(cache)/'dynamic'
    records = json.loads((root/'inventory.json').read_text())
    sources = [dict(path=str(Path(p).resolve()), size=Path(p).stat().st_size,
                    mtime=Path(p).stat().st_mtime_ns) for p in paths]
    def backend(operation, **kwargs):
        key = dict(version=1, sources=sources, config=config['dynamic'], operation=operation,
                   time=kwargs.get('time'), classes=kwargs.get('classes'))
        if key not in records:
            raise FileNotFoundError(f'Missing probe cache: {operation} {kwargs}; prepare it on GPU first')
        with np.load(root/f'item_{records.index(key):06d}.npz', allow_pickle=False) as data:
            if operation == 'segment_everything':
                h, w = map(int, data['shape'])
                return np.unpackbits(data['data'], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
            return data['data'].copy()
    return backend


def prepare(dataset, run, phase, cache, config, probe_report, output):
    names = inventories(Path(run))[phase]
    check_encoding(dataset)
    probe = json.loads(Path(probe_report).read_text())
    if (probe['phase'] != phase or not probe['complete'] or set(probe['names']) != set(names)
            or probe['backend_config'] != config
            or not any(abs(float(t)-THRESHOLD) < 1e-8 for t in probe['thresholds'])):
        raise ValueError('Require complete same-split probe_t13 report including threshold .22 and matching config')
    if phase == 'dev' and abs(probe['selected_threshold']-THRESHOLD) > 1e-8:
        raise ValueError('Development probe must have frozen threshold .22')
    output, dataset = Path(output), Path(dataset)
    provenance = dict(phase=phase, names=names, threshold=THRESHOLD, backend_config=config,
        dataset=str(dataset.resolve()), cache=str(Path(cache).resolve()), probe_report=str(Path(probe_report).resolve()))
    bind(output/'settings.json', provenance)
    component_counts = {}
    for name in names:
        path = output/'masks'/name
        if not path.exists():
            paths = [dataset/part/name for part in ('img1', 'img2')]
            with Image.open(paths[0]) as image:
                shape = (image.height, image.width)
            masks, scores = route(cached_pair(cache, config, paths), shape)
            save_prediction(path, proposal_union(masks[scores >= THRESHOLD], shape))
        component_counts[name] = len(components(mask(path))[1])
        print(name, flush=True)
    calls = sum(component_counts.values())
    complete = dict(provenance, component_counts=component_counts, calls_per_question=calls,
        estimated_cny_per_question=calls*request_cost('qwen3.7-flash',dict(input=8000,cached=0,output=128)))
    write_json(output/'complete.json', complete)
    return complete


def component_label(region, truth):
    """Positive iff >=50% of the component's pixels are change truth."""
    return int(2*int((region & truth).sum()) >= int(region.sum()))


def evaluate(images, answers, thresholds=P_THRESHOLDS):
    """Every image counts, including empty masks; unparsed answers are kept.

    p_yes missing: excluded from AUC, kept for threshold rules; never imputed.
    """
    direct, fixed = [], []
    swept = {str(t): [] for t in thresholds}
    positive, negative = [], []
    n = missing = unparsed = 0
    for name, pred, truth in images:
        labels, comps = components(pred)
        keep = np.zeros_like(pred)
        threshold_masks = {str(t): np.zeros_like(pred) for t in thresholds}
        for comp in comps:
            uid = f"{Path(name).stem}:{comp['id']}"
            answer = answers[uid]  # Missing responses must never be scored as completed.
            region = labels == comp['id']
            n += 1
            p_yes = answer.get('p_yes')
            if p_yes is None:
                missing += 1
            else:
                if not np.isfinite(p_yes) or not 0 <= p_yes <= 1:
                    raise ValueError(f'Invalid p_yes for {uid}')
                (positive if component_label(region, truth) else negative).append(p_yes)
            unparsed += answer.get('answer') not in ('yes', 'no')
            if answer.get('answer') != 'no':
                keep |= region
            for t in thresholds:
                if p_yes is None or p_yes >= t:
                    threshold_masks[str(t)] |= region
        fixed.append(counts(pred, truth))
        direct.append(counts(keep, truth))
        for t in thresholds:
            swept[str(t)].append(counts(threshold_masks[str(t)], truth))
    return dict(images=len(images), components=n, component_positive_scored=len(positive),
        component_negative_scored=len(negative), missing_p_yes=missing, unparsed=unparsed,
        auc=auc(positive, negative), fixed=score(fixed), delete_no=score(direct),
        thresholds={t: score(c) for t, c in swept.items()},
        per_image_delete_no=direct, component_label='truth overlap / component area >= 0.5',
        missing_probability='excluded from AUC; retained by probability threshold rules')


def select(reports):
    question = max(ASK_CANDIDATES, key=lambda q: reports[q]['delete_no']['f1_c'])
    thresholds = reports[question]['thresholds']
    threshold = max(thresholds, key=lambda t: (thresholds[t]['f1_c'], -float(t)))
    return dict(question=question, p_threshold=float(threshold),
        rule='highest complete-dev pooled delete-no F1; question ties use declared order; p threshold independently maximizes dev F1 with smaller-threshold tie break')


def answer_questions(dataset, run, phase, candidates, output, max_cost, settings=None, answer_fn=api_answer, concurrency=1):
    if max_cost <= 0:
        raise ValueError('Cost cap must be positive')
    names = inventories(Path(run))[phase]
    check_encoding(dataset)
    candidates, output, dataset = Path(candidates), Path(output), Path(dataset)
    source = json.loads((candidates/'complete.json').read_text())
    if (source['phase'] != phase or source['names'] != names or source['threshold'] != THRESHOLD
            or source['dataset'] != str(dataset.resolve())):
        raise ValueError('Candidates must match the frozen phase, dataset and inventory')
    frozen = json.loads(Path(settings).read_text()) if settings else None
    if phase == 'val':
        if (frozen is None or not frozen.get('complete') or frozen.get('phase') != 'dev'
                or frozen['names'] != inventories(Path(run))['dev']
                or frozen['candidates']['backend_config'] != source['backend_config']):
            raise ValueError('val requires a complete development selection with the same backend')
        questions = [frozen['selected']['question']]
    else:
        questions = list(ASK_CANDIDATES)
    policy = dict(phase=phase, names=names, candidates=source,
        masks=[dict(name=n,size=(candidates/'masks'/n).stat().st_size,
                    mtime=(candidates/'masks'/n).stat().st_mtime_ns) for n in names],
        questions={q: session_prompts(Session((1,1), target='any change', ask_variant=q)) for q in questions},
        parameters=VARIANTS['f37_corr'], source=ASK_SOURCE,
        selected=frozen['selected'] if phase == 'val' else None)
    if phase == 'val' and policy['questions'][questions[0]] != frozen['policy']['questions'][questions[0]]:
        raise ValueError('Frozen question text differs from current code')
    bind(output/'settings.json', policy)
    guard = RunGuard(output/'STOP.json', output/'budget/ledger.txt', max_cost)
    guard.check()
    guard.charge(0)  # Include spending from earlier processes before making another call.
    reports = {}
    for question in questions:
        dest = output/question
        dest.mkdir(parents=True, exist_ok=True)
        answers_path = dest/'answers.jsonl'
        done = {r['uid']: r for r in map(json.loads, answers_path.read_text().splitlines())} if answers_path.exists() else {}
        with answers_path.open('a') as file:
            for name in names:
                pred = mask(candidates/'masks'/name)
                labels, comps = components(pred)
                todo = [c for c in comps if f"{Path(name).stem}:{c['id']}" not in done]
                if not todo:
                    continue
                rgb = [np.asarray(Image.open(dataset/part/name).convert('RGB')) for part in ('img1','img2')]
                session = Session(pred.shape, target='any change', ask_variant=question)
                def one(comp):
                    guard.check()
                    uid = f"{Path(name).stem}:{comp['id']}"
                    return uid, answer_fn(ask_images(rgb, labels == comp['id'], comp['bbox'], True),
                        VARIANTS['f37_corr'], guard, sum(map(ord, uid)) % (2**31), session)
                # Requests may run in parallel; rows are written in component order by this thread only.
                with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                    for uid, result in pool.map(one, todo):
                        row = dict(uid=uid, **result)
                        file.write(json.dumps(row)+'\n')
                        file.flush()
                        done[uid] = row
                print(question, name, len(comps), flush=True)
        # Ground truth is opened only after all calls for this question have completed.
        images = [(n, mask(candidates/'masks'/n), binary_truth(*load_labels(dataset,n))) for n in names]
        thresholds = [frozen['selected']['p_threshold']] if phase == 'val' else P_THRESHOLDS
        report = evaluate(images, done, thresholds)
        report['cost_cny'] = sum(r.get('cost_cny',0) for r in done.values())
        report['question'] = policy['questions'][question]
        write_json(dest/'report.json', report)
        for name, pred, truth in images:
            labels, comps = components(pred)
            kept = [c['id'] for c in comps if done[f"{Path(name).stem}:{c['id']}"].get('answer') != 'no']
            save_mask_png(dest/'masks'/name, pred & np.isin(labels, kept))
        reports[question] = report
    result = dict(phase=phase, names=names, complete=True, candidates=source, policy=policy,
                  reports=reports, selected=frozen['selected'] if phase == 'val' else select(reports))
    # One component -> one request per question. Estimate 8k input + 128 output tokens.
    n = next(iter(reports.values()))['components']
    result['cost_estimate'] = dict(calls_per_question=n, input_tokens=8000, output_tokens=128,
        cny_per_question=n*request_cost('qwen3.7-flash',dict(input=8000,cached=0,output=128)))
    write_json(output/'selection.json' if phase == 'dev' else output/'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','answers'])
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--phase', choices=['dev','val'], default='dev')
    p.add_argument('--candidates-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--settings', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--vision-config', type=Path)
    p.add_argument('--probe-report', type=Path)
    p.add_argument('--max-cost-cny', type=float)
    p.add_argument('--concurrency', type=int, default=1)
    a = p.parse_args()
    if a.command == 'prepare':
        if not all((a.cache, a.vision_config, a.probe_report)):
            p.error('prepare requires --cache, --vision-config and --probe-report')
        prepare(a.dataset_root,a.run_root,a.phase,a.cache,json.loads(a.vision_config.read_text()),a.probe_report,a.candidates_dir)
    else:
        if not a.output_dir or a.max_cost_cny is None or a.max_cost_cny <= 0:
            p.error('answers requires --output-dir and positive --max-cost-cny')
        answer_questions(a.dataset_root,a.run_root,a.phase,a.candidates_dir,a.output_dir,a.max_cost_cny,a.settings,concurrency=a.concurrency)


if __name__ == '__main__':
    main()
