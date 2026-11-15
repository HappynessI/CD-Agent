"""SECOND SAM -> DINOv2 -> threshold. GPU inference only; no API.

4 CPUs, 32 GB RAM, one >=16 GB GPU; estimated 1-4 h per split, including
60 s/image oracle budget. Cache ~1-5 GB. See README_tools_t13.md for commands.
"""
import json
from pathlib import Path
import numpy as np
from agent_baseline.backends_t13 import WorkerPool
from agent_baseline.tools_t13 import Features, pooled, cosine
from agent_baseline.instances import from_dense
from .probe_t13_common import backend_config
from .second_labels import load_labels, pixel_counts
from .probe_t13_common import (parser, selected, pair_backend, binary_truth, metrics, proposal_union,
    coverage_upper, selection_upper, emit, shard_path, inventories, collect, fmt)

THRESHOLDS = np.linspace(0,2,101)


def route(backend,shape):
    masks = np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')],axis=0)
    rows = from_dense(masks,np.ones(len(masks)))
    features = [Features(backend('extract_features',time=t),shape,t) for t in ('t1','t2')]
    score = 1-cosine(pooled(features[0],rows),pooled(features[1],rows))
    return masks,score


def summarize(rows,thresholds,anychange_reason):
    counts = np.sum([r['counts'] for r in rows.values()],axis=0)
    scores = [metrics(c) for c in counts]
    covered = sum(r['coverage']['covered'] for r in rows.values())
    total = sum(r['coverage']['truth_pixels'] for r in rows.values())
    # Sum of per-image F1 upper * G is not a dataset F1 bound. The rigorous
    # aggregate relaxation allows perfect truth clipping across the whole split.
    result = dict(scores=[dict(threshold=float(t),**s) for t,s in zip(thresholds,scores)],
        coverage=covered/total if total else None,pixel_oracle_f1_upper=2*covered/(covered+total) if total else None,
        whole_proposal_oracles={name:r['oracle'] for name,r in rows.items()},
        whole_proposal_f1_upper_definition='Per image: maximum F1 among unions of whole proposals; MILP certified upper/lower, exact iff gap <= 1e-6. Aggregate pixel oracle permits truth clipping and is optimistic.',
        anychange=dict(reason=anychange_reason))
    if all('anychange' in r for r in rows.values()):
        result['anychange'] = metrics(np.sum([r['anychange'] for r in rows.values()],axis=0))
    return result


def markdown(report):
    lines = ['SAM 两期提议并集；DINOv2 区域均值的 1−cos；预测为 score >= 门限的整提议并集。',
        '真值：两期均非 0 且类别不同；数据集累计像素指标。',
        '覆盖率=真值与全部提议并集的交集/真值。像素 oracle F1 上限允许裁去假阳性；整提议最佳选择另报逐图 MILP 上下界，不把贪心解称为上限。',
        f"覆盖率：{fmt(report['summary']['coverage'])}%；像素 oracle F1 上限：{fmt(report['summary']['pixel_oracle_f1_upper'])}%。",'',
        '| 门限 | F1 | P | R |','|---:|---:|---:|---:|']
    for row in report['summary']['scores']:
        lines.append('| '+' | '.join([str(row['threshold']),*(fmt(row[k]) for k in ('f1','precision','recall'))])+' |')
    lines += ['', 'AnyChange: '+json.dumps(report['summary']['anychange'],ensure_ascii=False)]
    return '\n'.join(lines)


def main():
    p = parser(__doc__)
    p.add_argument('--settings',type=Path,help='Frozen complete 60-image development summary; required for val')
    p.add_argument('--merge',nargs='+',type=Path,help='Merge complete shards; no inference')
    reference = p.add_mutually_exclusive_group()
    reference.add_argument('--anychange',action='store_true',help='Include official H1 reference; failures abort')
    reference.add_argument('--anychange-skip-reason',help='Actual backend failure reason when rerunning without the reference')
    p.add_argument('--oracle-seconds',type=float,default=60)
    a = p.parse_args()
    names = selected(a)
    frozen = None
    if a.phase=='val':
        if not a.settings:
            p.error('val requires --settings from complete 60-image development merge')
        frozen = json.loads(a.settings.read_text())
        if frozen.get('phase')!='dev' or not frozen.get('complete') or frozen.get('names')!=inventories(a.run_root)['dev']:
            raise ValueError('Settings must describe the complete development inventory')
        thresholds = [frozen['selected_threshold']]
    else:
        thresholds = THRESHOLDS.tolist()
    config = backend_config(a,frozen)
    reason = a.anychange_skip_reason or '未请求 AnyChange；可用 --anychange 在 exp5-env 运行官方 H1。'
    if a.merge:
        reports,rows = collect(a.merge,a.phase,inventories(a.run_root)[a.phase])
        if any(r['thresholds']!=thresholds or r['anychange_requested']!=a.anychange or r['backend_config']!=config or r.get('anychange_skip_reason')!=a.anychange_skip_reason for r in reports):
            raise ValueError('Shard settings differ')
        names = inventories(a.run_root)[a.phase]
        complete = True
    else:
        if a.anychange and 'anychange' not in config:
            p.error('--anychange requires anychange worker config')
        pool = WorkerPool(config,a.cache)
        rows = {}
        try:
            for name in names:
                backend = pair_backend(pool,a,name)
                truth = binary_truth(*load_labels(a.dataset_root,name))
                masks,scores = route(backend,truth.shape)
                row = dict(coverage=coverage_upper(masks,truth),oracle=selection_upper(masks,truth,a.oracle_seconds),
                    counts=[pixel_counts(proposal_union(masks[scores>=t],truth.shape),truth) for t in thresholds])
                if a.anychange:
                    reference = backend('anychange')
                    row['anychange'] = pixel_counts(proposal_union(reference,truth.shape),truth)
                rows[name] = row
                print(name,flush=True)
        finally:
            pool.close()
        complete = a.shards==1
    report = dict(backend_config=config,phase=a.phase,names=names,complete=complete,thresholds=thresholds,anychange_requested=a.anychange,anychange_skip_reason=a.anychange_skip_reason,
        per_image=rows,summary=summarize(rows,thresholds,reason),selection_rule='Maximum development micro F1 on grid 0..2 step .02; tie chooses smaller threshold')
    if a.phase=='dev' and complete:
        report['selected_threshold'] = max(report['summary']['scores'],key=lambda s:(s['f1'] or 0,-s['threshold']))['threshold']
    if a.phase=='val':
        report['settings_source'] = str(a.settings)
    emit(shard_path(a,'probe_t13'),report,markdown(report))


if __name__=='__main__':
    main()
