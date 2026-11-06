"""SECOND SegEarth-OV per-class probe, annotated pixels only; GPU/no API.

4 CPUs, 24 GB RAM, >=12 GB GPU; estimate 1-3 h/split, 2-4 GB cache.
Run binary background-vs-class segmentation separately for each of six classes.
"""
import json
from pathlib import Path
import numpy as np
from agent_baseline.backends_t13 import WorkerPool
from .probe_t13_common import backend_config
from .second_labels import load_labels
from .probe_t13_common import (parser, selected, pair_backend, annotated_counts, metrics, emit,
    shard_path, inventories, collect, CLASSES, fmt)

LIMITATION = '只在本期非零标签区域算 P/R/IoU；不能据此推断未标注区域或全图质量。'


def summarize(rows):
    return {time:{text:metrics(np.sum([r[time][text] for r in rows.values()],axis=0))
                  for text in CLASSES.values()} for time in ('t1','t2')}


def recommendations(scores):
    """Predeclared development screen: both dates IoU >= .25; rank by worse date."""
    ranking = sorted(CLASSES.values(),key=lambda text:min(scores[t][text]['iou'] or 0 for t in ('t1','t2')),reverse=True)
    return [c for c in ranking if all((scores[t][c]['iou'] or 0)>=.25 for t in ('t1','t2'))]


def sam3_scores(path,names):
    saved = json.loads(path.read_text())
    if set(saved['per_image']) != set(names):
        raise ValueError('SAM3 report must cover exactly the same image inventory')
    return saved['scores']


def markdown(report):
    lines = [LIMITATION, '编码 2（N.V.G surface）的文本固定为 non-vegetated ground；其余：low vegetation、tree、water、building、playground。',
        '各类分别与 background 做二分类，概率 argmax 决定掩码；eval() 关闭上采样器 Dropout。SAM3 沿用 tile=512、pixel_thr=.85、score_thr=.5。', '',
        '| 时期 | 类别 | SegEarth P | SegEarth R | SegEarth IoU | SAM3 P | SAM3 R | SAM3 IoU |','|---|---|---:|---:|---:|---:|---:|---:|']
    for t in ('t1','t2'):
        for c in CLASSES.values():
            left = report['scores'][t][c]
            right = report['sam3_scores'][t][c] if report['sam3_scores'] is not None else {}
            values = [fmt(s.get(k)) for s in (left,right) for k in ('precision','recall','iou')]
            lines.append('| '+' | '.join([t,c,*values])+' |')
    lines += ['', 'SAM3 来源：'+report['sam3_source'],
              '开发集选择规则：两期 IoU 均至少 0.25，按较低一期排序；这是可用性筛选，非统计显著性判断。',
              '冻结建议类别：'+(', '.join(report['recommended_classes']) or '无类别通过预定筛选；查看逐类结果后再决定研究设定。')]
    return '\n'.join(lines)


def main():
    p = parser(__doc__)
    p.add_argument('--sam3-report',type=Path,help='SAM3 report for exactly the selected full split')
    p.add_argument('--settings',type=Path,help='Complete development report; required on val')
    p.add_argument('--merge',nargs='+',type=Path)
    a = p.parse_args()
    names = selected(a)
    frozen = None
    if a.phase=='val':
        if not a.settings:
            p.error('val requires --settings from complete development report')
        frozen = json.loads(a.settings.read_text())
        if frozen['phase']!='dev' or not frozen['complete'] or frozen['names']!=inventories(a.run_root)['dev']:
            raise ValueError('Incomplete/wrong development settings')
    config = backend_config(a,frozen)
    if a.merge:
        reports,rows = collect(a.merge,a.phase,inventories(a.run_root)[a.phase])
        if any(r['backend_config']!=config for r in reports):
            raise ValueError('Shard backend configurations differ')
        names,complete = inventories(a.run_root)[a.phase],True
    else:
        pool = WorkerPool(config,a.cache)
        rows = {}
        try:
            for name in names:
                labels = load_labels(a.dataset_root,name)
                backend = pair_backend(pool,a,name)
                rows[name] = {}
                for t,label in zip(('t1','t2'),labels):
                    rows[name][t] = {}
                    for cid,text in CLASSES.items():
                        prob = backend('classify',time=t,classes=[text])
                        rows[name][t][text] = annotated_counts(prob.argmax(0)==1,label,cid)
                print(name,flush=True)
        finally:
            pool.close()
        complete = a.shards==1
    scores = summarize(rows)
    comparison = sam3_scores(a.sam3_report,names) if a.sam3_report and complete else None
    if a.phase=='val' and complete and comparison is None:
        p.error('Complete val report requires --sam3-report for the parallel comparison table')
    report = dict(backend_config=config,phase=a.phase,names=names,complete=complete,per_image=rows,scores=scores,
        limitation=LIMITATION,prompts=CLASSES,sam3_scores=comparison,
        sam3_source=str(a.sam3_report) if comparison is not None else '未提供同图单的 SAM3 结果；表中留空，不伪造数值。',
        recommended_classes=frozen['recommended_classes'] if frozen else recommendations(scores) if complete else [])
    emit(shard_path(a,'probe_t12_segearth'),report,markdown(report))


if __name__=='__main__':
    main()
