"""Render recorded N3 results; never infer missing stages or launch inference."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import write_json


def read(path):
    if not path.exists():return dict(status='pending',path=str(path))
    try:return dict(status='available',path=str(path),data=json.loads(path.read_text()))
    except (ValueError,OSError) as exc:return dict(status='read_error',path=str(path),error=str(exc))


def usage(root):
    successes=[];failures=[];errors=[]
    for folder,target in (('usage',successes),('failures',failures)):
        for path in sorted((root/folder).rglob('*.json')):
            item=read(path)
            if item['status']!='available':errors.append(item);continue
            data=item['data']
            # Failed attempts have separate evidence files. The mutable usage
            # slot is counted only on success, preventing last-error duplication.
            if folder=='usage' and data.get('error_type'):continue
            target.append(dict(path=str(path),cost_cny=data.get('cost_cny'),
                model=data.get('model'),usage=data.get('usage'),http_status=data.get('http_status')))
    records=successes+failures
    return dict(successful_requests=len(successes),failed_attempts=len(failures),
        recorded_cost_cny=sum(r['cost_cny'] or 0 for r in records),
        missing_cost_records=sum(r['cost_cny'] is None for r in records),records=records,read_errors=errors,
        scope='new requests only; reused historical answers are not charged again')


def collect(root):
    gates=read(root/'candidate_gates_pilot/score_summary.json')
    report=dict(views=read(root/'score_summary.json'),smoke=read(root/'score_smoke.json'),
        selected_verifier=read(root/'selected_verifier.json'),gates=gates,
        whu_official=read(root/'candidate_gates_pilot/whu_official_summary.json'),
        views_usage=usage(root),rankings={},s4={},exact_cases=[])
    for row in gates.get('data',{}).get('images',[]):
        key=row['group']+'/'+row['split'];dest=report['s4'].setdefault(key,dict(images=0,unique_tp=0,unique_fp=0))
        dest['images']+=1;dest['unique_tp']+=row['s4_unique_tp'];dest['unique_fp']+=row['s4_unique_fp']
    for path in sorted(root.glob('rank_*')):
        if not path.is_dir():continue
        report['rankings'][path.name]=dict(scores=read(path/'score_summary.json'),usage=usage(path),settings=read(path/'settings.json'))
        for answer in sorted((path/'answers').glob('*.json')):
            row=read(answer)
            if row.get('data',{}).get('reused') or row.get('data',{}).get('reused_from'):
                report['exact_cases'].append(dict(path=str(answer),record=row['data']))
    report['new_recorded_cost_cny']=report['views_usage']['recorded_cost_cny']+sum(r['usage']['recorded_cost_cny'] for r in report['rankings'].values())
    report['limitations']=['Tune/confirmation image split is exploratory and historically exposed.',
        'TP>FP candidate proxy is not a semantic truth label.',
        'Gates add-all metrics are unverified, not Agent performance.',
        'Missing stages remain pending; recorded costs exclude unreported provider usage.',
        'WHU official-v3 audit rescored unchanged historical candidates; old GT results remain separate.']
    return report


def fmt(x):return '待完成/无值' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)


def render(report):
    lines=['# 夜间补漏研究实际结果','',*['- '+s for s in report['limitations']], '',
        '新请求已记录费用（含失败证据，不重复计复用答案）：¥'+fmt(report['new_recorded_cost_cny']), '',
        '## 阶段状态','', '| 阶段 | 状态 | 证据 |','|---|---|---|']
    for key in ('views','smoke','selected_verifier','gates','whu_official'):
        item=report[key];lines.append(f"| {key} | {item['status']} | {item['path']} |")
    lines+=['','## V0–V4 完整池结果','', '| 组/划分 | 臂 | K | 完整 | F1 / P / R | Δ初始及95%CI | ΔV0及95%CI |','|---|---|---|---|---|---|---|']
    for group,variants in report['views'].get('data',{}).get('groups',{}).items():
        for variant,ks in variants.items():
            for k,v in ks.items():
                m=v.get('metrics',{});scores=' / '.join(fmt(m.get(x)) for x in ('f1','precision','recall'))
                lines.append(f"| {group} | {variant} | {k} | {v.get('complete')} | {scores} | {fmt(v.get('delta_f1'))} {v.get('paired_image_bootstrap_95_percentile')} | {fmt(v.get('delta_vs_v0'))} {v.get('paired_vs_v0_bootstrap95')} |")
    lines+=['','## S5 门控（未经核验）','', '| 组 | 划分 | 门控 | 完整 | TP / FP | TP保留 / FP保留 |','|---|---|---|---|---|---|']
    for group,splits in report['gates'].get('data',{}).get('groups',{}).items():
        for split,gates in splits.items():
            for gate,v in gates.items():lines.append(f"| {group} | {split} | {gate} | {v.get('complete')} | {v['tp']} / {v['fp']} | {fmt(v.get('tp_retention'))} / {fmt(v.get('fp_retention'))} |")
    for title,key in [('冻结核验器','selected_verifier'),('S4 相对其余全部来源独有贡献','s4'),('WHU official-v3 独立审计','whu_official'),('门控与排名结果','rankings')]:
        value=report[key]
        if key=='rankings':value={k:dict(scores=v['scores'],usage={x:y for x,y in v['usage'].items() if x!='records'}) for k,v in value.items()}
        if key in ('whu_official','selected_verifier'):value=value.get('data',value)
        if key=='whu_official' and 'groups' in value:value=dict(value,groups={k:{x:y for x,y in v.items() if x!='records'} for k,v in value['groups'].items()})
        if key=='rankings':
            for v in value.values():
                if 'data' in v['scores']:v['scores']=dict(v['scores'],data={k:x for k,x in v['scores']['data'].items() if k!='images'})
        lines+=['','## '+title,'','```json',json.dumps(value,ensure_ascii=False,indent=2),'```']
    lines+=['','逐图净 ΔF1、候选 proxy 分母、错误和 exact-case 复用证据完整保留在配套 JSON 与原始路径；缺失排名结果表示尚未完成。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Generate report through Slurm')
    report=collect(a.out);write_json(a.out/'night_report.json',report)
    dest=a.out/'night_report.md';temp=dest.with_suffix('.tmp');temp.write_text(render(report));temp.replace(dest)


if __name__=='__main__':main()
