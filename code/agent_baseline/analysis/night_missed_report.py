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
        task_selection=read(root/'selected_verifier_by_task.json'),task_confirmation=read(root/'score_verifier_by_task.json'),
        matching_validation=read(root/'rematch_equivalence_train106.json'),packed_validation=read(root/'rematch_packed_equivalence_train106.json'),
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


def scores(value):return ' / '.join(fmt(value.get(k)) for k in ('f1','precision','recall'))


def interval(value):return '['+', '.join(fmt(x) for x in value)+']' if value else '待完成/无值'


def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    if not rows:lines.append('| '+' | '.join(['待完成']+['—']*(len(headers)-1))+' |')


def pooled_baseline(rows):
    if not rows:return {}
    tp,fp,fn=[sum(r['baseline'][key] for r in rows) for key in ('tp','fp','fn')]
    return dict(f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        precision=100*tp/(tp+fp) if tp+fp else 0,recall=100*tp/(tp+fn) if tp+fn else 0)


def render(report):
    """Readable primary contrasts; full records remain in the companion JSON."""
    decision=report['selected_verifier'].get('data',{});winner=decision.get('selected_verifier')
    lines=['# 夜间补漏研究实际结果','',
        '全局核验器：'+str(winner or '待冻结')+'。主排名方法预声明为 evidence@10；任务条件核验器仅作次级探索。确认集不用于重新选择核验器、门控、排序或 K。',
        '图像级调参/确认划分存在历史曝光；置信区间为探索性图像配对 bootstrap，不能据单图结果或区间宣称已获在线 Agent 验证。',
        '', '新请求已记录费用（含失败证据，复用答案不重复收费）：¥'+fmt(report['new_recorded_cost_cny'])]
    table(lines,['阶段','记录状态','证据'],[[key,report[key]['status'],report[key]['path']] for key in
        ('views','selected_verifier','task_selection','task_confirmation','gates','whu_official')])
    lines+=['','## 全局核验器：调参选择与确认表现']
    table(lines,['臂','K10 tune 三组等权 ΔF1 vs V0'],[[v,fmt(x['mean_delta_f1'])] for v,x in decision.get('scores',{}).items()])
    view_groups=report['views'].get('data',{}).get('groups',{});view_images=report['views'].get('data',{}).get('images',[])
    rows=[]
    for key,arms in view_groups.items():
        baseline=pooled_baseline([r for r in view_images if r['group']+'/'+r['split']==key])
        for v in ('V0','V1','V2','V3','V4'):
            x=arms.get(v,{}).get('10',{});m=x.get('metrics',{})
            rows.append([key,v,x.get('complete',False),scores(baseline),scores(m),fmt(x.get('delta_f1')),interval(x.get('paired_image_bootstrap_95_percentile')),fmt(x.get('delta_vs_v0')),interval(x.get('paired_vs_v0_bootstrap95'))])
    table(lines,['组/划分','臂','完整','初始 F1/P/R','K10 F1/P/R','Δ初始','95%CI','ΔV0','95%CI'],rows)
    lines+=['','## 任务条件核验器（次级探索）','每组仅用 K10 tune 选择一个臂，所有 K 沿用；确认结果只评价该冻结策略。']
    task=report['task_selection'].get('data',{});confirmed=report['task_confirmation'].get('data',{});rows=[]
    for group,item in task.get('groups',{}).items():
        for k in ('3','5','10','20'):
            x=confirmed.get('groups',{}).get(group,{}).get('confirmation',{}).get(k,{})
            rows.append([group,item['variant'],k,x.get('complete',False),scores(x.get('metrics',{})),fmt(x.get('delta_f1')),interval(x.get('paired_image_bootstrap_95_percentile')),fmt(x.get('delta_vs_v0')),interval(x.get('paired_vs_v0_bootstrap95'))])
    table(lines,['组','冻结臂','K','确认完整','F1/P/R','Δ初始','95%CI','ΔV0','95%CI'],rows)
    lines+=['','## S5 门控和完整来源','门控 add-all 是未核验候选诊断，不是 Agent 性能；TP 保留以原 S5 TP 为分母。S4 独有指相对其余全部来源。']
    gate=report['gates'].get('data',{});rows=[]
    for group,item in gate.get('selection',{}).items():
        for split in ('tune','confirm'):
            for name in dict.fromkeys(('baseline',item['gate'])):
                x=gate['groups'][group][split][name]
                rows.append([group,split,name,x['complete'],x['tp'],x['fp'],fmt(x.get('tp_retention')),fmt(x.get('fp_retention')),scores(x.get('add_all') or {}),x.get('matching_cache_mismatch_images',0)])
    table(lines,['组','划分','门控','完整','TP','FP','TP保留','FP保留','未核验add-all F1/P/R','匹配不一致图'],rows)
    rows=[]
    for key,x in report['s4'].items():
        source=[r for r in gate.get('images',[]) if r['group']+'/'+r['split']==key]
        rows.append([key,x['images'],sum(r['all_source_tp'] for r in source),sum(r['all_source_fp'] for r in source),sum(r['remaining_fn_pixels'] for r in source),x['unique_tp'],x['unique_fp']])
    table(lines,['组/划分','图数','全来源TP','全来源FP','初始剩余FN','S4独有TP','S4独有FP'],rows)
    lines+=['','## 固定 evidence@10 与配对对照','相对旧 balanced 池使用同一核验器和 K；相对 gated balanced 只改变排名，区分门控与排名效应。balanced/area 为对照，不据确认结果替换主方法。'];rows=[]
    for name,entry in report['rankings'].items():
        for group,splits in entry['scores'].get('data',{}).get('groups',{}).items():
            for split,rankings in splits.items():
                for ranking in ('evidence','balanced','area'):
                    x=rankings.get(ranking,{}).get('10',{})
                    rows.append([name,group,split,ranking,x.get('complete',False),scores(x.get('metrics',{})),fmt(x.get('delta_vs_ungated_same_verifier')),interval(x.get('paired_image_bootstrap95')),fmt(x.get('delta_vs_gated_balanced')),interval(x.get('paired_vs_gated_balanced_bootstrap95'))])
    table(lines,['池','组','划分','排序@10','完整','F1/P/R','Δ旧pool','95%CI','Δgated balanced','95%CI'],rows)
    lines+=['','## WHU 新官方 GT 独立审计','旧候选不重生成；C1 new/demolished 的 GT、初始答案和来源都使用 canonical ignore 的同一 valid。C0 单独评分。不得与历史旧 GT 增益混称官方结果。'];rows=[]
    for group,data in report['whu_official'].get('data',{}).get('groups',{}).items():
        for source,x in data['sources'].items():rows.append([group,source,data['images'],x['tp'],x['fp'],x['unique_tp'],x['unique_fp'],f"{x['missing_pixels_covered']}/{x['missing_pixels']}",f"{x['components_covered']}/{x['missing_components']}"])
    table(lines,['组','来源','图数','TP','FP','独有TP','独有FP','剩余FN像素覆盖','剩余FN组件覆盖≥50%'],rows)
    lines+=['','## 费用、复用和失败记录'];rows=[]
    for name,x in [('views',report['views_usage'])]+[(n,v['usage']) for n,v in report['rankings'].items()]:
        rows.append([name,x['successful_requests'],x['failed_attempts'],fmt(x['recorded_cost_cny']),x['missing_cost_records'],len(x['read_errors'])])
    table(lines,['范围','成功请求','失败尝试','已记录费用','缺失费用记录','读取错误'],rows)
    lines+=['','精确 mask/同核验器复用记录数：'+str(len(report['exact_cases']))+'。逐例 UID、原始响应、实际图像路径、proxy 分母和逐图净 ΔF1 保留在 night_report.json 及对应 scores/answers。',
        '', '## 具体案例与边界',
        '- `second_disappeared:train_1322:1`：原 four 回答 yes+rebuilt，与实际 scope 明确排除重建冲突；不是必须证明没有翻修才能识别的规则错误。见 [定向人审](manual_old_disappeared_contribution.md)。',
        '- `second_appeared:train_1065:4`：52像素屋顶局部在蓝色起点覆盖下被拒绝，提示对象粒度与上下文遮蔽问题；单图不能判定总体方法。见 manual_smoke_review.json。',
        '- `levir_any:train_106:3`：岸线类候选多臂拒绝，V0亦拒绝，不能把拒绝本身计作对V0改进。候选TP>FP只是收益proxy，不是人工语义标签。',
        '', '## 性能核查与下一轮',
        'load_npz 使用 np.load 读取压缩NPZ，并对每个数组解压/取值后 copy；load_seconds 包含这些CPU工作和文件读取，不能等同磁盘等待。packed 去重试验仅3308→3295 unique；旧总约696秒与新约466秒不是受控时间对照，不能宣称去重带来30%因果加速。原作业继续运行。',
        '下一轮只在确认评估和机制审查后决定是否做真实Agent三seed对照；保留初始答案与所有来源基线。优先检验方向scope遵循、混合/过碎组件、召回截断与核验误差的区分；不再按确认分数改选策略。',
        '', '<details><summary>全部 V0–V4 × K3/5/10/20 数值</summary>']
    rows=[]
    for key,arms in view_groups.items():
        for v,ks in arms.items():
            for k,x in ks.items():rows.append([key,v,k,x.get('complete',False),scores(x.get('metrics',{})),fmt(x.get('delta_f1')),interval(x.get('paired_image_bootstrap_95_percentile')),fmt(x.get('delta_vs_v0')),interval(x.get('paired_vs_v0_bootstrap95'))])
    table(lines,['组/划分','臂','K','完整','F1/P/R','Δ初始','95%CI','ΔV0','95%CI'],rows)
    lines+=['','</details>','', '<details><summary>全部排名 K3/5/10/20 配对结果</summary>']
    rows=[]
    for name,entry in report['rankings'].items():
        for group,splits in entry['scores'].get('data',{}).get('groups',{}).items():
            for split,rankings in splits.items():
                for ranking,ks in rankings.items():
                    for k,x in ks.items():rows.append([name,group,split,ranking,k,x.get('complete',False),scores(x.get('metrics',{})),fmt(x.get('delta_vs_ungated_same_verifier')),interval(x.get('paired_image_bootstrap95')),fmt(x.get('delta_vs_gated_balanced')),interval(x.get('paired_vs_gated_balanced_bootstrap95'))])
    table(lines,['池','组','划分','排序','K','完整','F1/P/R','Δ旧pool','95%CI','Δgated balanced','95%CI'],rows)
    lines+=['','</details>','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Generate report through Slurm')
    report=collect(a.out);write_json(a.out/'night_report.json',report)
    dest=a.out/'night_report.md';temp=dest.with_suffix('.tmp');temp.write_text(render(report));temp.replace(dest)


if __name__=='__main__':main()
