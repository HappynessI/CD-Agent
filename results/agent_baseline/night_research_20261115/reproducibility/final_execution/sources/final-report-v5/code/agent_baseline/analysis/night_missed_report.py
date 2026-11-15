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
        geometry_route_tune=read(root/'geometry_route/tune/summary.json'),geometry_route_confirm=read(root/'geometry_route/confirm/summary.json'),
        geometry=read(root/'candidate_geometry_tune/summary.json'),
        matching_validation=read(root/'rematch_equivalence_train106.json'),packed_validation=read(root/'rematch_packed_equivalence_train106.json'),
        whu_official=read(root/'candidate_gates_pilot/whu_official_summary.json'),
        same_time=read(root/'same_time_counterfactual/summary.json'),
        same_time_events=[read(p) for p in sorted((root/'same_time_counterfactual/events').glob('*.json'))],
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
    control_events=[r['data'] for r in report['same_time_events'] if r['status']=='available']
    report['same_time_cost_cny']=sum(e.get('cost_cny') or 0 for e in control_events)
    report['new_recorded_cost_cny']=report['views_usage']['recorded_cost_cny']+sum(r['usage']['recorded_cost_cny'] for r in report['rankings'].values())+report['same_time_cost_cny']
    report['limitations']=['Tune/confirmation image split is exploratory and historically exposed.',
        'Historical batch99 S3/S4 thresholds were selected using all development GT (SECOND 60 per direction; LEVIR 200), including current confirmation images. Current gate/verifier tune isolation does not make the candidate pool independently held out.',
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


def recorded_sum(values):
    return sum(values) if values and all(v is not None for v in values) else None


def recorded_recall_mean(rows,values):
    # Undefined recall on a zero-FN image is excluded, not replaced by zero.
    eligible=[(r,v) for r,v in zip(rows,values) if r.get('baseline',{}).get('fn')!=0]
    if any(r.get('baseline',{}).get('fn') is None or v is None for r,v in eligible):
        return '缺失'
    return f"{fmt(sum(v for _,v in eligible)/len(eligible))} (n={len(eligible)}/{len(rows)})" if eligible else '无初始FN图 (n=0)'


def render(report):
    """Readable primary contrasts; full records remain in the companion JSON."""
    decision=report['selected_verifier'].get('data',{});winner=decision.get('selected_verifier')
    lines=['# 夜间补漏研究实际结果','',
        '全局核验器：'+str(winner or '待冻结')+'。主排名方法预声明为 evidence@10；任务条件核验器仅作次级探索。确认集不用于重新选择核验器、门控、排序或 K。',
        '历史batch99的S3/S4阈值曾使用全部开发图GT（SECOND各方向60图、LEVIR200图），包含现在的确认图。本轮gate/verifier选择隔离成立，但候选池不是全新未见确认集，不能宣称独立泛化。详见 [候选来源限制](historical_candidate_selection_limit.md)。',
        '置信区间为探索性图像配对 bootstrap，不能据单图结果或区间宣称已获在线 Agent 验证。',
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
    lines+=['','### 共同候选确认 K10：候选代理混淆',
        '直接读取已有 candidate_proxy_counts，不重新评分。TP>FP 是候选像素收益代理，不是人工语义标签；下表 TP/FP/TN/FN 是候选数量，不是像素数量。代理精确率分母为接受候选数，召回分母为全部 TP>FP 候选数（含 unknown/missing）。unknown 与 truncated 可重叠，不能相加当作不同失败。上表最终答案 union precision 包含初始答案且对重叠去重，不是接受候选的像素精度或其分布。']
    rows=[]
    for key,arms in view_groups.items():
        if not key.endswith('/confirm'):continue
        for variant in ('V0','V1','V2','V3','V4'):
            result=arms.get(variant,{}).get('10',{})
            counts=result.get('candidate_proxy_counts',{})
            denominators=result.get('candidate_proxy_denominators',{})
            rows.append([key,variant,result.get('complete',False),
                '/'.join(str(counts.get(k,'待完成')) for k in ('proxy_true_yes','proxy_false_yes','proxy_true_no','proxy_false_no')),
                denominators.get('precision_accepted_candidates','待完成'),
                denominators.get('recall_all_tp_gt_fp_candidates','待完成'),
                counts.get('proxy_negative_total','待完成'),
                '/'.join(str(counts.get(k,'待完成')) for k in ('unknown','missing','truncated'))])
    table(lines,['组/划分','臂','完整','proxy TP/FP/TN/FN','接受数（P分母）','代理正例数（R分母）','代理负例数','unknown/missing/truncated'],rows)
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
    lines+=['','### 固定池、Top10截断与核验后的像素诊断',
        '仅读取排名 score_summary.json 的逐图数值。完整候选指选定S5门控后的全来源池、排名截断前；与上方原S5门控保留表合读，不把它称为门控前召回。召回列为有初始FN图的按图算术均值（0–1），括号显示有效图数/记录数，不是汇总像素召回。其余像素列为逐图union计数之和；接受后新增TP/FP为最终答案计数减同图baseline，非逐候选累加。missing/unknown为候选响应计数。字段缺失不补零，部分记录不冒充全量。完整池到Top10反映截断，Top10到接受新增像素反映核验筛选；这些是离线GT解释，不触发新选择。']
    rows=[]
    for name,entry in report['rankings'].items():
        data=entry['scores'].get('data',{});images=data.get('images',[])
        keys=sorted({(r['group'],r['split']) for r in images})
        for group,split in keys:
            selected=[r for r in images if r['group']==group and r['split']==split]
            baseline_fn=recorded_sum([r.get('baseline',{}).get('fn') for r in selected])
            full=recorded_recall_mean(selected,[r.get('pre_rank_fn_recall') for r in selected])
            for ranking in ('evidence','balanced'):
                values=[r.get('rankings',{}).get(ranking,{}).get('10',{}) for r in selected]
                top=recorded_recall_mean(selected,[v.get('candidate_fn_recall') for v in values])
                added=[]
                for field in ('tp','fp'):
                    differences=[]
                    for r,v in zip(selected,values):
                        final=(v.get('metrics') or {}).get(field);base=r.get('baseline',{}).get(field)
                        differences.append(final-base if final is not None and base is not None else None)
                    added.append(fmt(recorded_sum(differences)))
                complete=data.get('groups',{}).get(group,{}).get(split,{}).get(ranking,{}).get('10',{}).get('complete','缺失')
                rows.append([name,group,split,ranking,len(selected),complete,fmt(baseline_fn),full,top,
                    fmt(recorded_sum([v.get('candidate_tp') for v in values])),
                    fmt(recorded_sum([v.get('candidate_fp') for v in values])),*added,
                    fmt(recorded_sum([v.get('missing') for v in values])),
                    fmt(recorded_sum([v.get('unknown') for v in values]))])
    table(lines,['池','组','划分','排序@10','记录数','完整','baseline FN像素','完整池召回均值','Top10召回均值','Top10 TP像素','Top10 FP像素','接受新增TP','接受新增FP','missing','unknown'],rows)
    lines+=['','## WHU 新官方 GT 独立审计','旧候选不重生成；C1 new/demolished 的 GT、初始答案和来源都使用 canonical ignore 的同一 valid。C0 单独评分。不得与历史旧 GT 增益混称官方结果。'];rows=[]
    for group,data in report['whu_official'].get('data',{}).get('groups',{}).items():
        for source,x in data['sources'].items():rows.append([group,source,data['images'],x['tp'],x['fp'],x['unique_tp'],x['unique_fp'],f"{x['missing_pixels_covered']}/{x['missing_pixels']}",f"{x['components_covered']}/{x['missing_components']}"])
    table(lines,['组','来源','图数','TP','FP','独有TP','独有FP','剩余FN像素覆盖','剩余FN组件覆盖≥50%'],rows)
    lines+=['','## 固定 tune 几何诊断']
    geometry_data=report['geometry'].get('data',{});candidates=[c for r in geometry_data.get('images',[]) for c in r['candidates']]
    if candidates:
        multi=sum(c['geometry']['components']>1 for c in candidates)
        lines+=['',f'全部 {len(candidates)} 个候选中多组件 {multi} 个。固定池不支持多组件假说，不新增split工具。内部洞/邻接碎片也不能自动接受。']
    rows=[]
    for key,x in geometry_data.get('groups',{}).items():
        parts=key.split('/')
        if parts[1]=='20' and parts[2] in ('internal_hole','adjacent_edge_fragment'):
            e=x['errors'];rows.append([key,x['candidates'],f"{x['candidate_union_tp']}/{x['candidate_union_fp']}",f"{x['accepted_union_tp']}/{x['accepted_union_fp']}",'/'.join(str(e[k]) for k in ('true_yes','false_yes','true_no','false_no'))])
    table(lines,['组/K/几何/臂','候选','候选union TP/FP','接受union TP/FP','proxy TP/FP/TN/FN'],rows)
    lines+=['','分层像素可能重叠，不可跨层相加；完整诊断见 [几何解释](candidate_geometry_tune/interpretation.md)。',
        '', '## 几何路由次级验证（不推广）',
        '固定规则仅将新建内部洞/邻接碎片改用已有V0响应；其余沿用任务策略，零API。tune改善未在确认集泛化，不改变冻结主方法，不继续扫描几何阈值。逐UID审查见 [路由失败审查](geometry_route/failure_review.md)。'];rows=[]
    for split,key in (('tune','geometry_route_tune'),('confirm','geometry_route_confirm')):
        for k,x in report[key].get('data',{}).get('groups',{}).get('second_appeared',{}).items():
            rows.append([split,k,x.get('complete',False),scores(x.get('metrics',{})),fmt(x.get('delta_vs_task')),interval(x.get('paired95_vs_task')),str(x.get('pixel_delta_vs_task')),x.get('routed_candidates')])
    table(lines,['划分','K','完整','F1/P/R','Δ原任务','95%CI','ΔTP/FP/FN','路由候选数'],rows)
    lines+=['','## 固定六例同日期反事实（不参与方法选择）',
        '原review_pack六UID分别用T1/T1和T2/T2，共12槽；原V0仅作自然双期对照。此控制检验时序不变性，不是自然任务F1或在线晋级证据。两期相同不要求原V0回答也相同。']
    control=report.get('same_time',{});control_data=control.get('data',{})
    responses=control_data.get('responses',[])
    events=[r['data'] for r in report.get('same_time_events',[]) if r['status']=='available']
    if control.get('status')!='available':lines+=['','状态：pending（尚无完整摘要；如已有事件，费用仍计入下方独立账目）。']
    else:lines+=['',f"预定6UID/12槽；completed={control_data.get('completed','缺失')}，attempted={control_data.get('attempted','缺失')}，error事件={sum(bool(e.get('error_type')) for e in events)}，unknown已完成回答={sum(r.get('answer') not in ('yes','no') for r in responses)}。error与未完成槽不混作no。"]
    table(lines,['组','UID','控制输入','原V0','控制回答','finish_reason'],[
        [r['uid'].split(':')[0],r['uid'],r.get('condition'),r.get('original_answer'),r.get('answer'),r.get('finish_reason')] for r in responses])
    lines+=['',f"同日期控制独立usage费用：¥{fmt(report.get('same_time_cost_cny'))}；事件数={len(events)}，缺费用字段={sum(e.get('cost_cny') is None for e in events)}。不重复并入views或rank账目。"]
    lines+=['','## 费用、复用和失败记录'];rows=[]
    for name,x in [('views',report['views_usage'])]+[(n,v['usage']) for n,v in report['rankings'].items()]:
        rows.append([name,x['successful_requests'],x['failed_attempts'],fmt(x['recorded_cost_cny']),x['missing_cost_records'],len(x['read_errors'])])
    table(lines,['范围','成功请求','失败尝试','已记录费用','缺失费用记录','读取错误'],rows)
    lines+=['','精确 mask/同核验器复用记录数：'+str(len(report['exact_cases']))+'。逐例 UID、原始响应、实际图像路径、proxy 分母和逐图净 ΔF1 保留在 night_report.json 及对应 scores/answers。',
        '', '## 具体案例与边界',
        '冻结K10策略的确认区间、逐图净变化和收益集中度见 [确认审查](frozen_confirmation_review.md)。该审查不改变任何冻结选择。',
        '- `second_disappeared:train_1322:1`：原 four 回答 yes+rebuilt，与实际 scope 明确排除重建冲突；不是必须证明没有翻修才能识别的规则错误。见 [定向人审](manual_old_disappeared_contribution.md)。',
        '- `second_appeared:train_1065:4`：52像素屋顶局部在蓝色起点覆盖下被拒绝，提示对象粒度与上下文遮蔽问题；单图不能判定总体方法。见 manual_smoke_review.json。',
        '- `levir_any:train_106:3`：岸线类候选多臂拒绝，V0亦拒绝，不能把拒绝本身计作对V0改进。候选TP>FP只是收益proxy，不是人工语义标签。',
        '', '## 性能核查与下一轮',
        'load_npz 使用 np.load 读取压缩NPZ，并对每个数组解压/取值后 copy；load_seconds 包含这些CPU工作和文件读取，不能等同磁盘等待。packed 去重试验仅3308→3295 unique；旧总约696秒与新约466秒不是受控时间对照，不能宣称去重带来30%因果加速。原186417后在3h时限超时，保留29/30断点，186610已恢复至30/30；失败日志和后续完成记录均保留。',
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
