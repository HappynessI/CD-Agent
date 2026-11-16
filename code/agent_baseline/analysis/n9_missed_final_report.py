"""Single-turn N9 final confirmation report, predictions and pixel-level evidence."""
import json,os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from agent_baseline.evaluation import load_reference
from .n8_missed_prepare import load_masks
from .n8_missed_assets import read,write
from .n9_missed_s5_only import OUT,cumulative_cost

LABELS={'levir_any':'LEVIR / 无方向','whu_any':'WHU / 无方向','whu_appeared':'WHU / 新建',
        'whu_disappeared':'WHU / 拆除','second_appeared':'SECOND / 新建','second_disappeared':'SECOND / 拆除'}
def ratio(n,d):return 100*n/d if d else None
def fmt(x,empty='无可补漏真值'):return empty if x is None else f'{x:.2f}'

def diagnostics(out,score):
    assets={(x['group'],x['stem']):x for x in read(out/'fm1_manifest.json')}
    records=[]
    for rows in score['cases'].values():
        for row in rows:
            dst=out/'confirm_diagnostics'/row['group']/(row['stem']+'.json')
            if dst.exists():
                prior=read(dst)
                if prior['config']!=row['config']:raise ValueError('Diagnostic checkpoint config differs')
                records.append(prior);continue
            item=assets[row['group'],row['stem']]
            gt,valid=load_reference(row['gt_path'],row.get('gt_ignore_path'))
            with np.load(item['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
            masks=load_masks(item['masks']);full=np.zeros_like(base);top=np.zeros_like(base);pred=base.copy()
            for mask in masks:full|=mask
            choices=[]
            for index in row['submitted_indexes']:
                mask=masks[index];top|=mask
                uid=f"{row['group']}__{row['stem']}__n9_{index:04d}"
                answer_path=out/'answers'/row['config']['verifier']/(uid+'.json');answer=read(answer_path)
                if index in row['accepted_indexes']:pred|=mask
                net=mask & ~base & valid
                c=item['candidates'][index]
                choices.append(dict(uid=uid,index=index,bbox=c['bbox'],sources=c['sources'],
                    semantic_delta=c['semantic_delta'],dino=c['dino'],area=c['area'],
                    new_tp_pixels=int((net&gt).sum()),new_fp_pixels=int((net&~gt).sum()),
                    answer=answer['answer'],finish_reason=answer.get('finish_reason'),
                    reason=answer.get('parsed',{}).get('reason'),answer_path=str(answer_path),
                    request_path=str(out/'inputs'/row['config']['verifier']/uid/'request.json')))
            prediction=out/'confirm_predictions'/row['group']/(row['stem']+'.png');prediction.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pred.astype('uint8')*255).save(prediction)
            missed=gt & ~base & valid
            record=dict(group=row['group'],stem=row['stem'],config=row['config'],baseline=row['baseline'],after=row['after'],
                delta_f1=row['delta_f1'],missing_gt_pixels=int(missed.sum()),
                all_source_recalled_pixels=int((full&missed).sum()),topk_recalled_pixels=int((top&missed).sum()),
                added_tp=row['added_tp'],added_fp=row['added_fp'],choices=choices,
                accepted_all_fp_candidates=sum(c['answer']=='yes' and c['new_tp_pixels']==0 and c['new_fp_pixels']>0 for c in choices),
                invalid_candidates=sum(c['answer'] not in ('yes','no') for c in choices),
                image_t1=row['image_t1'],image_t2=row['image_t2'],gt_path=row['gt_path'],
                gt_ignore_path=row.get('gt_ignore_path'),prediction_path=str(prediction),
                labels_role='offline evaluation and diagnosis only; not sent to verifier')
            write(dst,record);records.append(record)
    return records

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    out=OUT;frozen=read(out/'frozen_profiles.json');score=read(out/'scores_confirm.json')
    lines={}
    for value in score['summaries'].values():
        for group,result in value['lines'].items():
            if group in lines:raise ValueError('Exactly one frozen configuration per confirmation line')
            lines[group]=result
    assert len(lines)==6 and all(r['tasks']==30 for r in lines.values())
    diag=diagnostics(out,score);sens=read(out/'image_sensitivity.json')
    actual=cumulative_cost(out);answer_counts=Counter();finish_counts=Counter()
    for path in (out/'answers').glob('*/*.json'):
        r=read(path);answer_counts[r.get('answer','unknown')]+=1;finish_counts[r.get('finish_reason','unknown')]+=1
    transport=Counter();new_calls=0;new_cost=0
    for p in (out/'usage').glob('*.json'):
        r=read(p);new_calls+=1;new_cost+=r.get('cost_cny',0)
        if r.get('error_type'):transport[str(r.get('http_status'))+'/'+r['error_type']]+=1
    diagnostics_by_line={};failures={}
    for group in sorted(lines):
        cases=[r for r in diag if r['group']==group]
        missing=sum(r['missing_gt_pixels'] for r in cases);pool=sum(r['all_source_recalled_pixels'] for r in cases);top=sum(r['topk_recalled_pixels'] for r in cases)
        diagnostics_by_line[group]=dict(missing_gt_pixels=missing,pool_recalled_pixels=pool,topk_recalled_pixels=top,
            pool_recall_percent=ratio(pool,missing),topk_recall_percent=ratio(top,missing),
            added_precision_percent=ratio(lines[group]['added_tp'],lines[group]['added_tp']+lines[group]['added_fp']),
            accepted_all_fp_candidates=sum(r['accepted_all_fp_candidates'] for r in cases),
            invalid_candidates=sum(r['invalid_candidates'] for r in cases),
            fp_increase_tasks=sum(r['added_fp']>0 for r in cases),
            pure_fp_addition_tasks=sum(r['added_fp']>0 and r['added_tp']==0 for r in cases),
            calls=lines[group]['calls'],invalid_rate=lines[group]['invalid_calls']/max(1,lines[group]['calls']))
        losses=sorted([r for r in cases if r['delta_f1']<0 or (r['added_fp']>0 and r['added_tp']==0)],key=lambda r:(r['delta_f1'],-r['added_fp'],r['stem']))[:3]
        failures[group]=[dict(stem=r['stem'],delta_f1=r['delta_f1'],added_tp=r['added_tp'],added_fp=r['added_fp'],
            case_path=str(out/'confirm_diagnostics'/group/(r['stem']+'.json')),
            accepted_all_fp_candidates=r['accepted_all_fp_candidates']) for r in losses]
    summary=dict(status='single_turn_tune_and_confirm_complete',profiles=frozen['profiles'],confirm_lines=lines,
        candidate_diagnostics=diagnostics_by_line,failures=failures,
        cost=dict(parent_and_revision_actual_cny=actual,revision_new_calls=new_calls,revision_new_actual_cny=new_cost,
            tune=read(out/'verify_tune.json'),confirm=read(out/'verify_confirm.json')),
        answers=dict(counts=dict(answer_counts),finish_reasons=dict(finish_counts),transport_errors=dict(transport)),
        online_agent_runs=0,formal_test_used=False,new_pending_review_data_used=False,
        limitation='Historically exposed development data; frozen tune/confirm membership; no online-Agent efficacy claim',
        recommended_next_lines=[g for g,r in lines.items() if r['delta_f1']>0 and sens[g]['pooled_delta_ci'][0]>0],
        offline_positive_lines=[g for g,r in lines.items() if r['delta_f1']>0])
    write(out/'final_summary.json',summary)
    text=['# find_missed 分任务单轮核验：最终开发集结果','',
        '状态：三轨道调参、冻结和独立确认完成。只运行单轮 qwen3.7-flash 核验；未运行多轮在线 Agent，未使用报告/测试集或待审核新任务。',
        '每个任务行 tune30 / confirm30；各任务行独立累计像素，禁止跨数据集或跨任务pooled F1。开发图有历史实验暴露，不能声称全新盲测集。',
        '起点为已冻结固定路线经过历史f37_corr核验规则过滤的start掩码，不是在线Agent的最终答案。补漏按单轮核验yes追加候选，GT仅用于事后评分。','',
        '## 冻结参数','', '| 任务档 | 来源 | 语义门槛 | 净候选DINO硬筛 | 面积指数 | 核验器 | K |','|---|---|---:|---|---:|---|---:|']
    for profile in ('any','build','demolish'):
        c=frozen['profiles'][profile]['config'];sources='S1+S2+S3+S4+S5' if c['sources']=='all' else 'S1+S2+S5'
        text.append(f"| {profile} | {sources} | {c['semantic']} | {c['dino'] if c['dino'] is not None else '无'} | {c['alpha']} | {c['verifier']} | {c['k']} |")
    text+=['','S5构造DINO≥0.2仍保留；build/demolish只取消额外的全来源净候选DINO硬筛。按合法任务方向选择参数，不读取数据集名或GT。排序公式、候选净区域与图像证据保持不变。',
        '参数仅按各profile的两条tune任务行ΔF1等权平均选择；最终确认与后续建议逐行独立判断，不用一条任务的收益抵消另一条的退化。','',
        '## 冻结确认：F1与新增像素','', '| 确认任务行 | 图数 | 基线F1 | 补漏F1 | ΔF1 | 新增TP | 新增FP | 胜/负/平 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for g,r in sorted(lines.items()):text.append(f"| {LABELS[g]} | {r['tasks']} | {r['baseline']['f1']:.4f} | {r['after']['f1']:.4f} | {r['delta_f1']:+.4f} | {r['added_tp']} | {r['added_fp']} | {r['wins']}/{r['losses']}/{r['ties']} |")
    text+=['','| 确认任务行 | 基线P/R | 补漏P/R | 宏平均ΔF1 | ΔF1的95%图像bootstrap区间 |','|---|---|---|---:|---|']
    for g,r in sorted(lines.items()):
        ci=sens[g]['pooled_delta_ci'];text.append(f"| {LABELS[g]} | {r['baseline']['precision']:.4f}/{r['baseline']['recall']:.4f} | {r['after']['precision']:.4f}/{r['after']['recall']:.4f} | {r['macro_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    text+=['','## 候选召回与核验诊断','',
        '召回分母为初始预测遗漏的有效GT像素；候选池召回是上界，不是工具实际效果。核验后实际效果见上表。纯FP候选指接受候选中新增有效TP=0且FP>0，不对含部分TP的混合候选武断判错。','',
        '| 确认任务行 | 全候选池召回% | Top-K召回% | 新增像素精确率% | 接受纯FP候选 | 核验调用 | 无效回复 |','|---|---:|---:|---:|---:|---:|---:|']
    for g,r in diagnostics_by_line.items():text.append(f"| {LABELS[g]} | {fmt(r['pool_recall_percent'])} | {fmt(r['topk_recall_percent'])} | {fmt(r['added_precision_percent'],'无新增有效像素')} | {r['accepted_all_fp_candidates']} | {r['calls']} | {r['invalid_candidates']} |")
    text+=['','空GT图的单图F1可能在新增误检后仍为0，不能用胜负平或宏F1掩盖误检增加。下面另列空图新增FP及纯FP增量任务数。','',
        '| 确认任务行 | 空GT图数 | 空图新增FP像素 | 新增FP图数 | 仅新增FP无新增TP图数 |','|---|---:|---:|---:|---:|']
    for g,r in sorted(lines.items()):
        d=diagnostics_by_line[g];text.append(f"| {LABELS[g]} | {r['empty_gt_tasks']} | {r['empty_gt_added_fp']} | {d['fp_increase_tasks']} | {d['pure_fp_addition_tasks']} |")
    text+=['','## 可核查失败样本','']
    for g,rows in failures.items():
        text.append('### '+LABELS[g]);text.append('')
        if not rows:text.append('本确认行没有F1下降或纯FP增量样本；不代表具有显著提升。')
        for r in rows:text.append(f"- {r['stem']}：ΔF1 {r['delta_f1']:+.4f}，新增TP/FP {r['added_tp']}/{r['added_fp']}；证据：`confirm_diagnostics/{g}/{r['stem']}.json`（含候选ID、框、来源、请求/答案路径与最终预测）。")
        text.append('')
    text+=['## 成本、失败处理与复现','',f"修正版新增请求 {new_calls} 次，实际¥{new_cost:.6f}；含此前原版核验的全程累计¥{actual:.6f}。精确复用答案不重复计费。", 
        '截断导致解析/决策无效时保留unknown并弃权，不添加候选；有效yes/no仍按原值处理；不按残缺yes/no前缀猜答案，不无限重试。原始回复、HTTP错误状态及usage均保留。',
        '复现：`run_n9_missed_s5_only.sbatch tune` → 冻结 → `run_n9_missed_s5_only.sbatch confirm` → `python -m agent_baseline.analysis.n9_missed_final_report`；均通过Slurm/exp-loop，单轮API使用workspace CSV端点和gpu42直连。',
        '完整参数、每图结果、原始请求、答案、计费与最终掩码保留在本目录；原有N7/N8和额外DINO门槛旧版敏感性结果不覆盖。','',
        '## 下一步边界','',
        '只有确认行自身ΔF1为正才列为后续候选；区间跨零的微小提升不称稳定收益。退化任务行不晋级，不用其他任务行收益抵消；部署前需基于合法可观测状态解决差异，不能根据数据集名称开关工具。',
        '本次授权仅完成单轮核验及离线实验，在线Agent收益尚未验证，不能由离线Oracle或固定候选结果代替。下一轮可针对有稳定证据的任务行提出独立在线配对方案，未提交在线作业。','']
    (out/'final_report.md').write_text('\n'.join(text))
    print(json.dumps(dict(status=summary['status'],confirm_tasks=sum(r['tasks'] for r in lines.values()),cost_cny=actual,online_agent_runs=0)))
if __name__=='__main__':main()
