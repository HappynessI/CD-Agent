"""Read existing tune JSON only: no image, GT, model or parameter selection."""
import argparse,json,os
from pathlib import Path
from .io_retry import read_text,write_json,write_text

BASE='baseline_R1_delta_top8'
METHODS=(BASE,'clean_delta_top5','clean_delta_top8','clean_joint_top5','clean_joint_top8')

def f1(value):
    tp,fp,fn=value['counts'];return 200*tp/max(1,2*tp+fp+fn)

def aggregate(rows,method,reference=BASE):
    counts=[sum(r['methods'][method]['counts'][i] for r in rows) for i in range(3)];tp,fp,fn=counts
    deltas=[f1(r['methods'][method])-f1(r['methods'][reference]) for r in rows]
    return dict(tasks=len(rows),images=len({r['query_image'] for r in rows}),counts=counts,
        precision=100*tp/max(1,tp+fp),recall=100*tp/max(1,tp+fn),pooled_f1=200*tp/max(1,2*tp+fp+fn),
        macro_f1=sum(f1(r['methods'][method]) for r in rows)/max(1,len(rows)),
        wins=sum(d>1e-9 for d in deltas),ties=sum(abs(d)<=1e-9 for d in deltas),losses=sum(d< -1e-9 for d in deltas),
        empty_predictions=sum(r['methods'][method]['predicted_pixels']==0 for r in rows),reference=reference)

def selection(inference,method):
    if method==BASE:indices=inference['original_delta_order'][:8];original_ids=indices
    else:
        _,rank,k=method.split('_');indices=inference['clean_orders'][rank][:int(k[3:])]
        original_ids=[inference['clean_original_ids'][i] for i in indices]
    return dict(pool_ids_zero_based=indices,original_geometry_ids_zero_based=original_ids,
        raw_sam_ids_zero_based=[inference['r1_raw_ids'][i] for i in original_ids])

def case(row,method,reference,source):
    path=source/'tasks'/row['id']/'inference.json';inference=json.loads(read_text(path))
    gt=row['gt_pixels'];original=row['original_pool_covered'];clean=row['clean_pool_covered'];tp=row['methods'][method]['counts'][0]
    pool=original if method==BASE else clean
    evidence=[]
    if gt>original:evidence.append('原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。')
    if original>clean:evidence.append('去自匹配/净区处理后池GT覆盖下降：有可量化候选支持损失，但JSON不能区分具体被剔除候选语义。')
    if pool>tp:evidence.append('候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。')
    if not evidence:evidence.append('现有覆盖统计不能解释误差；缺少逐候选语义/图像证据，无法进一步归因。')
    return dict(id=row['id'],task_type=row['task_type'],group=row['group'],query_image=row['query_image'],method=method,reference=reference,
        f1=f1(row['methods'][method]),reference_f1=f1(row['methods'][reference]),delta_f1=f1(row['methods'][method])-f1(row['methods'][reference]),
        counts=row['methods'][method]['counts'],reference_counts=row['methods'][reference]['counts'],gt_pixels=gt,
        original_pool_covered=original,clean_pool_covered=clean,original_pool_uncovered_gt=gt-original,cleaning_coverage_loss=original-clean,
        selected_pool_covered_but_not_tp=pool-tp,selected=selection(inference,method),reference_selected=selection(inference,reference),
        inference_path=str(path),metrics_path=str(path.parent/'metrics.json'),attribution=evidence)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    summary=json.loads(read_text(a.source/'summary_T31_T33.json'));rows=summary['records']
    if any(r['split']!='tune' for r in rows):raise ValueError('Diagnostics only accepts existing dev_tune records')
    table=[];examples=[]
    for kind in ('T3.1','T3.3'):
        typed=[r for r in rows if r['task_type']==kind]
        for group in ('全部','目标类','地表类'):
            chosen=[r for r in typed if group=='全部' or r['group']==group];gt=sum(r['gt_pixels'] for r in chosen)
            original=sum(r['original_pool_covered'] for r in chosen);clean=sum(r['clean_pool_covered'] for r in chosen)
            table.append(dict(task_type=kind,group=group,tasks=len(chosen),images=len({r['query_image'] for r in chosen}),gt_pixels=gt,
                original_pool_covered=original,clean_pool_covered=clean,original_pool_coverage_recall=original/max(1,gt),clean_pool_coverage_recall=clean/max(1,gt),cleaning_coverage_loss=original-clean,
                methods={m:aggregate(chosen,m) for m in METHODS},
                top8_vs_top5={rank:aggregate(chosen,f'clean_{rank}_top8',f'clean_{rank}_top5') for rank in ('delta','joint')}))
        comparisons=[(m,BASE) for m in METHODS[1:]]+[(f'clean_{rank}_top8',f'clean_{rank}_top5') for rank in ('delta','joint')]
        for method,reference in comparisons:
            ordered=sorted(typed,key=lambda r:(f1(r['methods'][method])-f1(r['methods'][reference]),r['id']))
            losses=[r for r in ordered if f1(r['methods'][method])<f1(r['methods'][reference])-1e-9][:2]
            wins=[r for r in reversed(ordered) if f1(r['methods'][method])>f1(r['methods'][reference])+1e-9][:2]
            for outcome,selected in (('loss',losses),('win',wins)):
                for row in selected:examples.append(dict(case(row,method,reference,a.source),outcome=outcome))
    write_json(a.output/'diagnostics.json',dict(status='exploratory_tune_diagnostic_only',source=str(a.source/'summary_T31_T33.json'),selection_used_for_new_parameters=False,
        table=table,examples=examples,limitations=['只复用既有JSON，不读GT/RGB/NPZ，不新建模型或参数。','例子按每项对照F1差值两端各最多2例，事后诊断选择，不是额外验证集。','任务共享图像，macro与胜负数按任务，不是独立图像显著性检验。','池覆盖不足和TopK未取用支持是像素数量证据；无逐候选真值与视觉分析时不作语义因果判断。']))
    lines=['# N9 tune 固定检索探索诊断','', '仅复用已计算JSON，无新推理、GT读取或调参；人工批准前不执行正式confirm/online。任务共享图像，任务macro/胜负不是独立图像统计检验。','', '| 类型 | 分组 | 任务/图 | 方法 | pooled F1 | macro F1 | 胜/平/负（对原Top8） | 空预测 |','|---|---|---|---|---|---|---|---|']
    for item in table:
        for method,m in item['methods'].items():lines.append(f"| {item['task_type']} | {item['group']} | {item['tasks']}/{item['images']} | {method} | {m['pooled_f1']:.5f} | {m['macro_f1']:.5f} | {m['wins']}/{m['ties']}/{m['losses']} | {m['empty_predictions']} |")
    lines+=['','## 候选池覆盖与Top8/Top5','']
    for item in table:
        lines.append(f"- {item['task_type']} {item['group']}：原/净池GT覆盖召回 {item['original_pool_coverage_recall']:.4%}/{item['clean_pool_coverage_recall']:.4%}，覆盖损失 {item['cleaning_coverage_loss']} 像素；GT总量 {item['gt_pixels']}。")
        for rank,m in item['top8_vs_top5'].items():lines.append(f"  - {rank} Top8相对Top5：任务胜/平/负 {m['wins']}/{m['ties']}/{m['losses']}。")
    lines+=['','## 典型得失（每类型/对照按F1差两端各最多2例）','']
    for e in examples:
        lines += [f"### {e['id']} · {e['method']} vs {e['reference']} · {e['outcome']}",'',f"{e['group']}；F1 {e['reference_f1']:.5f} → {e['f1']:.5f}；TP/FP/FN {e['reference_counts']} → {e['counts']}。原/净池GT覆盖 {e['original_pool_covered']}/{e['clean_pool_covered']}，原池未覆盖 {e['original_pool_uncovered_gt']}，清理覆盖损失 {e['cleaning_coverage_loss']}，当前TopK未取用的池覆盖 {e['selected_pool_covered_but_not_tp']}。",f"候选ID（0起始）：{json.dumps(e['selected'],ensure_ascii=False)}；参照：{json.dumps(e['reference_selected'],ensure_ascii=False)}。",' '.join(e['attribution']),f"证据：[inference]({e['inference_path']})；[metrics]({e['metrics_path']})。",'']
    write_text(a.output/'diagnostics.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
