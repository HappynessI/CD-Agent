"""N8 offline report and ten-case diagnostics in the existing G3 visual style."""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from .n8_missed_assets import read,write,OUT
from .n8_missed_prepare import load_masks,union,truth,aggregate
from .n8_missed_pipeline import cost_total
from .n8_missed_score import config_id

def bootstrap(rows):
    keys=sorted({r['image_t1'] for r in rows});clusters={key:[r for r in rows if r['image_t1']==key] for key in keys}
    def point(values):
        return dict(pooled_delta=aggregate([r['metrics'] for r in values])['f1']-aggregate([r['baseline'] for r in values])['f1'],
                    macro_delta=float(np.mean([r['delta_f1'] for r in values])))
    rng=np.random.default_rng(1016);samples=[point([r for i in indexes for r in clusters[keys[i]]]) for indexes in rng.integers(0,len(keys),size=(2000,len(keys)))]
    return dict(tasks=len(rows),images=len(keys),point=point(rows),cluster_bootstrap95={k:np.percentile([s[k] for s in samples],[2.5,97.5]).tolist() for k in ('pooled_delta','macro_delta')},
        leave_one_image_out=[dict(image=key,**point([r for r in rows if r['image_t1']!=key])) for key in keys])

def choose_cases(rows):
    selected=[];seen=set()
    def add(row,reason):
        key=row['group'],row['stem']
        if key not in seen:selected.append(dict(row,selection_reason=reason));seen.add(key)
    for group,stem in (('levir_any','train_121'),('second_disappeared','train_1322')):
        for r in rows:
            if (r['group'],r['stem'])==(group,stem):add(r,'N7预声明灾难例')
    for r in sorted(rows,key=lambda r:(-r['delta_f1'],r['group'],r['stem']))[:4]:add(r,'确认ΔF1最高四例')
    for r in sorted(rows,key=lambda r:(r['delta_f1'],r['group'],r['stem']))[:4]:add(r,'确认ΔF1最低四例')
    for r in sorted(rows,key=lambda r:(-abs(r['delta_f1']),r['group'],r['stem'])):
        if len(selected)>=10:break
        add(r,'按绝对ΔF1补足十例')
    return selected[:10]

def diagnostics(a,rows):
    from agent_baseline.make_demo import CSS,CURRENT_CSS,figure,jpeg,esc
    folder=a.out/'diagnostics';folder.mkdir(exist_ok=True)
    selected=choose_cases(rows);index=[];metadata=[]
    tau=read(a.out/'frozen_semantic.json')['threshold']
    manifest={(r['group'],r['stem']):r for r in read(a.out/'fm1_manifest.json')}
    for row in selected:
        item=manifest[row['group'],row['stem']]
        with np.load(row['baseline_path'],allow_pickle=False) as z:base=z['start'].astype(bool)
        masks=load_masks(row['candidate_masks']);pred=base|union([masks[i] for i in row['accepted_indexes']],base.shape)
        gt=truth(row['gt_path'],row['group']);t1=Image.open(row['image_t1']).convert('RGB');t2=Image.open(row['image_t2']).convert('RGB')
        def overlay(mask):
            image=(np.asarray(t2).astype(float)*.22).astype(np.uint8)
            image[mask&gt]=[44,206,120];image[mask&~gt]=[239,80,80];image[~mask&gt]=[69,141,255]
            return Image.fromarray(image)
        sem=union([masks[c['index']] for c in item['candidates'] if c['semantic_delta']>=tau],base.shape)
        submitted=union([masks[i] for i in row['submitted_indexes']],base.shape)
        panels=''.join(figure(jpeg(image,limit=512,quality=75),caption) for image,caption in
            ((t1,'T1 原图'),(t2,'T2 原图'),(overlay(base),'初始答案：绿TP / 红FP / 蓝FN'),(overlay(pred),'N8答案：绿TP / 红FP / 蓝FN'),
             (overlay(base|sem),'FM1全部门控后候选并入（诊断上界池，不是输出）'),(overlay(base|submitted),'Top-K送审候选并入（核验前诊断）')))
        title=row['group']+' / '+row['stem'];filename=row['group']+'__'+row['stem']+'.html'
        detail=f"{row['selection_reason']}；F1 {row['baseline']['f1']:.4f} → {row['metrics']['f1']:.4f}；Δ {row['delta_f1']:+.4f}；新增TP/FP {row['added_tp']}/{row['added_fp']}；残差面积/块数 {row['residual_area']}/{row['residual_components']}；核验触发 {row['trigger']}；送审/接受 {len(row['submitted_indexes'])}/{len(row['accepted_indexes'])}。"
        cards=[]
        for i in row['submitted_indexes']:
            c=item['candidates'][i];uid=f"{row['group']}__{row['stem']}__n8_{i:04d}"
            answer=read(a.out/'answers'/row['config']['variant']/(uid+'.json'))
            cards.append(f"<tr><td>{i}</td><td>{esc(','.join(c['sources']))}</td><td>{c['area']}</td><td>{c['semantic_delta']:.4f}</td><td>{c['dino']:.4f}</td><td>{esc(answer['answer'])}</td><td>{esc(answer.get('parsed',{}).get('reason',''))}</td></tr>")
        body=f'<header><h1>{esc(title)}</h1><p>{esc(detail)}</p><p><a href="index.html">返回十例诊断</a> · GT仅用于离线诊断，模型不可见</p></header><main class="hub"><section class="panel"><div class="pair">{panels}</div></section><section class="panel"><h2>实际候选核验</h2><table><thead><tr><th>候选</th><th>来源</th><th>面积</th><th>ΔP</th><th>DINO</th><th>判定</th><th>理由</th></tr></thead><tbody>{"".join(cards)}</tbody></table></section></main>'
        zoom='<dialog id="zoom"><button onclick="this.parentNode.close()">关闭</button><img id="zoomimg" alt=""></dialog><script>document.querySelectorAll(".zoom").forEach(b=>b.onclick=()=>{document.getElementById("zoomimg").src=b.querySelector("img").src;document.getElementById("zoom").showModal()});</script>'
        document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+CURRENT_CSS+'</style></head><body>'+body+zoom+'</body></html>'
        (folder/filename).write_text(document)
        index.append(f'<li><a href="{filename}">{esc(title)}</a> — {esc(detail)}</li>')
        metadata.append(dict(group=row['group'],stem=row['stem'],reason=row['selection_reason'],file=filename,delta_f1=row['delta_f1']))
    (folder/'index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>N8补漏诊断</title><style>'+CSS+CURRENT_CSS+'</style></head><body><header><h1>N8补漏十例诊断</h1><p>复用G3回放配色与图像组件；不是新站点。确认评分后选例，仅用于解释，不反馈到配置选择。</p></header><main class="hub"><section class="panel"><ol>'+''.join(index)+'</ol></section></main></body></html>')
    write(a.out/'diagnostic_selection.json',metadata)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    frozen=read(a.out/'frozen_config.json');decision=read(a.out/'offline_decision.json');key=config_id(frozen)
    tune=read(a.out/'scores_tune.json');confirm=read(a.out/'scores_confirm.json');rows=confirm['cases'][key]
    write(a.out/'image_cluster_sensitivity.json',bootstrap(rows))
    lines=['# N8 find_missed：重构候选、残差触发与证据核验','',
        '完整执行FM1→FM2→FM3；新候选经真实qwen3.7-flash V0/V0_evidence核验。原始候选源重建，未以N7旧表代替。',
        f'冻结配置：{key}；语义门槛 {frozen["semantic_threshold"]}。实际usage累计费用 ¥{cost_total(a.out):.6f}。',
        f'N8 pooled与三组pooled均值晋级条件：{decision["promote_primary"]}；同时要求三组task-macro均值也提高：{decision["promote_primary_strict_both_group_definitions"]}。','',
        '| split/config | group | pooled F1/P/R | macro F1 | pooled Δ | macro Δ | added TP/FP | VLM tasks | W/L/T |',
        '|---|---|---|---|---|---|---|---|---|']
    means=[]
    for split,data in (('tune',tune),('confirm',confirm)):
        for cid,s in data['summaries'].items():
            if split=='tune' and cid!=key:continue
            for group,g in [('all',s),*s['groups'].items()]:
                m=g['pooled'];lines.append(f"| {split}/{cid} | {group} | {m['f1']:.4f}/{m['precision']:.4f}/{m['recall']:.4f} | {g['macro_f1']:.4f} | {g['pooled_delta']:+.4f} | {g['macro_delta']:+.4f} | {g['added_tp']}/{g['added_fp']} | {g['vlm_trigger_tasks']}/{g['tasks']} | {g['wins']}/{g['losses']}/{g['ties']} |")
            means.append(f"{split}/{cid}：三组pooled F1均值 {s['group_pooled_mean_f1']:.4f}（基线{s['baseline_group_pooled_mean_f1']:.4f}）；三组task-macro均值 {s['group_macro_mean_f1']:.4f}（基线{s['baseline_group_macro_mean_f1']:.4f}）。")
    lines+=['',*means]
    lines+=['','## 证据边界与成本','候选生成/特征读取在所有任务执行（find_missed生成率100%）；VLM触发率另列。残差信号在FM1之后获得，不声称提前节省候选生成成本。',
        '15任务只对应10张物理影像；cluster区间与逐图剔除在image_cluster_sensitivity.json。macro为任务等权，pooled为累计像素，指标均为百分数。',
        '历史候选设置及confirm已曝光，本轮是开发池内验证。GT仅进入标定、评分、诊断；不得称独立泛化。',
        '全部API尝试含失败分别保留usage/，request输入在inputs/，解析答案在answers/；逐配置answer-cost为反事实选中答案费用，不等于整个搜索实际支出。',
        '中间候选逐阶段TP/FP在candidate_diagnostics_tune/confirm.json；每门槛召回变化在frozen_semantic.json及fm1_tune_scores.json。',
        '复现顺序：run_n8_missed_inventory.sbatch → run_n8_missed_fill.sbatch → run_n8_missed_prepare.sbatch → run_n8_missed_verify.sbatch tune → run_n8_missed_score.sbatch tune → verify confirm → score confirm → run_n8_missed_report.sbatch。由主Agent统一exp-loop Slurm提交。']
    if not decision['promote_primary']:
        diagnostics(a,rows);lines+=['','未达到在线红线，停止在线推进。十例图像诊断见[诊断索引](diagnostics/index.html)，保留成败及预声明灾难例。']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.out/'metric.json',dict(stage='report_complete',status='complete',completed=15,cost_cny=cost_total(a.out),cumulative=True,
        primary=confirm['summaries'][key]['pooled']['f1'],gain=confirm['summaries'][key]['pooled_delta']))
    print('N8 report complete; promotion='+str(decision['promote_primary']))

if __name__=='__main__':main()
