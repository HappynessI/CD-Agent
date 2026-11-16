"""Render the actual single-session record using G3 CSS and local PNG assets."""
import json,os,re,html,shutil
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw
from agent_baseline.evaluation import load_reference
from agent_baseline.make_demo import read_steps,mask_path
from .n8_missed_prepare import load_masks,union,components
from .n8_missed_assets import ROOT,read,write

OUT=ROOT/'CD-Agent/results/agent_baseline/demo_find_missed_replay'
COLORS={'old':(44,206,120),'new':(255,193,7),'fp':(239,80,80),'fn':(69,141,255)}
def esc(x):return html.escape(str(x),quote=True)
def code(x):return '<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2) if not isinstance(x,str) else x)+'</pre>'
def metric(m,g,v):
    tp=int((m&g&v).sum());fp=int((m&~g&v).sum());fn=int((~m&g&v).sum())
    return dict(tp=tp,fp=fp,fn=fn,f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,precision=100*tp/(tp+fp) if tp+fp else 0,recall=100*tp/(tp+fn) if tp+fn else 0)
def paint(rgb,m,color,opacity=.62):
    a=np.array(rgb).copy();a[m]=(a[m]*(1-opacity)+np.array(color)*opacity).astype('uint8');return Image.fromarray(a)
def scored(rgb,m,initial,g,v):
    im=rgb.copy()
    for mask,key in [(m&g&initial&v,'old'),(m&g&~initial&v,'new'),(m&~g&v,'fp'),(~m&g&v,'fn')]:im=paint(im,mask,COLORS[key])
    return im

def main():
    assert os.environ.get('SLURM_JOB_ID')
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    finals=list((OUT/'run').glob('**/final.json'));assert len(finals)==1, 'Exactly one real online session required'
    session=finals[0].parent;final=read(finals[0]);steps=read_steps(session)
    calls=[(r,c) for r in steps for c in r.get('calls',[]) if c['tool']=='find_missed' and c['ok']]
    assert calls,'No successful real find_missed call; cannot invent replay'
    task=json.loads((OUT/'task.jsonl').read_text());runtime=read(OUT/'runtime.json')
    rgb=[Image.open(task[k]).convert('RGB') for k in ('image_t1','image_t2')]
    gt,valid=load_reference(task['gt_mask'],task.get('gt_ignore_mask'))
    initial=np.asarray(Image.open(OUT/'initial.png'))>0
    pred=np.asarray(Image.open(session/final['mask_path']))>0
    def save(name,img):
        img.save(assets/(name+'.png'));return 'assets/'+name+'.png'
    def fig(name,img,label):
        path=save(name,img)
        return f'<figure><button class="zoom" type="button" aria-label="放大 {esc(label)}"><img src="{path}" alt="{esc(label)}"></button><figcaption>{esc(label)}</figcaption></figure>'
    def stats(mask,n):return f'候选条目 {n} · 合并连通域 {len(components(mask))} · 去重像素 {int(mask.sum()):,}'
    def card(title,body):return '<section class="panel section"><h2>'+title+'</h2>'+body+'</section>'
    before=metric(initial,gt,valid);after=metric(pred,gt,valid)
    metrics=''.join(f'<div class="metric"><span>{label} · 初始 → Agent最终</span><strong>{before[k]:.2f} → {after[k]:.2f}</strong><span>变化 {after[k]-before[k]:+.2f} 个百分点</span></div>' for k,label in [('f1','F1'),('precision','Precision'),('recall','Recall')])
    overview='<div class="overview">'+fig('t1',rgb[0],'T1 · 较早时相')+fig('t2',rgb[1],'T2 · 较晚时相')+fig('initial',paint(rgb[1],initial,(190,90,220)),'初始掩码 · 真实固定路线起点')+fig('final',paint(rgb[1],pred,(255,193,7)),'Agent最终提交掩码')+fig('gt_compare',scored(rgb[1],pred,initial,gt,valid),'GT真值对照 · 新增正确像素与剩余错误')+'</div>'
    legend='<div class="legend">'+''.join(f'<span><i style="background:rgb{COLORS[k]}"></i>{name}</span>' for k,name in [('old','原有正确 TP'),('new','新增找回 TP'),('fp','误检 FP'),('fn','仍漏检 FN')])+'</div>'
    panels={};funnel_summaries=[]
    for callnum,(step,call) in enumerate(calls,1):
        folder=session/'n9_missed'/f'call_{callnum:03d}';meta=read(folder/'funnel.json');generation=read(folder/'generation.json')
        with np.load(folder/'funnel_stages.npz',allow_pickle=False) as z:stages={k:z[k].astype(bool) for k in z.files}
        masks=load_masks(folder/'funnel_candidates.npz');current=stages['current'];shape=current.shape
        raw=stages['original_all'];survivors=union([masks[i] for i in meta['surviving_indexes']],shape);top=union([masks[i] for i in meta['selected_indexes']],shape)
        rawcount=sum(len(components(m)) for k,m in stages.items() if k.startswith('source_'))
        start=paint(rgb[1],current,(190,90,220));draw=ImageDraw.Draw(start)
        for m in components(gt&~current&valid,100):
            yy,xx=np.nonzero(m);x0,x1=int(xx.min()),int(xx.max());y0,y1=int(yy.min()),int(yy.max())
            for x in range(x0,x1,10):draw.line((x,y0,min(x+5,x1),y0),fill='yellow',width=2);draw.line((x,y1,min(x+5,x1),y1),fill='yellow',width=2)
            for y in range(y0,y1,10):draw.line((x0,y,x0,min(y+5,y1)),fill='yellow',width=2);draw.line((x1,y,x1,min(y+5,y1)),fill='yellow',width=2)
        pre=metric(current,gt,valid)
        body='<p class="warning">此面板来自本次真实工具调用。虚线框及TP/FP统计仅在运行结束后用GT生成，用于讲解，未输入Agent或核验器。工具返回提议，由Agent决定采用，不自动改写主掩码。</p>'
        body+=fig(f'call{callnum}_start',start,f'输入 M_init：{int(current.sum()):,}像素；F1 {pre["f1"]:.2f}；黄虚线为事后标出的遗漏GT')
        body+='<h3>01 多源候选池 · S1–S5</h3><p>'+stats(raw,rawcount)+'</p>'
        body+=fig(f'call{callnum}_raw',paint(rgb[1],raw,(220,100,220)),'全部来源合并；原始指本工具读取的既有图像候选池，并非未经任何处理的模型输出')
        body+='<details><summary>展开五种来源</summary><div class="five">'
        for k,m in stages.items():
            if k.startswith('source_'):body+=fig(f'call{callnum}_{k}',paint(rgb[1],m,(220,100,220)),k.replace('source_','')+f' · {len(components(m))}域 / {int(m.sum()):,}像素')
        body+='</div></details><h3>02 物理硬筛 · 扣除已检区域并筛选</h3>'
        filtered=paint(paint(rgb[1],raw&~survivors,(130,130,130),.4),survivors,(40,220,210))
        removed=100*(1-survivors.sum()/max(1,raw.sum()))
        body+='<p>'+stats(survivors,len(meta['surviving_indexes']))+f'；候选覆盖像素减少 {removed:.1f}%（不是误检减少率）。</p>'
        body+=fig(f'call{callnum}_filtered',filtered,'灰：被筛掉的候选区域；青：幸存候选')
        body+='<p>实际规则：起点膨胀半径3像素后扣除；净组件≥25像素；本次新建保留全部来源；S5构造DINO差≥0.2；净候选语义差≥0.05；无全来源DINO硬阈值。没有在图示中虚构未启用的门控。</p>'
        body+='<h3>03 证据排序 · Top-K</h3><p>'+stats(top,len(meta['selected_indexes']))+'</p>'
        ranked=paint(rgb[1],top,(255,193,7));d=ImageDraw.Draw(ranked)
        table='<table><tr><th>排名</th><th>候选ID</th><th>得分</th><th>面积</th><th>来源</th></tr>'
        for rank,i in enumerate(meta['selected_indexes'],1):
            c=meta['candidates'][i];x0,y0,x1,y1=c['bbox'];d.rectangle((x0,y0,x1-1,y1-1),outline='yellow',width=2);d.text((x0,max(0,y0-12)),f'#{rank}',fill='yellow',stroke_width=1,stroke_fill='black')
            score=max(c['semantic_delta'],0)*max(c['dino'],0)*c['area']**runtime['config']['alpha']
            table+=f'<tr><td>#{rank}</td><td>{i}</td><td>{score:.4f}</td><td>{c["area"]}</td><td>{esc(", ".join(c["sources"]))}</td></tr>'
        body+=fig(f'call{callnum}_ranked',ranked,'真正送入VLM的候选编号')+table+'</table><p>排序得分 = max(语义差,0) × max(DINO差,0) × 面积^0.5；冻结K=10，候选不足时按实际数量。</p>'
        body+='<h3>04 视觉核验 · 真实输入与逐项裁决</h3><div class="candidate-grid">';accepted=[];verdicts=[]
        for rank,i in enumerate(meta['selected_indexes'],1):
            directory=folder/f'candidate_{rank:02d}';a=read(directory/'answer.json');yes=a['answer']=='yes'
            if yes:accepted.append(masks[i])
            verdicts.append(dict(rank=rank,index=i,answer=a['answer']))
            body+=f'<article class="candidate {"yes" if yes else "no"}"><h3>#{rank} · {esc(a["answer"])}</h3><div class="pair">'
            for j,label in enumerate(['T1局部轮廓','T2局部轮廓','T1宽视野轮廓','T2宽视野轮廓']):
                im=Image.open(directory/f'{j}.jpg').convert('RGB');body+=fig(f'call{callnum}_candidate{rank}_{j}',im,label+' · 实际核验输入')
            body+='</div><p>'+esc(a.get('parsed',{}).get('reason','未返回可解析理由'))+'</p>'
            local=rgb[1].crop(tuple(meta['candidates'][i]['bbox']))
            merged=paint(rgb[1],current|union(accepted,shape),(255,193,7)).crop(tuple(meta['candidates'][i]['bbox']))
            body+='<details><summary>候选局部原图 / 截至本项yes并集示意</summary><div class="pair">'+fig(f'call{callnum}_local{rank}',local,'局部原图')+fig(f'call{callnum}_localafter{rank}',merged,'yes并集示意，非Agent已执行写回')+'</div></details>'
            req=read(directory/'request.json');body+='<details><summary>实际核验请求与回复</summary>'+code(req)+code(a)+'</details></article>'
        accepted_mask=union(accepted,shape);proposal=current|accepted_mask
        body+='</div><p>'+stats(accepted_mask,len(accepted))+'；yes候选并集仅为工具建议，实际提交以Agent轨迹为准。</p><div class="pair">'
        body+=fig(f'call{callnum}_yes',scored(rgb[1],proposal,current,gt,valid),'工具yes建议并入起点后的效果（建议结果）')+fig(f'call{callnum}_actual',scored(rgb[1],pred,current,gt,valid),'Agent实际最终提交（真实结果）')+'</div>'
        panels[step['step']]=card('find_missed 漏斗机制可视化拆解',body)
        funnel_summaries.append(dict(call=callnum,raw_entries=rawcount,raw_pixels=int(raw.sum()),filtered_entries=len(meta['surviving_indexes']),filtered_pixels=int(survivors.sum()),top_entries=len(meta['selected_indexes']),top_pixels=int(top.sum()),yes_entries=len(accepted),yes_pixels=int(accepted_mask.sum()),verdicts=verdicts,pre=pre,yes_proposal=metric(proposal,gt,valid)))
    timeline='';numbered=[r for r in steps if r['step']>0]
    for row in steps:
        n=row['step'];body=f'<div class="stephead"><h2>{"输入起点" if n==0 else "Step "+str(n)}</h2><span class="badge">{esc(row.get("current_mask",""))}</span></div>'
        thought=(row.get('planner') or {}).get('thought')
        body+='<div class="thought">'+esc(thought or row.get('report') or '本步未记录Planner文字摘要。')+'</div>'
        for c in row.get('calls',[]):
            body+=f'<div class="call"><h3>{esc(c["tool"])} · {"成功" if c["ok"] else "错误"}</h3>'+code(c.get('args') or c.get('raw_arguments'))
            body+='<details><summary>实际返回 · 句柄与统计（展开）</summary>'+code(c.get('result_text',''))+'</details></div>'
        handle=row.get('current_mask')
        if handle and mask_path(session,handle).exists():
            m=np.asarray(Image.open(mask_path(session,handle)))>0
            body+='<div class="thumb">'+fig(f'step{n}',scored(rgb[1],m,initial,gt,valid),'本步当前答案与GT对照')+'</div>'
        timeline+=f'<article class="step" id="step{n}" data-step="{n}">'+body+panels.get(n,'')+'</article>'
    css='\n'.join((ROOT/'CD-Agent/code/agent_baseline/site_assets'/p).read_text() for p in ['hub.css','site.css'])
    extra='main{max-width:1600px;margin:auto;padding:24px}.overview,.five{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.section{margin:18px 0}.section>figure{max-width:620px;margin:16px auto}.section h3{margin-top:20px}.candidate-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.candidate{padding:12px;border:2px solid var(--down)}.candidate.yes{border-color:var(--up)}table{width:100%;border-collapse:collapse}td,th{padding:6px;text-align:left;border-bottom:1px solid var(--line)}.controls{position:sticky;top:0;background:var(--panel);padding:12px;z-index:5;border:1px solid var(--line)}@media(max-width:900px){.overview,.five{grid-template-columns:repeat(2,minmax(0,1fr))}.candidate-grid{grid-template-columns:1fr}}'
    js='''const steps=[...document.querySelectorAll('.step[data-step]')];let current=0;function go(i){current=Math.max(0,Math.min(steps.length-1,i));steps.forEach((e,j)=>e.classList.toggle('active',j===current));steps[current].scrollIntoView({block:'start'});document.querySelector('#position').textContent='Step '+steps[current].dataset.step;}document.querySelector('#prev').onclick=()=>go(current-1);document.querySelector('#next').onclick=()=>go(current+1);const dialog=document.querySelector('dialog');document.querySelectorAll('.zoom').forEach(b=>b.onclick=()=>{dialog.querySelector('img').src=b.querySelector('img').src;dialog.showModal();});dialog.querySelector('button').onclick=()=>dialog.close();document.querySelector('#theme').onclick=()=>{const e=document.documentElement;e.dataset.theme=e.dataset.theme==='dark'?'light':'dark';};'''
    page='<!doctype html><html lang="zh-CN" data-theme="light"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>find_missed · SECOND train_2255 真实回放</title><style>'+css+extra+'</style><header><div class="eyebrow">CD-AGENT / G3 REPLAY STYLE / SINGLE CASE</div><h1>从漏检到找回：find_missed 四级漏斗</h1><p>SECOND · train_2255 · 新建建筑 · qwen3.7-flash · 冻结build参数</p><p class="warning">按离线收益挑选的组会展示样本；本次真实在线运行1次。指令明确要求演示find_missed，不能据此证明Agent自主触发能力或总体平均收益。</p><div class="metrics">'+metrics+'</div></header><main>'+card('全局对照',overview+legend)+'<div class="controls"><button id="prev">← 上一步</button><button id="next">下一步 →</button><span id="position">完整轨迹</span><button id="theme">切换明暗</button><a href="#funnel-link">定位补漏步骤</a></div><h2>Agent完整执行轨迹</h2>'+timeline.replace(f'id="step{calls[0][0]["step"]}"',f'id="step{calls[0][0]["step"]}"')+'<span id="funnel-link"></span>'+card('运行与真实性记录',code(dict(sample='train_2255',termination=final['termination_reason'],submitted_by=final['submitted_by'],cost_cny=final['cost_cny'],calls=final['tool_calls'],rounds=final['planning_rounds'],instruction=task['instruction'],gt_role='仅用于事后评分与虚线框，不进入Agent/核验器',source_scope=meta['source_note'])))+'</main><dialog><button>关闭</button><img alt="放大图"></dialog><script>'+js+'</script></html>'
    page=page.replace('href="#funnel-link"',f'href="#step{calls[0][0]["step"]}"')
    (OUT/'index.html').write_text(page)
    # No remote requests, external scripts/fonts or filesystem-absolute image paths.
    refs=re.findall(r'<img[^>]+src="([^"]+)"',page);assert refs and all(x.startswith('assets/') and (OUT/x).is_file() for x in refs)
    for x in refs:
        with Image.open(OUT/x) as im:im.verify()
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=',page)
    summary=dict(sample='train_2255',before=before,after=after,delta_f1=after['f1']-before['f1'],funnel=funnel_summaries,steps=len(numbered),image_references=len(refs),cost_cny=final['cost_cny'],submitted_by=final['submitted_by'],final_record=str(finals[0]),status='rendered_and_static_links_validated')
    write(OUT/'page_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
