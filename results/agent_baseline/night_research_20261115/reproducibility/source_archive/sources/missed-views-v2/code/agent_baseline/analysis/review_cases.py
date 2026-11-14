"""Build complete replay pages for an explicitly selected, manually reviewed case set."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw
from agent_baseline.make_demo import render, render_hub, resolve_asset
from agent_baseline.make_g3_pages import audit_html
from .coordinate_audit import write_json


def contact(directory, meta, steps, final, output):
    entries=[('T1',Path(meta['source_images'][0])),('T2',Path(meta['source_images'][1]))]
    if meta.get('gt_mask_path'):entries.append(('GT (offline)',Path(meta['gt_mask_path'])))
    seen=set()
    for row in steps:
        handles=[row.get('current_mask')]+[h for c in row.get('calls',[]) for h in c.get('output_handles',[]) if h.startswith('mask#')]
        for handle in handles:
            if handle and handle not in seen:
                path=directory/'masks'/(handle.replace('#','_')+'.png')
                if path.exists():entries.append((f's{row["step"]} {handle}',path));seen.add(handle)
    if final.get('mask_path'):entries.append(('FINAL',directory/final['mask_path']))
    cols=6;tile=180;canvas=Image.new('RGB',(cols*tile,((len(entries)+cols-1)//cols)*(tile+24)),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for i,(label,path) in enumerate(entries):
        image=Image.open(resolve_asset(directory,path)).convert('RGB');image.thumbnail((tile,tile))
        x=i%cols*tile;y=i//cols*(tile+24);canvas.paste(image,(x,y+24));draw.text((x+4,y+4),label,fill='black')
    canvas.save(output)


def build(selection,output,reviews=None):
    cases=json.loads(selection.read_text());assert len(cases)==20 and len({c['session'] for c in cases})==20
    assert set(Counter(c['category'] for c in cases).values())=={5}
    output.mkdir(parents=True,exist_ok=True);review_map={r['case_id']:r for r in json.loads(reviews.read_text())} if reviews else {}
    manifest=[];hub=[]
    for case in cases:
        cid=case['case_id'];directory=Path(case['session']);meta=json.loads((directory/'input.json').read_text());final=json.loads((directory/'final.json').read_text());steps=[json.loads(l) for l in (directory/'steps.jsonl').read_text().splitlines() if l.strip()]
        review=review_map.get(cid);page=output/f'{cid}.html';checkpoint=selection.parent/f'{cid}.build.json'
        # Initial expensive page generation is resumable; annotation reuses its cached HTML.
        base=selection.parent/f'{cid}.base.html'
        if not base.exists():
            document=render(directory,case['split'],note='按行为异常候选挑选；非随机样本，不代表总体错误率。',image_quality=55)
            base.write_text(document)
        document=base.read_text()
        trigger=case['trigger_event'].get('step');links=[trigger] if trigger is not None else []
        if review:links=review['key_steps']
        panel='<section class="panel" id="review"><h2>轨迹人工审查 · '+html.escape(cid)+'</h2>'
        panel+='<p>入选信号：'+html.escape(case['category']+' / '+case['trigger'])+'</p>'
        panel+='<p>'+''.join(f'<a href="#step-{s}">第{s}步</a>　' for s in links)+'</p>'
        if review:
            for title,key in [('判定','verdict'),('根因归属','attribution'),('完整轨迹审查','review'),('可能改进与风险','proposal')]:panel+='<h3>'+title+'</h3><p>'+html.escape(review[key])+'</p>'
        else:panel+='<p>待逐条人工复核；以下是完整原始会话回放，自动信号不构成已确认错误。</p>'
        panel+='</section>';document=document.replace('</header>','</header>'+panel,1)
        for row in steps:
            raw=(row.get('planner') or {}).get('raw') or {}
            choices=raw.get('choices',[]) if isinstance(raw,dict) else []
            requested=choices[0].get('message',{}).get('tool_calls',[]) if choices else []
            if requested and not row.get('calls'):
                anchor=f'id="step-{row["step"]}">'
                document=document.replace(anchor,anchor+'<details><summary>已请求但未执行的工具调用</summary><pre>'+html.escape(json.dumps(requested,ensure_ascii=False,indent=2))+'</pre></details>',1)
        audit_html(document)
        assert all(f'id="step-{r["step"]}"' in document for r in steps)
        page.write_text(document)
        if not (output/f'{cid}.png').exists():contact(directory,meta,steps,final,output/f'{cid}.png')
        # All rows/calls retained for review; no content-length truncation.
        packet=[]
        for row in steps:
            packet.append(dict(step=row['step'],thought=(row.get('planner') or {}).get('thought'),current=row.get('current_mask'),eval=row.get('eval'),report=row.get('report'),note=row.get('note'),
                calls=[{k:c.get(k) for k in ['tool','args','ok','error_code','result_text','current_before','current_after','input_handles','output_handles','eval','candidate_eval','set_current']} for c in row.get('calls',[])]))
        write_json(output/f'{cid}.trace.json',dict(input=meta,final=final,steps=packet))
        md=[f'# {cid} 完整轨迹',f'[交互回放]({cid}.html#review) · [审查总表](20_case_review.md)',f'![全部候选掩码]({cid}.png)']
        if review:md += [review['verdict'],review['review'],review['proposal']]
        for row in packet:
            md += [f'## 第 {row["step"]} 步',f'当前答案：{row["current"]}；离线评价：{row["eval"]}',row['thought'] or '',row['report'] or '']
            for call in row['calls']:
                md += ['<details><summary>'+html.escape(call['tool'])+' · 完整参数和返回</summary>','', '```json',json.dumps(call,ensure_ascii=False,indent=2),'```','','</details>']
        (output/f'{cid}.md').write_text('\n\n'.join(md)+'\n')
        record=dict(case,rounds=len(steps),calls=sum(len(r.get('calls',[])) for r in steps),final_f1=final.get('eval',{}).get('f1'),termination=final.get('termination_reason'),page=page.name,review=review)
        write_json(checkpoint,record);manifest.append(record)
        hub.append(dict(file=page.name+('#review' if review else ''),label=cid+' · '+case['category'],task=case['dataset']+' '+case['task'],run=case['config'],score=f'{100*record["final_f1"]:.2f}',why=review['verdict'] if review else case['trigger']+'（待审查）'))
        print(cid+' page/complete trace/contact saved',flush=True)
    (output/'index.html').write_text(render_hub(dict(title='20 条 Agent 轨迹行为审查',lead='四类各5条，跨数据集、任务和Harness版本。每页保留完整工具调用、参数、返回结果及掩码变化。',rows=hub,notes=['所有GT/F1仅用于离线审查。','目的抽样，不估计总体错误率；历史报告样例不用于新实验选参。'])).replace('四个回放','20 条完整回放'))
    write_json(output/'selection.json',manifest)
    write_json(output/'metric.json',dict(cases=len(manifest),reviewed=sum(r['review'] is not None for r in manifest),cost_cny=0))
    if reviews:
        assert len(review_map)==20
        write_json(output/'20_case_review.json',manifest)
        lines=['# 20 条完整轨迹人工审查','', '审查者：Codex逐条检查；无独立第二审阅者，一致性未做双人测量。目的抽样，类别可重叠，不据20例估计总体频率。','']
        lines += ['入选信号审查分布：'+json.dumps(dict(Counter(r['review']['trigger_assessment'] for r in manifest)),ensure_ascii=False),'', 'HTML页为自包含交互回放，可下载后直接打开；每例Markdown页可在GitHub直接查看完整步骤与掩码图。','']
        for r in manifest:
            v=r['review'];lines += [f'## {r["case_id"]} · {r["category"]}',f'- 会话：`{r["session"]}`',f'- [完整回放]({r["page"]}#review) / [GitHub可直接阅读的全步骤]({r["case_id"]}.md)；{r["rounds"]}条步骤记录/{r["calls"]}次调用；最终F1={100*r["final_f1"]:.2f}',f'- 判定：{v["verdict"]}；归属：{v["attribution"]}',f'- 关键步骤：{v["key_steps"]}',v['review'],f'改进与风险：{v["proposal"]}','']
        (output/'20_case_review.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reviews',type=Path);a=p.parse_args();build(a.selection,a.output,a.reviews)
if __name__=='__main__':main()
