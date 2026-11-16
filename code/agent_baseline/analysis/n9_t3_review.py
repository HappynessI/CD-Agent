"""Separate Chinese six-case task-data review, not an Agent replay or approval."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from .io_retry import read_image,write_json,write_text,save_image
from .n9_t3_data import CN

REASONS={'example_area_below64':'示例少于64像素','example_area_above4096':'示例超过4096像素',
'example_hole_ratio_above0.1':'示例孔洞比例超过10%','example_bbox_fill_below0.25':'示例边框填充率低于25%',
'example_not_one_8connected_component':'示例不是单一八连通域','query_pixels_below64':'查询目标少于64像素',
'query_no_component32':'查询没有至少32像素的目标组件'}


def select_cases(old,manifests):
    # Query performance is never consulted. Prefer development scenes; fixed ID tie-breaking.
    old=[r for r in old if r['split']!='report'];selected=[];used=set();classes=set();images=set()
    def transition_key(row):
        return next(v for v in row['id'].split('_') if 'to' in v and v.split('to')[0].isdigit())
    def add(candidates,version):
        candidates=[r for r in candidates if r['id'] not in used]
        candidates.sort(key=lambda r:(Path(r['original_task']['image_t1']).name in images,transition_key(r) in classes,r['id']))
        if not candidates:raise ValueError('Insufficient actual cases for six-case review; do not fabricate')
        r=candidates[0];selected.append((version,r));used.add(r['id']);classes.add(transition_key(r));images.add(Path(r['original_task']['image_t1']).name)
    add([r for r in old if r['task_type']=='T3.1' and r['query_support']['pixels']==0],'旧版')
    bad=[r for r in old if r['task_type']=='T3.1' and any(x.startswith('example_') for x in r['reasons'])]
    add(bad,'旧版');add([r for r in old if r['task_type']=='T3.3'],'旧版')
    fresh=[dict(id=t['id'],task_type=t['task_type'],split=p,geometry=t['meta']['geometry'],query_support=t['meta']['query_support'],
                reasons=[],original_task=t) for kind in manifests for p in ('tune','confirm') for t in manifests[kind][p]]
    add([r for r in fresh if r['task_type']=='T3.1'],'新版候选')
    add([r for r in fresh if r['task_type']=='T3.3'],'新版候选')
    add([r for r in fresh if r['task_type']=='T3.1'],'新版候选')
    return selected


def overlay(rgb,mask,color):
    out=rgb.copy();inside=np.asarray(mask,bool)
    out[inside]=(out[inside]*.60+np.asarray(color)*.40).astype(np.uint8)
    edge=inside&~ndi.binary_erosion(inside)
    out[edge]=color
    return out


def example_images(task):
    if task['task_type']=='T3.1':
        return [read_image(task[k],rgb=True) for k in ('image_t1','image_t2')],read_image(task['exemplar_mask'])>0
        # Kept on query coordinates.
    ex=task['examples'][0];mask=read_image(ex['mask'])>0
    provenance=task['meta']['example_provenance'][0]
    if 'crop_box' in provenance:
        dataset=Path(task['image_t1']).parent.parent;name=provenance['name']
        images=[read_image(dataset/d/name,rgb=True) for d in ('img1','img2')]
        native=np.zeros(images[0].shape[:2],bool);x0,y0,x1,y1=provenance['crop_box'];native[y0:y1,x0:x1]=mask
        return images,native
    return [read_image(ex[k],rgb=True) for k in ('image_t1','image_t2')],mask


def render_review(old,manifests,out,review):
    selected=select_cases(old,manifests);records=[];cards=[]
    for n,(version,row) in enumerate(selected,1):
        task=row['original_task'];ident=f'N9审查{n:02d}';folder=review/f'case_{n:02d}'
        images,ex=example_images(task);query=[read_image(task[k],rgb=True) for k in ('image_t1','image_t2')]
        gt=read_image(task['gt_mask'])>0;remaining=gt&~ex if task['task_type']=='T3.1' else gt
        assets={}
        def save(name,array,title):
            path=folder/(name+'.png');save_image(path,array);assets[name]=dict(path=str(path.relative_to(review)),title=title)
        for t in range(2):
            save(f'example_t{t+1}',images[t],f'示例图 T{t+1} 原图（完整原生视野）')
            save(f'example_overlay_t{t+1}',overlay(images[t],ex,[255,45,45]),f'示例图 T{t+1} 红色支持区；孔洞不着色')
            save(f'query_t{t+1}',query[t],f'查询图 T{t+1} 原图')
            save(f'query_remaining_t{t+1}',overlay(query[t],remaining,[0,220,180]),f'查询图 T{t+1} 可检索真值（绿色，仅审核）')
        save('example_mask',ex.astype(np.uint8)*255,'示例二值掩码：白色有效，黑色不属于示例')
        save('query_gt',gt.astype(np.uint8)*255,'完整转换真值：白色目标（仅审核）')
        save('query_remaining_gt',remaining.astype(np.uint8)*255,'T3.1扣示例后目标 / T3.3完整查询目标（仅审核）')
        transition=task['meta'].get('transition_cn')
        if not transition:
            source,target=map(int,task['id'].rsplit('_',1)[-1].split('to'));transition=f'{CN[source]}→{CN[target]}'
        reason='；'.join(REASONS.get(r,r) for r in row['reasons']) or '通过固定自动几何筛选；仍待人工审核，未获正式评测批准'
        description=('同图示例：绿色查询目标已扣除红色示例E。' if task['task_type']=='T3.1' else '跨图双时相示例：示例图与查询图分别显示；查询GT不扣示例。')
        provenance=task['meta'].get('example_provenance',[])
        record=dict(review_id=ident,version=version,task_type=task['task_type'],id=task['id'],split=row['split'],
                    transition_cn=transition,geometry=row['geometry'],query_support=row['query_support'],
                    automatic_reasons=row['reasons'],explanation_cn=reason,review_status='pending_review',
                    decision_cn='待人工审核',note_cn='',source_task=task,assets=assets)
        write_json(folder/'case.json',record);records.append(record)
        g=row['geometry'];q=row['query_support'];esc=html.escape
        part_cn={'tune':'开发调参','confirm':'开发确认','report':'独立报告'}[row['split']]
        cards.append(f'<section class="card" id="case-{n}"><h2>{ident} · {version} · {task["task_type"]}</h2>'
                     f'<p>{esc(task["id"])} ｜ 查询图：{esc(Path(task["image_t1"]).name)} ｜ 划分：{part_cn}</p>'
                     f'<p><strong>目标变化：{transition}</strong>。{description}</p>'
                     f'<p>示例面积 {g["area"]} 像素；孔洞占比 {g["hole_ratio"]:.3%}；边框填充率 {g["bbox_fill"]:.3%}。'
                     f'查询剩余目标 {q["pixels"]} 像素 / {q["components"]} 连通组件，最大组件 {q["largest_component"]} 像素。</p>'
                     f'<p>自动结果：{esc(reason)}</p><p>请检查示例是否指向真实目标、孔洞异类是否误导、是否是建筑群局部或细碎模糊区域；不要仅凭类别文字判断。</p>'
                     '<div class="grid">'+''.join(f'<figure><a href="{v["path"]}"><img loading="lazy" src="{v["path"]}" alt="{v["title"]}"></a><figcaption>{v["title"]}</figcaption></figure>' for v in assets.values())+'</div>'
                     f'<p>示例来源：{esc(json.dumps(provenance,ensure_ascii=False)) if provenance else "与查询同图，原任务exemplar_mask"}</p>'
                     f'<details><summary>原始路径与标注来源（只读）</summary><pre>{esc(json.dumps(task,ensure_ascii=False,indent=2))}</pre></details>'
                     f'<label>人工意见 <select data-review="{ident}"><option>待人工审核</option><option>合理</option><option>不合理</option><option>存疑</option></select></label>'
                     f'<label>中文备注 <textarea data-note="{ident}" placeholder="说明示例指向、目标可检索性及疑问"></textarea></label></section>')
    write_json(review/'review_manifest.json',records)
    blank=[dict(审核编号=r['review_id'],任务编号=r['id'],状态='待人工审核',意见='',备注='') for r in records]
    write_json(review/'review_records.json',blank)
    js='''const key='n9-t3-review-20261116';let saved=JSON.parse(localStorage.getItem(key)||'{}');
function save(){document.querySelectorAll('[data-review]').forEach(e=>{let id=e.dataset.review;saved[id]={状态:e.value,备注:document.querySelector('[data-note="'+id+'"]').value}});localStorage.setItem(key,JSON.stringify(saved));document.getElementById('notice').textContent='已保存到当前浏览器本地；尚未上传，也未批准任何数据集。'}
document.querySelectorAll('[data-review]').forEach(e=>{let id=e.dataset.review;if(saved[id]){e.value=saved[id].状态;document.querySelector('[data-note="'+id+'"]').value=saved[id].备注||''}e.onchange=save});document.querySelectorAll('[data-note]').forEach(e=>e.oninput=save);
document.getElementById('export').onclick=()=>{save();let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}));a.download='N9_T3_人工意见.json';a.click();URL.revokeObjectURL(a.href)};'''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N9 T3 独立任务数据审核</title><style>
body{margin:0;background:#f2f5f8;color:#172b3a;font:16px/1.7 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:24px}.card,header{background:white;border:1px solid #dce4eb;border-radius:12px;padding:24px;margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{padding:8px;background:#f5f7fa}textarea{display:block;width:95%;min-height:90px}select,button{font-size:16px;padding:8px;margin:8px}pre{white-space:pre-wrap;word-break:break-all}h1,h2{line-height:1.3}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:10px}.card{padding:14px}}
</style><main><header><h1>N9 T3 任务数据审核：旧3例＋新3例</h1><p>这是任务数据审查，不是Agent回放。所有新任务均为候选，默认待人工审核；自动QC和本页六例意见都不等于整个clean数据集获批准。真值及目标类别仅供本页审核，禁止输入Agent。</p><p>前景八连通；背景四连通填洞；64≤示例面积≤4096；孔洞占比≤10%；边框填充率≥25%；查询目标≥64且有≥32像素组件。图像完整原生视野，点击可打开原图。未按模型成绩选例。</p><p>意见只保存在当前浏览器localStorage，不会写回服务器。请导出意见文件交给研究负责人；也可下载<a href="review_records.json">空白审核记录</a>填写。明确批准前不得正式confirm或online。</p><button id="export">导出本地中文审核意见</button><p id="notice">尚未提交人工批准</p></header>'''+''.join(cards)+'<script>'+js+'</script></main></html>'
    write_text(review/'index.html',page)
    write_json(out/'review_selection.json',dict(policy='Fixed development-only ordering: prefer unused query image, then unused transition, then task ID; old empty, old bad geometry, old cross-image; new two same-image and one cross-image. No model score.',
        cases=[dict(review_id=r['review_id'],id=r['id'],version=r['version']) for r in records],review_directory=str(review),status='pending_review'))
