"""Static Chinese audit cards using relative image assets; no review backend."""
import html,json,os,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/agent_baseline/task_reconstruct_audit'
ASSETS=OUT/'assets'
BLUE=(60,130,245);CYAN=(15,220,225);GREEN=(70,230,110);ORANGE=(230,140,60);YELLOW=(255,220,40);RED=(245,65,65)
def read(path):return json.loads(Path(path).read_text())
def rgb(path):
    with Image.open(path) as im:return im.convert('RGB')
def mask(path):
    with Image.open(path) as im:return np.asarray(im)>0
def overlay(image, layers):
    a=np.asarray(image).astype(float).copy()
    for m,c,alpha in layers:a[m]=a[m]*(1-alpha)+np.asarray(c)*alpha
    return Image.fromarray(a.clip(0,255).astype('uint8'))
def marked(image,box,point=None):
    result=image.copy();d=ImageDraw.Draw(result);x0,y0,x1,y1=box
    x1-=1;y1-=1
    for x in range(x0,x1,12):
        d.line((x,y0,min(x+6,x1),y0),fill=YELLOW,width=3);d.line((x,y1,min(x+6,x1),y1),fill=YELLOW,width=3)
    for y in range(y0,y1,12):
        d.line((x0,y,x0,min(y+6,y1)),fill=YELLOW,width=3);d.line((x1,y,x1,min(y+6,y1)),fill=YELLOW,width=3)
    if point:
        x,y=point;d.ellipse((x-5,y-5,x+5,y+5),fill=RED,outline='white',width=2)
    return result
def extent(box,shape,padding=24):
    return (max(0,box[0]-padding),max(0,box[1]-padding),min(shape[0],box[2]+padding),min(shape[1],box[3]+padding))
def save(image,name,title):
    target=ASSETS/(name+'.png');image.save(target)
    return f'<figure><a href="assets/{target.name}" target="_blank"><img src="assets/{target.name}" alt="{html.escape(title)}" loading="lazy"></a><figcaption>{html.escape(title)}</figcaption></figure>'
def grid(images,names,prefix,cls=''):
    return '<div class="grid '+cls+'">'+''.join(save(im,f'{prefix}_{i}',title) for i,(im,title) in enumerate(zip(images,names)))+'</div>'
def details(pairs):
    return '<dl>'+''.join(f'<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>' for k,v in pairs)+'</dl>'
def controls(ident):
    return '<div class="review"><label>审查判断 <select data-id="'+ident+'"><option value="pending_review">待审核</option><option value="reasonable">合理</option><option value="unreasonable">不合理</option><option value="uncertain">存疑</option></select></label><textarea data-id="'+ident+'" placeholder="请记录具体问题或修改意见"></textarea></div>'
def card(row,body):
    ident=row['id'];row.setdefault('image_id',row.get('stem',''));return f'<article id="{ident}"><h2>{html.escape(row["dataset"])} · {html.escape(row["image_id"])}</h2><p class="badge">'+('不满足完整性 · 仅疑义诊断，不计合格样本' if row.get('diagnostic_only') else '候选数据 · 待人工审核')+'</p>'+('<p class="notice">'+html.escape(row.get('qc_issue',''))+'</p>' if row.get('qc_issue') else '')+body+controls(ident)+'</article>'
def t5(row):
    a,b=rgb(row['image_t1']),rgb(row['image_t2']);pred=mask(row['m_init']);gt=mask(row['gt_mask']);region=mask(row['lesion_mask'])
    valid=~mask(row['gt_ignore_mask']) if row.get('gt_ignore_mask') else np.ones_like(gt)
    box=row['bbox'];point=row['point'];base=b if row['direction']=='appeared' else a
    init=marked(overlay(base,[(pred,BLUE,.55)]),box,point)
    truth=overlay(base,[(pred&gt&valid,GREEN,.55),(region&gt&valid,YELLOW,.7),(pred&~gt&valid,RED,.6)])
    truth=marked(truth,box)
    images=[marked(a,box),marked(b,box),init,truth]
    body=grid(images,['前期影像 T1 · 目标框','后期影像 T2 · 同一目标框','真实初始掩码：蓝；反馈点：红','已对：绿；病灶内GT：黄；误检：红'],row['id']+'_full')
    crop=extent(box,a.size)
    local=[im.crop(crop).resize((int((crop[2]-crop[0])*1.8),int((crop[3]-crop[1])*1.8)),Image.Resampling.NEAREST) for im in images]
    body+=grid(local,['局部 T1 · 1.8倍','局部 T2 · 1.8倍','局部初始掩码与反馈 · 1.8倍','局部真值对照 · 1.8倍'],row['id']+'_local')
    body+=details([('任务','同一区块的 T5.2 点框反馈 / T5.3 单句反馈'),('方向',{'appeared':'新建','disappeared':'拆除'}.get(row['direction'],row['direction'])),('真实弱预测方法','未核验 SAM3 建筑跨期固定预测：像素阈值0.85、实例阈值0.5、分块512、重叠0.25；没有人工扰动GT'),('病灶类型',{'FN-Build':'新建漏检','FN-Demolish':'拆除漏检','FP':'误检','Coarse-Merge':'粗边界粘连'}[row['lesion_type']]),('病灶 / 实际错误像素',f"{row['lesion_pixels']} / {row['error_pixels']}"),('初始整图 F1 / 局部理想修复后 F1',f"{row['initial_f1']:.2f}% / {row['ideal_local_f1']:.2f}%（后者仅离线诊断）"),('T5.2 坐标参数',f"框 {box}；点 {point}；操作：{row['operation']}；像素坐标，框右下边界不含"),('T5.3 自然语言反馈',row['feedback_text']),('预测来源',row['source'].get('prediction') or row['source'].get('cache_root','见候选元数据'))])
    return card(row,body)
def t44(row):
    a,b=rgb(row['image_t1']),rgb(row['image_t2']);paths=row['evaluation']
    anchor=mask(paths['anchor_mask']);buffer=mask(paths['buffer_mask']);target=mask(paths['target_mask']);other=mask(paths['distractor_mask'])
    body=grid([a,b,overlay(b,[(buffer,YELLOW,.25),(anchor,CYAN,.65)]),overlay(b,[(other,ORANGE,.65),(target,GREEN,.7)])],['前期影像 T1','后期影像 T2','参照物：青；固定缓冲区：淡黄','命中整实例：绿；排除同类干扰：橙'],row['id'])
    body+=details([('任务','T4.4 × C0'),('纯净查询指令',row['instruction']),('参照物',row['anchor_class_cn']),('空间规则','参照物32像素欧氏缓冲区；建筑重叠比例≥10%时整实例保留'),('命中实例 / 像素',f"{row['target_instances']} / {row['target_pixels']}"),('排除干扰实例 / 像素',f"{row['distractor_instances']} / {row['distractor_pixels']}"),('填洞新增像素',row['filled_pixels']),('边界情况','命中目标均不触边；部分排除干扰位于图像边界，仍完整保留其原图可见区域，未删减GT' if row.get('all_gt_touches_border') else '命中与排除的建筑变化实例均不触边'),('参照来源','两期 SegEarth 同类最高概率且均≥0.55；仍需目视确认参照物真实存在、含义唯一')])
    return card(row,body)
def t3(row):
    a,b=rgb(row['image_t1']),rgb(row['image_t2']);e=mask(row['example_mask']);full=mask(row['gt_full']);rem=mask(row['gt_rem']);box=row['bbox']
    base=b if row['example_time']=='t2' else a
    body=grid([marked(a,box),marked(b,box),overlay(base,[(e,CYAN,.7)]),overlay(base,[(rem,GREEN,.65),(e,CYAN,.7)])],['前期影像 T1 · 示例框','后期影像 T2 · 示例框','单栋示例：青','完整同类真值：示例青，其余绿'],row['id']+'_full')
    instances=row['instances'];example=next(x for x in instances if x['is_example']);others=[next(x for x in instances if x['id']==ident) for ident in row['sample_instance_ids']]
    views=[];titles=[]
    for i,obj in enumerate([example]+others):
        crop=extent(obj['bbox'],a.size,12)
        for t,image in [('T1',a),('T2',b)]:
            view=image.crop(crop);view=view.resize((int(view.width*1.8),int(view.height*1.8)),Image.Resampling.BICUBIC)
            views.append(view);titles.append(('示例' if i==0 else f'同款目标 {i}')+' · '+t)
    body+=grid(views,titles,row['id']+'_local','six')
    body+=details([('任务','T3.1 同图视觉示例；仅候选'),('类别与转换方向',{'building':'建筑','playground':'运动场'}[row['category']]+' / '+{'build':'出现','demolish':'消失'}[row['direction']]),('示例存在时相','前期 T1' if row['example_time']=='t1' else '后期 T2'),('示例面积 / 原孔洞率 / 紧凑度',f"{row['example_area']} / {row['hole_ratio']:.3f} / {row['compactness']:.3f}"),('剩余目标实例 / 像素',f"{row['remaining_instances']} / {row['remaining_pixels']}"),('同质簇占完整同类变化面积',f"{100*row['cluster_area_fraction']:.2f}%"),('筛选口径','单期 DINOv2 余弦≥0.75；RGB均值欧氏距离 / (255√3) ≤0.15；非在线输入'),('未来任务指令',row['instruction'])])
    body+='<table><thead><tr><th>实例</th><th>面积</th><th>单期 DINO 余弦</th><th>归一化 RGB 差</th><th>同款筛选</th></tr></thead><tbody>'
    for v in instances:body+=f'<tr><td>{html.escape(str(v["id"]))}</td><td>{v["area"]}</td><td>{v["cosine"]:.4f}</td><td>{v["rgb_distance"]:.4f}</td><td>'+('示例' if v['is_example'] else '通过' if v['is_match'] else '未通过')+'</td></tr>'
    return card(row,body+'</tbody></table>')
CSS='''body{margin:0;background:#101722;color:#e7edf7;font:15px/1.65 system-ui,sans-serif}header,main{max-width:1500px;margin:auto;padding:24px}header{position:sticky;top:0;background:#101722f5;z-index:10;border-bottom:1px solid #314158}h1{font-size:26px;margin:0}h2{font-size:20px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:15px}button,select,textarea{background:#1d2b40;color:#e7edf7;border:1px solid #4b607c;border-radius:6px;padding:9px}button{cursor:pointer}button.active{border-color:#59c9da;color:#85e7df}article{padding:22px;margin:24px 0;border:1px solid #314158;border-radius:12px;background:#172233}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}.grid.six{grid-template-columns:repeat(6,minmax(0,1fr))}figure{margin:0;background:#0c1420;padding:6px}img{width:100%;height:auto;max-height:550px;object-fit:contain}figcaption{padding:5px;font-size:13px;color:#c3d4e9}dl{display:grid;grid-template-columns:200px 1fr;gap:5px 12px}dt{color:#9db6d4}dd{margin:0;overflow-wrap:anywhere}.badge{color:#f7ce71}.review{display:flex;gap:15px;align-items:start;border-top:1px solid #314158;padding-top:18px}textarea{flex:1;min-height:65px}a{color:#7cdbe2}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #314158}.notice{color:#b6c7dc}section[hidden]{display:none}@media(max-width:900px){.grid,.grid.six{grid-template-columns:repeat(2,minmax(0,1fr))}dl{grid-template-columns:1fr}.review{display:block}textarea{width:95%}}'''
JS='''const key='cd-task-reconstruct-review-v1';let records={};try{records=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};function persist(){try{localStorage.setItem(key,JSON.stringify(records))}catch(e){document.getElementById('storage').textContent='浏览器未允许本地保存，请及时导出审核记录。'}}document.querySelectorAll('[data-id]').forEach(el=>{const id=el.dataset.id;records[id]??={id,status:'pending_review',note:''};el.value=el.tagName==='SELECT'?records[id].status:records[id].note;el.addEventListener('change',()=>{records[id][el.tagName==='SELECT'?'status':'note']=el.value;records[id].updated_at=new Date().toISOString();persist()})});document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==b.dataset.tab);document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b))});document.getElementById('export').onclick=()=>{document.querySelectorAll('textarea').forEach(e=>records[e.dataset.id].note=e.value);const blob=new Blob([JSON.stringify({scope:'candidate_visual_review_only',records:Object.values(records)},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='人工审核记录.json';a.click();URL.revokeObjectURL(a.href)};'''
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Render via Slurm')
    ASSETS.mkdir(parents=True,exist_ok=True)
    selected={};summaries={}
    for module in ('t5','t44','t3'):
        base=OUT/module
        if module=='t5':
            if (base/'visual_selection/selected.json').exists():base=base/'visual_selection'
            elif (base/'revision_actual_error/selected.json').exists():base=base/'revision_actual_error'
        if module=='t3' and (base/'revision_interior_examples/selected.json').exists():base=base/'revision_interior_examples'
        if module=='t44':
            if (base/'road_stage/revision_interior_targets/selected.json').exists():base=base/'road_stage/revision_interior_targets'
            elif (base/'road_stage/selected.json').exists():base=base/'road_stage'
        selected[module]=read(base/'selected.json');summaries[module]=read(base/'pool_summary.json')
    if 'source_scan_summary' in summaries['t44']:
        current=summaries['t44'];summaries['t44']=dict(current['source_scan_summary'],**{k:v for k,v in current.items() if k!='source_scan_summary'})
        summaries['t44']['qualified_images']=current['interior_target_configurations']
        summaries['t44']['qualified_configurations']=current['interior_target_configurations']
    if 'strict_eligible_candidates' in summaries['t3']:
        summaries['t3']['eligible_image_direction_candidates']=summaries['t3']['strict_eligible_candidates']
        summaries['t3']['eligible_physical_images']=summaries['t3']['strict_eligible_physical_images']
    labels={'t5':'T5.2 / T5.3 掩码优化','t44':'T4.4 空间关系查询','t3':'T3 视觉示例'}
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>核心任务数据重构 · 人工视觉审核</title><style>'+CSS+'</style><header><h1>核心任务数据重构 · 人工视觉审核</h1><p class="notice">仅离线候选，不代表正式批准。GT、理论F1与特征分数仅用于审核，不能进入未来Agent输入。所有图片位于相对路径assets，可下载整个目录后离线打开。</p><nav>'
    for k in labels:body+=f'<button data-tab="{k}" class="'+('active' if k=='t5' else '')+'">'+labels[k]+f'（{len(selected[k])} 张）</button>'
    body+='<button id="export">导出人工审核记录</button><a href="review_records.json">初始审核记录</a><a href="audit_summary.md">池统计与缺口</a></nav><small id="storage">判断和备注保存在本浏览器；请导出记录交回。页面操作不会自动批准正式评测。</small></header><main>'
    for module,render in [('t5',t5),('t44',t44),('t3',t3)]:
        body+=f'<section id="{module}" '+('hidden' if module!='t5' else '')+'><h2>'+labels[module]+'</h2>'
        if not selected[module]:body+='<p>当前筛选条件下暂无合格候选；缺口如实保留，详见审核摘要。</p>'
        for row in selected[module]:body+=render(row)
        body+='</section>'
    body+='</main><script>'+JS+'</script></html>'
    (OUT/'index.html').write_text(body)
    t5s=summaries['t5'];t44s=summaries['t44'];t3s=summaries['t3']
    lines=['# 核心任务重构：候选池与人工抽样审核','',
        '本轮仅离线数据构造。所有新样本均为 pending_review；没有启动在线 Agent，没有修改原始 GT 或历史预测。',
        '入口：`index.html`。配图均在相对目录 `assets/`；下载本目录后可离线打开。不需要审核后端。','',
        '## 池规模与抽样','', '| 模块 | 筛选前 | 自动筛选池（待人工审核） | 抽样卡片 |','|---|---:|---:|---:|',
        f"| T5.2 / T5.3 配对 | {len(t5s['source_records'])} 个图像×方向预测 | {len(t5s['candidates'])} 个数值门槛病灶（未全量目视批准） | {t5s.get('qualified_selected',len(selected['t5']))} 合格 + {t5s.get('diagnostic_selected',0)} 疑义诊断（每卡配对两任务） |",
        f"| T4.4 | {sum(t44s['scanned_images'].values())} 张开发图 | {t44s['qualified_images']} 张图 / {t44s['qualified_configurations']} 个关系配置 | {len(selected['t44'])} |",
        f"| T3.1 | {t3s['images_scanned']} 张开发图 | {t3s['eligible_physical_images']} 张图 / {t3s['eligible_image_direction_candidates']} 个类别方向候选 | {len(selected['t3'])} |",'',
        '## T5：真实弱预测与局部病灶','',
        '固定使用现有未核验 SAM3 建筑跨期预测（像素0.85、实例0.5、分块512、重叠0.25），不按每图GT更换弱版本、不腐蚀或挖除GT。WHU使用官方变化约束的方向GT及ignore。整图初始F1须在20–75%，真实错误像素WHU≥800、SECOND≥400。',
        '候选病灶完整填洞，但误检计数严格取原预测前景；早期填洞计数错误版本保留在t5根目录，计数修正版本为revision_actual_error，最终目视选例为visual_selection。','',
        '| 数据集 / 病灶 | 数值门槛候选数量 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in t5s['candidate_distribution'].items()]
    lines += ['', '数值筛选后逐张查看两期影像，换掉触边及多屋粘连样本。当前WHU2例有效、SECOND3例有效；WHU拆除缺1例：train916官方拆除GT只覆盖原棚屋下部，作为不合格诊断展示，不计合格。其GT原样保留。局部理论F1按补检/剔除/修边各自操作计算，不是Agent结果。', '',
        '## T4.4：纯净拓扑查询','',
        '建筑变化不分新建/拆除，连通实例填洞后过滤<150像素；固定32像素欧氏缓冲、实例面积重叠≥10%则整栋保留；目标≥400像素，同时至少一栋被排除的同类别干扰。',
        '首轮六类语义缓存没有道路通道，120张开发图的稳定水体/运动场未产生合格查询。随后用本地SegEarth显式追加道路类别；两期同类最高概率且均≥0.55，参照面积≥400，道路跨度≥图边长60%。不将普通硬化地表直接改称道路。',
        '道路语义初筛9配置；整条排除目标触边配置后剩5条，其中整组变化实例均不触边1条。选择3条完整目标查询，所有干扰原样保留，部分干扰位于影像边缘；页面明确标注。连通域是否包含多栋相邻建筑及参照物唯一性仍待人工审查。','',
        '## T3.1：完整同质实例簇','',
        '仅WHU建筑与SECOND建筑/运动场，使用各60张已有开发图；单期DINOv2缓存按实例池化，新建看T2、拆除看T1。E面积200–5000、原孔洞≤5%、最小旋转矩形填充≥0.45。至少另外2个完整实例余弦≥0.75且RGB均值欧氏距离/(255√3)≤0.15。',
        '严格送审层要求整组GT均不接触图像边界，避免裁断建筑。WHU当前开发池严格条件下无合格同质簇，不借报告集补数。gt_full保留同类同方向全部完整实例，包括未通过外观匹配的实例；gt_rem仅扣除示例，不偷偷缩小真值。优先同质簇占总面积≥85%的图；三例互不重复，尽可能覆盖两个数据集。',
        f"达到85%面积优先层的类别方向候选：{t3s['candidates_at_least_85_percent']}。",'',
        '| 数据集 / 类别 / 方向 | 原实例 | 合格示例 | 合格图 |','|---|---:|---:|---:|']
    lines += [f"| {k} | {v['original_instances']} | {v['eligible_examples']} | {v['eligible_images']} |" for k,v in t3s['groups'].items()]
    lines += ['', '## 缺口、成本与停止点','',
        f"目标合格卡片6/3/3；实际{t5s.get('qualified_selected',len(selected['t5']))}/{len(selected['t44'])}/{len(selected['t3'])}，另保留{t5s.get('diagnostic_selected',0)}张明确不合格诊断。逐项排除统计及来源保留在各模块pool_summary.json和selected.json。", 
        '数据构造API费用¥0；本地CPU/GPU计算日志另行归档。补漏研究的API费用单独记账，不计入本页面。',
        '本页只用于人工抽样视觉审核，不导出正式评测任务。合理/不合理/存疑及备注可在浏览器保存并导出；标记合理也不会自动批准或触发实验。交付后停止数据构造，等待人工审核。','']
    (OUT/'audit_summary.md').write_text('\n'.join(lines))
    records=[dict(id=r['id'],status='pending_review',note='') for rows in selected.values() for r in rows]
    (OUT/'review_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    errors=[]
    paths=re.findall(r'(?:src|href)="(assets/[^"]+)"',body)
    for path in set(paths):
        try:
            with Image.open(OUT/path) as im:im.verify()
        except Exception as exc:errors.append(dict(path=path,error=str(exc)))
    qc=dict(cards={k:len(v) for k,v in selected.items()},unique_images=len(set(paths)),image_errors=errors,external_dependencies=[],api_calls=0,online_agent_runs=0,status='complete' if not errors else 'failed')
    (OUT/'page_qc.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n')
    if errors:raise RuntimeError('Broken images; inspect page_qc')
    for module, rows in selected.items():
        if not rows:continue
        sheet=Image.new('RGB',(900,330*len(rows)),(20,30,45));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows):
            draw.text((5,j*330+3),row['id'],fill='white')
            if module=='t5': m=mask(row['lesion_mask'])
            elif module=='t44':m=mask(row['evaluation']['target_mask'])
            else:m=mask(row['gt_full'])
            source=[rgb(row['image_t1']),rgb(row['image_t2']),overlay(rgb(row['image_t2']),[(m,GREEN,.6)])]
            for i,im in enumerate(source):im.thumbnail((295,295));sheet.paste(im,(i*300,j*330+28))
        sheet.save(ASSETS/(module+'_contact.png'))
    print(json.dumps(qc,ensure_ascii=False))
if __name__=='__main__':main()
