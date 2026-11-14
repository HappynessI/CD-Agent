#!/usr/bin/env python3
"""Explain fixed routes, deletion rules and Agent using cached artifacts only.

Never imports or invokes a backend. Mask arithmetic below is display/scoring
of saved predictions and verifier answers; missing intermediate steps stay missing.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.dont_write_bytecode=True
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent_baseline.make_demo import (CSS, esc, figure, jpeg, read_json, read_text,
    read_mask, open_image, resolve_asset, mask_path, retry_read)
from common import components

IMC = '两期 segment_text("building", tile=512, score_thr=0.5, pixel_thr=0.85) → match_instances(overlap_thr=0.25，原始实例) → to_mask(出现 ∪ 消失)'
RULE = '固定候选分成 8 连通组件；同一 flash 验证器、相同裁图与问法逐组件回答，no 删除、yes 保留；未解析保留。无规划器，不补漏。'
DEFINITIONS = [
    ('固定路线','程序写死的工具链，不用任何多模态大模型。同样的原子工具，按固定顺序、固定参数调用。参数原则上在开发集上选，历史选择偏差见下。'),
    ('规则','固定路线的候选 + 验证器。把候选拆成 8 连通组件，每个组件问一次和 Agent 同一个验证器（同样的裁图和问法），答 no 就删，答 yes 就留。没有规划器，不会补漏，只会删。'),
    ('Agent','同样的工具和验证器，由规划器自己决定调哪些工具、问哪些组件、答案采不采纳、何时提交；固定核验阶段仍会自动核验提交候选。'),
]

# Numbers transcribed from the cited read-only README, never fitted here.
# Paths are relative to the externally supplied run root.
SPECS = {
 'levir_c0':dict(task='T1.1 × C0',dataset='LEVIR',fixed='init/sam3_imc_p85/val',answers='ask_eval/val_sam3_imc_p85/f37_corr.jsonl',description=IMC,rule_description=RULE,dev='LEVIR val 64 张选 SAM3/匹配与早期问法；dev2 训练 200 张比较 L2（最终保留 L1）',scores='固定 83.15；规则 86.35；Agent G3 85.76（64 张 × 3）',section='§10.5、§10.10',script='fixed_routes.py；analysis/vs_rule.py'),
 'levir_t51':dict(task='T5.1',dataset='LEVIR',fixed='init/sam3_imc_p85/val',answers='ask_eval/val_sam3_imc_p85/f37_corr.jsonl',description='给定掩码原样提交；给定掩码由 '+IMC,rule_description=RULE,dev='沿用 LEVIR val 64 张和 dev2 200 张设置，无单独重选',scores='固定/给定 83.15；规则 86.35；Agent G3 86.23（64 张 × 3）',section='§10.5、§10.10',script='fixed_routes.py；analysis/vs_rule.py'),
 'second_c1':dict(task='T1.1 × C1 新建',dataset='SECOND',fixed='second/fixed/appeared',rule='second/rule/appeared/masks',answers='second/rule/appeared/answers.jsonl',description=IMC.replace('出现 ∪ 消失','仅 appeared'),rule_description=RULE+' 按新建方向核验，原址重建不算。',dev='数值沿用 LEVIR val 64；范围卡在 second_dev 60 张上选（与报告 128 张不重叠，但问题先在报告试跑中发现）',scores='固定 57.04；规则 63.74；Agent G3 71.31（128 张 × 3）',section='§10.6、§10.10',script='fixed_routes.py；analysis/second_rule.py'),
 'second_c2':dict(task='T1.1 × C2',dataset='SECOND',fixed='second/box_baselines/c2/fixed_any',rule='second/box_baselines/c2/rule',answers='second/rule/any/answers.jsonl',original='second/fixed/any',description=IMC+' → 与 region#0 逐像素求交',rule_description=RULE+' 先按整图组件删，再与 region#0 求交。',dev='沿用 LEVIR 数值及 SECOND 范围卡 60 张设置；框按种子 1016 冻结，无新阈值',scores='固定 ∩ 框 54.67；规则 ∩ 框 60.69；Agent G3 62.77；事后 Agent ∩ 框 66.71（128 张 × 3）',section='§10.7、§10.10',script='analysis/second_box_baselines.py'),
 'second_c4':dict(task='T1.1 × C4',dataset='SECOND',fixed='second/box_baselines/c4/fixed_direction',rule='second/box_baselines/c4/rule',answers='second/rule/appeared/answers.jsonl',original='second/fixed/appeared',description=IMC.replace('出现 ∪ 消失','仅 appeared')+' → 与 region#0 求交',rule_description=RULE+' 新建方向；先删整图组件，再与 region#0 求交。',dev='沿用 LEVIR 数值及 SECOND 范围卡 60 张设置；固定框，无新阈值',scores='带方向固定 ∩ 框 62.63；规则 ∩ 框 67.99；Agent G3 71.94；事后 Agent ∩ 框 76.12（128 张 × 3）',section='§10.7、§10.10',script='analysis/second_box_baselines.py'),
 'levir_t52':dict(task='T5.2 点反馈 k=3',dataset='LEVIR',fixed='t52/fixed_bothdates/k3',description='固定响应 C：误检删点击所在给定组件；漏检分别在 T1/T2 做 SAM ViT-H 点提示，各取质量最高候选，再取面积较小者并入给定掩码。',rule_description='本任务无规则对照；主要对照是固定响应 C，不是验证器逐组件删除。',dev='B/C 是看过固定响应 A 在 LEVIR val 的失败后增加，未调门限；报告 64 张，属事后选择',scores='给定 83.15；固定响应 C 84.73；规则：无；Agent G3 84.59（64 张 × 3）',section='§10.8、§10.10',script='analysis/t52_fixed.py'),
 'levir_t53':dict(task='T5.3 语言反馈',dataset='LEVIR',fixed='t53/fixed/largest/masks',response='t53/fixed/verifier/masks',trace='t53/fixed/verifier/traces',description='固定响应 1：按候选最远点所属 3×3 方位格定位，选格内面积最大完整组件；漏检来自两期 SAM3、给定覆盖 <50%，误检来自给定组件；执行一次增删。',rule_description='本任务无规则对照；固定响应 2 用验证器 p_yes 挑候选（漏检最高、误检最低），不是对全部候选答 no 就删。',dev='沿用 LEVIR 设置，无新增数值门限；10 张试跑修正语言句子后报告 64 张',scores='给定 83.15；固定响应 1 82.73；固定响应 2 83.22；Agent G3 82.89（64 张 × 3）',section='§10.10',script='analysis/t53_fixed.py'),
 'second_t13':dict(task='T1.3 不限类别',dataset='SECOND',fixed='t13_ask/candidates/val/masks',rule='t13_ask/val/describe/masks',answers='t13_ask/val/describe/answers.jsonl',description='两期 SAM ViT-H 不限类别提议 → DINOv2 两期区域均值 1−cos ≥ 0.22 → 候选并集。',rule_description=RULE+' 问法 describe，标准规则只删 no；p_yes<0.9 是另一个变体。',dev='second_dev/names.txt 60 张选 0.22；同 60 张 1649 组件选 describe 问法',scores='固定 60.88；规则 61.11；Agent G3 46.53，G4(a) 47.71（128 张 × 3）',section='§10.10、§10.11',script='analysis/t13_ask_select.py；analysis/probe_t13.py'),
 'second_t31':dict(task='T3.1 图内示例',dataset='SECOND',fixed='t3/fixed_sub/masks',description='两期 SAM 提议 → DINOv2 T2−T1 区域变化向量 → 与示例余弦 ≥0.6 的提议取并集；预测和真值都剔除示例像素。',rule_description='本任务无规则对照；对照是示例相似度固定路线。',dev='second_dev 60 张选 0.6；报告 128 图每图抽 1 个任务，种子 1016 不看结果',scores='固定 19.39；规则：无；Agent G3 20.34（128 个任务 × 3）；703 任务全集固定 19.80',section='§10.10',script='analysis/t3_fixed.py；analysis/t3_metrics.py'),
 'whu_c0':dict(task='T1.1 × C0 / C1',dataset='WHU-CD',fixed='whu/fixed/any',rule='whu/rule/any/masks',answers='whu/rule/any/answers.jsonl',description=IMC+'；C1 分别只取 appeared/disappeared。',rule_description=RULE+' 三方向独立核验。',dev='WHU train 中冻结报告128、开发60（种子1016）；零样本迁移，不调设置',scores='C0：固定56.59 / 规则62.75 / Agent64.27；新建62.25 / 70.26 / 72.86；拆除21.71 / 22.17 / 25.86（各128张 × 3）',section='§10.11',script='fixed_routes.py；analysis/second_rule.py'),
 'second_t12':dict(task='T1.2 建筑 + 低矮植被',dataset='SECOND',fixed='t12/fixed/sam3_min64/val',rule='t12/rule/sam3_min64/any/masks',answers='t12/rule/sam3_min64/any/answers.jsonl',description='建筑用 SAM3 固定变化；低矮植被用 SegEarth 两期语义异或；两者并集，删除面积<64像素组件。',rule_description=RULE,dev='second_dev 60 张选 sam3_min64',scores='固定46.26；规则47.50；Agent G3 41.22 / G4(b) 38.95（128张 × 3）',section='§10.10、§10.11',script='analysis/t12_fixed.py'),
 'second_c1t':dict(task='T1.1 × C1 转换',dataset='SECOND',fixed='c1t/fixed/component0.25_min0/val',rule='c1t/rule/component0.25_min0/appeared/masks',answers='c1t/rule/component0.25_min0/appeared/answers.jsonl',description='SAM3 新建组件 ∩ T1 低矮植被/非植被地表语义；覆盖比例≥25%保留完整组件，不设最小面积。',rule_description=RULE+' 只接受规定的前期类别到建筑转换。',dev='second_dev 60 张选 component0.25_min0',scores='固定62.82；规则62.70；Agent G3 64.84（128张 × 3）',section='§10.10',script='analysis/c1t_fixed.py'),
 'whu_t54':dict(task='T5.4 旧地图输入',dataset='WHU-CD',fixed='t54/fixed/val/xor',rule='t54/rule/val/any/masks',description='SAM3 分割 2016 建筑（tile=512、score_thr=0.5、pixel_thr=0.85）→ 与 2012 旧地图异或；min_area=0，不删小组件。',rule_description=RULE+' 核验器对旧地图理解差，82%组件答 no；主要对照采用固定路线。',dev='WHU dev 60 张选择 min_area=0（t54/selection.json）；报告128张；其余 SAM3 参数沿用冻结设置',scores='固定59.40；规则30.97；Agent 未跑全量（报告128张）',section='§10.11、§10.12',script='analysis/t54_fixed.py；analysis/t54_rule.py'),
 'levir_t41':dict(task='T4.1 尺寸',dataset='LEVIR',description='固定路线或规则输出 → 8连通组件面积过滤；分辨率0.5m/像素，只保留面积严格大于260m²（>1040像素）的组件，相连建筑按一个组件。',rule_description='规则 + 同样的面积过滤。',dev='dev2 训练200张的组件面积分布选260m²，使约1/3组件达标；不看报告集',scores='64张：固定77.13 / 规则78.48；Agent 未跑全量。10张单次试跑：Agent82.52 / 固定83.17 / 规则83.52',section='§10.11、§10.12',script='analysis/make_t41_tasks.py'),
 'second_t32':dict(task='T3.2 外部目标示例',dataset='SECOND',description='SAM候选 → DINOv2 T2 外观与示例 T2 外观余弦排序 → 分数≥0.55取并集。',rule_description='本任务无规则对照。',dev='912个开发任务联合选各任务阈值，并列取小；T3.2报告401任务',scores='固定14.44；规则：无；Agent 未跑全量',section='§10.11',script='analysis/t3x_fixed.py'),
 'second_t33':dict(task='T3.3 外部变化示例',dataset='SECOND',description='SAM候选 → DINOv2两期变化向量与外部示例余弦 → 分数≥0.15取并集。',rule_description='本任务无规则对照。',dev='912个开发任务联合选各任务阈值，并列取小；T3.3报告711任务',scores='固定15.43；规则：无；Agent 未跑全量',section='§10.11',script='analysis/t3x_fixed.py'),
 'second_t34':dict(task='T3.4 正负示例',dataset='SECOND',description='SAM候选 → 正例余弦−负例余弦 → 分数≥0.20取并集。',rule_description='本任务无规则对照。',dev='912个开发任务联合选各任务阈值，并列取小；T3.4报告711任务',scores='固定12.84；规则：无；Agent 未跑全量',section='§10.11',script='analysis/t3x_fixed.py'),
}

BASELINE_CSS='''
.baseline-panel{margin:18px 24px;padding:14px}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.comparison>div{min-width:0}.comparison img{max-height:320px;object-fit:contain}.comparison-label{font-variant-numeric:tabular-nums}.baseline-panel table,.baseline-overview table{border-collapse:collapse;width:100%;font-size:13px}.baseline-panel td,.baseline-panel th,.baseline-overview td,.baseline-overview th{padding:7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.baseline-overview td{min-width:150px}.baseline-overview td:nth-child(3){min-width:310px}.lane{width:100%;min-width:900px;display:block}.lane text{fill:var(--text);font-size:14px}.lane rect{fill:var(--panel);stroke:var(--line)}.lane .tool{fill:var(--tint);stroke:var(--accent)}.lane .verifier{fill:var(--warnbg);stroke:var(--warn)}.lane .planner{fill:var(--tint);stroke:var(--up)}.lane path{stroke:var(--muted);fill:none}.cache-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.component-map{max-width:600px}.baseline-panel .deleted{color:var(--warn)}@media(max-width:650px){.comparison{grid-template-columns:1fr}.baseline-panel{margin:12px 16px}}
'''


@lru_cache(maxsize=16)
def answer_ledger(path):
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def image_answers(path, image):
    if not path.exists(): return []
    out=[]
    for value in answer_ledger(str(path)):
        uid=value.get('uid','')
        if uid.rsplit(':',1)[0] == image:
            answer=value.get('answer')
            row=answer if isinstance(answer,dict) else value
            out.append(dict(row,id=int(uid.rsplit(':',1)[1])))
    return out


def load_baselines(root, record, directory=None):
    """Load saved masks/answers. Only LEVIR rule has to be recovered from its ledger."""
    root=Path(root); image=record['image']; key=record['page']
    spec=SPECS.get(key,dict(description='固定路线未找到',rule_description='产物未找到'))
    result=dict(spec=spec,key=key,fixed=None,rule=None,response=None,answers=[],sources={},labels=None)
    for kind in ('fixed','rule','response','original'):
        value=spec.get(kind)
        if not value: continue
        path=root/value/(image+'.png')
        result['sources'][kind]=str(path.relative_to(root))
        if path.exists(): result[kind]=read_mask(path)
    fixed=result['fixed']
    if fixed is not None:
        original=result.get('original',fixed)
        labels,comps=components(original)
        result['labels']=labels
        ledger=image_answers(root/spec['answers'],image) if spec.get('answers') else []
        by_id={r['id']:r for r in ledger}
        if spec.get('answers'):
            result['answers']=[dict(id=c['id'],area=c['area'],answer=by_id.get(c['id'],{}).get('answer','未记录'),reason=by_id.get(c['id'],{}).get('reason','未记录')) for c in comps]
        if spec.get('answers'): result['sources']['answers']=spec['answers']
        if key.startswith('levir_') and key in ('levir_c0','levir_t51') and ledger:
            removed=[a['id'] for a in ledger if a.get('answer')=='no']
            result['rule']=fixed & ~np.isin(labels,removed)
            result['sources']['rule']='按已缓存 f37_corr 答案删除 no 组件恢复（无后端调用）'
        if spec.get('trace'):
            path=root/spec['trace']/(image+'.json')
            result['sources']['trace']=str(path.relative_to(root))
            if path.exists():
                trace=read_json(path)
                result['answers']=[dict(id=c['uid'],area=c['area'],answer=(c.get('verification') or {}).get('answer','未记录'),reason=(c.get('verification') or {}).get('reason','未记录')) for c in trace.get('candidates',[])]
    if directory and key=='second_t31':
        info=read_json(Path(directory)/'input.json')
        ex=(info.get('initial_view') or {}).get('overlay')
        if ex and mask_path(Path(directory),ex).exists(): result['ignore']=read_mask(mask_path(Path(directory),ex))
    return result


def measures(pred,truth):
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
    return dict(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)


def scored_image(pred,truth,background,deleted=None):
    small=lambda x: np.asarray(Image.fromarray(x.astype('uint8')*255).resize(background.size,Image.Resampling.NEAREST))>127
    p,t=small(pred),small(truth)
    canvas=(np.array(background)*.22).astype('uint8')
    canvas[p&t]=[44,206,120];canvas[p&~t]=[239,80,80];canvas[~p&t]=[69,141,255]
    if deleted is not None and deleted.any():
        im=Image.fromarray(small(deleted).astype('uint8')*255)
        edge=np.asarray(im.filter(ImageFilter.MaxFilter(3)))!=np.asarray(im.filter(ImageFilter.MinFilter(3)))
        yy,xx=np.indices(edge.shape)
        canvas[edge & ((xx+yy)//4%2==0)]=[255,218,65]
    return Image.fromarray(canvas)


def answers_table(rows, note=''):
    if not rows: return '<p>组件答案：产物未找到。</p>'
    cells=''.join(f'<tr><td>{esc(r["id"])}</td><td>{esc(r["area"])}</td><td>{esc(r.get("answer","未记录"))}</td><td>{esc(r.get("reason","未记录"))}</td></tr>' for r in rows)
    return f'<details><summary>组件答案表（{len(rows)} 个）</summary><p>{esc(note)}</p><div class="tablewrap"><table><thead><tr><th>组件号</th><th>面积 / 像素</th><th>验证器答案</th><th>理由</th></tr></thead><tbody>{cells}</tbody></table></div></details>'


def render_comparison(data,agent,truth,t2,encode=jpeg,link=True):
    spec=data['spec'];ignore=data.get('ignore')
    if ignore is not None:
        truth=truth&~ignore;agent=agent&~ignore
    columns=[]
    fixed,rule=data.get('fixed'),data.get('rule')
    secondary=rule if rule is not None else data.get('response')
    deleted=(fixed&~rule) if fixed is not None and rule is not None else None
    for title,pred in [('固定响应 1' if data['key']=='levir_t53' else '固定响应 C' if data['key']=='levir_t52' else '固定路线',fixed),('固定响应 2（验证器挑选，非规则）' if data.get('response') is not None else '规则',secondary),('Agent',agent)]:
        if pred is None:
            label='本任务无规则对照' if title=='规则' and not spec.get('rule') and not spec.get('answers') else '产物未找到'
            columns.append(f'<div><h3>{title}</h3><p class="empty">{label}</p></div>');continue
        if ignore is not None: pred=pred&~ignore
        metric=measures(pred,truth)
        caption=' · '.join(f'{name} {metric[key]*100:.2f}%' for name,key in [('F1','f1'),('P','precision'),('R','recall')])
        picture=scored_image(pred,truth,t2,deleted if title=='规则' else None)
        columns.append(f'<div><h3>{title}</h3>{figure(encode(picture),title+" · TP绿 / FP红 / FN蓝")}<p class="comparison-label">{caption}</p></div>')
    description=spec['description']+' '+spec['rule_description']
    body='<section class="panel baseline-panel"><h2>对照：同一张图上的固定路线和规则</h2><p>'+esc(description)+'</p>'
    if link: body+=f'<p><a href="baselines.html#{esc(data["key"])}">查看三种做法与本任务对照口径</a></p>'
    body+='<div class="comparison">'+''.join(columns)+'</div>'
    if rule is not None: body+='<p class="deleted">黄色虚线轮廓：固定候选中被规则删除的组件；颜色只表示事后与真值的比较。</p>'
    if ignore is not None: body+='<p>本面板按 T3.1 报告口径，从三种预测和真值中剔除示例像素；页头与步骤保留记录器原始 F1，二者口径不同。</p>'
    if spec.get('original'): body+='<p>组件号和面积沿用裁剪前的固定候选；虚线只显示本区域内删除部分，不将裁剪后的新组件编号套到旧答案上。</p>'
    if spec.get('answers') or spec.get('trace'):
        body+=answers_table(data.get('answers',[]),'未记录/未解析不等于 no；离线规则与 Agent 在线核验是独立请求，答案可能不同。')
    body+='<details><summary>对照产物来源（相对运行目录）</summary>'+''.join(f'<p>{esc(k)}：{esc(v)}</p>' for k,v in data.get('sources',{}).items())+'</details></section>'
    return body


def lanes_svg():
    bits=['<div class="tablewrap"><svg class="lane" viewBox="0 0 1120 370" role="img" aria-label="固定路线、规则、Agent三泳道；相同工具与验证器保持同色，只换决策者"><defs><marker id="lane-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10"/></marker></defs>']
    lanes=[('固定路线（程序）', [('输入',''),('固定顺序与参数',''),('同一套原子工具','tool'),('无验证器',''),('输出掩码','')]),('规则（程序 + 验证器）',[('相同输入',''),('固定候选 → 8连通组件','tool'),('同一验证器逐组件回答','verifier'),('程序：no删 / yes留',''),('输出掩码','')]),('Agent（规划器决策）',[('相同输入',''),('规划器选工具、看返回','planner'),('同一套原子工具','tool'),('同一验证器 → 规划器复核','verifier'),('submit → 输出','')])]
    for i,(name,nodes) in enumerate(lanes):
        y=35+i*115
        bits.append(f'<text x="12" y="{y}">{esc(name)}</text>')
        for j,(text,cls) in enumerate(nodes):
            x=12+j*221
            bits.append(f'<rect class="{cls}" x="{x}" y="{y+15}" width="205" height="48" rx="6"/><text x="{x+102}" y="{y+45}" text-anchor="middle">{esc(text)}</text>')
            if j<4: bits.append(f'<path d="M{x+205} {y+39}h16" marker-end="url(#lane-arrow)"/>')
        if i==2: bits.append(f'<path d="M775 {y+65}v20H335v-20" marker-end="url(#lane-arrow)"/><text x="550" y="{y+100}" text-anchor="middle">验证是证据：规划器可采纳、反对、再编辑、补漏</text>')
    bits.append('</svg></div>')
    return ''.join(bits)


def render_overview():
    rows=[]
    for key,spec in SPECS.items():
        rows.append(f'<tr id="{key}"><td>{esc(spec["task"])}</td><td>{esc(spec["dataset"])}</td><td>{esc(spec["description"])}</td><td>{esc(spec["rule_description"])}</td><td>{esc(spec["dev"])}</td><td>{esc(spec["scores"])}</td><td>README_ai_readonly.md {esc(spec["section"])}<br>{esc(spec["script"])}</td></tr>')
    return '<section class="panel baseline-overview"><h2>各任务固定路线与规则</h2><p>以下报告分数照录 README 的指定章节，单位为百分数；不是上方示例的单图 F1。Agent 全量结果为 3 次运行的合并 F1；原 README 部分比较报告三次整集 F1 的均值，WHU 使用平均像素计数，具体区间和汇总口径见原章节。10 张试跑只有一次，不冒充全量结果。</p><div class="tablewrap"><table><thead><tr><th>任务</th><th>数据</th><th>固定步骤与参数</th><th>规则 / 无规则原因</th><th>在哪选参数</th><th>报告集 F1 / Agent 状态</th><th>依据</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>'


def component_figure(data,t2):
    labels=data.get('labels')
    if labels is None: return '<p>组件编号：产物未找到。</p>'
    canvas=t2.copy(); w,h=canvas.size
    ids={r['id']:r.get('answer') for r in data['answers']}
    raw=Image.fromarray((labels>0).astype('uint8')*255).resize(canvas.size,Image.Resampling.NEAREST)
    edge=np.asarray(raw.filter(ImageFilter.MaxFilter(3)))!=np.asarray(raw.filter(ImageFilter.MinFilter(3)))
    arr=np.array(canvas);arr[edge]=[60,220,140]
    removed=np.isin(labels,[k for k,v in ids.items() if v=='no'])
    small=Image.fromarray(removed.astype('uint8')*255).resize(canvas.size,Image.Resampling.NEAREST)
    noedge=np.asarray(small.filter(ImageFilter.MaxFilter(3)))!=np.asarray(small.filter(ImageFilter.MinFilter(3)))
    arr[noedge]=[255,70,70];canvas=Image.fromarray(arr);draw=ImageDraw.Draw(canvas)
    for ident in sorted(ids):
        yy,xx=np.nonzero(labels==ident)
        if len(xx):
            x=int(xx.mean()*w/labels.shape[1]);y=int(yy.mean()*h/labels.shape[0])
            draw.text((x,y),str(ident),fill='white',stroke_width=1,stroke_fill='black',anchor='mm')
    return '<div class="component-map">'+figure(jpeg(canvas),'固定候选组件编号；红色为答 no 被删，绿色为其余轮廓')+'</div>'


def cached_levir_steps(root,image):
    """Read saved SAM3 soft instances; do not recalculate matching outputs."""
    path=Path(root)/'sam3_soft/t512s256/val'/(image+'.npz')
    steps=[]
    if path.exists():
        def load():
            with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as z:
                shape=tuple(map(int,z['shape']));out=[]
                for time in (0,1):
                    dense=np.zeros(int(np.prod(shape)),bool)
                    scores,origins,offsets=z[f'scores{time}'],z[f'origins{time}'],z[f'offs{time}']
                    pixels,values=z[f'idx{time}'],z[f'val{time}']
                    n=0
                    for i in np.flatnonzero(scores>=.5):
                        if origins[i][0]%512 or origins[i][1]%512: continue
                        idx=pixels[offsets[i]:offsets[i+1]][values[offsets[i]:offsets[i+1]]>=round(.85*255)]
                        if len(idx): dense[idx]=True;n+=1
                    out.append((dense.reshape(shape),n))
                return out
        for i,(mask,n) in enumerate(retry_read(load)):
            steps.append(figure(jpeg(Image.fromarray(mask.astype('uint8')*255)),f'T{i+1} SAM3 缓存实例并集（{n} 个）；score≥0.5 / pixel≥0.85'))
    else: steps+=['<p>T1 实例：未缓存，只显示最终掩码。</p>','<p>T2 实例：未缓存，只显示最终掩码。</p>']
    steps+=['<p>匹配后的出现：未缓存，只显示最终掩码。不以 Agent 的其他门限调用代替。</p>', '<p>匹配后的消失：未缓存，只显示最终掩码。不重新执行匹配。</p>']
    fixed=Path(root)/'init/sam3_imc_p85/val'/(image+'.png')
    if fixed.exists(): steps.append(figure(jpeg(open_image(fixed)),'出现 ∪ 消失：已缓存的最终固定掩码'))
    return '<div class="cache-steps">'+''.join(steps)+'</div>'


def select_example(root,records):
    """Apply the original task median, restricting candidates to an observed deletion."""
    root=Path(root)
    ledger=root/'ask_eval/val_sam3_imc_p85/f37_corr.jsonl'
    deleted={r['uid'].rsplit(':',1)[0] for r in answer_ledger(str(ledger)) if r.get('answer')=='no'} if ledger.exists() else set()
    # The closest overall median session is sufficient if it meets the condition.
    typical=next(r for r in records if r['page']=='levir_c0')
    if typical['image'] in deleted:
        return dict(typical,example_rule='三种子任务中位数不变；候选必须是 model 提交且规则至少删除 1 个组件；按原并列规则选择。')
    from agent_baseline.make_demo import read_steps,CATEGORIES
    candidates=[]
    for n in (1,2,3):
        run=f'json_qwen37flash_g3_s{n}'
        for path in sorted((root/run/'T1.1xC0/val').glob('*/final.json')):
            f=read_json(path)
            if path.parent.name in deleted and f.get('submitted_by')=='model' and f.get('eval',{}).get('f1') is not None:
                cats={CATEGORIES.get(c['tool'],'其他') for row in read_steps(path.parent) for c in row.get('calls',[]) if c.get('ok') and c.get('status')!='skipped_due_to_submit'}
                candidates.append(dict(run=run,image=path.parent.name,f1=f['eval']['f1'],category_count=len(cats)))
    if not candidates: raise ValueError('满足规则有删除且模型提交的 LEVIR 示例未找到')
    middle=typical['median'];distance=min(abs(c['f1']-middle) for c in candidates)
    tied=[c for c in candidates if abs(abs(c['f1']-middle)-distance)<=1e-12]
    chosen=min(tied,key=lambda c:(-c['category_count'],c['image'],c['run']))
    return dict(typical,**chosen,file='baseline_example.html')


def render_baselines(root,records,example=None):
    root=Path(root);example=example or select_example(root,records)
    d=root/example['run']/example['task']/'val'/example['image']
    info=read_json(d/'input.json');final=read_json(d/'final.json')
    data=load_baselines(root,example,d)
    # LEVIR truth is located from the recorded source imagery's sibling label directory.
    if info.get('gt_mask_path'): truth=read_mask(resolve_asset(d,info['gt_mask_path']))
    else: truth=read_mask(Path(info['source_images'][0]).parent.parent/'label'/(example['image']+'.png'))
    agent=read_mask(resolve_asset(d,final['mask_path']))
    t2=open_image(resolve_asset(d,info['source_images'][1]));t2.thumbnail((512,512))
    intro=''.join(f'<p><strong>{a}：</strong>{b}</p>' for a,b in DEFINITIONS)
    body='<header><h1>三种做法：固定路线、规则与 Agent</h1><p><a href="index.html">九个典型回放</a></p>'+intro+'<p>三者共享同一套工具；规则和 Agent 共享同一个验证器，固定路线不调用验证器。差别在谁做决定。规则是 Agent 不做任何判断时的下限参照，并非数学性能下界；固定路线衡量不用 MLLM 能到多少。</p></header><main class="hub">'
    body+='<section class="panel"><h2>只换决策者</h2>'+lanes_svg()+'<p>蓝色是同一套工具，黄色是同一验证器，绿色边框是规划器。规则的答案只用于程序删留；Agent 可补漏、修改和重新选择。在线与离线是独立请求，不保证逐次回答相同。</p></section>'
    body+=f'<section class="panel"><h2>同一张图：{esc(example["image"])}</h2><p>选例：{esc(example["run"])}；最终 F1 {example["f1"]*100:.4f}%，任务中位数 {example["median"]*100:.4f}%。先按三种子全部逐会话 F1 的中位数选典型会话，再要求规则至少删过一个组件；并列沿用工具类别数、图名、运行名。</p><p><a href="{esc(example["file"])}">打开同图 Agent 逐步回放</a></p><h3>固定路线每一步（仅缓存）</h3>'+cached_levir_steps(root,example['image'])+'<h3>规则：逐组件问，再按 no 删除</h3>'+component_figure(data,t2)+answers_table(data['answers'])+'</section>'
    body+=render_comparison(data,agent,truth,t2,link=False)
    body+=render_overview()
    body+='<section class="panel"><h2>解释分数时的限制</h2><p>固定路线和规则的参数也在验证集上看过：LEVIR 的 SAM3 门限、匹配和早期验证器问法都受验证结果影响；SECOND 的范围卡问题也先在报告试跑中发现。后续用独立开发图选设置，不能消除已有选择偏差（README §10.3、§10.5、§10.6）。</p><p>全量 Agent 分数来自 3 次运行的合并 F1，区间在 README 指定章节；不把单图示例当整集分数，也不把三次整集 F1 的均值与逐会话 F1 中位数混用。T4.1 的 Agent 仅10张单次试跑；T3.2–3.4、T5.4 没有 Agent 全量。没有跑新实验。</p></section></main>'
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三种做法 · 固定路线、规则与 Agent</title><style>'+CSS+BASELINE_CSS+'</style></head><body>'+body+'</body></html>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--selections',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    records=read_json(args.selections)
    doc=render_baselines(args.run_root,records)
    from agent_baseline.make_g3_pages import audit_html
    report=audit_html(doc)
    if report['bytes']>3_000_000: raise ValueError('baselines.html 超过 3 MB')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(doc,encoding='utf-8')
    print(f'{args.out.name}: {report["bytes"]} bytes')

if __name__=='__main__': main()
