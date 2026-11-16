"""Export manually inspected T5 examples; preserve numerical pool and source masks."""
import argparse,json,os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from . import task_reconstruct_t5_actual as base
from .io_retry import write_json,save_image

CHOICES=[
('WHU-CD_train_30_appeared_FN-Build_1','画面下中部、道路右侧的棚屋漏检了，请参照后期影像补全。',True),
('WHU-CD_train_916_disappeared_FN-Demolish_0','画面中下部原有棚屋已被拆除，但官方真值仅覆盖旧棚屋局部；本卡仅用于诊断，不作为合格任务。',False),
('WHU-CD_train_155_appeared_FP_3','画面西侧的附属屋顶在前期已存在，却被误标为新建变化，请剔除该误检。',True),
('SECOND_train_127_appeared_FN-Build_3','画面中下部、河道右侧的矩形新屋顶漏检了，请参照后期影像补全。',True),
('SECOND_train_1754_disappeared_FN-Demolish_1','画面左中部的蓝顶旧屋已拆除，当前掩码漏检了该建筑，请参照前期影像边界补全。',True),
('SECOND_train_2677_appeared_FP_5','画面左下方厂房前的大片地表区域被误标为建筑变化，请剔除该误检。',True),
]
ISSUE='官方拆除GT仅覆盖旧棚屋下部局部（候选框y=319至369），前期影像可见屋顶还向上延伸至约y=288；不满足完整独立实例要求。保留原GT与预测，不能自动补全GT。本卡不计入合格样本，WHU拆除样本缺口为1。'

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm only')
    pool=json.loads((a.source/'pool_summary.json').read_text());lookup={r['id']:r for r in pool['candidates']}
    statistics=Counter();by_dataset={};shapes={}
    for row in pool['candidates']:
        path=row['image_t1']
        if path not in shapes:
            with Image.open(path) as im:shapes[path]=im.size
        w,h=shapes[path];x0,y0,x1,y1=row['bbox'];touch=x0<=0 or y0<=0 or x1>=w or y1>=h
        key='touching_border' if touch else 'not_touching_border';statistics[key]+=1
        by_dataset.setdefault(row['dataset'],Counter())[key]+=1
    selected=[]
    for ident,text,passed in CHOICES:
        meta=dict(lookup[ident]);dataset=meta['dataset'];root=Path(meta['image_t1']).parent.parent;name=meta['image_id'];direction=meta['direction']
        pred=base.read(meta['source']['prediction'])>0
        truth=base.load_truth(root,name) if dataset=='WHU-CD' else base.building_truth(*base.load_labels(root,name))
        gt=truth['new' if direction=='appeared' else 'demolished'];valid=~truth['ignore'] if dataset=='WHU-CD' else np.ones_like(gt)
        minimum=800 if dataset=='WHU-CD' else 400
        region,kind,operation,error=next(v for index,v in enumerate(base.lesions(pred,gt,valid,direction,minimum)) if index==meta['lesion_index'])
        if (kind,operation,error)!=(meta['lesion_type'],meta['operation'],meta['error_pixels']):raise ValueError('Selected candidate no longer matches original numeric pool')
        initial=base.score(pred,gt,valid);ideal=base.score(base.ideal_edit(pred,gt,region,operation),gt,valid)
        if abs(initial-meta['initial_f1'])>1e-8 or abs(ideal-meta['ideal_local_f1'])>1e-8:raise ValueError('Source prediction/GT differs from numeric candidate')
        folder=a.output/'samples'/ident
        for filename,value in [('m_init.png',pred),('gt.png',gt),('lesion.png',region),('ignore.png',~valid)]:save_image(folder/filename,value.astype(np.uint8)*255)
        paths=dict(m_init=str(folder/'m_init.png'),gt_mask=str(folder/'gt.png'),lesion_mask=str(folder/'lesion.png'),gt_ignore_mask=str(folder/'ignore.png'))
        meta.update(paths);meta.update(feedback_text=text,qc_passed=passed,diagnostic_only=not passed,qc_issue='' if passed else ISSUE,
            status='pending_review',visual_selection_source='Parent-approved manual visual selection after viewing original T1/T2; user final approval still pending',
            numeric_source=str(a.source/'pool_summary.json'),gt_policy='official WHU change with reliable semantic direction; no repair of original GT' if dataset=='WHU-CD' else 'SECOND directed building labels unchanged',
            input_fields=dict(image_t1=meta['image_t1'],image_t2=meta['image_t2'],m_init=paths['m_init']),
            evaluation_fields=dict(gt_mask=paths['gt_mask'],gt_ignore_mask=paths['gt_ignore_mask'],lesion_mask=paths['lesion_mask'],initial_f1=initial,ideal_local_f1=ideal),
            paired_tasks=[dict(task_type='T5.2',feedback=dict(bbox=meta['bbox'],point=meta['point'],operation=operation),diagnostic_only=not passed),dict(task_type='T5.3',feedback=text,diagnostic_only=not passed)])
        selected.append(meta)
    write_json(a.output/'selected.json',selected)
    summary=dict(pool)
    summary.update(status='pending_review',source_numeric_pool=str(a.source/'pool_summary.json'),candidate_pool_definition='仅数值F1/错误面积门槛候选；全池未逐一核实地物物理完整性，不能宣称237例全部合格',
        candidate_count=len(pool['candidates']),border_statistics=dict(statistics),border_statistics_by_dataset={k:dict(v) for k,v in by_dataset.items()},
        border_definition='bbox触及图像任一边：x0<=0或y0<=0或x1>=实际width或y1>=实际height；非触边不等于实例物理完整',
        selected_count=len(selected),selected_by_dataset=dict(Counter(r['dataset'] for r in selected)),qualified_selected=5,diagnostic_selected=1,
        qualified_by_dataset=dict(Counter(r['dataset'] for r in selected if r['qc_passed'])),shortage={'WHU-CD/FN-Demolish':1},complete=False,
        selection_policy='人工目视选取5例加1例明确不合格诊断，未改GT/预测或数值门槛。qc_passed仅本轮视觉初筛，全部仍pending_review。',
        selected_ids=[r['id'] for r in selected],diagnostic_issue=ISSUE,api_calls=0)
    write_json(a.output/'pool_summary.json',summary)
    print(json.dumps(dict(cards=6,qualified_selected=5,diagnostic_selected=1,border_statistics=dict(statistics),status='pending_review'),ensure_ascii=False))

if __name__=='__main__':main()
