"""Offline fixed-dev60 exemplar support/holes audit; no rendering or API changes."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from render import crop_box, SIZE
from .io_retry import read_image, read_text, write_json, write_text, exists
from .t3_routes import load_dev, group, pair_id, GROUPS
from .t3_routes_match import bbox
from agent_baseline.exemplar_semantics import canonical, CLASSES


def touches(mask):
    return dict(top=bool(mask[0].any()),bottom=bool(mask[-1].any()),
        left=bool(mask[:,0].any()),right=bool(mask[:,-1].any()))


def geometry(exemplar, truth):
    exemplar=np.asarray(exemplar,bool); truth=np.asarray(truth,bool)
    if not exemplar.any(): raise ValueError('Exemplar must be nonempty')
    # Four-connected background: a hole cannot reach the image exterior by a
    # horizontal/vertical path. Same definition is used for fill and counting.
    connectivity=ndi.generate_binary_structure(2,1)
    holes=ndi.binary_fill_holes(exemplar,structure=connectivity) & ~exemplar
    _,hole_count=ndi.label(holes,structure=connectivity)
    h,w=exemplar.shape; count=int(exemplar.sum()); box=bbox(exemplar)
    crops={}
    for name,kwargs in [('close',{}),('wide',dict(scale=10,min_side=256))]:
        x0,y0,x1,y1=crop_box(box,exemplar.shape,**kwargs)
        region=exemplar[y0:y1,x0:x1]
        display=np.asarray(Image.fromarray(region.astype('uint8')*255).resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
        crops[name]=dict(box=[x0,y0,x1,y1],pixels=int(region.size),support_pixels=int(region.sum()),
            support_fraction=float(region.mean()),exemplar_visible_fraction=float(region.sum()/count),
            displayed_support_fraction=float(display.mean()),support_touches_crop_edge=touches(region),
            crop_touches_image_edge=dict(left=x0==0,top=y0==0,right=x1==w,bottom=y1==h))
    return dict(shape=list(exemplar.shape),exemplar_pixels=count,gt_pixels=int(truth.sum()),
        remaining_gt_pixels=int((truth & ~exemplar).sum()),hole_count=int(hole_count),hole_pixels=int(holes.sum()),
        has_holes=bool(hole_count),support_bbox=box,support_touches_image_edge=touches(exemplar),crops=crops)


def semantic_pair(root, task_id):
    folder=root/'tasks'/task_id
    if exists(folder/'metrics.json'):
        row=json.loads(read_text(folder/'metrics.json'))
        return row['interpretation']['pair'],str(folder/'metrics.json')
    paths=[folder/'interpretation'/time/'answer.json' for time in ('t1','t2')]
    if not all(exists(path) for path in paths): return None,None
    reverse={name:i for i,name in CLASSES.items()}
    return [reverse.get(canonical(json.loads(read_text(path))['parsed'].get('class'))) for path in paths],str(folder/'interpretation')


def cross_summary(records):
    rows=[]
    for label in ('全部',*GROUPS):
        for holes in ('全部','有孔洞','无孔洞'):
            chosen=[r for r in records if (label=='全部' or r['group']==label)
                and (holes=='全部' or r['has_holes']==(holes=='有孔洞'))]
            available=[r for r in chosen if r['semantic_available']]
            agreed=sum(r['pair_label_agreement'] for r in available)
            rows.append(dict(group=label,holes=holes,tasks=len(chosen),semantic_available=len(available),
                pair_label_agreed=agreed,pair_label_agreement=agreed/len(available) if available else None,
                empty_remaining_gt=sum(r['remaining_gt_pixels']==0 for r in chosen),
                mean_close_support_fraction=float(np.mean([r['crops']['close']['support_fraction'] for r in chosen])) if chosen else None,
                mean_wide_support_fraction=float(np.mean([r['crops']['wide']['support_fraction'] for r in chosen])) if chosen else None))
    return rows


def markdown(report):
    lines=['# T3 dev60示例几何与解析关联审计','',
        '孔洞定义为无法经四邻域背景路径连到图像外部的非支持区域。close/wide严格使用现有ask_images的crop_box默认参数与scale=10/min_side=256；显示支持占比使用现有384像素最近邻支持图，未计红线覆盖。未改任何模型输入或GT。',
        '', '标签一致率是解析类别对与既有SECOND标签的对应程度；以下仅为描述性关联，不证明孔洞导致解析失败，也不代表人工裁决后的识别准确率。缺失解析单列，unknown保留为不一致。', '',
        '| 分组 | 孔洞 | 任务 | 已有双时相解析 | 类别对一致 | 一致率 | 剩余GT为空 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in report['cross_summary']:
        rate='—' if row['pair_label_agreement'] is None else f"{100*row['pair_label_agreement']:.1f}%"
        lines.append(f"| {row['group']} | {row['holes']} | {row['tasks']} | {row['semantic_available']} | {row['pair_label_agreed']} | {rate} | {row['empty_remaining_gt']} |")
    lines += ['', '逐例计数、裁图框、两种支持占比、触边、解析来源见[audit.json](audit.json)。图像极性/区域指向需继续结合原始二值掩码与VLM所见图人工审查。', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--semantic-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Execute offline image processing through Slurm')
    records=[]
    for row in load_dev(args.run):
        stats=geometry(read_image(row['exemplar_mask'])>0,read_image(row['gt_mask'])>0)
        pair,source=semantic_pair(args.semantic_root,row['id'])
        records.append(dict(id=row['id'],group=group(row),**stats,semantic_available=pair is not None,
            predicted_pair=pair,true_pair=list(pair_id(row)),
            pair_label_agreement=(pair==list(pair_id(row))) if pair is not None else None,
            semantic_source=source,exemplar_path=row['exemplar_mask']))
    report=dict(tasks=len(records),semantic_root=str(args.semantic_root),
        interpretation='descriptive association only; no causal attribution',
        hole_connectivity='4-connected background',cross_summary=cross_summary(records),records=records)
    write_json(args.output/'audit.json',report)
    write_text(args.output/'audit.md',markdown(report))


if __name__=='__main__': main()
