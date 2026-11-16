"""Read-only source audit: reject whole spatial configurations with clipped targets."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from agent_baseline.evaluation import load_reference

MAIN=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent')
SOURCE=MAIN/'results/agent_baseline/task_reconstruct_audit/t44/road_stage'
OUT=SOURCE/'revision_interior_targets'

def touches_border(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())

def components(binary, minimum):
    labels,n=ndi.label(binary,np.ones((3,3)))
    return [filled for i in range(1,n+1) if (filled:=ndi.binary_fill_holes(labels==i)).sum()>=minimum]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    evaluation=json.loads((MAIN/'results/agent_baseline/n9_missed_20261116/evaluation_manifest.json').read_text())
    tasks={r['task']['gt_mask']:r['task'] for r in evaluation}
    records=[]
    for scan in sorted((SOURCE/'scan').glob('*.json')):
        for original in json.loads(scan.read_text())['candidates']:
            row=dict(original)
            truth=np.logical_or.reduce([load_reference(g,tasks[g].get('gt_ignore_mask'))[0] for g in row['source_gt']])
            buildings=components(truth,150)
            target=np.logical_or.reduce([buildings[i] for i in row['target_instance_ids']])
            assert int(target.sum())==row['target_pixels'],(row['id'],'source reconstruction mismatch')
            row['instances']=[]
            for i,b in enumerate(buildings):
                y,x=np.where(b)
                row['instances'].append(dict(id=i,area=int(b.sum()),bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],touches_border=touches_border(b),is_target=i in row['target_instance_ids']))
            row['target_touches_border']=touches_border(target)
            row['all_gt_touches_border']=any(v['touches_border'] for v in row['instances'])
            row['eligible_interior_target']=not row['target_touches_border']
            row['physical_instance_review']='pending: connected component may contain multiple adjoining buildings'
            folder=OUT/'masks'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            probs=[]
            for time in ('t1','t2'):
                with np.load(row['source_semantic'][time],allow_pickle=False) as z:probs.append(z['data'])
            ci={'water':4,'playground':6,'road':7}[row['anchor_class']]
            stable=np.logical_and.reduce([(p.argmax(0)==ci)&(p[ci]>=.55) for p in probs])
            anchor=components(stable,400)[row['anchor_index']]
            values=dict(target_mask=target,distractor_mask=np.logical_or.reduce([buildings[i] for i in row['distractor_instance_ids']]),anchor_mask=anchor,buffer_mask=ndi.distance_transform_edt(~anchor)<=32)
            row['evaluation']={}
            for name,value in values.items():
                path=folder/(name+'.png');Image.fromarray(value.astype('uint8')*255).save(path);row['evaluation'][name]=str(path)
            row['inference_input']={k:row[k] for k in ('image_t1','image_t2','instruction')}
            records.append(row)
    eligible=sorted([r for r in records if r['eligible_interior_target']],key=lambda r:(-min(r['target_pixels'],r['distractor_pixels']),-r['anchor_pixels'],r['id']))
    selected=[];used=set()
    for row in eligible:
        key=row['dataset'],row['image_id']
        if key in used:continue
        used.add(key);selected.append(row)
        if len(selected)==3:break
    write(OUT/'configuration_audit.json',records)
    write(OUT/'selected.json',selected)
    write(OUT/'pool_summary.json',dict(source_scan_summary=json.loads((SOURCE/'pool_summary.json').read_text()),audited_configurations=len(records),interior_target_configurations=len(eligible),whole_gt_interior_configurations=sum(not r['all_gt_touches_border'] for r in records),selected=len(selected),gap=3-len(selected),policy='Reject entire configuration if any target touches image boundary; no GT instance removed; original ranking preserved; physical connected-building review still pending',api_cost_cny=0))
    print(json.dumps(dict(stage='t44_interior_audit',audited=len(records),eligible=len(eligible),selected=len(selected),gap=3-len(selected))))
if __name__=='__main__':main()
