"""Offline T5 paired-feedback candidate construction from genuine SAM3 predictions."""
import argparse,json,os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from agent_baseline.whu_gt import load_truth
from .second_labels import building_truth,load_labels
from .io_retry import write_json,save_image

FG=np.ones((3,3),bool);BG=ndi.generate_binary_structure(2,1)

def read(path,rgb=False):
    with Image.open(path) as im:return np.array(im.convert('RGB') if rgb else im)
def score(pred,gt,valid):
    p=pred&valid;g=gt&valid;tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
    return 200*tp/max(1,2*tp+fp+fn)
def ideal_edit(pred,gt,region,operation):
    if operation=='剔除':return pred&~region
    if operation=='补检':return pred|(gt&region)
    return (pred&~region)|(gt&region)
def components(mask):
    labels,n=ndi.label(mask,FG)
    for i,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        support=np.zeros_like(mask);support[box]=ndi.binary_fill_holes(labels[box]==i,structure=BG);yield support

def geometry(mask):
    y,x=np.nonzero(mask);distance=ndi.distance_transform_edt(mask);point=np.unravel_index(np.argmax(distance),mask.shape)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],[int(point[1]),int(point[0])]

def lesions(pred,gt,valid,direction,minimum):
    g=gt&valid;objects=list(components(g))
    for obj in objects:
        missing=int((obj&g&~pred).sum())
        if missing>=minimum:yield obj,'FN-Build' if direction=='appeared' else 'FN-Demolish','补检',missing
    for obj in components(pred):
        error=int((obj&pred&~gt&valid).sum());tp=int((obj&pred&g).sum())
        if error<minimum:continue
        region=obj.copy()
        # Keep entire intersected reference instances for the offline ideal repair diagnostic.
        for instance in objects:
            if (instance&region).any():region|=instance
        region=ndi.binary_fill_holes(region,structure=BG)
        yield region,'FP' if tp==0 else 'Coarse-Merge','剔除' if tp==0 else '修边',error

def feedback(kind,bbox,shape):
    x=(bbox[0]+bbox[2])/2/shape[1];y=(bbox[1]+bbox[3])/2/shape[0]
    loc=('上方' if y<1/3 else '下方' if y>2/3 else '中部') if 1/3<=x<=2/3 else ('左' if x<1/3 else '右')+('上方' if y<1/3 else '下方' if y>2/3 else '侧中部')
    tail={'FN-Build':'新建建筑漏检了，请参照后期影像补全。','FN-Demolish':'原有建筑后期已拆除但漏检了，请参照前期影像补全。','FP':'当前变化掩码中的该片区域为误检，请剔除。','Coarse-Merge':'当前建筑变化掩码与周围区域粘连，请按建筑边界修整。'}[kind]
    return '画面'+loc+'的'+tail

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm only')
    datasets={'WHU-CD':Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/WHU-CD'),'SECOND':Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/SECOND')}
    candidates=[];audit=[];selected=[]
    write_json(a.output/'protocol.json',dict(status='pending_review',initial_f1_percent=[20,75],minimum_error_pixels={'WHU-CD':800,'SECOND':400},weak_method='Fixed unverified SAM3 IMC p0.85 score0.5 tile512 overlap0.25, same version on every image; never GT corruption',alternative_not_used='base/*.npz start is fixed connected components after historical f37_corr verifier rejection; yes is accepted components. Neither is selected per image.',connectivity=8,hole_fill_background_connectivity=4,selection='per dataset choose one FN-Build, FN-Demolish and FP/Coarse-Merge on different images; prioritize error area then stable ID',api_calls=0,online_runs=0))
    protocol=json.loads((a.output/'protocol.json').read_text())
    protocol.update(revision='actual_prediction_error_counts',error_definition='FP/TP always intersect the unchanged real prediction; filled holes do not create predicted pixels',ideal_edit_definition='remove: M_init minus region; add: M_init union GT inside region; refine: replace region by local GT. GT-assisted ideal edits are audit only.')
    write_json(a.output/'protocol.json',protocol)
    for dataset,root in datasets.items():
        names=(a.run/('whu/names_dev60.txt' if dataset=='WHU-CD' else 'second_dev/names.txt')).read_text().split()
        forbidden=set((a.run/('whu/names_val128.txt' if dataset=='WHU-CD' else 'second/second_val128.txt')).read_text().split())
        if set(names)&forbidden or any(n.startswith('test_') for n in names):raise ValueError('Development list overlaps forbidden report/test')
        minimum=800 if dataset=='WHU-CD' else 400
        for name in names:
            paths=[str(root/k/name) for k in ('img1','img2')]
            for direction in ('appeared','disappeared'):
                checkpoint=a.output/'checkpoints'/dataset/direction/(Path(name).stem+'.json')
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_text());audit.append(saved['audit']);candidates.extend(saved['candidates']);continue
                pred_path=(a.run/'second_dev/second/fixed'/direction/name if dataset=='SECOND' else a.run.parent/'batch99_find_missed_pre/artifacts/whu_fixed'/('whu_'+direction)/name)
                source=dict(method='SAM3 building IMC fixed p0.85 score0.5 tile512 overlap0.25, unverified',prediction=str(pred_path),generated=False,settings=str(pred_path.parent/'settings.json'),baseline_builder='agent_baseline.analysis.missed_sources:build_baseline uses this fixed mask before historical verifier filtering')
                if not pred_path.exists():
                    record=dict(dataset=dataset,image_id=name,direction=direction,status='missing_real_prediction',source=source)
                    write_json(checkpoint,dict(audit=record,candidates=[]));audit.append(record);continue
                pred=read(pred_path)>0
                truth=load_truth(root,name) if dataset=='WHU-CD' else building_truth(*load_labels(root,name))
                gt=truth['new' if direction=='appeared' else 'demolished'];valid=~truth['ignore'] if dataset=='WHU-CD' else np.ones_like(gt)
                initial=score(pred,gt,valid);record=dict(dataset=dataset,image_id=name,direction=direction,initial_f1=initial,source=source,status='eligible_initial' if 20<=initial<=75 else 'initial_f1_outside_20_75');audit.append(record);found=[]
                if 20<=initial<=75:
                    for index,(region,kind,operation,error) in enumerate(lesions(pred,gt,valid,direction,minimum)):
                        ideal=ideal_edit(pred,gt,region,operation)
                        ideal_f1=score(ideal,gt,valid)
                        if ideal_f1<=initial:continue
                        bbox,_=geometry(region);error_support=(region&gt&~pred&valid) if kind.startswith('FN-') else (region&pred&~gt&valid);_,point=geometry(error_support)
                        ident=f'{dataset}_{Path(name).stem}_{direction}_{kind}_{index}'
                        found.append(dict(id=ident,dataset=dataset,image_id=name,image_t1=paths[0],image_t2=paths[1],direction=direction,lesion_type=kind,lesion_index=index,operation=operation,bbox=bbox,point=point,lesion_pixels=int(region.sum()),error_pixels=error,initial_f1=initial,ideal_local_f1=ideal_f1,weak_method=source['method'],source=source,feedback_text=feedback(kind,bbox,gt.shape),status='pending_review'))
                candidates.extend(found);write_json(checkpoint,dict(audit=record,candidates=found))
            write_json(a.output/'progress.json',dict(dataset=dataset,last_image=name,prediction_records=len(audit),candidate_count=len(candidates)))
        used=set()
        for slot in ('FN-Build','FN-Demolish','FP/Coarse-Merge'):
            options=[v for v in candidates if v['dataset']==dataset and v['image_id'] not in used and (v['lesion_type']==slot or slot=='FP/Coarse-Merge' and v['lesion_type'] in ('FP','Coarse-Merge'))]
            if not options:continue
            meta=sorted(options,key=lambda v:(-v['error_pixels'],v['id']))[0];used.add(meta['image_id']);folder=a.output/'samples'/meta['id'];name=meta['image_id'];direction=meta['direction']
            pred=read(meta['source']['prediction'])>0;truth=load_truth(root,name) if dataset=='WHU-CD' else building_truth(*load_labels(root,name));gt=truth['new' if direction=='appeared' else 'demolished'];valid=~truth['ignore'] if dataset=='WHU-CD' else np.ones_like(gt)
            region=next(item[0] for index,item in enumerate(lesions(pred,gt,valid,direction,minimum)) if index==meta['lesion_index'])
            for filename,mask in [('m_init.png',pred),('gt.png',gt),('lesion.png',region),('ignore.png',~valid)]:save_image(folder/filename,mask.astype(np.uint8)*255)
            paths_out=dict(m_init=str(folder/'m_init.png'),gt_mask=str(folder/'gt.png'),lesion_mask=str(folder/'lesion.png'),gt_ignore_mask=str(folder/'ignore.png'))
            meta=dict(meta,**paths_out,input_fields=dict(image_t1=meta['image_t1'],image_t2=meta['image_t2'],m_init=paths_out['m_init']),evaluation_fields=dict(gt_mask=paths_out['gt_mask'],gt_ignore_mask=paths_out['gt_ignore_mask'],lesion_mask=paths_out['lesion_mask'],initial_f1=meta['initial_f1'],ideal_local_f1=meta['ideal_local_f1']),paired_tasks=[dict(task_type='T5.2',feedback=dict(bbox=meta['bbox'],point=meta['point'],operation=meta['operation'])),dict(task_type='T5.3',feedback=meta['feedback_text'])],gt_policy='official WHU change with semantic direction and ambiguous direction ignored' if dataset=='WHU-CD' else 'SECOND directed building transition; unchanged/unlabelled excluded')
            selected.append(meta)
    write_json(a.output/'selected.json',selected)
    write_json(a.output/'pool_summary.json',dict(status='pending_review',source_records=audit,candidates=candidates,selected_count=len(selected),selected_by_dataset=dict(Counter(r['dataset'] for r in selected)),candidate_distribution=dict(Counter(r['dataset']+'/'+r['lesion_type'] for r in candidates)),missing_real_predictions=sum(r['status']=='missing_real_prediction' for r in audit),required_samples=6,complete=len(selected)==6,api_calls=0,notes=['真实M_init不做任何GT扰动；仅病灶实例填洞。','反馈由离线GT辅助构造，仅bbox/point/操作或自然语言进入未来输入；GT和理论F1仅审核。','候选不足如实报告，不借用report/test，不放宽F1与病灶阈值。']))

if __name__=='__main__':main()
