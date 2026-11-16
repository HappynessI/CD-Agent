"""Validate actual sampled assets without approving any candidate."""
import json,os,re,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from agent_baseline.whu_gt import load_truth
from .second_labels import building_truth,load_labels
from .task_reconstruct_page import OUT,mask,read

def border(m):return bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
def main():
    assert os.environ.get('SLURM_JOB_ID')
    selections={k:read(OUT/p/'selected.json') for k,p in dict(t5='t5/visual_selection',t3='t3/revision_interior_examples',t44='t44/road_stage/revision_interior_targets').items()}
    if (OUT/'t44/visual_selection/selected.json').exists():selections['t44']=read(OUT/'t44/visual_selection/selected.json')
    results=[];errors=[]
    for module,rows in selections.items():
        for row in rows:
            checks={};diagnostic=row.get('diagnostic_only',False)
            with Image.open(row['image_t1']) as im:shape=(im.height,im.width)
            with Image.open(row['image_t2']) as im:checks['paired_image_shape']=(im.height,im.width)==shape
            checks['not_auto_approved']=row.get('status',row.get('review_status'))=='pending_review'
            if module=='t5':
                p=mask(row['m_init']);gt=mask(row['gt_mask']);region=mask(row['lesion_mask']);valid=~mask(row['gt_ignore_mask'])
                checks['real_prediction_unchanged']=np.array_equal(p,mask(row['source']['prediction']))
                data=Path(row['image_t1']).parent.parent;name=Path(row['image_t1']).name
                truth=load_truth(data,name) if row['dataset']=='WHU-CD' else building_truth(*load_labels(data,name))
                checks['official_direction_gt_unchanged']=np.array_equal(gt,truth['new' if row['direction']=='appeared' else 'demolished'])
                checks['initial_f1_in_range']=20<=row['initial_f1']<=75
                error=region & valid & ((gt & ~p) if row['operation']=='补检' else (p & ~gt))
                checks['actual_error_count']=int(error.sum())==row['error_pixels']
                checks['error_size']=row['error_pixels']>=(800 if row['dataset']=='WHU-CD' else 400)
                x,y=row['point'];checks['feedback_point_on_error']=bool(error[y,x])
                checks['lesion_connected']=ndi.label(region,np.ones((3,3)))[1]==1
                checks['lesion_hole_free']=np.array_equal(region,ndi.binary_fill_holes(region))
                checks['paired_tasks']=len(row['paired_tasks'])==2
            elif module=='t3':
                e=mask(row['example_mask']);full=mask(row['gt_full']);rem=mask(row['gt_rem'])
                checks['gt_full_rem_consistent']=np.array_equal(rem,full & ~e) and not (e & ~full).any()
                checks['whole_gt_interior']=not border(full)
                checks['single_example']=ndi.label(e,np.ones((3,3)))[1]==1
                checks['example_hole_free']=np.array_equal(e,ndi.binary_fill_holes(e))
                checks['example_geometry']=200<=e.sum()<=5000 and row['hole_ratio']<=.05 and row['compactness']>=.45
                checks['two_visual_matches']=sum(not x['is_example'] and x['is_match'] and x['cosine']>=.75 and x['rgb_distance']<=.15 for x in row['instances'])>=2
                checks['remaining_area']=int(rem.sum())>=400
            else:
                ev=row['evaluation'];t=mask(ev['target_mask']);d=mask(ev['distractor_mask']);a=mask(ev['anchor_mask'])
                checks['distinct_target_distractor']=not (t&d).any() and bool(d.any()) and t.sum()>=400
                checks['anchor_present']=bool(a.any())
                checks['no_direction_or_attribute_words']=not any(word in row['instruction'] for word in ('新建','拆除','红色','蓝色','大面积','小型','大型'))
                checks['whole_target_interior']=not border(t)
            entry=dict(module=module,id=row['id'],diagnostic_only=diagnostic,checks={k:bool(v) for k,v in checks.items()})
            results.append(entry)
            for k,v in checks.items():
                if not v and not diagnostic:errors.append(dict(id=row['id'],check=k))
    html=(OUT/'index.html').read_text();paths=re.findall(r'(?:src|href)="([^"]+)"',html)
    for path in paths:
        if '://' in path:errors.append(dict(path=path,check='external_dependency'))
        elif path and not path.startswith('#') and not (OUT/path).is_file():errors.append(dict(path=path,check='missing_local_link'))
    js_check='node_unavailable'
    if shutil.which('node'):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.js';p.write_text(re.search(r'<script>(.*?)</script>',html,re.S).group(1))
            checked=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
            js_check='passed' if checked.returncode==0 else checked.stderr
            if checked.returncode:errors.append(dict(check='javascript_syntax',error=checked.stderr))
    report=dict(status='passed' if not errors else 'failed',samples=results,errors=errors,javascript_syntax=js_check,
        offline_links_checked=len(paths),browser_runtime_test='not_run',human_approval=False,api_calls=0)
    (OUT/'sample_qc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False))
    if errors:raise RuntimeError('Sample QC failed; see sample_qc.json')
if __name__=='__main__':main()
