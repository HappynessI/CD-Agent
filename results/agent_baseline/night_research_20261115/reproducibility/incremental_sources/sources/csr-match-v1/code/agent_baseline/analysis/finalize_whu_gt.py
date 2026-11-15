"""Full frozen WHU C0/C1 correction, validation, offline scoring and gated cleanup.
Run on a Slurm CPU node. No inference or source-dataset mutation.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from agent_baseline.whu_gt import VERSION, GT_DIR, load_truth, read_raw, paths, update_task
from agent_baseline.evaluation import load_reference, pixel_counts
from agent_baseline.analysis.make_whu_tasks import generate
from agent_baseline.analysis.revise_whu_official import metrics
from agent_baseline.tasks import read_tasks


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def frozen(run):
    result={s:(run/'whu'/f'names_{n}.txt').read_text().split() for s,n in [('val','val128'),('dev','dev60')]}
    if len(result['val'])!=128 or len(result['dev'])!=60 or set(result['val'])&set(result['dev']):
        raise ValueError('Expected unchanged disjoint report128 / dev60')
    return result


def refresh_inventory(a):
    """Discover runnable JSONL manifests, including those outside tasks directories."""
    inventory=json.loads((a.output/'inventory.json').read_text())
    known={r['path'] for r in inventory['manifests']};extra=[];checked=0
    for root in (a.run,a.repo/'results'):
        for folder,dirs,files in os.walk(root):
            # Session archives and assets contain evidence, not runnable task manifests.
            if 'final.json' in files or ('input.json' in files and 'steps.jsonl' in files):
                dirs[:]=[]
                continue
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','images','masks','insts','cache','budget','ask_eval') and not d.startswith('code_')]
            for filename in files:
                if not filename.endswith('.jsonl') or filename in ('steps.jsonl','messages.jsonl'):continue
                p=Path(folder)/filename;checked+=1
                if str(p) in known or a.output in p.parents:continue
                with p.open() as f:
                    first=f.readline(1000000)
                try:head=json.loads(first)
                except ValueError:continue
                if not isinstance(head,dict) or not {'task_type','gt_mask','image_t2'}<=head.keys():continue
                rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                whu=[r for r in rows if 'WHU-CD' in r.get('image_t2','')]
                if not whu:continue
                item=dict(path=str(p),rows=len(whu),tasks=sorted({r['task_type'] for r in whu}))
                inventory['manifests'].append(item);extra.append(item);known.add(str(p))
    write(a.output/'inventory.json',inventory)
    write(a.output/'additional_manifest_audit.json',dict(jsonl_checked=checked,additional=extra))
    history=a.output/'manifest_history.json'
    if history.exists():
        archive=json.loads(history.read_text());saved={r['path'] for r in archive['manifests']}
        for r in extra:
            if r['path'] not in saved and any(t in ('T1.1xC0','T1.1xC1') for t in r['tasks']):
                archive['manifests'].append(dict(path=r['path'],rows=[json.loads(x) for x in Path(r['path']).read_text().splitlines() if x.strip()]))
        write(history,archive)


def migrate(a):
    inventory=json.loads((a.output/'inventory.json').read_text())
    history=a.output/'manifest_history.json'
    if not history.exists():
        write(history,dict(archive_only=True,not_an_evaluation_manifest=True,manifests=[
            dict(path=r['path'],rows=[json.loads(x) for x in Path(r['path']).read_text().splitlines() if x.strip()])
            for r in inventory['manifests'] if any(t in ('T1.1xC0','T1.1xC1') for t in r['tasks'])]))
        write(a.output/'legacy_preparation_record.json',json.loads((a.run/'whu/preparation.json').read_text()))
    generate(a.dataset,a.run)
    changed=[]
    for item in inventory['manifests']:
        if not any(t in ('T1.1xC0','T1.1xC1') for t in item['tasks']):continue
        p=Path(item['path']);rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        revised=[update_task(r,a.run) if 'WHU-CD' in r.get('image_t2','') and r['task_type'] in ('T1.1xC0','T1.1xC1') else r for r in rows]
        tmp=p.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in revised));tmp.replace(p)
        changed.append(dict(path=str(p),rows=len(rows)))
    # Portable Git mirror: same rasters, references resolve relative to manifests.
    shutil.copytree(a.run/'whu'/GT_DIR,a.output/GT_DIR,dirs_exist_ok=True)
    for p in (a.run/'whu/tasks').glob('*.jsonl'):
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for r in rows:
            gt,ignore=paths(a.run,r['id'],r['change_direction'])
            r['gt_mask']='../'+GT_DIR+'/'+gt.parent.name+'/'+gt.name
            r['gt_ignore_mask']='../'+GT_DIR+'/ignore/'+gt.name if ignore else None
        target=a.output/'tasks'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(a.output/'manifest_migration.json',dict(version=VERSION,updated=changed,
         t54='Independent map-update semantics not changed without an explicit decision'))
    return changed


def validate(a, migrated):
    split_names=frozen(a.run);rows=[]
    for split,names in split_names.items():
        for name in names:
            expected=load_truth(a.dataset,name)
            for kind,want in expected.items():
                got=read_raw(a.run/'whu'/GT_DIR/kind/name)>0
                mirror=read_raw(a.output/GT_DIR/kind/name)>0
                if not np.array_equal(got,want) or not np.array_equal(mirror,want):raise ValueError(f'GT mismatch: {name}/{kind}')
            g,n,d,i=(expected[k] for k in ('c0','new','demolished','ignore'))
            assert np.array_equal(n|d|i,g) and not (n&d).any() and not ((n|d)&i).any()
            assert not (i&~g).any()
            for part in ('img1','img2','slabel1','slabel2','label_unch0chi1'):
                with Image.open(a.dataset/part/name) as im:
                    if im.size!=(g.shape[1],g.shape[0]):raise ValueError(f'Shape mismatch: {part}/{name}')
            rows.append(dict(name=name,split=split,total_pixels=g.size,official_pixels=int(g.sum()),
                appeared_pixels=int(n.sum()),disappeared_pixels=int(d.sum()),ignore_pixels=int(i.sum()),
                official_empty=not bool(g.any()),tiny_official_positive=0<int(g.sum())<64,
                direction_fully_ambiguous=bool(g.any()) and not bool((n|d).any())))
            write(a.output/'images'/f'{Path(name).stem}.json',rows[-1])
    manifest_rows=0
    for p in [Path(x['path']) for x in migrated]+list((a.output/'tasks').glob('*.jsonl')):
        loaded=read_tasks(p)
        for row in loaded:
            for field in ['image_t1','image_t2','gt_mask']:
                if not Path(row[field]).is_file():raise FileNotFoundError(row[field])
            gt,valid=load_reference(row['gt_mask'],row.get('gt_ignore_mask'))
            if row['change_direction']!='any' and not row.get('gt_ignore_mask'):
                raise ValueError('Direction manifest lacks ignore reference')
            if row['meta'].get('whu_gt_version')!=VERSION:raise ValueError('Stale manifest version')
            if '/whu/gt/' in row['gt_mask']:raise ValueError('Legacy GT reference still active')
            manifest_rows+=1
    summary={}
    for split in split_names:
        rr=[x for x in rows if x['split']==split]
        summary[split]=dict(images=len(rr),**{k:sum(x[k] for x in rr) for k in ['total_pixels','official_pixels','appeared_pixels','disappeared_pixels','ignore_pixels','official_empty','tiny_official_positive','direction_fully_ambiguous']},
            images_with_ignore=sum(x['ignore_pixels']>0 for x in rr))
    result=dict(version=VERSION,passed=True,images=len(rows),splits=summary,checked_manifest_rows=manifest_rows,
        doubts=[x for x in rows if x['tiny_official_positive'] or x['direction_fully_ambiguous']],
        gt_root=str(a.run/'whu'/GT_DIR),git_mirror=str(a.output/GT_DIR))
    write(a.output/'validation.json',result);return result


def rescore(a):
    # Reuse the complete already-audited prediction inventory, not model calls.
    source=a.repo/'results/agent_baseline/data_revision_20261113/v2/whu/rescored_predictions.json'
    inventory=json.loads(source.read_text());cache={};journal=a.output/'rescored_predictions.jsonl'
    done={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                r=json.loads(line);done[(r['config'],r['direction'],r['prediction'])]=r
    missing=[]
    with journal.open('a') as f:
        for item in inventory:
            key=(item['config'],item['direction'],item['prediction'])
            if key in done:continue
            predpath=Path(item['prediction'])
            if not predpath.is_file():missing.append(str(predpath));continue
            name=item['name'];direction=item['direction'];ref=(name,direction)
            if ref not in cache:
                gtpath,ignore=paths(a.run,name,direction);cache[ref]=load_reference(gtpath,ignore)
            gt,valid=cache[ref];pred=read_raw(predpath)>0
            c=pixel_counts(pred,gt,valid)
            row=dict(config=item['config'],split=item['split'],direction=direction,name=name,prediction=str(predpath),
                gt_version=VERSION,corrected=metrics(c),ignored_pixels=int((~valid).sum()),
                ignored_prediction_pixels=int((pred&~valid).sum()),evaluated_pixels=int(valid.sum()),
                historical_score=item.get('historical_eval',item.get('old')))
            f.write(json.dumps(row)+'\n');f.flush();done[key]=row
    write(a.output/'missing_predictions.json',missing)
    if missing or len(done)!=len(inventory):raise ValueError('Prediction inventory incomplete; partial scores preserved')
    groups=defaultdict(list)
    for row in done.values():groups[(row['config'],row['split'],row['direction'])].append(row)
    report=[]
    for (config,split,direction),rows in groups.items():
        c={k:sum(r['corrected'][k] for r in rows) for k in ['tp','fp','fn']}
        report.append(dict(config=config,split=split,direction=direction,predictions=len(rows),**metrics(c),
            mean_image_f1=float(np.mean([r['corrected']['f1'] for r in rows])),
            ignored_prediction_pixels=sum(r['ignored_prediction_pixels'] for r in rows)))
    write(a.output/'rescore_summary.json',report)
    return dict(predictions=len(done),groups=len(report),missing=0,
        by_direction={d:sum(r['direction']==d for r in done.values()) for d in ('any','appeared','disappeared')})


def cleanup(a):
    validation=json.loads((a.output/'validation.json').read_text())
    summary=json.loads((a.output/'summary.json').read_text())
    if not validation['passed'] or validation['images']!=188 or summary['rescore']['missing']:
        raise ValueError('Validated full coverage and scoring required before cleanup')
    migration=json.loads((a.output/'manifest_migration.json').read_text())
    validate(a,migration['updated'])
    # Exact derived GT subtree only. Raw semantic/official labels and predictions are outside it.
    root=a.run/'whu/gt';deleted=[]
    if not root.exists() and summary.get('cleanup_complete'):
        write(a.output/'metric.json',dict(stage='cleanup',**summary))
        return
    if root.is_symlink():raise ValueError('Legacy derived GT directory unexpectedly symlinked')
    for kind in ['c0','new','demolished']:
        folder=root/kind
        for p in sorted(folder.glob('*.png')):
            if p.is_symlink():raise ValueError('Refuse to remove an unexpected source-label link')
            deleted.append(str(p));p.unlink()
        if folder.exists():folder.rmdir()
    if root.exists() and not any(root.iterdir()):root.rmdir()
    write(a.output/'cleanup.json',dict(deleted_gt_files=len(deleted),paths=deleted,
        raw_dataset_touched=False,historical_predictions_touched=False,
        historical_records='Existing input.json and published review pages retain provenance, not active scoring references',
        t54='Independent map-update truth retained pending explicit scope decision'))
    summary['cleanup_complete']=True;summary['deleted_gt_files']=len(deleted)
    write(a.output/'summary.json',summary);write(a.output/'metric.json',dict(stage='cleanup',**summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cleanup',action='store_true')
    a=p.parse_args()
    if a.cleanup:cleanup(a);return
    refresh_inventory(a)
    migrated=migrate(a);v=validate(a,migrated);scores=rescore(a)
    summary=dict(version=VERSION,images=v['images'],splits=v['splits'],updated_manifests=len(migrated),
        updated_manifest_rows=sum(x['rows'] for x in migrated),canonical_task_rows=188*3,
        gt_root=v['gt_root'],git_mirror=v['git_mirror'],rescore=scores,api_calls=0,cost_cny=0,cleanup_complete=False)
    write(a.output/'summary.json',summary);write(a.output/'metric.json',dict(stage='build_and_rescore',**summary));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
