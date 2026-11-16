"""N9 checkpointed data audit: clean construction completes before legacy I/O."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from . import n9_t3_data as base
from .io_retry import write_json,write_text

def load(path):return json.loads(path.read_text())
def dump_manifests(output,manifests):
    for kind,parts in manifests.items():
        for part,rows in parts.items():
            p=output/f'{kind.lower().replace(".","_")}_clean_{part}'
            write_json(p.with_suffix('.json'),rows)
            write_text(p.with_suffix('.jsonl'),''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def clean(a,query,sources,old):
    pool={p:defaultdict(list) for p in base.PARTS};source_audit=[]
    for part in base.PARTS:
        for name in sources[part]:
            ck=a.output/'checkpoints/sources'/part/(Path(name).stem+'.json')
            if ck.exists():records=load(ck)
            else:
                single={p:([name] if p==part else []) for p in base.PARTS}
                _,records=base.catalogue(a.dataset,single);write_json(ck,records)
            source_audit.extend(records)
            for r in records:
                if not r['reasons']:pool[part][(r['source'],r['target'])].append({k:r[k] for k in ('name','source','target','component','geometry')})
    write_json(a.output/'external_component_audit.json',source_audit)
    manifests={k:{p:[] for p in base.PARTS} for k in ('T3.1','T3.3')};audit=[];mapping=[]
    for part in base.PARTS:
        for name in query[part]:
            ck=a.output/'checkpoints/queries'/part/(Path(name).stem+'.json')
            if ck.exists():value=load(ck)
            else:
                single={p:([name] if p==part else []) for p in base.PARTS}
                m,r,links=base.rebuild(a.dataset,single,pool,old,a.output)
                value=dict(manifests=m,audit=r,mapping=links);write_json(ck,value)
            for kind in manifests:manifests[kind][part].extend(value['manifests'][kind][part])
            audit.extend(value['audit']);mapping.extend(value['mapping'])
            # Small partial progress is durable; no rebuilding completed images after timeout.
            write_json(a.output/'clean_progress.json',dict(status='building',last_query=name,completed_query_checkpoints=len(list((a.output/'checkpoints/queries').glob('*/*.json')))))
    dump_manifests(a.output,manifests);write_json(a.output/'rebuild_audit.json',audit);write_json(a.output/'old_new_mapping.json',mapping)
    parts={name:p for p in base.PARTS for name in query[p]}
    metadata=[dict(task_type=r['task_type'],split=parts[Path(r['image_t1']).name],original_task=r) for r in old]
    summary=base.summarize(metadata,audit,source_audit,manifests)
    summary.update(legacy_audit_status='pending_separate_stage',checkpointed=True)
    write_json(a.output/'summary.json',summary);write_json(a.output/'clean_progress.json',dict(status='complete',tasks=sum(len(v) for p in manifests.values() for v in p.values())))
    write_text(a.output/'qc_report.md','# N9 clean 数据构造\n\n六manifest已落盘，全部pending_review；自动几何QC不等于人工批准。旧任务逐例审计另行checkpoint，未完成旧审计不会被记作成功。完整计数见summary.json；失败/延期见legacy_io_status.json。\n')

def legacy(a,query,old):
    parts={name:p for p in base.PARTS for name in query[p]};records=[];deferred=[]
    for row in old:
        ck=a.output/'checkpoints/legacy'/row['task_type']/(row['id']+'.json')
        if ck.exists():record=load(ck)
        elif row['id'] in a.defer_old_id:
            deferred.append(dict(id=row['id'],status='deferred_io',reason='Explicit known legacy I/O stall; original mask not replaced',original_task=row));continue
        else:
            write_json(a.output/'legacy_io_status.json',dict(status='reading',current_id=row['id'],completed=len(records),deferred=deferred,original_task=row))
            try:
                result,_=base.audit_old([row],parts);record=result[0];write_json(ck,record)
            except OSError as exc:
                deferred.append(dict(id=row['id'],status='io_error',error=str(exc),original_task=row));continue
        records.append(record)
        write_json(a.output/'old_task_audit.json',records)
        write_json(a.output/'t3_empty_abstention_diag.json',dict(status='sealed_not_evaluated',tasks=[r for r in records if r['query_support']['pixels']==0]))
    write_json(a.output/'legacy_io_status.json',dict(status='incomplete' if deferred else 'complete',completed=len(records),expected=len(old),deferred=deferred))
    summary=load(a.output/'summary.json');summary['legacy_audit_status']='incomplete' if deferred else 'complete';summary['legacy_audit_completed']=len(records);write_json(a.output/'summary.json',summary)

def main():
    p=argparse.ArgumentParser()
    for key in ('run','night','output','review'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=base.DATASET);p.add_argument('--phase',choices=['clean','old','review'],required=True);p.add_argument('--defer-old-id',action='append',default=[]);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    query,sources=base.inventory(a.run,a.night);old=base.old_tasks(a.run)
    write_json(a.output/'partition.json',dict(query=query,external_sources=sources))
    write_json(a.output/'predeclaration.json',dict(**base.POLICY,query_images=query,external_source_images=sources,execution_order='checkpointed clean first, legacy audit second; no replacement legacy pixels'))
    if a.phase=='clean':base.check_encoding(a.dataset);clean(a,query,sources,old)
    elif a.phase=='old':legacy(a,query,old)
    else:
        from .n9_t3_review import render_review
        manifests={k:{part:load(a.output/f'{k.lower().replace(".","_")}_clean_{part}.json') for part in base.PARTS} for k in ('T3.1','T3.3')}
        render_review(load(a.output/'old_task_audit.json'),manifests,a.output,a.review)
        # The legacy renderer currently emits this extra README; no persistent duplicate document.
        (a.review/'README.md').unlink(missing_ok=True)

if __name__=='__main__':main()
