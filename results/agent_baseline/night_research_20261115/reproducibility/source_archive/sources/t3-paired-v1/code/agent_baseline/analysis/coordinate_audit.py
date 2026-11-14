"""Strict, read-only coordinate error audit and development-only paired task selection."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import itertools
from pathlib import Path


def public(call):
    try:
        value=json.loads(call.get('result_text') or '{}')
        return value if isinstance(value,dict) else {}
    except (ValueError,TypeError): return {}


def coordinate_error(call,shape):
    """Require both the coordinate-specific rejection and numerical out-of-bounds.

    An unordered/empty in-bounds box, a large-box guard, a malformed field, or an
    unrelated bad_args with stray out-of-bounds arguments is not counted.
    """
    if call.get('error_code')!='bad_args' or not shape or len(shape)!=2:return None
    message=public(call).get('message','')
    args=call.get('args') or {};h,w=shape
    fields=[]
    if 'ordered box inside the full image' in message:
        field=message.split()[0];value=args.get(field)
        if isinstance(value,(list,tuple)) and len(value)==4 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in value):
            if any(x<0 or x>maximum for x,maximum in zip(value,(w,h,w,h))):fields.append(dict(field=field,value=value))
    elif message in ('points must lie inside the full image.','Point must be inside the image.'):
        if isinstance(args.get('exemplar'),dict):args=args['exemplar']
        field='points' if 'points' in args else 'point'
        values=args.get(field,[]);values=values if field=='points' else [values]
        for point in values:
            if isinstance(point,(list,tuple)) and len(point)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in point):
                if not (0<=point[0]<w and 0<=point[1]<h):fields.append(dict(field=field,value=point))
    return dict(message=message,shape=list(shape),fields=fields) if fields else None


def sessions(root,configs=None,include_test=False):
    """Walk only config/task/split/session, never descend into image/model caches."""
    for cfg in sorted(root.iterdir(),key=lambda p:('t13' not in p.name,p.name)):
        if configs and cfg.name not in configs:continue
        if cfg.name.startswith(('.', 'code')) or not cfg.is_dir():continue
        if not include_test and 'test' in cfg.name.lower():continue
        for task in cfg.iterdir():
            if not task.name.startswith('T') or not task.is_dir():continue
            for split in task.iterdir():
                if not split.is_dir() or (not include_test and split.name=='test'):continue
                for item in split.iterdir():
                    if item.is_dir():yield cfg.name,task.name,split.name,item


def empty_session(directory):
    """Observed initialization-only directory; missing metadata with real logs remains an error."""
    return not (directory/'input.json').exists() and not (directory/'final.json').exists() and all(
        not (directory/name).exists() or (directory/name).stat().st_size==0
        for name in ('steps.jsonl','messages.jsonl'))


def audit(item):
    config,task,split,directory=item
    meta=json.loads((directory/'input.json').read_text())
    sources=' '.join(meta.get('source_images') or [])
    dataset=next((x for x in ('SECOND','LEVIR','WHU','xBD') if x.lower() in sources.lower()),'unknown')
    shape=meta.get('image_shape');events=[];bad=0;last=None;retry=0
    steps_path=directory/'steps.jsonl'
    if steps_path.exists():
        with steps_path.open() as f:
            for line_number,line in enumerate(f,1):
                row=json.loads(line)
                for ordinal,call in enumerate(row.get('calls',[]),1):
                    if call.get('error_code')=='bad_args':bad+=1
                    error=coordinate_error(call,shape)
                    signature=(call.get('tool'),json.dumps(call.get('args'),sort_keys=True))
                    if error:
                        repeated=signature==last
                        retry+=int(repeated)
                        events.append(dict(step=row['step'],call=ordinal,line=line_number,tool=call['tool'],args=call.get('args'),unchanged_retry=repeated,**error))
                    last=signature if error else None
    final_path=directory/'final.json';final=json.loads(final_path.read_text()) if final_path.exists() else {}
    return dict(coordinate_classifier_version=2,session=str(directory),config=config,task=task,split=split,dataset=dataset,
        name=Path((meta.get('source_images') or [directory.name])[0]).stem,
        harness=meta.get('backends',{}),shape=shape,coordinate_bad_args=len(events),other_bad_args=bad-len(events),
        unchanged_coordinate_retries=retry,events=events,complete=bool(final),
        termination=final.get('termination_reason'),f1=(final.get('eval') or {}).get('f1'),
        cost_cny=final.get('cost_cny'),evidence=str(steps_path))


def load_journal(path):
    """Recover only our interrupted last write, keeping its original bytes as evidence."""
    if not path.exists():return []
    rows=[];end=0
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line:break
            try:row=json.loads(line)
            except ValueError:
                remainder=f.read()
                if remainder:raise ValueError('Invalid journal record before the final line: '+str(path))
                path.with_suffix('.partial').write_bytes(line)
                with path.open('r+b') as out:out.truncate(end)
                break
            rows.append(row);end=f.tell()
    return rows


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def report(rows,out,failures):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['dataset'],row['task'],row['config'],row['split'])].append(row)
    groups=[]
    for key,items in sorted(grouped.items()):
        groups.append(dict(zip(('dataset','task','config','split'),key),sessions=len(items),
            coordinate_bad_args=sum(r['coordinate_bad_args'] for r in items),
            affected_sessions=sum(bool(r['coordinate_bad_args']) for r in items),
            other_bad_args=sum(r['other_bad_args'] for r in items),
            examples=[dict(session=r['session'],evidence=r['evidence'],event=e) for r in items for e in r['events']][:3]))
    result=dict(sessions=len(rows),groups=groups,read_failures=failures,definition=coordinate_error.__doc__,
        test_logs='Existing logs only if explicitly included; never used for development task selection or new inference')
    write_json(out/'summary.json',result)
    return result


def select_dev(rows,manifest,out):
    tasks=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    by_name={Path(t['image_t1']).stem:t for t in tasks};counts=Counter();evidence=defaultdict(list)
    for row in rows:
        if row['task']=='T1.3' and row['name'] in by_name and row['split']!='test':
            counts[row['name']]+=row['coordinate_bad_args']
            if row['events']:evidence[row['name']].append(dict(session=row['session'],count=row['coordinate_bad_args']))
    chosen=sorted(by_name,key=lambda name:(-counts[name],name))[:10]
    (out/'paired_tasks.jsonl').write_text(''.join(json.dumps(by_name[n],ensure_ascii=False)+'\n' for n in chosen))
    write_json(out/'selection.json',dict(source_manifest=str(manifest),count=len(chosen),
        rationale='Rank historical coordinate bad_args within the existing development inventory; fill deterministically if fewer than 10 affected images',
        tasks=[dict(name=n,historical_coordinate_errors=counts[n],evidence=evidence[n]) for n in chosen]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--include-test-logs',action='store_true');parser.add_argument('--config',action='append')
    parser.add_argument('--dev-manifest',type=Path)
    parser.add_argument('--recheck-other-errors',action='store_true',help='Reclassify only previously non-coordinate bad_args, preserving all prior journal entries.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    journal=args.output/'sessions.jsonl';rows=list({r['session']:r for r in load_journal(journal)}.values());done={r['session'] for r in rows};failures=[]
    excluded_path=args.output/'empty_sessions.json'
    excluded=json.loads(excluded_path.read_text()) if excluded_path.exists() else []
    def work(item):
        try:
            if empty_session(item[3]):return None,dict(session=str(item[3]),excluded='initialization_only_no_metadata_or_steps')
            return audit(item),None
        except (OSError,ValueError,KeyError,TypeError) as error:return None,dict(session=str(item[3]),error=str(error))
    pending=(iter((r['config'],r['task'],r['split'],Path(r['session'])) for r in rows if r['other_bad_args'] and r.get('coordinate_classifier_version',0)<2) if args.recheck_other_errors else
             (item for item in sessions(args.root,args.config,args.include_test_logs) if str(item[3]) not in done))
    with journal.open('a') as sink,ThreadPoolExecutor(max_workers=args.workers) as pool:
        while batch:=list(itertools.islice(pending,args.workers*4)):
            for row,error in pool.map(work,batch):
                if error and error.get('excluded'):
                    if error not in excluded:excluded.append(error)
                    write_json(excluded_path,excluded);continue
                if error:
                    failures.append(error);write_json(args.output/'read_failures.json',failures);continue
                sink.write(json.dumps(row,ensure_ascii=False)+'\n');sink.flush();rows.append(row)
                if len(rows)%100==0:print('sessions_saved='+str(len(rows)),flush=True)
    rows=list({r['session']:r for r in rows}.values())
    write_json(args.output/'read_failures.json',failures)
    report(rows,args.output,failures)
    if args.dev_manifest:select_dev(rows,args.dev_manifest,args.output)
    write_json(args.output/'metric.json',dict(sessions=len(rows),read_failures=len(failures),empty_sessions=len(excluded),cost_cny=0))
    print(f'Completed sessions={len(rows)} read_failures={len(failures)}',flush=True)
    if failures:raise SystemExit('Read failures retained; investigate before calling coverage complete')


if __name__=='__main__':main()
