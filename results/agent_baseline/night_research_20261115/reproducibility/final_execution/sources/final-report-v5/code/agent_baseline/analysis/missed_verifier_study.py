"""Frozen development candidates; labels are read exclusively by score()."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from .io_retry import read_image, write_json, write_text, save_image
from .missed_sources import load_json, load_npz, save_npz, specs, rows_for, unpack, pack, pixel_counts, split_components

SOURCES = ('S1_02', 'S2', 'S3', 'S4', 'S5')
GROUPS = ('second_appeared', 'second_disappeared', 'levir_any')
VARIANTS = ('four', 'som', 'structured')
MODEL = 'qwen3.7-flash'


def progress(args, stage, completed=0, failures=0):
    root=args.out.parent
    charges=[load_json(p).get('cost_cny',0) for p in root.glob('*/answers/*/usage/*.json')]
    write_json(root/'metric.json',dict(stage=stage,cost_cny=sum(charges),cumulative=True,
        completed=completed,failures=failures,result_path=str(args.out),model=MODEL))


def ranked_candidates(by_source):
    """Round robin sources, descending area within source; exact duplicates once."""
    queues = {s: sorted(enumerate(by_source[s]), key=lambda x: (-int(x[1].sum()), x[0])) for s in SOURCES}
    found = []; seen={}
    for i in range(max((len(v) for v in queues.values()), default=0)):
        for source in SOURCES:
            if i >= len(queues[source]):
                continue
            original, mask = queues[source][i]
            identity=np.packbits(mask).tobytes()
            identical=seen.get(identity)
            if identical is not None:
                identical['sources'].append(source)
                continue
            found.append(dict(source=source, sources=[source], original=original, area=int(mask.sum()), mask=mask))
            seen[identity]=found[-1]
    return found


def union(masks, shape):
    out = np.zeros(shape, bool)
    for mask in masks:
        out |= mask
    return out


def prepare(args):
    settings = dict(version=1, groups=GROUPS, sources=SOURCES, ranking='source_round_robin_area_desc_original_index',
                    image_selection='lexicographic_names', images_per_group=args.images, top=20, model=MODEL,
                    source_work=str(args.work.resolve()), run=str(args.run.resolve()))
    path = args.out/'settings.json'
    if path.exists() and load_json(path) != json.loads(json.dumps(settings)):
        raise ValueError('Settings differ; use a separate output directory')
    write_json(path, settings)
    manifest = []
    for spec in specs(args.run, args.work):
        if spec['group'] not in GROUPS:
            continue
        rows = sorted(rows_for(args.run, spec), key=lambda r:r['id'])
        if args.images:
            rows = rows[:args.images]
        for row in rows:
            stem = Path(row['id']).stem
            dest = args.out/'prepared'/spec['group']/f'{stem}.json'
            if dest.exists():
                manifest.append(load_json(dest)); continue
            by_source = {}
            for source in SOURCES:
                z = load_npz(args.work/'candidates'/spec['group']/source/f'{stem}.npz')
                by_source[source] = list(unpack(z['masks'], tuple(z['shape'])))
            ranked = ranked_candidates(by_source)
            chosen = ranked[:20]
            shape = tuple(int(v) for v in z['shape'])
            masks_path = dest.with_suffix('.npz')
            save_npz(masks_path, masks=pack([r['mask'] for r in chosen], shape), shape=shape)
            records = []
            for i, record in enumerate(chosen):
                y,x = np.nonzero(record['mask'])
                records.append(dict(uid=f"{spec['group']}:{stem}:{i+1}", rank=i+1, source=record['source'],
                    sources=record['sources'], original=record['original'], area=record['area'],
                    bbox=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]))
            # GT paths/labels are deliberately excluded from the frozen inference manifest.
            item = dict(group=spec['group'], stem=stem, shape=shape, candidates=records,
                before_dedup=sum(len(v) for v in by_source.values()), after_dedup=len(ranked),
                masks=str(masks_path), image_t1=row['image_t1'], image_t2=row['image_t2'],
                direction=spec['direction'], scope=spec['scope'])
            write_json(dest,item); manifest.append(item)
    write_json(args.out/'manifest.json',manifest)
    write_json(args.out/'prepare_summary.json',dict(images=len(manifest), candidates=sum(len(r['candidates']) for r in manifest),
        requests_three_arms=3*sum(len(r['candidates']) for r in manifest), status='complete'))
    reuse_smoke_answers(args,manifest)
    progress(args,'prepared',len(manifest))


def reuse_smoke_answers(args, manifest):
    """Same frozen candidates and requests: reuse paid smoke responses once."""
    if args.out.name != 'pilot': return
    source=args.out.parent/'smoke'
    if not (source/'manifest.json').exists(): return
    old_settings=load_json(source/'settings.json');new_settings=load_json(args.out/'settings.json')
    for key in ('version','ranking','top','model','source_work','run'):
        if old_settings[key]!=new_settings[key]:raise ValueError('Smoke request policy differs')
    old={(r['group'],r['stem']):r for r in load_json(source/'manifest.json')}
    reused=[]
    for item in manifest:
        prior=old.get((item['group'],item['stem']))
        if prior is None:continue
        for field in ('candidates','image_t1','image_t2','direction','scope'):
            if item[field]!=prior[field]:raise ValueError('Smoke candidates or inference inputs differ')
        for candidate in item['candidates']:
            name=candidate['uid'].replace(':','__')+'.json'
            for variant in VARIANTS:
                origin=source/'answers'/variant/name;dest=args.out/'answers'/variant/name
                if not origin.exists():continue
                if not dest.exists():
                    response=load_json(origin);event_path=origin.parent/'usage'/name
                    if event_path.exists() and load_json(event_path).get('raw'):
                        response.update(parse_response(load_json(event_path)['raw']))
                    response.update(reused_from=str(origin),original_cost_cny=response.get('cost_cny',0),cost_cny=0.)
                    write_json(dest,response)
                reused.append(str(dest))
    write_json(args.out/'reuse_smoke.json',dict(reused_requests=len(reused),responses=reused,
        billing='No new requests; original usage remains charged once in smoke'))


def marked_images(rgb, mask, bbox, variant):
    from agent_baseline.view import ask_images
    from render import crop_box
    images = ask_images(rgb, mask, bbox, True)
    if variant != 'four':
        for i, image in enumerate(images):
            box = crop_box(bbox, mask.shape, **(dict(scale=10,min_side=256) if i>=2 else {}))
            y,x = np.nonzero(mask)
            closest=int(np.argmin((x-x.mean())**2+(y-y.mean())**2))
            px = int((x[closest]-box[0])*image.width/(box[2]-box[0]))
            py = int((y[closest]-box[1])*image.height/(box[3]-box[1]))
            draw = ImageDraw.Draw(image)
            # Small numbered mark with leader; same candidate and geometry in both dates.
            tx,ty = max(2,min(image.width-18,px+10)), max(2,min(image.height-15,py-18))
            draw.line((px,py,tx,ty+7),fill='yellow',width=1)
            draw.rectangle((tx,ty,tx+15,ty+13),fill='yellow')
            draw.text((tx+3,ty), '1',fill='black')
    return images


def messages(item, record, mask, variant, directory):
    from agent_baseline.state import Session
    from agent_baseline.scope_t13 import session_prompts
    session = Session(mask.shape,target='building',change_direction=item['direction'],scope=item['scope'])
    prompts = session_prompts(session)
    question = prompts['CORRESPONDENCE']
    if variant != 'four':
        question += '\nThe yellow mark 1 identifies the candidate inside the existing red contour in all four views; it is an annotation, not image evidence.'
    if variant == 'structured':
        question += ('\nUse the same change definition above. Before the final change decision, describe the object at T1 and T2, '
            'whether they are the same persisting object, and any appearance-only explanation. '
            'Return JSON with t1_state, t2_state, same_object, appearance_only, change, type, reason. '
            'The change field must still be yes or no, using the same decision rule as above.')
    rgb = [read_image(item['image_'+t],rgb=True) for t in ('t1','t2')]
    images = marked_images(rgb,mask,record['bbox'],variant)
    content=[]
    directory.mkdir(parents=True,exist_ok=True)
    for i,(label,im) in enumerate(zip(('T1 close-up','T2 close-up','T1 wider view','T2 wider view'),images)):
        path=directory/f'{i}.jpg';im.save(path,quality=92)
        content.extend([dict(type='text',text=label),dict(type='image_url',image_url=dict(url=str(path)))])
    content.append(dict(type='text',text=question))
    return [dict(role='system',content=prompts['SYSTEM']),dict(role='user',content=content)]


def parse_response(raw):
    from mllm import parse_json
    choice=raw['choices'][0]; content=choice['message'].get('content') or ''
    parsed=parse_json(content)
    valid=isinstance(parsed,dict)
    parsed=parsed if valid else {}
    return dict(answer=parsed.get('change','unknown'),parsed=parsed,raw=content,
        finish_reason=choice.get('finish_reason'),parse_failure=not valid,
        unknown=parsed.get('change') not in ('yes','no'))


def verify(args):
    from agent_baseline.planner import Planner, usage_tokens, request_cost
    from mllm import parse_json
    directory=args.out/'answers'/args.variant
    stop=directory/'STOP.json'
    if stop.exists():
        if not args.clear_stop:
            raise RuntimeError('Resolve saved failure then explicitly use --clear-stop')
        stop.replace(directory/'previous_STOP.json')
    for item in load_json(args.out/'manifest.json'):
        if args.group and item['group'] != args.group:
            continue
        z=load_npz(Path(item['masks'])); masks=list(unpack(z['masks'],tuple(z['shape'])))
        for record,mask in zip(item['candidates'],masks):
            dest=directory/(record['uid'].replace(':','__')+'.json')
            if dest.exists():
                continue
            event_path=directory/'usage'/(record['uid'].replace(':','__')+'.json')
            if event_path.exists():
                event=load_json(event_path)
                if event.get('raw'):
                    write_json(dest,dict(uid=record['uid'],**parse_response(event['raw']),
                        usage=event['usage'],tokens=event['tokens'],cost_cny=event['cost_cny'],model=MODEL)); continue
            try:
                def capture(event):
                    tokens=usage_tokens(event.get('usage'))
                    body=event.get('error_body')
                    if body:
                        for variable in ('DASHSCOPE_API_KEY',):
                            secret=os.environ.get(variable)
                            if secret: body=body.replace(secret,'[credential omitted]')
                    write_json(event_path,dict(model=MODEL,usage=event.get('usage'),tokens=tokens,cost_cny=request_cost(MODEL,tokens),
                        raw=event.get('raw'),http_status=event.get('http_status'),error_type=event.get('error_type'),error_body=body))
                    if event.get('error_type'):
                        raise RuntimeError('API request failed; saved usage and status, no automatic retry')
                planner=Planner(seed=sum(map(ord,record['uid']))%(2**31), max_attempts=1, record=capture,
                    parameters=dict(model=MODEL,temperature=0,max_tokens=256,enable_thinking=False,
                                    response_format={'type':'json_object'},logprobs=True,top_logprobs=5))
                inputs=directory/'inputs'/record['uid'].replace(':','__')
                payload=messages(item,record,mask,args.variant,inputs)
                write_json(inputs/'request.json',dict(messages=payload,parameters=planner.parameters,uid=record['uid'],variant=args.variant))
                result=planner.complete(payload,[],1)
                tokens=usage_tokens(result['usage'])
                write_json(dest,dict(uid=record['uid'],**parse_response(result['raw']),
                    usage=result['usage'],tokens=tokens,cost_cny=request_cost(MODEL,tokens),model=MODEL,input_path=str(inputs)))
            except Exception as exc:
                write_json(stop,dict(uid=record['uid'],error_type=type(exc).__name__,http_status=getattr(exc,'code',None)))
                progress(args,'verify_failed',failures=1)
                raise RuntimeError(f"Verifier stopped at {record['uid']}: {type(exc).__name__}") from None
        progress(args,'verifying_'+args.variant)
    answers=[load_json(p) for p in directory.glob('*.json') if p.name not in ('STOP.json','previous_STOP.json','summary.json')]
    expected=sum(len(i['candidates']) for i in load_json(args.out/'manifest.json'))
    write_json(directory/'summary.json',dict(status='complete' if len(answers)==expected else 'partial',
        completed_group=args.group,expected=expected,answers=len(answers),cost_cny=sum(r['cost_cny'] for r in answers)))
    progress(args,'verified_'+args.variant,len(answers))


def metrics(counts):
    tp,fp,fn=map(int,counts)
    return dict(tp=tp,fp=fp,fn=fn,f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                precision=100*tp/(tp+fp) if tp+fp else 0, recall=100*tp/(tp+fn) if tp+fn else 0)


def recall(mask, missing):
    components=split_components(missing)
    return dict(pixels=int((mask&missing).sum()),missing_pixels=int(missing.sum()),
                pixel_recall=float((mask&missing).sum()/missing.sum()) if missing.any() else None,
                components_hit=sum(int((m&mask).sum())>=.5*int(m.sum()) for m in components),components=len(components))


def aggregate_recall(values):
    result={key:sum(v[key] for v in values) for key in ('pixels','missing_pixels','components_hit','components')}
    result['pixel_recall']=result['pixels']/result['missing_pixels'] if result['missing_pixels'] else None
    result['component_recall']=result['components_hit']/result['components'] if result['components'] else None
    return result


def sources(args):
    """Full frozen development source analysis, independent of verifier truncation."""
    summaries={}
    for spec in specs(args.run,args.work):
        group=spec['group']
        if group not in GROUPS or (args.group and group!=args.group): continue
        records=[]
        for row in rows_for(args.run,spec):
            stem=Path(row['id']).stem
            dest=args.out/'full_sources'/group/f'{stem}.json'
            if dest.exists(): records.append(load_json(dest));continue
            gt=read_image(row['gt_mask'])>(127 if spec['dataset']=='levir' else 0)
            base=load_npz(args.work/'base'/group/f'{stem}.npz')['start']
            by_source={}
            for source in SOURCES:
                z=load_npz(args.work/'candidates'/group/source/f'{stem}.npz')
                by_source[source]=union(unpack(z['masks'],tuple(z['shape'])),base.shape)
            per={}
            for source,mask in by_source.items():
                unique=mask&~union([m for s,m in by_source.items() if s!=source],base.shape)
                beyond=mask&~by_source['S1_02']
                per[source]=dict(tp=int((mask&gt).sum()),fp=int((mask&~gt).sum()),
                    unique_vs_all_tp=int((unique&gt).sum()),unique_vs_all_fp=int((unique&~gt).sum()),
                    beyond_s1_tp=int((beyond&gt).sum()),beyond_s1_fp=int((beyond&~gt).sum()),
                    recall=recall(mask,gt&~base),add_all=metrics(pixel_counts(base|mask,gt)))
            pairwise={}
            for i,s in enumerate(SOURCES):
                for t in SOURCES[i+1:]:
                    overlap=by_source[s]&by_source[t]
                    pairwise[s+'+'+t]=dict(tp=int((overlap&gt).sum()),fp=int((overlap&~gt).sum()))
            record=dict(stem=stem,baseline=metrics(pixel_counts(base,gt)),sources=per,pairwise_overlap=pairwise,
                        full_recall=recall(union(by_source.values(),base.shape),gt&~base))
            write_json(dest,record);records.append(record)
        aggregated={}
        for source in SOURCES:
            vals=[r['sources'][source] for r in records]
            aggregated[source]={k:sum(v[k] for v in vals) for k in ('tp','fp','unique_vs_all_tp','unique_vs_all_fp','beyond_s1_tp','beyond_s1_fp')}
            aggregated[source]['contributing_images_vs_all']=sum(v['unique_vs_all_tp']>0 for v in vals)
            aggregated[source]['contributing_images_vs_s1']=sum(v['beyond_s1_tp']>0 for v in vals)
            aggregated[source]['add_all']=metrics([sum(v['add_all'][k] for v in vals) for k in ('tp','fp','fn')])
            aggregated[source]['recall']=aggregate_recall([v['recall'] for v in vals])
            contributions=sorted([dict(stem=r['stem'],pixels=r['sources'][source]['beyond_s1_tp']) for r in records],key=lambda r:(-r['pixels'],r['stem']))
            total=aggregated[source]['beyond_s1_tp']
            aggregated[source]['top5_vs_s1']=contributions[:5]
            aggregated[source]['top2_share_vs_s1']=sum(r['pixels'] for r in contributions[:2])/total if total else None
        summaries[group]=dict(images=len(records),sources=aggregated,records=records,
            baseline=metrics([sum(r['baseline'][k] for r in records) for k in ('tp','fp','fn')]),
            full_recall=aggregate_recall([r['full_recall'] for r in records]))
        write_json(args.out/'full_sources'/f'{group}.json',summaries[group])
        progress(args,'sources_'+group,len(records))
    write_json(args.out/'full_sources_summary.json',dict(groups=summaries,gt_role='scoring_only',
        inherited_thresholds='S3/S4 historically selected on these development labels; descriptive evidence only'))


def online_selection(args):
    """Freeze a proposed genuine Agent comparison; no inference is launched."""
    from agent_baseline.tasks import INSTRUCTIONS
    selection={}
    for spec in specs(args.run,args.work):
        if spec['group'] not in GROUPS:continue
        rows=sorted(rows_for(args.run,spec),key=lambda r:r['id'])
        chosen=random.Random(1016).sample(rows,2)
        for row in chosen:
            stem=Path(row['id']).stem
            initial=args.out/'online_selection'/'initial_masks'/spec['group']/(stem+'.png')
            start=load_npz(args.work/'base'/spec['group']/(stem+'.npz'))['start']
            save_image(initial,(start.astype(np.uint8)*255))
            if spec['dataset']=='levir':
                task='T5.1';instruction=INSTRUCTIONS['T5.1']
            else:
                task='T5.1xC1'
                direction='newly built' if spec['direction']=='appeared' else 'demolished'
                instruction=('Here is an initial building change detection mask (mask#0). Check it against T1 (earlier) and T2 (later), '
                    'correct it where needed, and keep only buildings that were '+direction+'. Submit the final binary change mask and briefly summarize your decision.')
            row.update(task_type=task,instruction=instruction,constraint=None,init_mask=str(initial.resolve()),
                meta=dict(protocol='fixed_historical_initial_answer_development_supplementation',
                          group=spec['group'],initial_source=str(args.work/'base'/spec['group']/(stem+'.npz'))))
        write_text(args.out/'online_selection'/(spec['group']+'.jsonl'),''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
        selection[spec['group']]=[r['id'] for r in chosen]
    write_json(args.out/'online_selection'/'design.json',dict(selection_seed=1016,seed_offsets=[1016,2027,3038],
        selected=selection,arms=['frozen_baseline','find_missed_opt_in'],sessions=36,
        status='proposed_not_launched_pending_offline_evidence',gt_role='recorder_scoring_only',
        protocol='T5.1 correction from matched historical base masks; candidates already exclude that base; not from-scratch discovery',
        baseline='freeze full planner/tool/harness settings before any Agent launch'))


def score(args):
    manifest=load_json(args.out/'manifest.json'); details=[]
    group_specs={s['group']:s for s in specs(args.run,args.work)}
    truth_rows={g:{Path(r['id']).stem:r for r in rows_for(args.run,s)} for g,s in group_specs.items() if g in GROUPS}
    for item in manifest:
        group,stem=item['group'],item['stem']; spec=group_specs[group]
        gt=read_image(truth_rows[group][stem]['gt_mask'])>(127 if spec['dataset']=='levir' else 0)
        base=load_npz(args.work/'base'/group/f'{stem}.npz')['start']; missing=gt&~base
        by_source={}
        for source in SOURCES:
            z=load_npz(args.work/'candidates'/group/source/f'{stem}.npz')
            by_source[source]=union(unpack(z['masks'],tuple(z['shape'])),base.shape)
        full=union(by_source.values(),base.shape)
        source_stats={}
        for source,mask in by_source.items():
            others=union([m for s,m in by_source.items() if s!=source],base.shape)
            unique=mask&~others; beyond=mask&~by_source['S1_02']
            source_stats[source]=dict(total_tp=int((mask&gt).sum()),total_fp=int((mask&~gt).sum()),
                unique_vs_all_tp=int((unique&gt).sum()),unique_vs_all_fp=int((unique&~gt).sum()),
                beyond_s1_tp=int((beyond&gt).sum()),beyond_s1_fp=int((beyond&~gt).sum()),recall=recall(mask,missing))
        z=load_npz(Path(item['masks'])); masks=list(unpack(z['masks'],tuple(z['shape'])))
        record=dict(group=group,stem=stem,baseline=metrics(pixel_counts(base,gt)),sources=source_stats,
            full_source_recall=recall(full,missing),budgets={},failures=[])
        for k in (5,10,20):
            selected=union(masks[:k],base.shape)
            row=dict(candidate_recall=recall(selected,missing),unverified_add_all=metrics(pixel_counts(base|selected,gt)),variants={})
            for variant in VARIANTS:
                accepted=[]; answered=0; unknown=0; truncated=0; parse_failures=0
                errors=dict(tp_fp_proxy_false_accept=0,tp_fp_proxy_false_reject=0)
                for c,m in zip(item['candidates'][:k],masks[:k]):
                    path=args.out/'answers'/variant/(c['uid'].replace(':','__')+'.json')
                    if not path.exists(): continue
                    response=load_json(path)
                    if 'finish_reason' not in response:
                        event_path=path.parent/'usage'/path.name
                        if event_path.exists():
                            raw=load_json(event_path).get('raw')
                            if raw: response.update(parse_response(raw))
                    answer=response['answer']; answered+=1
                    truncated+=response.get('finish_reason')=='length'
                    parse_failures+=bool(response.get('parse_failure'))
                    tp=int((m&gt).sum()); fp=int((m&~gt).sum())
                    if answer=='yes': accepted.append(m)
                    if answer not in ('yes','no'): unknown+=1
                    error='tp_fp_proxy_false_accept' if answer=='yes' and tp<=fp else 'tp_fp_proxy_false_reject' if answer=='no' and tp>fp else None
                    if error:
                        errors[error]+=1
                        if k==20: record['failures'].append(dict(uid=c['uid'],variant=variant,error=error,tp=tp,fp=fp,
                            criterion='candidate TP>FP pixel-benefit proxy, not a semantic change label',
                            semantic_error='requires_visual_review',input_path=response.get('input_path')))
                complete=answered==len(masks[:k])
                row['variants'][variant]=dict(answered=answered,expected=len(masks[:k]),unknown=unknown,complete=complete,
                    truncated=truncated,parse_failures=parse_failures,verifier_errors=errors,
                    metrics=metrics(pixel_counts(base|union(accepted,base.shape),gt)) if complete else None)
            record['budgets'][str(k)]=row
        details.append(record)
        write_json(args.out/'scores'/group/f'{stem}.json',record)
    summary={}
    for group in GROUPS:
        rows=[r for r in details if r['group']==group]; aggregate={}
        for k in (5,10,20):
            aggregate[str(k)]=dict(candidate_recall=aggregate_recall([r['budgets'][str(k)]['candidate_recall'] for r in rows]))
            for variant in VARIANTS:
                values=[r['budgets'][str(k)]['variants'][variant]['metrics'] for r in rows]
                aggregate[str(k)][variant]=metrics([sum(v[key] for v in values) for key in ('tp','fp','fn')]) if all(v is not None for v in values) else None
        summary[group]=dict(images=len(rows),baseline=metrics([sum(r['baseline'][key] for r in rows) for key in ('tp','fp','fn')]),budgets=aggregate,
            full_source_recall=aggregate_recall([r['full_source_recall'] for r in rows]))
    write_json(args.out/'score_summary.json',dict(groups=summary,scope='frozen common Top20 subset; source recall uses all cached candidates',
        source_thresholds='S3/S4 inherited historical development-selected thresholds; descriptive, not held-out',details=details))
    progress(args,'scored',len(details))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','verify','score','sources','online_selection'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--images',type=int,default=0)
    parser.add_argument('--variant',choices=VARIANTS,default='four');parser.add_argument('--clear-stop',action='store_true')
    parser.add_argument('--group',choices=GROUPS)
    args=parser.parse_args();globals()[args.command](args)

if __name__=='__main__': main()
