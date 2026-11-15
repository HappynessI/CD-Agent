"""Ten full-image post-confirmation diagnostic cases; GT used only for offline display."""
import argparse
import html
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .io_retry import read_text,read_image,write_json,write_text,save_image
from .t3_routes import load_dev,pooled
from .t3_routes_cache import read_npz
from .n8_t3_common import packed_load


def select_cases(records,clean_records,clean_method):
    def delta(r):return pooled([r['methods']['N8_verified']])['f1_c']-pooled([r['methods']['original_R1_top8']])['f1_c']
    chosen=[];seen=set()
    def add(rows,n,reason):
        count=0
        for r in rows:
            if r['id'] in seen:continue
            chosen.append((r,reason));seen.add(r['id']);count+=1
            if count==n:break
    add(sorted([r for r in records if 0<r['remaining_gt_pixels']<16],key=lambda r:r['id']),2,'tiny remaining-GT diagnostic requested after Step1 audit')
    def clean_gain(r):return pooled([clean_records[r['id']]['methods'][clean_method]])['f1_c']-pooled([r['methods']['original_R1_top8']])['f1_c']
    add(sorted(records,key=lambda r:(-clean_gain(r),r['id'])),2,'largest remaining clean-instance gain under tune-frozen instance configuration')
    add(sorted(records,key=lambda r:(delta(r),r['id'])),3,'largest remaining image-F1 harm (or smallest gain if no harm)')
    add(sorted(records,key=lambda r:(-delta(r),r['id'])),1,'largest remaining image-F1 gain (or smallest harm if no gain)')
    add(sorted([r for r in records if r['remaining_gt_pixels']==0],key=lambda r:(-r['methods']['N8_verified']['counts'][1],r['id'])),2,'largest remaining empty-target FP')
    if len(chosen)<10:add(sorted(records,key=lambda r:r['id']),10-len(chosen),'deterministic fill after distinct case selection')
    return chosen[:10]


def tinted(image,mask,color):
    out=np.asarray(image).copy();out[mask]=(out[mask]*.55+np.asarray(color)*.45).astype(np.uint8);return out


def comparison(image,pred,truth,ex):
    out=image.copy();pred=pred&~ex;truth=truth&~ex
    for mask,color in [(pred&truth,(0,230,0)),(pred&~truth,(255,0,160)),(~pred&truth,(0,200,255)),(ex,(255,220,0))]:out=tinted(out,mask,color)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--night',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Image processing requires Slurm')
    final=json.loads(read_text(a.out/'step3/verification/confirm/summary.json'))
    frozen=json.loads(read_text(a.out/'step3/final_frozen.json'));rank=frozen['system']['instance']['rank'];instk=frozen['system']['instance']['k'];k=frozen['k']
    clean_records={r['id']:r for r in json.loads(read_text(a.out/'step1/summary.json'))['records']}
    chosen=select_cases(final['records'],clean_records,f'clean_{rank}_top{instk}');rows={r['id']:r for r in load_dev(a.run)}
    out=a.out/'diagnostics10';cases=[];font=ImageFont.load_default()
    for record,reason in chosen:
        row=rows[record['id']];folder=out/record['id'];folder.mkdir(parents=True,exist_ok=True)
        t1=read_image(row['image_t1'],rgb=True);t2=read_image(row['image_t2'],rgb=True)
        ex=read_image(row['exemplar_mask'])>0;truth=read_image(row['gt_mask'])>0
        old=read_npz(a.night/'retrieval/B0/tasks'/row['id']/'prepared.npz');ids=old['order1'].tolist()[:8]
        baseline=old['r1'][ids].any(axis=0) if ids else np.zeros_like(ex)
        clean,saved=packed_load(a.out/'step1/tasks'/row['id']/'clean_pool.npz');indices=saved['order_'+rank].tolist()[:instk]
        inst=clean[indices].any(axis=0) if indices else np.zeros_like(ex)
        step2,meta=packed_load(a.out/'step2/confirm/tasks'/row['id']/'predictions.npz')
        selected=step2[meta['methods'].tolist().index(frozen['system']['system_method'])]
        prep=a.out/'step3/prepare/confirm/tasks'/row['id'];pool,meta=packed_load(prep/'pool.npz');order=meta['order'].tolist()
        fulltop=pool[order[:5]].any(axis=0) if order else np.zeros_like(ex)
        verified,meta=packed_load(a.out/'step3/verification/confirm/tasks'/row['id']/'predictions.npz');names=meta['methods'].tolist()
        gate=verified[names.index(f'gate_top{k}')];pred=verified[names.index(f'verified_top{k}')]
        inference=json.loads(read_text(prep/'inference.json'))
        for name,array in [('T1',t1),('T2',t2),('example',ex.astype(np.uint8)*255),('GT_remaining',(truth&~ex).astype(np.uint8)*255),
                           ('baseline',baseline.astype(np.uint8)*255),('N8_prediction',pred.astype(np.uint8)*255)]:save_image(folder/f'{name}.png',array)
        panels=[('T1 original',t1),('T2 original',t2),('Example support / T1',tinted(t1,ex,(255,220,0))),
                ('Original R1 Top8',comparison(t2,baseline,truth,ex)),('Clean instance selected',comparison(t2,inst,truth,ex)),
                ('Step2 frozen route/hybrid',comparison(t2,selected,truth,ex)),('Rebuilt pool Top5',comparison(t2,fulltop,truth,ex)),
                ('Frozen gate unverified',comparison(t2,gate,truth,ex)),('N8 evidence verified',comparison(t2,pred,truth,ex)),('Remaining GT',tinted(t2,truth&~ex,(0,230,0)))]
        h,w=ex.shape;sheet=Image.new('RGB',(w*5,(h+24)*2),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,array) in enumerate(panels):
            x=(i%5)*w;y=(i//5)*(h+24);draw.text((x+4,y+4),label,fill='black',font=font);sheet.paste(Image.fromarray(array),(x,y+24))
        sheet.save(folder/'panel.png')
        candidates=[]
        for position,index in enumerate(order[:5],1):
            mask=pool[index]
            save_image(folder/f'candidate_{position}_mask.png',mask.astype(np.uint8)*255)
            save_image(folder/f'candidate_{position}_T1.png',tinted(t1,mask,(255,0,160)))
            save_image(folder/f'candidate_{position}_T2.png',tinted(t2,mask,(255,0,160)))
            candidates.append(dict(rank=position,**inference['candidates'][index],answer=record['verification'].get(str(index)),sources=inference['sources'][index]))
        detail=dict(id=row['id'],selection_reason=reason,scope='Post hoc confirmation diagnosis only; no inference/selection updates',
            metrics=record,observable=inference['observable'],candidates=candidates,
            legend='Green TP; magenta FP; cyan FN; yellow example. All panels retain full native image extent; source T1/T2 saved separately.')
        write_json(folder/'case.json',detail);cases.append(detail)
        print('N8 diagnostic',len(cases),row['id'],flush=True)
    write_json(out/'manifest.json',dict(tasks=len(cases),cases=[dict(id=c['id'],reason=c['selection_reason']) for c in cases],
        selection='2 tiny positive GT + 2 clean-instance gains + 3 final harms + 1 final gain + 2 empty-FP cases, all distinct with deterministic fill. Revised after Step1 audit by main-Agent request; post hoc diagnosis, not tuning.'))
    lines=['<html><meta charset="utf-8"><title>N8 T3 ten-case diagnostics</title><body>',
           '<h1>N8 T3 post-confirmation diagnostics</h1><p>Full native image extent. Green TP, magenta FP, cyan FN, yellow example. Offline GT display only.</p>']
    for case in cases:
        ident=html.escape(case['id']);lines.extend([f'<h2>{ident}</h2><p>{html.escape(case["selection_reason"])}</p>',
            f'<a href="{ident}/case.json">metrics / candidate evidence</a> | <a href="{ident}/T1.png">original T1</a> | <a href="{ident}/T2.png">original T2</a>',
            f'<p><a href="{ident}/panel.png"><img width="1280" src="{ident}/panel.png"></a></p>'])
    write_text(out/'index.html','\n'.join(lines+['</body></html>']))


if __name__=='__main__':main()
