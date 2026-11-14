"""Two fixed post-result mechanism pages using the existing blind-review layout/conventions."""
import argparse
import html
import json
import os
import shutil
from pathlib import Path
import numpy as np
from .night_t3_blind_review import Image, crop_box, SIZE, bbox
from .io_retry import read_text, read_image, write_json, write_text, exists
from .t3_routes import load_dev, pixel_counts
from .t3_routes_cache import read_npz
from .night_t3_retrieval import unpack

CASES=('train_1808_2to1','train_1745_5to2')
DISCLAIMER=('Post-result mechanism diagnosis: these two cases were selected after seeing results. '
            'This is not blind review or independent human truth. GT is shown only for offline inspection. '
            'No thresholds, labels, candidate choices or model inputs are changed by this page.')


def unpack_proposals(saved,key):
    shape=tuple(map(int,saved['shape']));bits=saved[key]
    return np.unpackbits(bits,axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Image generation and mask counts require Slurm')
    root=a.night/'positive_endpoints_tune';out=root/'review_pages'
    if exists(out/'index.html'):raise ValueError('Keep existing review pages; do not overwrite completed review')
    rows={r['id']:r for r in load_dev(a.run)};parses={r['id']:r for r in json.loads(read_text(a.night/'B2/summary.json'))['records']}
    sections=[];metadata=[]
    for name in CASES:
        row=rows[name];folder=out/name;folder.mkdir(parents=True,exist_ok=True)
        record_path=root/'B2/tasks'/name/'metrics.json';record=json.loads(read_text(record_path));saved=read_npz(record_path.parent/'proposals.npz')
        exemplar=read_image(row['exemplar_mask'])>0;truth=read_image(row['gt_mask'])>0;shape=exemplar.shape
        figures=[];panels=[]
        def figure(filename,label):
            return f'<figure><a href="{name}/{filename}"><img loading="lazy" src="{name}/{filename}"></a><figcaption>{html.escape(label)}</figcaption></figure>'
        for time in ('t1','t2'):
            destination=folder/f'{time}_original.png';shutil.copyfile(row['image_'+time],destination)
            figures.append(figure(destination.name,f'{time.upper()} original RGB, full native image (click to inspect)'))
        def panel(label,mask,source):
            filename=label+'.png';Image.fromarray(mask.astype(np.uint8)*255).save(folder/filename)
            counts=dict(panel=label,native_pixels=int(mask.sum()),remaining_pixels=int((mask&~exemplar).sum()),
                remaining_tp_fp_fn=pixel_counts(mask,truth,exemplar),source=source)
            panels.append(counts);figures.append(figure(filename,f'{label}; white=true, black=false; {counts["native_pixels"]} native pixels'))
        panel('example_support',exemplar,row['exemplar_mask'])
        # Same crop_box and nearest-neighbor binary support display as grounding_blind_v1.
        for label,kwargs in [('close',{}),('wide',dict(scale=10,min_side=256))]:
            l,t,r,b=crop_box(bbox(exemplar),shape,**kwargs)
            filename=f'example_support_{label}.png'
            Image.fromarray(exemplar[t:b,l:r].astype(np.uint8)*255).resize((SIZE,SIZE),Image.Resampling.NEAREST).save(folder/filename)
            figures.append(figure(filename,f'Example {label} support; display crop box {(l,t,r,b)}; other masks remain full-image coordinates'))
        if len(record['cache_sources'])!=4:raise ValueError('Fixed B2 cases require four endpoint caches')
        for label,path in zip(('source_T1','source_T2','target_T1','target_T2'),record['cache_sources']):
            stack=unpack(Path(path))
            if stack.shape[1:]!=shape:raise ValueError('Endpoint coordinates differ')
            union=stack.any(axis=0) if len(stack) else np.zeros(shape,bool)
            panel(label+'_union',union,path)
        for label,key,rankkey in [('strict','strict','strict_rank'),('positive_only','positive_only','positive_rank')]:
            masks=unpack_proposals(saved,key)
            union=masks.any(axis=0) if len(masks) else np.zeros(shape,bool)
            ids=saved[rankkey].tolist()[:8]
            top=masks[ids].any(axis=0) if ids else np.zeros(shape,bool)
            panel(label+'_all_native',union,str(record_path.parent/'proposals.npz')+'::'+key)
            panel(label+'_all_scored',union&~exemplar,'above union minus example support')
            panel(label+'_top8_scored',top&~exemplar,str(record_path.parent/'proposals.npz')+'::'+rankkey)
        panel('GT_original_offline_only',truth,row['gt_mask'])
        panel('GT_remaining_offline_only',truth&~exemplar,'original GT minus example support')
        evidence=dict(id=name,review_scope=DISCLAIMER,images={t:row['image_'+t] for t in ('t1','t2')},
            original_exemplar=row['exemplar_mask'],original_gt_offline_only=row['gt_mask'],metrics_source=str(record_path),
            predicted_pair=record['predicted_pair'],interpretation=parses[name]['interpretation'],source_cache_paths=record['cache_sources'],
            panels=panels,existing_scores=record['scores'])
        write_json(folder/'evidence.json',evidence);metadata.append(evidence)
        sections.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(DISCLAIMER)}</p>'
            f'<pre>B2 prediction: {html.escape(json.dumps(parses[name]["interpretation"],ensure_ascii=False,indent=2))}</pre>'
            '<div class="images">'+''.join(figures)+'</div>'
            f'<h3>Existing metrics and cache provenance</h3><pre>{html.escape(json.dumps(evidence,ensure_ascii=False,indent=2))}</pre></section>')
    write_json(out/'evidence.json',dict(scope=DISCLAIMER,cases=metadata,api_calls=0))
    # Plain figure/flex layout reused from night_t3_blind_review; no separate viewer system.
    style='body{font-family:system-ui;max-width:1400px;margin:24px auto}section{border-top:2px solid #555;padding:20px}figure{margin:8px}img{width:300px;max-width:100%}.images{display:flex;flex-wrap:wrap}figcaption{max-width:300px;font-size:12px}pre{white-space:pre-wrap;background:#eee;padding:12px}'
    write_text(out/'index.html','<!doctype html><meta charset="utf-8"><title>Positive endpoint mechanism review</title><style>'+style+'</style><h1>Fixed two-case mechanism evidence</h1><p>'+html.escape(DISCLAIMER)+'</p>'+''.join(sections))


if __name__=='__main__':main()
