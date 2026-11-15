"""Offline inference adapters for missed_sources; launch ONLY inside a GPU job.

The host never imports torch. Persistent JSON-line workers run in existing model
environments. Existing experiment caches are read-only; new caches stay in work.
"""
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from agent_baseline import instances
from agent_baseline.segment_cache import manifest_cache_path
from agent_baseline.tools_t13 import Features, pool_features, cosine
from .io_retry import read_image, read_text, retry_io, save_image, write_json
from .missed_sources import (VERSION, SOURCES, exists, load_json, load_npz, save_npz,
    pack, unpack, stack, union, change_candidates, selected_rows, ensure_settings,
    reference, audit_settings)

SYNONYMS = ('house', 'roof', 'factory', 'shed')


class LineWorker:
    def __init__(self, command, folder):
        self.command, self.folder = command, Path(folder)
        self.proc = self.log = None

    def __call__(self, request):
        if self.proc is None:
            retry_io(lambda: self.folder.mkdir(parents=True, exist_ok=True))
            self.log = retry_io(lambda: (self.folder/'stderr.log').open('a'))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
            self.proc = subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,text=True,env=env)
        self.proc.stdin.write(json.dumps(request)+'\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Inference worker exited; inspect its stderr.log')
        result = json.loads(line)
        if not result['ok']:
            # Worker traceback remains in the staging log; no credentials in host errors.
            raise RuntimeError('Inference worker failed; inspect its stderr.log')
        return load_npz(result['path'])

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait()
            self.log.close()


class Sam3:
    def __init__(self,args,run,work):
        values = (args.sam3_python,args.sam3_root,args.sam3_checkpoint,args.sam3_bpe)
        if any(x is None for x in values):
            raise ValueError('GPU mode needs --sam3-python/root/checkpoint/bpe')
        self.run,self.work,self.args = run,work,args
        self.worker = LineWorker([str(args.sam3_python),'-B','-m',
            'agent_baseline.workers.missed_sources_worker','sam3',
            '--sam3-root',str(args.sam3_root),'--checkpoint',str(args.sam3_checkpoint),
            '--bpe',str(args.sam3_bpe)],work/'workers/sam3')

    def text(self,spec,row,time,text,threshold):
        # conf=.1 can serve both S1 thresholds. Never read a truncated cache for .2.
        for root in spec['cache_roots']:
            path = manifest_cache_path(root,row['image_'+time],text,512,.1,.85)
            if exists(path):
                data=load_npz(path)
                masks=unpack(data['masks'],tuple(map(int,data['shape'])))
                return masks[data['scores'] >= threshold]
        # Existing LEVIR soft data contains no instances below .3.
        if spec['dataset']=='levir' and text=='building' and threshold >= .3:
            path=self.run/'sam3_soft/t512s256/train'/(Path(row['id']).stem+'.npz')
            if exists(path):
                from agent_baseline.backends import read_soft
                rows=retry_io(lambda:read_soft('train',row['id'],0 if time=='t1' else 1,.85,threshold,self.run))
                shape=tuple(map(int,load_npz(path)['shape']))
                return stack([instances.to_mask([r],shape) for r in rows],shape)
        path=self.work/'gpu_cache/text'/spec['dataset']/Path(row['id']).stem/text/(time+'.npz')
        request=dict(operation='text',query=row['image_'+time],text=text,tile=512,
                     conf=.1,pixel_thr=.85,output=str(path))
        data=self.cached(path,request)
        return unpack(data['masks'],tuple(map(int,data['shape'])))[data['scores'] >= threshold]

    def cached(self,path,request):
        identity=dict(version=VERSION,operation=request,
            checkpoint=reference(self.args.sam3_checkpoint,self.run),
            model_root=reference(self.args.sam3_root,self.run),resolution=1008)
        # Input stats detect ordinary source replacement without content digests.
        identity['inputs']=[dict(path=reference(request[k],self.run),
            size=retry_io(lambda p=Path(request[k]):p.stat()).st_size,
            mtime_ns=retry_io(lambda p=Path(request[k]):p.stat()).st_mtime_ns)
            for k in ('query','reference') if request.get(k)]
        # Keep all cache metadata paths relative to R.
        identity['operation']={k:reference(v,self.run) if k in ('query','reference','output') else v
                               for k,v in request.items()}
        meta=path.with_suffix('.json')
        if exists(path):
            if not exists(meta) or load_json(meta)!=identity:
                raise ValueError('SAM3 cache metadata differs; use a fresh work directory')
            return load_npz(path)
        data=self.worker(request)
        write_json(meta,identity)
        return data

    def visual(self,spec,row,start,yes):
        """Positive boxes on verified yes components only, no truth in prompts.

        On the seed date use one image and add_geometric_prompt. To decide
        presence on the opposite date retrieve with the same exemplar in a
        reference/query mosaic; never put an exemplar box on empty opposite land.
        """
        found={'t1':[],'t2':[]}
        refs=('t2',) if spec['direction']=='appeared' else ('t1',) if spec['direction']=='disappeared' else ('t1','t2')
        for ref in refs:
            for i,seed in enumerate(yes,1):
                y,x=np.nonzero(seed)
                box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
                for time in ('t1','t2'):
                    path=(self.work/'gpu_cache/visual'/spec['group']/Path(row['id']).stem/
                          f'{ref}_{i:04d}_{time}.npz')
                    request=dict(operation='visual',query=row['image_'+time],
                        reference=row['image_'+ref],same_image=time==ref,box=box,
                        conf=.1,score_thr=.5,pixel_thr=.85,output=str(path))
                    data=self.cached(path,request)
                    found[time].extend(unpack(data['masks'],tuple(map(int,data['shape']))))
        return change_candidates(found['t1'],found['t2'],start,spec['direction'])

    def close(self):
        self.worker.close()


class Dynamic:
    def __init__(self,args,work):
        if args.vision_config is None:
            raise ValueError('generate needs --vision-config for existing local model paths')
        self.cfg=load_json(args.vision_config)['dynamic']
        self.work=work
        self.run=args.run.resolve()
        self.worker=LineWorker([self.cfg['python'],'-B','-m',
            'agent_baseline.workers.missed_sources_worker','dynamic'],work/'workers/dynamic')

    def for_row(self,spec,row):
        def infer(operation,time):
            path=self.work/'gpu_cache/dynamic'/spec['dataset']/Path(row['id']).stem/f'{operation}_{time}.npz'
            request=dict(operation=operation,paths=[row['image_t1'],row['image_t2']],time=time,
                         config=self.cfg,classes=['building'],output=str(path))
            # Explicit per-image cache makes resume independent of shard count.
            config={k:reference(v,self.run) if isinstance(v,str) and Path(v).is_absolute() else v
                    for k,v in self.cfg.items()}
            settings=dict(version=VERSION,operation=operation,time=time,config=config,
                inputs=[dict(path=reference(p,self.run),size=retry_io(lambda p=Path(p):p.stat()).st_size,
                             mtime_ns=retry_io(lambda p=Path(p):p.stat()).st_mtime_ns) for p in request['paths']])
            meta=path.with_suffix('.json')
            if exists(path):
                if not exists(meta) or load_json(meta)!=settings:
                    raise ValueError('Dynamic cache settings changed')
                z=load_npz(path)
            else:
                z=self.worker(request)
                write_json(meta,settings)
            return unpack(z['masks'],tuple(z['shape'])) if operation=='segment_everything' else z['data']
        return infer

    def close(self):
        self.worker.close()


def feature_sources(dynamic,start,direction):
    """Class-agnostic SAM proposals, DINO difference, semantic direction gate.

    SegEarth building-vs-background probabilities >.5 determine building gain
    and loss. This label-free gate prevents reverse direction/nonbuilding-only
    residuals; its omissions are part of the measured S3/S4 recall.
    """
    from PIL import Image
    shape=start.shape
    features=[Features(dynamic('extract_features',t),shape,t) for t in ('t1','t2')]
    semantic=[]
    for t in ('t1','t2'):
        p=dynamic('classify',t)[1].astype(np.float32)
        # Worker semantic probabilities use a 448 grid. Resize scores, then threshold.
        resized=np.asarray(Image.fromarray(p,mode='F').resize((shape[1],shape[0]),Image.Resampling.BILINEAR))
        semantic.append(resized > .5)
    loss=semantic[0] & ~semantic[1]
    gain=semantic[1] & ~semantic[0]
    output={s:[] for s in ('S3','S4')}; scores={s:[] for s in output}
    times=('t2',) if direction=='appeared' else ('t1',) if direction=='disappeared' else ('t1','t2')
    for time in times:
        gate=gain if time=='t2' else loss
        for mask in dynamic('segment_everything',time):
            idx=np.flatnonzero(mask)
            if not len(idx):
                continue
            difference=float(1-cosine(pool_features(features[0],idx),pool_features(features[1],idx)))
            residual=mask & gate & ~start
            if not residual.any():
                continue
            output['S4'].append(residual); scores['S4'].append(difference)
            if mask.sum() > .1*mask.size:
                output['S3'].append(residual); scores['S3'].append(difference)
    return output,scores


def generate_one(spec,row,start,yes,sam3,dynamic):
    output={}
    for source,threshold in (('S1_03',.3),('S1_02',.2)):
        output[source]=change_candidates(sam3.text(spec,row,'t1','building',threshold),
            sam3.text(spec,row,'t2','building',threshold),start,spec['direction'])
    found={t:[] for t in ('t1','t2')}
    for word in SYNONYMS:
        for t in found:
            found[t].extend(sam3.text(spec,row,t,word,.5))
    output['S2']=change_candidates(found['t1'],found['t2'],start,spec['direction'])
    feature_masks,scores=feature_sources(dynamic,start,spec['direction'])
    output.update(feature_masks)
    output['S5']=sam3.visual(spec,row,start,yes)
    arrays={s:pack(output[s],start.shape) for s in SOURCES}
    arrays.update({s+'_scores':np.asarray(v,np.float32) for s,v in scores.items()})
    arrays['shape']=np.asarray(start.shape)
    arrays['yes_seed_count']=np.asarray(len(yes))
    return arrays


def run_gpu(args,run,work,chosen):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU generation must run in an allocated Slurm job')
    sam3=Sam3(args,run,work)
    dynamic=None
    try:
        if args.command=='generate':
            dynamic=Dynamic(args,work)
        for spec in chosen:
            if args.command=='fixed' and spec['dataset']!='whu':
                continue
            folder=work/('whu_fixed' if args.command=='fixed' else 'raw')/spec['group']
            settings=dict(version=VERSION,direction=spec['direction'],scope=spec['scope'],
                pixel_thr=.85,score_thr=.5,tile=512,overlap_thr=.25,names=spec['names'],
                synonyms=list(SYNONYMS),s1=[.3,.2],s5='all explicitly-yes components; same-date boxes; opposite-date mosaic',
                feature_direction='SegEarth building/background>.5 gain/loss, not truth',
                sam3_checkpoint=reference(args.sam3_checkpoint,run))
            ensure_settings(folder/'settings.json',settings)
            for row in selected_rows(run,spec,args.shard,args.shards):
                stem=Path(row['id']).stem
                dest=folder/(stem+('.png' if args.command=='fixed' else '.npz'))
                if args.resume and exists(dest):
                    continue
                if args.command=='fixed':
                    shape=read_image(row['image_t1'],rgb=True).shape[:2]
                    empty=np.zeros(shape,bool)
                    masks=change_candidates(sam3.text(spec,row,'t1','building',.5),
                        sam3.text(spec,row,'t2','building',.5),empty,spec['direction'])
                    save_image(dest,union(masks,shape).astype(np.uint8)*255)
                else:
                    z=load_npz(work/'base'/spec['group']/(stem+'.npz'))
                    yes=unpack(z['yes'],tuple(z['shape']))
                    arrays=generate_one(spec,row,z['start'],yes,sam3,dynamic.for_row(spec,row))
                    save_npz(dest,**arrays)
                print(spec['group'],stem,'complete',flush=True)
    finally:
        sam3.close()
        if dynamic:
            dynamic.close()
