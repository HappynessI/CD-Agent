"""Four fixed train124 NPZ files: timing attribution, not a speedup benchmark."""
import argparse
from io import BytesIO
import os
from pathlib import Path
from time import perf_counter,process_time
import zipfile
import numpy as np
from .io_retry import write_json

INDICES=(0,119,239,359)


def timed(call):
    wall,cpu=perf_counter(),process_time();value=call()
    return value,dict(wall_seconds=perf_counter()-wall,cpu_seconds=process_time()-cpu)


def materialize(source):
    z,opening=timed(lambda:np.load(source,allow_pickle=False));arrays={};keys={}
    try:
        for key in z.files:
            value,reading=timed(lambda key=key:z[key])
            arrays[key],copying=timed(value.copy)
            keys[key]=dict(materialize=reading,copy=copying,shape=list(value.shape),dtype=str(value.dtype),nbytes=int(value.nbytes))
    finally:_,closing=timed(z.close)
    return arrays,dict(open_directory=opening,keys=keys,close=closing)


def read_once(path):
    file,opening=timed(lambda:path.open('rb'))
    try:data,reading=timed(file.read)
    finally:_,closing=timed(file.close)
    return data,dict(open=opening,read=reading,close=closing,bytes=len(data))


def compare(a,b):
    return dict(keys_equal=set(a)==set(b),fields={key:bool(np.array_equal(a[key],b[key])) for key in a if key in b})


def decode_stages(arrays):
    masks,unpack_time=timed(lambda:np.unpackbits(arrays['masks'],axis=1,count=int(np.prod(arrays['shape']))).reshape(-1,*tuple(arrays['shape'])))
    indexes,index_time=timed(lambda:[np.flatnonzero(mask).astype(np.int32) for mask in masks])
    def unique():
        seen=set();nonempty=0
        for idx in indexes:
            if not len(idx):continue
            identity=idx.tobytes()
            if identity not in seen:seen.add(identity);nonempty+=1
        return nonempty
    count,dedup_time=timed(unique)
    return dict(unpack=unpack_time,flatnonzero=index_time,bytes_set=dedup_time,
        rows=len(indexes),unique_nonempty=count,unpacked_nbytes=int(masks.nbytes))


def diagnostic(path,order,stage):
    result=dict(path=str(path),order=order,timing_scope='process CPU is current process only; wall includes filesystem waits')
    data=None;arms={};arrays={}
    for arm in order:
        stage('begin_'+arm,result)
        if arm=='A':arrays[arm],arms[arm]=materialize(path)
        else:
            data,raw=read_once(path);stage('read_bytes',dict(result,raw_file=raw))
            arrays[arm],arms[arm]=materialize(BytesIO(data));arms[arm]['raw_file']=raw
        result['arms']=arms;stage('complete_'+arm,result)
    with zipfile.ZipFile(BytesIO(data)) as archive:
        result['zip_members']=[dict(name=x.filename,compressed_size=x.compress_size,expanded_size=x.file_size) for x in archive.infolist()]
    result['equal']=compare(arrays['A'],arrays['B']);stage('compared',result)
    result['decode']=decode_stages(arrays['A']);stage('decoded',result)
    if not result['equal']['keys_equal'] or not all(result['equal']['fields'].values()):raise RuntimeError('Saved array mismatch')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run diagnostic through Slurm')
    folder=a.work/'gpu_cache/visual/levir_any/train_124';files=sorted(folder.glob('*.npz'))
    if len(files)!=360:raise ValueError('Expected frozen 360-file train124 cache')
    policy=dict(stem='train_124',indices=INDICES,orders=['AB','BA','AB','BA'],api_requests=0,gt_read=False,
        scope='Four deterministic files only; cache/order effects remain; do not extrapolate overall acceleration',
        path_A='np.load(path) once; each key materialized once then copied',
        path_B='one open/read/close to bytes, then BytesIO np.load; each key materialized once then copied',
        cache_modified=False)
    write_json(a.out/'settings.json',policy);records=[]
    for n,index in enumerate(INDICES):
        path=files[index];out=a.out/f'{index:03d}.json';order='AB' if n%2==0 else 'BA'
        def stage(name,value):write_json(out,dict(value,index=index,stage=name))
        record=diagnostic(path,order,stage);record.update(index=index,stage='complete');write_json(out,record);records.append(record)
        write_json(a.out/'progress.json',dict(completed=len(records),expected=4,last_index=index,api_requests=0))
    write_json(a.out/'summary.json',dict(policy=policy,files=records))


if __name__=='__main__':main()
