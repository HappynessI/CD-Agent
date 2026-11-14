"""One ABBA batch/row unpack comparison on the same four train124 arrays."""
import argparse
from io import BytesIO
import os
from pathlib import Path
import numpy as np
from .io_retry import write_json
from .night_npz_io_diagnostic import INDICES,timed,read_once,materialize


def layout(value):
    return dict(shape=list(value.shape),strides=list(value.strides),dtype=str(value.dtype),
        c_contiguous=bool(value.flags.c_contiguous),f_contiguous=bool(value.flags.f_contiguous),
        aligned=bool(value.flags.aligned),owns_data=bool(value.flags.owndata),nbytes=int(value.nbytes))


def unpack(bits,shape,method):
    pixels=int(np.prod(shape))
    if method=='A':return np.unpackbits(bits,axis=1,count=pixels).reshape(-1,*shape)
    result=np.empty((len(bits),*shape),dtype=np.uint8)
    for i,row in enumerate(bits):result[i]=np.unpackbits(row,axis=None,count=pixels).reshape(shape)
    return result


def compare_unpack(bits,shape,stage=lambda result:None):
    report=dict(input=layout(bits),first_row=layout(bits[0]) if len(bits) else None,order='ABBA',rounds=[])
    reference=None
    for index,method in enumerate('ABBA'):
        value,elapsed=timed(lambda:unpack(bits,shape,method))
        if reference is None:reference=value
        same=bool(np.array_equal(reference,value))
        indices,index_time=timed(lambda:[np.flatnonzero(mask).astype(np.int32) for mask in value])
        report['rounds'].append(dict(round=index+1,method=method,unpack=elapsed,index=index_time,
            output=layout(value),pixels_set=sum(len(x) for x in indices),equal_to_first=same))
        stage(report)
        if not same:raise RuntimeError('Saved batch/row mismatch')
        del value,indices
    report['all_equal']=all(r['equal_to_first'] for r in report['rounds'])
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    files=sorted((a.work/'gpu_cache/visual/levir_any/train_124').glob('*.npz'))
    if len(files)!=360:raise ValueError('Expected frozen 360-file cache')
    policy=dict(indices=list(INDICES),order='ABBA once per file',A='batch unpackbits(axis=1) then reshape',
        B='preallocated output; row unpackbits(axis=None) then assign',
        timing='unpack includes output allocation/row assignment; index measured separately; loading excluded',
        limitation='Four fixed files; order/allocation/cache effects remain. No overall speedup claim.',api_requests=0,gt_read=False)
    write_json(a.out/'settings.json',policy);records=[]
    for index in INDICES:
        path=files[index];dest=a.out/f'{index:03d}.json';write_json(dest,dict(index=index,path=str(path),stage='load'))
        data,raw=read_once(path);arrays,loading=materialize(BytesIO(data))
        def stage(result):write_json(dest,dict(index=index,path=str(path),stage='unpack',raw_read=raw,loading=loading,comparison=result))
        result=compare_unpack(arrays['masks'],tuple(arrays['shape']),stage)
        record=dict(index=index,path=str(path),stage='complete',raw_read=raw,loading=loading,comparison=result)
        write_json(dest,record);records.append(record)
        write_json(a.out/'progress.json',dict(completed=len(records),expected=4,last_index=index))
    write_json(a.out/'summary.json',dict(policy=policy,files=records))


if __name__=='__main__':main()
