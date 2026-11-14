"""Experimental integer CSR intersections, isolated from online tool primitives."""
from time import perf_counter
import numpy as np
from scipy import sparse


def instance_csr(rows,pixels):
    lengths=np.asarray([len(r['idx']) for r in rows],dtype=np.int64)
    indptr=np.concatenate(([0],np.cumsum(lengths)))
    indices=np.concatenate([r['idx'] for r in rows]).astype(np.int32,copy=False) if rows else np.empty(0,np.int32)
    # Input idx is the unique sorted set from flatnonzero, exactly as in match.
    return sparse.csr_matrix((np.ones(len(indices),np.int32),indices,indptr),shape=(len(rows),pixels),dtype=np.int32)


def sparse_overlaps(found,shape,stats=None,progress=None):
    values={};started=perf_counter();pixels=int(np.prod(shape))
    a=instance_csr(found['t1'],pixels);b=instance_csr(found['t2'],pixels)
    values.update(build_seconds=perf_counter()-started,input_nnz=int(a.nnz+b.nnz),
        input_csr_bytes=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (a,b)))
    if progress:progress('csr_built',values)
    started=perf_counter();intersections=(a@b.T).tocsr();intersections.sort_indices()
    values.update(multiply_seconds=perf_counter()-started,intersection_pairs=int(intersections.nnz),
        product_bytes=int(intersections.data.nbytes+intersections.indices.nbytes+intersections.indptr.nbytes),dtype=str(intersections.dtype))
    if progress:progress('csr_multiplied',values)
    started=perf_counter();pairs=[]
    for i in range(len(found['t1'])):
        for pos in range(intersections.indptr[i],intersections.indptr[i+1]):
            j=int(intersections.indices[pos]);inter=int(intersections.data[pos])
            if inter:pairs.append((i,j,inter,found['t1'][i]['area'],found['t2'][j]['area']))
    values['pairs_seconds']=perf_counter()-started
    if stats is not None:stats.update(values)
    return pairs
