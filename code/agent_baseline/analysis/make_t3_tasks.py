"""Deterministic SECOND T3.1 task generation; label-derived exemplar only.

No inference/API. Intended for GPU-job preparation alongside the probe, 1 CPU,
4 GB RAM; <5 min/split estimate. Eight-connected nonzero category transitions;
one task per (image, directed transition), minimum exemplar area 16 pixels; truth retains every instance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from .probe_t13_common import RUN, RESULTS, CLASSES, binary_truth, inventories, emit
from .second_labels import DATASET, load_labels, check_encoding


SEED = 1016
MIN_AREA = 16


def transition_instances(a,b,min_area=MIN_AREA):
    changed = binary_truth(a,b)
    result = {}
    for source in CLASSES:
        for target in CLASSES:
            if source==target:
                continue
            labels,n = ndi.label(changed&(a==source)&(b==target),np.ones((3,3),bool))
            items = [np.flatnonzero(labels==i) for i in range(1,n+1)]
            if items and any(len(idx)>=min_area for idx in items):
                result[(source,target)] = items
    return result


def make_tasks(a,b,name,seed=SEED,min_area=MIN_AREA):
    result = []
    for (source,target),items in transition_instances(a,b,min_area).items():
        # Local deterministic seed: independent of shard count and file processing order.
        local_seed = seed + sum((i+1)*ord(c) for i,c in enumerate(name)) + source*101+target*1009
        rng = np.random.default_rng(local_seed)
        eligible = [i for i,idx in enumerate(items) if len(idx)>=min_area]
        exemplar = int(rng.choice(eligible))
        key = f'{Path(name).stem}_{source}to{target}'
        result.append(dict(id=key,name=name,source=source,target=target,
            instances=[idx.tolist() for idx in items],exemplar_index=exemplar,seed=local_seed,shape=list(a.shape)))
    return result


def generate(args):
    check_encoding(args.dataset_root)
    all_names = inventories(args.run_root)[args.phase]
    if not 0<=args.shard<args.shards:
        raise ValueError('Require 0 <= shard < shards')
    names = all_names[args.shard::args.shards]
    output = args.output_root/f't3_tasks_{args.phase}_{args.shard}of{args.shards}'
    output.mkdir(parents=True,exist_ok=True)
    rows,manifest,counts = [],[],{}
    for name in names:
        for task in make_tasks(*load_labels(args.dataset_root,name),name,args.seed,args.min_area):
            rows.append(task)
            masks = []
            for indices in task['instances']:
                m = np.zeros(task['shape'],np.uint8)
                m.ravel()[indices] = 255
                masks.append(m)
            exemplar = masks[task['exemplar_index']]
            gt = np.maximum.reduce(masks)
            exemplar_path,gt_path = output/(task['id']+'_exemplar.png'),output/(task['id']+'_gt.png')
            Image.fromarray(exemplar).save(exemplar_path)
            Image.fromarray(gt).save(gt_path)
            transition = f"{CLASSES[task['source']]} -> {CLASSES[task['target']]}"
            counts[transition] = counts.get(transition,0)+1
            # The public instruction never names the transition, dataset or truth path.
            # Exemplar is loaded by a future harness as a non-current mask handle.
            manifest.append(dict(id=task['id'],task_type='T3.1',target='any change',change_direction='any',
                instruction='Find all land-cover changes with the same transition as the marked example. Output a binary mask.',
                constraint=None,image_t1=str((args.dataset_root/'img1'/name).resolve()),
                image_t2=str((args.dataset_root/'img2'/name).resolve()),init_mask=None,
                exemplar_mask=str(exemplar_path.resolve()),gt_mask=str(gt_path.resolve()),meta=dict(transition=transition)))
    (output/'tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    report = dict(phase=args.phase,names=names,seed=args.seed,min_area=args.min_area,
        rule='One task per image and directed class transition; All 8-connected same-transition components form truth, including tiny components. Uniform deterministic exemplar among components of area >= min_area; skip transition if none eligible. Truth includes exemplar.',
        tasks=rows,counts=counts,total=len(rows))
    lines = [report['rule'],'','| 转换 | 任务数 |','|---|---:|']+[f'| {key} | {value} |' for key,value in sorted(counts.items())]
    emit(output/'inventory.json',report,'\n'.join(lines))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DATASET)
    p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--output-root',type=Path,default=RESULTS)
    p.add_argument('--phase',choices=['dev','val'],default='dev')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--min-area',type=int,default=MIN_AREA)
    generate(p.parse_args())


if __name__=='__main__':
    main()
