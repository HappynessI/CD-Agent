"""Offline persistent SAM3/DynamicEarth worker, no server or implicit download."""
import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from agent_baseline.analysis.io_retry import read_image, retry_io
from agent_baseline.analysis.missed_sources import save_npz, pack
from agent_baseline.workers.t3_sam3_worker import normalized_box, prompt_state


def visual_state(processor,query,reference,box,same_image):
    if same_image:
        h,w=query.shape[:2]
        return processor.add_geometric_prompt(box=normalized_box(box,w,h),label=True,
                                              state=processor.set_image(Image.fromarray(query)))
    return prompt_state(processor,query,reference=reference,box=box)


class Sam3Model:
    def __init__(self,args):
        sys.path.insert(0,str(args.sam3_root))
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not torch.cuda.is_available():
            raise RuntimeError('SAM3 requires GPU')
        self.torch=torch
        model=build_sam3_image_model(bpe_path=args.bpe,checkpoint_path=args.checkpoint,
            device='cuda',load_from_HF=False,enable_inst_interactivity=False)
        model.to('cuda'); model._device=torch.device('cuda')
        self.processor=Sam3Processor(model,resolution=1008,device='cuda',confidence_threshold=.1)

    def infer(self,req):
        torch=self.torch
        query=read_image(req['query'],rgb=True)
        h,w=query.shape[:2]
        masks,scores=[],[]
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
            if req['operation']=='text':
                tile=req['tile']
                for y in range(0,h,tile):
                    for x in range(0,w,tile):
                        self.processor.confidence_threshold=req['conf']
                        state=self.processor.set_text_prompt(prompt=req['text'],state=
                            self.processor.set_image(Image.fromarray(query[y:y+tile,x:x+tile])))
                        local=(state['masks_logits'][:,0] > req['pixel_thr']).cpu().numpy()
                        values=state['scores'].float().cpu().numpy()
                        for m,s in zip(local,values):
                            if s < req['conf'] or not m.any():
                                continue
                            full=np.zeros((h,w),bool)
                            full[y:y+m.shape[0],x:x+m.shape[1]]=m
                            masks.append(full); scores.append(float(s))
            else:
                reference=read_image(req['reference'],rgb=True)
                state=visual_state(self.processor,query,reference,req['box'],req['same_image'])
                local=(state['masks_logits'][:,0] > req['pixel_thr']).cpu().numpy()
                if not req['same_image']:
                    local=local[:,:,w:]
                values=state['scores'].float().cpu().numpy()
                for m,s in zip(local,values):
                    if s >= req['score_thr'] and m.any():
                        masks.append(m); scores.append(float(s))
        return dict(masks=pack(masks,(h,w)),scores=np.asarray(scores,np.float32),shape=[h,w])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['sam3','dynamic'])
    p.add_argument('--sam3-root');p.add_argument('--checkpoint');p.add_argument('--bpe')
    args=p.parse_args()
    wire=sys.stdout
    backend=None
    for line in sys.stdin:
        try:
            req=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if args.kind=='sam3':
                    if backend is None:
                        backend=Sam3Model(args)
                    arrays=backend.infer(req)
                else:
                    import torch
                    from agent_baseline.t13_worker import DynamicBackend
                    if backend is None:
                        backend=DynamicBackend(req['config'],Path(req['output']).parent)
                    with torch.inference_mode():
                        # Retry image I/O without replaying successful GPU inference.
                        images=[read_image(q,rgb=True) for q in req['paths']]
                        image=images[0 if req['time']=='t1' else 1]
                        if req['operation']=='segment_everything':
                            masks=backend.proposals(image)
                            arrays=dict(masks=pack(masks,image.shape[:2]),shape=image.shape[:2])
                        elif req['operation']=='extract_features':
                            arrays=dict(data=backend.features(image))
                        else:
                            arrays=dict(data=backend.semantic(image,req['classes']))
            save_npz(req['output'],**arrays)
            response=dict(ok=True,path=req['output'])
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response=dict(ok=False,error='Model or local I/O failure; inspect stderr')
        wire.write(json.dumps(response)+'\n');wire.flush()


if __name__=='__main__':
    main()
