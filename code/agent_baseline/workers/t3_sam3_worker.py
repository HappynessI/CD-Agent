"""T3 diagnostic SAM3 worker: JSON stdin/stdout, text or positive visual box.

A reference/query mosaic lets the documented same-image geometric prompt retrieve
objects in the other date without putting a box around a nonexistent object.
No HTTP service, no online-tool changes, no implicit model download.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from agent_baseline.analysis.io_retry import read_image, retry_io


def normalized_box(box, width, height):
    x0, y0, x1, y1 = box
    return [(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height]


def prompt_state(processor, query, text=None, reference=None, box=None):
    """Pure orchestration; testable without importing torch or real models."""
    if reference is None:
        return processor.set_text_prompt(prompt=text, state=processor.set_image(Image.fromarray(query)))
    h, w = query.shape[:2]
    mosaic = np.concatenate([reference, query], axis=1)
    state = processor.set_image(Image.fromarray(mosaic))
    return processor.add_geometric_prompt(box=normalized_box(box, 2*w, h), label=True, state=state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bpe', required=True)
    p.add_argument('--sam3-root', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.sam3_root)
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model = build_sam3_image_model(bpe_path=a.bpe, checkpoint_path=a.checkpoint, device=a.device,
                                      load_from_HF=False, enable_inst_interactivity=False)
        model.to(a.device)
        model._device = torch.device(a.device)
        processor = Sam3Processor(model, resolution=1008, device=a.device, confidence_threshold=.1)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            query = read_image(req['query'], rgb=True)
            reference = read_image(req['reference'], rgb=True) if req.get('reference') else None
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.autocast('cuda', dtype=dtype):
                    state = prompt_state(processor, query, req.get('text'), reference, req.get('box'))
            masks = (state['masks_logits'][:, 0] > .85).cpu().numpy()
            scores = state['scores'].float().cpu().numpy()
            masks = masks[scores >= .5]
            if reference is not None:
                masks = masks[:, :, query.shape[1]:]
            masks = masks[np.any(masks, axis=(1, 2))]
            path = Path(req['output'])
            def save():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp.npz')
                np.savez(temp, masks=np.packbits(masks.reshape(len(masks), query.shape[0]*query.shape[1]), axis=1), shape=query.shape[:2])
                temp.replace(path)
            retry_io(save)
            result = dict(ok=True, path=str(path))
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
        wire.write(json.dumps(result)+'\n')
        wire.flush()


if __name__ == '__main__':
    main()
