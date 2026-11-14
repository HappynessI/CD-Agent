"""GPU-only, offline, lazy worker. JSON stdin/stdout; diagnostics go to stderr."""
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from .analysis.io_retry import write_text


class DynamicBackend:
    def __init__(self, cfg, cache):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('T1.3 inference requires a GPU compute node')
        self.cfg, self.cache = cfg, Path(cache)
        de = Path(cfg['dynamic_earth']).resolve()
        sys.path[:0] = [str(de),str(de/'third_party/SegEarth_OV'),str(de/'third_party/segment_anything')]
        self.sam, self.dino, self.processor, self.identifier = None,None,None,None
        self.vocabulary = None

    def proposals(self, image):
        if self.sam is None:
            from segment_anything import sam_model_registry
            from dynamic_earth.sam_ext import MaskProposal
            self.sam = MaskProposal()
            model = sam_model_registry['vit_h'](checkpoint=self.cfg['sam_checkpoint']).to('cuda').eval()
            self.sam.make_mask_generator(model=model,points_per_side=32,points_per_batch=64,
                pred_iou_thresh=.5,stability_score_thresh=.95,stability_score_offset=.9,
                box_nms_thresh=.7,min_mask_region_area=0)
            self.sam.set_hyperparameters(match_hist=False,area_thresh=.8)
        from segment_anything.utils.amg import rle_to_mask
        data,_ = self.sam.proposal(image)
        return np.asarray([rle_to_mask(r) for r in data['rles']],bool).reshape(-1,*image.shape[:2])

    def features(self, image):
        if self.dino is None:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(self.cfg['dinov2'],do_resize=False,do_center_crop=False,local_files_only=True)
            self.dino = AutoModel.from_pretrained(self.cfg['dinov2'],local_files_only=True).to('cuda').eval()
        from dynamic_earth.comparator.bi_match import extract_features
        return extract_features(image,self.dino,self.processor,dict(model_type='DINOv2',feature_dim=768,patch_size=14),'cuda')[0].permute(1,2,0).float().cpu().numpy()

    def semantic(self, image, classes):
        import torch
        from torchvision import transforms
        # SegEarth-OV ships a top-level package named `prompts`; our ovcd_v0/prompts.py module may already be loaded.
        ours = sys.modules.get('prompts')
        if ours is not None and not hasattr(ours, '__path__'):
            del sys.modules['prompts']
        saved = list(sys.path)
        # Directories holding a plain prompts.py would shadow SegEarth-OV's package during this import.
        sys.path[:] = [p for p in sys.path if not os.path.isfile(os.path.join(p or '.', 'prompts.py'))]
        try:
            from dynamic_earth.identifier.segearth_ov_ext import SegEarth_OV
            import open_clip_custom.openai as clip_loader
        finally:
            sys.path[:] = saved
            if ours is not None and not hasattr(ours, '__path__'):
                sys.modules['prompts'] = ours
        vocabulary = ('background',*classes)
        if self.vocabulary != vocabulary:
            # A local names file avoids get_identifier's implicit ./class_name.txt write.
            names = self.cache/f'classes_{os.getpid()}.txt'
            write_text(names, '\n'.join(vocabulary)+'\n')
            original = clip_loader.download_pretrained_from_url
            def local_download(url, cache_dir=None):
                if not url.endswith('/ViT-B-16.pt'):
                    raise ValueError('Unexpected CLIP checkpoint requested')
                path = Path(self.cfg['clip_checkpoint'])
                if not path.is_file():
                    raise FileNotFoundError(path)
                return str(path)
            clip_loader.download_pretrained_from_url = local_download
            try:
                self.identifier = SegEarth_OV(clip_type='CLIP',vit_type='ViT-B/16',model_type='SegEarth',
                    name_path=str(names),device='cuda',ignore_residual=True,feature_up=True,
                    feature_up_cfg=dict(model_name='jbu_one',model_path=self.cfg['simfeatup_checkpoint']),cls_token_lambda=-.3).eval()
            finally:
                clip_loader.download_pretrained_from_url = original
            self.vocabulary = vocabulary
        processor = transforms.Compose([transforms.ToTensor(),transforms.Normalize(
            [0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),transforms.Resize((448,448))])
        inputs = processor(image).unsqueeze(0).to('cuda').half()
        logits = self.identifier.forward_slide(inputs,[dict(ori_shape=image.shape[:2])],self.identifier.slide_stride,self.identifier.slide_crop)
        return (logits[0]*self.identifier.logit_scale).softmax(0).float().cpu().numpy()

    def infer(self, request):
        from PIL import Image
        image = np.asarray(Image.open(request['paths'][0 if request['time']=='t1' else 1]).convert('RGB'))
        operation = request['operation']
        if operation == 'segment_everything':
            return self.proposals(image)
        if operation == 'extract_features':
            return self.features(image)
        if operation == 'classify':
            return self.semantic(image,request['classes'])
        raise ValueError('Unknown dynamic operation')


class AnyChangeBackend:
    def __init__(self,cfg,cache):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('AnyChange requires a GPU compute node')
        from torchange.models.segment_any_change import AnyChange
        self.model = AnyChange('vit_h',sam_checkpoint=cfg['sam_checkpoint'])
        self.model.make_mask_generator(points_per_side=32,stability_score_thresh=.95)
        self.model.set_hyperparameters(change_confidence_threshold=145,use_normalized_feature=True,bitemporal_match=True)

    def infer(self, request):
        from PIL import Image
        from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask
        images = [np.asarray(Image.open(p).convert('RGB')) for p in request['paths']]
        try:
            masks,_,_ = self.model.forward(*images)
            return np.asarray([rle_to_mask(r) for r in masks['rles']],bool).reshape(-1,*images[0].shape[:2])
        finally:
            self.model.clear_cached_embedding()


def cache_path(request):
    """Readable indexed cache inventory; identity includes source stat and all config."""
    root = Path(request['cache'])/sys.argv[1]
    root.mkdir(parents=True,exist_ok=True)
    sources = [dict(path=p,size=Path(p).stat().st_size,mtime=Path(p).stat().st_mtime_ns) for p in request['paths']]
    key = dict(version=1,sources=sources,config=request['config'],operation=request['operation'],time=request.get('time'),classes=request.get('classes'))
    import fcntl
    lock = (root/'inventory.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    inventory = root/'inventory.json'
    records = json.loads(inventory.read_text()) if inventory.exists() else []
    if key not in records:
        records.append(key)
        temporary = inventory.with_suffix('.tmp')
        temporary.write_text(json.dumps(records,indent=1))
        temporary.replace(inventory)
    lock.close()
    return root/f"item_{records.index(key):06d}.npz"


def main():
    backend = None
    wire = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            path = cache_path(request)
            import fcntl
            with path.with_suffix('.lock').open('a') as item_lock:
                fcntl.flock(item_lock,fcntl.LOCK_EX)
                if not path.exists():
                    with contextlib.redirect_stdout(sys.stderr):
                        import torch
                        if backend is None:
                            cls = AnyChangeBackend if sys.argv[1]=='anychange' else DynamicBackend
                            backend = cls(request['config'],path.parent)
                        with torch.inference_mode():
                            result = backend.infer(request)
                    temporary = path.with_suffix('.tmp.npz')
                    if request['operation'] in ('segment_everything','anychange'):
                        np.savez(temporary,data=np.packbits(result.reshape(len(result),int(np.prod(result.shape[1:]))),axis=1),shape=result.shape[1:])
                    else:
                        np.savez(temporary,data=result)
                    temporary.replace(path)
            response = dict(ok=True,path=str(path))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = dict(ok=False,error=f'{type(exc).__name__}: {exc}')
        wire.write(json.dumps(response)+'\n')
        wire.flush()


if __name__ == '__main__':
    main()
