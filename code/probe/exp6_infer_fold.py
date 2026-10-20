"""Save out-of-fold SECOND train predictions in the existing probe prediction layout."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from exp6_common import CFG, CV, PRED, VENDOR, read_fold, write_json

sys.path.insert(0, str(VENDOR))
from changedetection.configs.config import get_config
from changedetection.datasets.common import img_loader
from changedetection.datasets.imutils import normalize_img, to_channel_first
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from changedetection.script.script_utils import get_vssm_kwargs


class HeldoutImages(Dataset):
    def __init__(self, fold):
        self.root = Path(fold['root']) / 'heldout'
        self.images = fold['heldout_images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        pair = [np.ascontiguousarray(to_channel_first(normalize_img(img_loader(self.root / phase / name))))
                for phase in ('T1', 'T2')]
        return *pair, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv_root', type=Path, default=CV)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, default=PRED)
    parser.add_argument('--no_sem', action='store_true', help='Omit two 7-class float16 softmax maps')
    args = parser.parse_args()
    _, fold = read_fold(args.cv_root, args.fold)
    root = Path(fold['root'])
    summary = json.loads((root / 'training_complete.json').read_text())
    if not summary['completed'] or summary['manifest'] != str(args.cv_root.absolute() / 'manifest.json'):
        raise ValueError('Need a completed training run for this manifest')
    report_path = args.output / f'fold{args.fold}_complete.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    torch.backends.cudnn.benchmark = False
    checkpoint = root / 'ckpt/final.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    snapshot = payload['args_snapshot']
    if snapshot['fold'] != args.fold or snapshot['manifest'] != summary['manifest']:
        raise ValueError('Checkpoint belongs to another fold/manifest')
    cfg = get_config(SimpleNamespace(cfg=str(CFG), opts=None))
    model = ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None, **get_vssm_kwargs(cfg))
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.cuda().eval()
    for subdir in ('prob', 'mask', 'sem'):
        if subdir != 'sem' or not args.no_sem:
            (args.output / subdir).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(HeldoutImages(fold), batch_size=1, shuffle=False, num_workers=4)
    started = time.perf_counter()
    count = 0
    with torch.inference_mode():
        for before, after, names in loader:
            cd, t1, t2 = model(before.cuda(), after.cuda())
            prob = cd.softmax(1)[0, 1].float().cpu().numpy()
            name, stem = names[0], Path(names[0]).stem
            np.save(args.output / 'prob' / f'{stem}.npy', prob.astype(np.float16))
            Image.fromarray((prob > 0.5).astype(np.uint8) * 255).save(args.output / 'mask' / name)
            if not args.no_sem:
                np.savez(args.output / 'sem' / f'{stem}.npz',
                         t1=t1.softmax(1)[0].cpu().numpy().astype(np.float16),
                         t2=t2.softmax(1)[0].cpu().numpy().astype(np.float16))
            count += 1
            if count % 100 == 0:
                print(f'fold{args.fold}: {count}/{len(loader)} images', flush=True)
    report = dict(dataset='second_cv', detector='mambascd_base', fold=args.fold,
                  images=fold['heldout_images'], count=count, checkpoint=str(checkpoint),
                  best_iteration=summary['best_iteration'], threshold=0.5,
                  threshold_before_float16=True, semantics=not args.no_sem,
                  semantic_format='npz keys t1/t2; each float16 (7,H,W); unmasked softmax',
                  manifest=summary['manifest'], elapsed_seconds=time.perf_counter() - started)
    write_json(report_path, report)
    print(f'Completed fold{args.fold}: {count} predictions; report={report_path}', flush=True)


if __name__ == '__main__':
    main()
