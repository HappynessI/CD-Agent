"""SAM ViT-H box/point prompting (tools T5 and T6).

The image encoder runs once per image on the GPU (`embed`), and the embeddings are cached as fp16.
Prompted decoding then runs on the CPU inside the agent loop, so any mask state can be refined
without another GPU pass.

    python sam_tools.py embed --split val --shard 0 --shards 1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from common import RUN, SAM_CKPT, load_rgb, names

EMB = RUN / 'sam_emb'


def build(device):
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry['vit_h'](checkpoint=str(SAM_CKPT)).to(device).eval()
    return SamPredictor(sam)


def embed(a):
    pred = build('cuda')
    todo = names(a.split)[a.shard::a.shards]
    for name in todo:
        out = EMB / a.split / (Path(name).stem + '.pt')
        if out.exists():
            continue
        feats = []
        for im in load_rgb(a.split, name):
            pred.set_image(im)
            feats.append(pred.features.half().cpu())
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(features=torch.cat(feats), original_size=pred.original_size,
                        input_size=pred.input_size), out)
        print(name, flush=True)


class Prompter:
    """CPU SAM decoder on cached embeddings."""

    def __init__(self, device='cpu'):
        self.pred = build(device)
        # Only the prompt encoder + mask decoder run here; keep img_size for postprocess_masks.
        stub = torch.nn.Module()
        stub.img_size = self.pred.model.image_encoder.img_size
        self.pred.model.image_encoder = stub
        self.device = device
        self.key = None

    def set(self, split, name, t):
        key = (split, name, t)
        if key != self.key:
            d = torch.load(EMB / split / (Path(name).stem + '.pt'))
            self.pred.reset_image()
            self.pred.features = d['features'][t:t + 1].float().to(self.device)
            self.pred.original_size, self.pred.input_size = d['original_size'], d['input_size']
            self.pred.interm_features = None  # SAM-HQ fork API; the standard vit_h decoder ignores it
            self.pred.is_image_set = True
            self.key = key

    def box(self, split, name, t, box):
        self.set(split, name, t)
        masks, iou, _ = self.pred.predict(box=np.asarray(box, float), multimask_output=False)
        return masks[0], float(iou[0])

    def points(self, split, name, t, pts, labels):
        self.set(split, name, t)
        masks, iou, _ = self.pred.predict(point_coords=np.asarray(pts, float),
                                          point_labels=np.asarray(labels), multimask_output=True)
        i = int(np.argmax(iou))
        return masks[i], float(iou[i])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['embed'])
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    embed(p.parse_args())
