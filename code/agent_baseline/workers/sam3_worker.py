"""Serialized SAM3 text inference in its own environment and cache namespace."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'agent_baseline.workers'
    import agent_baseline

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from urllib.parse import quote
import numpy as np
from PIL import Image
from common import load_rgb


class Segmenter:
    def __init__(self, processor, engine, run):
        self.processor, self.engine, self.run = processor, engine, Path(run)
        self.lock = threading.Lock()

    def infer(self, split, name, time, text, tile, conf=.1, pixel_thr=.5):
        if time not in ('t1', 't2') or tile not in (0, 256, 512) or not 0 <= conf <= 1 or not 0 < pixel_thr < 1:
            raise ValueError('Invalid segmentation parameters')
        text = ' '.join(text.lower().split())
        slug = quote(text, safe='')
        path = self.run / 'sam3_cache' / (f'{slug}_t{tile}_c{conf:g}' + (f'_p{pixel_thr:g}' if pixel_thr != .5 else '')) / split / f'{Path(name).stem}_{time}.npz'
        started = __import__('time').monotonic()
        with self.lock:
            queued = __import__('time').monotonic() - started
            if path.exists():
                return dict(ok=True, path=str(path), cache_hit=True, seconds=0, queue_seconds=queued)
            import torch
            image = load_rgb(split, name)[0 if time == 't1' else 1]
            h, w = image.shape[:2]
            self.processor.confidence_threshold = conf
            step_y, step_x = (tile, tile) if tile else (h, w)
            packed, scores = [], []
            for y in range(0, h, step_y):
                for x in range(0, w, step_x):
                    with torch.inference_mode(), self.engine.autocast():
                        state = self.processor.set_image(Image.fromarray(image[y:y+step_y, x:x+step_x]))
                        state = self.processor.set_text_prompt(prompt=text, state=state)
                    if not len(state['scores']):
                        continue
                    masks = (state['masks'] if pixel_thr == .5 else state['masks_logits'] > pixel_thr)[:, 0].cpu().numpy().astype(bool)
                    values = state['scores'].float().cpu().numpy()
                    for mask, score in zip(masks, values):
                        full = np.zeros((h, w), bool)
                        full[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
                        packed.append(np.packbits(full.ravel()))
                        scores.append(float(score))
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp.npz')
            np.savez(temp, masks=np.asarray(packed, np.uint8).reshape(len(packed), (h*w+7)//8),
                     scores=np.asarray(scores, np.float32), shape=[h,w], text=text, tile=tile, conf=conf)
            temp.replace(path)
        return dict(ok=True, path=str(path), cache_hit=False,
                    seconds=__import__('time').monotonic()-started-queued, queue_seconds=queued)


def handler(segmenter):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, result):
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if self.path == '/health' else 404, dict(ok=self.path == '/health'))

        def do_POST(self):
            if self.path != '/infer':
                return self.respond(404, dict(ok=False))
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.respond(200, segmenter.infer(**payload))
            except Exception:
                traceback.print_exc()
                self.respond(500, dict(ok=False, error='Segmentation failed.'))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--run', default=os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
    args = parser.parse_args()
    sys.path[:0] = ['/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD',
                    '/guisongxia01/pangchao/wangyihan/Chang-Agent/probe']
    import sam3_tools
    segmenter = Segmenter(sam3_tools.build_processor(), sam3_tools, args.run)
    ThreadingHTTPServer(('127.0.0.1', args.port), handler(segmenter)).serve_forever()


if __name__ == '__main__':
    main()
