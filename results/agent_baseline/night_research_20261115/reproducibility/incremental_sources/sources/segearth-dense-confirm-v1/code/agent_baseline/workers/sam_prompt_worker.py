"""One SAM ViT-H prompt decoder per job, shared by every Agent process through localhost HTTP.

Two paths, the same as backends.SharedPrompter:
- cached: LEVIR split/name with stored image embeddings (sam_tools.Prompter);
- image: any other image. The client sends the exact pixel array the session holds; embeddings are kept
  in a small LRU keyed by a hash of the pixels, so the three repeats of one image are encoded once.
Point prompts take the highest-quality of three masks; box prompts use one mask, as before.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'agent_baseline.workers'
    import agent_baseline

import argparse
import base64
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import traceback
import numpy as np


def decode_array(spec):
    data = np.frombuffer(base64.b64decode(spec['data']), dtype=np.dtype(spec['dtype']))
    return data.reshape(spec['shape'])


def encode_mask(mask):
    mask = np.asarray(mask, bool)
    return dict(shape=list(mask.shape), bits=base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode('ascii'))


class PromptServer:
    def __init__(self, image_factory=None, cached_factory=None, cache_size=32):
        self.lock = threading.Lock()
        self.image_factory, self.cached_factory = image_factory, cached_factory
        self.image_model = self.cached = None
        self.embeddings, self.cache_size = OrderedDict(), cache_size
        self.current = None
        self.stats = dict(calls=0, encodes=0, cache_hits=0)

    def _image_model(self):
        if self.image_model is None:
            if self.image_factory is None:
                import torch
                from sam_tools import build
                self.image_factory = lambda: build('cuda' if torch.cuda.is_available() else 'cpu')
            self.image_model = self.image_factory()
        return self.image_model

    def _set_image(self, pred, rgb):
        key = hashlib.sha1(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
        if key == self.current:
            self.stats['cache_hits'] += 1
            return
        saved = self.embeddings.get(key)
        if saved is not None:
            self.embeddings.move_to_end(key)
            pred.reset_image()
            pred.features, pred.original_size, pred.input_size = saved
            pred.is_image_set = True
            self.stats['cache_hits'] += 1
        else:
            pred.set_image(rgb)
            self.embeddings[key] = (pred.features, pred.original_size, pred.input_size)
            if len(self.embeddings) > self.cache_size:
                self.embeddings.popitem(last=False)
            self.stats['encodes'] += 1
        self.current = key

    def infer(self, payload):
        box, points, labels = payload.get('box'), payload.get('points'), payload.get('point_labels')
        started = time.monotonic()
        with self.lock:
            queued = time.monotonic() - started
            self.stats['calls'] += 1
            if payload.get('mode') == 'image':
                pred = self._image_model()
                self._set_image(pred, decode_array(payload['image']))
                if box is not None:
                    masks, quality, _ = pred.predict(box=np.asarray(box, float), multimask_output=False)
                else:
                    masks, quality, _ = pred.predict(point_coords=np.asarray(points, float),
                        point_labels=np.asarray(labels), multimask_output=True)
                i = int(np.argmax(quality))
                mask, q = masks[i], float(quality[i])
            else:
                if self.cached is None:
                    if self.cached_factory is None:
                        from sam_tools import Prompter
                        self.cached_factory = lambda: Prompter(device='cuda')
                    self.cached = self.cached_factory()
                t = 0 if payload['time'] == 't1' else 1
                if box is not None:
                    mask, q = self.cached.box(payload['split'], payload['name'], t, box)
                else:
                    mask, q = self.cached.points(payload['split'], payload['name'], t, points, labels)
            mask = np.array(mask, copy=True)
        return dict(ok=True, mask=encode_mask(mask), quality=q, queue_seconds=queued,
                    seconds=time.monotonic() - started)


def handler(server):
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
            if self.path == '/health':
                self.respond(200, dict(ok=True, **server.stats))
            else:
                self.respond(404, dict(ok=False))

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.respond(200, server.infer(payload))
            except Exception:
                traceback.print_exc()
                self.respond(500, dict(ok=False, error='inference failed'))
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--cache-size', type=int, default=32)
    a = p.parse_args()
    ThreadingHTTPServer(('127.0.0.1', a.port), handler(PromptServer(cache_size=a.cache_size))).serve_forever()


if __name__ == '__main__':
    main()
