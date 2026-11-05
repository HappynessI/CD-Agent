"""Threaded HTTP ingress, one inference thread and <=8-pair/50-ms microbatches."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'agent_baseline.workers'
    import agent_baseline

import argparse
import base64
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import queue
import threading
import time
import traceback
from PIL import Image
from prompts import VERIFY


class MicroBatch:
    def __init__(self, model, max_batch=8, wait=.05):
        self.model, self.max_batch, self.wait = model, max_batch, wait
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, pair):
        future = Future()
        self.queue.put((pair, future, time.monotonic()))
        return future

    def run(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                batch.append(item)
            started = time.monotonic()
            try:
                answers = self.model.verify([item[0] for item in batch])
                if len(answers) != len(batch):
                    raise RuntimeError('Verifier answer count differs from request count')
                elapsed = time.monotonic() - started
                for (_, future, queued), answer in zip(batch, answers):
                    future.set_result(dict(answer, status='parsed' if answer.get('answer') in ('yes', 'no') else 'unparsed',
                        queue_seconds=started-queued, inference_seconds=elapsed, batch_size=len(batch)))
            except Exception as exc:
                traceback.print_exc()
                for _, future, _ in batch:
                    future.set_exception(exc)

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=2)


def handler(batcher):
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
                if payload.get('question') != VERIFY:
                    return self.respond(400, dict(ok=False, error='Only the fixed verifier question is supported.'))
                pairs = [[Image.open(BytesIO(base64.b64decode(s))).convert('RGB') for s in pair] for pair in payload['pairs']]
                if any(len(p) != 2 for p in pairs):
                    raise ValueError('Each region needs two images')
                futures = [batcher.submit(pair) for pair in pairs]
                rows = [f.result() for f in futures]
                self.respond(200, dict(ok=True, results=rows,
                    queue_seconds=max((r['queue_seconds'] for r in rows), default=0),
                    inference_seconds=sum(r['inference_seconds']/r['batch_size'] for r in rows)))
            except Exception:
                traceback.print_exc()
                self.respond(500, dict(ok=False, error='Verifier failed.'))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--target', default='building')
    parser.add_argument('--change-direction', default='any', choices=['any', 'appeared', 'disappeared'])
    args = parser.parse_args()
    from common import QWEN
    from mllm import Qwen
    import prompts as verifier
    settings = verifier.verifier_prompts(args.target, args.change_direction)
    # One local worker serves one task scope; its existing microbatch stays homogeneous.
    global VERIFY
    VERIFY = settings['VERIFY']
    verifier.SYSTEM, verifier.VERIFY = settings['SYSTEM'], VERIFY
    batcher = MicroBatch(Qwen(QWEN))
    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), handler(batcher)).serve_forever()
    finally:
        batcher.close()


if __name__ == '__main__':
    main()
