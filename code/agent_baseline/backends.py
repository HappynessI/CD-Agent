"""Adapters for existing caches, one shared SAM decoder, and optional HTTP workers."""
import base64
from io import BytesIO
import json
import math
import threading
import time
import urllib.error
import urllib.request
import numpy as np
from prompts import SYSTEM, VERIFY
from . import instances
from .state import ToolError
from .planner import Planner, usage_tokens, token_cost


def post(url, payload, timeout=180):
    request = urllib.request.Request(url.rstrip('/') + '/infer', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.URLError:
        raise ToolError('backend_unavailable', 'The worker is unavailable.') from None
    if not result.get('ok'):
        raise RuntimeError('Worker inference failed')
    return result


def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=92)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class SegmentBackend:
    def __init__(self, split, name, worker=None):
        self.split, self.name, self.worker = split, name, worker

    def __call__(self, time, text, tile, score_thr):
        import sam3_t7
        index = 0 if time == 't1' else 1
        cached = ''.join(text.lower().split()) == 'building' and ((tile == 512 and score_thr >= .1) or (tile == 0 and score_thr >= .5))
        if cached:
            conf = .1 if tile == 512 else .5
            try:
                masks, scores = sam3_t7.read(self.split, self.name, tile, conf=conf, min_score=score_thr)[index]
                return instances.from_dense(masks, scores), dict(cache_hit=True, backend='sam3_cache', tile=tile, conf=conf)
            except FileNotFoundError:
                if self.worker is None:
                    raise ToolError('backend_unavailable', 'The requested segmentation cache is unavailable.') from None
        if self.worker is None:
            raise ToolError('backend_unavailable', 'This text or threshold needs a SAM3 worker.')
        # A lower requested threshold must never read a cache truncated above it.
        conf = min(.1, score_thr)
        response = post(self.worker, dict(split=self.split, name=self.name, time=time, text=text, tile=tile, conf=conf))
        with np.load(response['path']) as z:
            h, w = (int(x) for x in z['shape'])
            scores = z['scores']
            keep = scores >= score_thr
            masks = np.unpackbits(z['masks'][keep], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
            rows = instances.from_dense(masks, scores[keep])
        return rows, dict(cache_hit=response.get('cache_hit', False), backend='sam3_worker',
                          worker_seconds=response.get('seconds'), cache_path=response['path'], conf=conf)


class SharedPrompter:
    def __init__(self, factory=None):
        self.lock = threading.Lock()
        self.model = None
        self.factory = factory

    def for_image(self, split, name):
        def call(time, box=None, points=None, point_labels=None):
            start = time_module()
            with self.lock:
                queued = time_module() - start
                if self.model is None:
                    if self.factory:
                        self.model = self.factory()
                    else:
                        from sam_tools import Prompter
                        self.model = Prompter()
                t = 0 if time == 't1' else 1
                # Lock covers set -> predict inside box/points, not just set().
                if box is not None:
                    mask, quality = self.model.box(split, name, t, box)
                else:
                    mask, quality = self.model.points(split, name, t, points, point_labels)
                mask = np.array(mask, copy=True)
            return mask, quality, dict(backend='sam_prompt', queue_seconds=queued)
        return call


def time_module():
    return time.monotonic()


class LocalAsk:
    def __init__(self, url):
        self.url = url

    def __call__(self, pairs, question):
        if not self.url:
            raise ToolError('backend_unavailable', 'No ask worker is configured.')
        response = post(self.url, dict(pairs=[[encode_image(im) for im in pair] for pair in pairs], question=question))
        return response['results'], dict(backend='local_ask', queue_seconds=response.get('queue_seconds'),
                                          inference_seconds=response.get('inference_seconds'))


def yes_probability(choice):
    for entry in ((choice.get('logprobs') or {}).get('content') or []):
        if entry.get('token', '').strip().strip('"').lower() in ('yes', 'no'):
            mass = dict(yes=0., no=0.)
            for alt in entry.get('top_logprobs') or [entry]:
                key = alt.get('token', '').strip().strip('"').lower()
                if key in mass:
                    mass[key] += math.exp(alt['logprob'])
            total = sum(mass.values())
            return mass['yes'] / total if total else None
    return None


class ApiAsk:
    def __init__(self, client=None):
        self.client = client

    def __call__(self, pairs, question):
        from mllm import parse_json
        answers, events = [], []
        planner = Planner(client=self.client, record=events.append, max_attempts=96,
            parameters=dict(model='qwen3-vl-plus', logprobs=True, top_logprobs=5,
                            response_format={'type': 'json_object'}, parallel_tool_calls=False, max_tokens=256))
        try:
            for n, pair in enumerate(pairs, 1):
                content = []
                for t, im in enumerate(pair, 1):
                    content.extend([dict(type='text', text=f'T{t} close-up'), dict(type='image_url',
                        image_url=dict(url='data:image/jpeg;base64,' + encode_image(im)))])
                content.append(dict(type='text', text=question))
                result = planner.complete([dict(role='system', content=SYSTEM), dict(role='user', content=content)], [], n)
                choice = result['raw']['choices'][0]
                raw = result['message'].get('content', '')
                parsed = parse_json(raw)
                parsed = parsed if isinstance(parsed, dict) else {}
                answers.append(dict(answer=parsed.get('change'), type=parsed.get('type'), reason=parsed.get('reason'),
                    p_yes=yes_probability(choice), raw=raw, usage=result['usage']))
        except Exception as exc:
            # Partial API usage remains accountable even when a later region fails.
            exc.backend_meta = dict(backend='api_ask', attempts=events, tokens=planner.tokens, cost_cny=token_cost(planner.tokens))
            raise
        return answers, dict(backend='api_ask', attempts=events, tokens=planner.tokens, cost_cny=token_cost(planner.tokens))
