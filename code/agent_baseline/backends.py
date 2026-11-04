"""Adapters for existing caches, one shared SAM decoder, and optional HTTP workers."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import numpy as np
from prompts import PSEUDO_TYPES, SYSTEM, VERIFY
from . import instances
from .state import ToolError
from .planner import Planner


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


def read_soft(split, name, index, pixel_thr, score_thr, root=None, tile=512):
    """Instances of one tile grid from the soft cache (sam3_soft.py, step = tile / 2), binarized at pixel_thr."""
    root = Path(root or os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
    with np.load(root / 'sam3_soft' / f't{tile}s{tile // 2}' / split / (Path(name).stem + '.npz')) as z:
        scores, origins, offsets = z[f'scores{index}'], z[f'origins{index}'], z[f'offs{index}']
        pixels, values, (h, w) = z[f'idx{index}'], z[f'val{index}'], z['shape']
    rows = []
    for i in np.flatnonzero(scores >= score_thr):
        if origins[i][0] % tile or origins[i][1] % tile:
            continue
        idx = pixels[offsets[i]:offsets[i + 1]][values[offsets[i]:offsets[i + 1]] >= round(pixel_thr * 255)]
        if len(idx):
            idx = np.sort(idx).astype(np.int32)
            ys, xs = idx // w, idx % w
            rows.append(dict(id=len(rows) + 1, idx=idx, area=int(idx.size), score=float(scores[i]),
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return rows


class SegmentBackend:
    def __init__(self, split, name, worker=None, pixel_thr=.5):
        self.split, self.name, self.worker, self.pixel_thr = split, name, worker, pixel_thr

    def __call__(self, time, text, tile, score_thr):
        import sam3_t7
        index = 0 if time == 't1' else 1
        building = ''.join(text.lower().split()) == 'building'
        if self.pixel_thr != .5:
            # The soft cache holds scores from 0.3 and values from 0.2.
            if building and tile in (256, 512) and score_thr >= .3:
                try:
                    return read_soft(self.split, self.name, index, self.pixel_thr, score_thr, tile=tile), dict(
                        cache_hit=True, backend='sam3_soft_cache', tile=tile, pixel_thr=self.pixel_thr)
                except FileNotFoundError:
                    pass
        elif building and ((tile == 512 and score_thr >= .1) or (tile == 0 and score_thr >= .5)):
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
        response = post(self.worker, dict(split=self.split, name=self.name, time=time, text=text, tile=tile, conf=conf, pixel_thr=self.pixel_thr))
        with np.load(response['path']) as z:
            h, w = (int(x) for x in z['shape'])
            scores = z['scores']
            keep = scores >= score_thr
            masks = np.unpackbits(z['masks'][keep], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
            rows = instances.from_dense(masks, scores[keep])
        return rows, dict(cache_hit=response.get('cache_hit', False), backend='sam3_worker',
                          worker_seconds=response.get('seconds'), cache_path=response['path'], conf=conf, pixel_thr=self.pixel_thr)


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


# Bounds simultaneous verifier requests per process, whatever the number of sessions.
ASK_SLOTS = threading.Semaphore(int(os.environ.get('AGENT_ASK_SLOTS', 4)))
CROP_WORDS = ('T1 close-up', 'T2 close-up', 'T1 wider view', 'T2 wider view')
CONTEXT_NOTE = 'The first two images are close-ups and the last two show the same place with more surroundings. '
CORRESPONDENCE = ('The first two images are close-ups and the last two show the same place with more surroundings. '
    'The red contour marks the same region in all of them. First find the object inside the contour at the same '
    'position in both dates, and compare its outline, roof structure and neighbouring fixed objects. Is there evidence '
    'that a building was newly constructed, demolished, or replaced by a different building? If both dates show the '
    'same building and only its color, shadow, occlusion or visibility differs, answer no. Do not read poor visibility '
    'as demolition. The reason must name the evidence of correspondence or change. Reply with JSON only: '
    '{"change": "yes" or "no", "type": one of ' + str(PSEUDO_TYPES).replace("'", '"') + ', "reason": one short sentence}')


EXAMPLES = Path(__file__).parent / 'verifier_examples.json'
_example_blocks = None
_example_lock = threading.Lock()


def example_blocks():
    """Fixed solved examples for the thorough check: the same bytes in every request, so the provider can cache the prefix."""
    global _example_blocks
    with _example_lock:
        if _example_blocks is None:
            from common import load_rgb
            from .view import ask_images
            blocks = [dict(type='text', text='Solved examples from other places. They show how to apply the question; they are not part of the region you will answer about.')]
            for n, row in enumerate(json.loads(EXAMPLES.read_text()), 1):
                rgb = load_rgb('train', row['name'])
                region = np.zeros(rgb[0].shape[:2], bool)
                flat = region.ravel()
                for start, length in row['runs']:
                    flat[start:start + length] = True
                blocks.append(dict(type='text', text=f'Example {n}.'))
                for word, image in zip(CROP_WORDS, ask_images(rgb, region, row['bbox'], True)):
                    blocks += [dict(type='text', text=word), dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
                blocks.append(dict(type='text', text='Answer: ' + json.dumps(dict(change=row['answer'], type=row['type'], reason=row['reason']))))
            blocks.append(dict(type='text', text='Now the region to answer about.'))
            _example_blocks = blocks
    return _example_blocks


class ApiAsk:
    """Independent verifier: each region is one fresh request that knows nothing of the planner's reasoning."""
    context = True

    def __init__(self, client=None, guard=None, model='qwen3.7-plus', max_attempts=12, correspondence=False):
        self.client, self.guard, self.model, self.max_attempts = client, guard, model, max_attempts
        self.correspondence = correspondence

    def one(self, images, question, number, level='L1'):
        from mllm import parse_json
        events = []
        # Region crops are reproducible from the mask, so the archive keeps the reply and not the images.
        planner = Planner(client=self.client, guard=self.guard, max_attempts=self.max_attempts,
            record=lambda event: events.append({k: v for k, v in event.items() if k != 'request'}),
            parameters=dict(model=self.model, logprobs=True, top_logprobs=5, response_format={'type': 'json_object'},
                            parallel_tool_calls=False, max_tokens=256))
        content = list(example_blocks()) if level == 'L2' else []
        for word, image in zip(CROP_WORDS, images):
            content += [dict(type='text', text=word), dict(type='image_url',
                        image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
        content.append(dict(type='text', text=CORRESPONDENCE if self.correspondence else (CONTEXT_NOTE if len(images) > 2 else '') + question))
        try:
            with ASK_SLOTS:
                result = planner.complete([dict(role='system', content=SYSTEM), dict(role='user', content=content)], [], number)
        except Exception as exc:
            exc.partial = dict(attempts=events, tokens=planner.tokens, cost=planner.cost)
            raise
        raw = result['message'].get('content') or ''
        parsed = parse_json(raw)
        parsed = parsed if isinstance(parsed, dict) else {}
        answer = dict(answer=parsed.get('change'), type=parsed.get('type'), reason=parsed.get('reason'), level=level,
                      p_yes=yes_probability(result['raw']['choices'][0]), raw=raw, usage=result['usage'])
        return answer, events, planner.tokens, planner.cost

    def __call__(self, pairs, question, level='L1'):
        total = dict(input=0, cached=0, output=0)
        cost = 0.
        attempts = []
        answers = [None] * len(pairs)
        failure = None
        with ThreadPoolExecutor(min(16, len(pairs)) or 1) as pool:
            futures = [pool.submit(self.one, images, question, n, level) for n, images in enumerate(pairs, 1)]
            for n, future in enumerate(futures):
                try:
                    answers[n], events, tokens, spent = future.result()
                    cost += spent
                except Exception as exc:
                    # Keep the first failure and let the other requests finish, so all spending is accounted.
                    failure = failure or exc
                    partial = getattr(exc, 'partial', dict(attempts=[], tokens=dict(input=0, cached=0, output=0), cost=0.))
                    events, tokens = partial['attempts'], partial['tokens']
                    cost += partial['cost']
                attempts.extend(events)
                for key in total:
                    total[key] += tokens[key]
        meta = dict(backend='api_ask', model=self.model, correspondence=self.correspondence, attempts=attempts, tokens=total, cost_cny=cost)
        if failure is not None:
            failure.backend_meta = meta
            raise failure
        return answers, meta
