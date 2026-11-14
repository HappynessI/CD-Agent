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
from .scope_t13 import scope_card, verifier_prompts, session_prompts
from . import instances
from .segment_cache import manifest_cache_path, read_instances
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


def read_soft(split, name, index, pixel_thr, score_thr, root=None):
    """Instances of the 512 grid from the soft cache (sam3_soft.py, tile 512, step 256), binarized at pixel_thr."""
    root = Path(root or os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
    with np.load(root / 'sam3_soft' / 't512s256' / split / (Path(name).stem + '.npz')) as z:
        scores, origins, offsets = z[f'scores{index}'], z[f'origins{index}'], z[f'offs{index}']
        pixels, values, (h, w) = z[f'idx{index}'], z[f'val{index}'], z['shape']
    rows = []
    for i in np.flatnonzero(scores >= score_thr):
        if origins[i][0] % 512 or origins[i][1] % 512:
            continue
        idx = pixels[offsets[i]:offsets[i + 1]][values[offsets[i]:offsets[i + 1]] >= round(pixel_thr * 255)]
        if len(idx):
            idx = np.sort(idx).astype(np.int32)
            ys, xs = idx // w, idx % w
            rows.append(dict(id=len(rows) + 1, idx=idx, area=int(idx.size), score=float(scores[i]),
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return rows


class SegmentBackend:
    def __init__(self, split, name, worker=None, pixel_thr=.5, target="building", image_paths=None):
        self.target, self.image_paths = target, image_paths
        self.cache_config = scope_card(target)["sam3_cache"]
        self.split, self.name, self.worker, self.pixel_thr = split, name, worker, pixel_thr

    def __call__(self, time, text, tile, score_thr):
        import sam3_t7
        index = 0 if time == 't1' else 1
        cached_class = isinstance(self.target, str) and ''.join(text.lower().split()) == ''.join(self.target.lower().split())
        cache = self.cache_config if cached_class else {}
        if self.image_paths is not None:
            cache = {}  # Explicit images use the worker's source-path namespace.
            conf = min(.1, score_thr)
            root = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
            path = manifest_cache_path(root, self.image_paths[index], text, tile, conf, self.pixel_thr)
            if path.is_file():
                return read_instances(path, score_thr), dict(cache_hit=True, backend='sam3_manifest_cache',
                    tile=tile, conf=conf, pixel_thr=self.pixel_thr)
        if self.pixel_thr != .5:
            # The soft cache holds scores from 0.3 and values from 0.2.
            if cache and tile == 512 and score_thr >= cache['soft_min_score']:
                try:
                    return read_soft(self.split, self.name, index, self.pixel_thr, score_thr), dict(
                        cache_hit=True, backend='sam3_soft_cache', tile=tile, pixel_thr=self.pixel_thr)
                except FileNotFoundError:
                    pass
        elif cache and ((tile == 512 and score_thr >= cache['tile_min_score']) or (tile == 0 and score_thr >= cache['full_min_score'])):
            conf = cache['tile_min_score'] if tile == 512 else cache['full_min_score']
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
        payload = dict(split=self.split, name=self.name, time=time, text=text, tile=tile, conf=conf, pixel_thr=self.pixel_thr)
        if self.image_paths is not None:
            payload['image_path'] = self.image_paths[index]
        response = post(self.worker, payload)
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
        self.image_model = None
        self.factory = factory

    def for_image(self, split, name, rgb=None):
        def call(time, box=None, points=None, point_labels=None):
            start = time_module()
            with self.lock:
                queued = time_module() - start
                if rgb is not None:
                    # Arbitrary images cannot use embeddings indexed by dataset split/name.
                    if self.image_model is None:
                        # Images without stored embeddings are encoded per call; use the job's GPU when there is one.
                        import torch
                        from sam_tools import build
                        self.image_model = build('cuda' if torch.cuda.is_available() else 'cpu')
                    pred = self.image_model
                    pred.set_image(rgb[0 if time == 't1' else 1])
                    if box is not None:
                        masks, quality, _ = pred.predict(box=np.asarray(box, float), multimask_output=False)
                    else:
                        masks, quality, _ = pred.predict(point_coords=np.asarray(points, float),
                            point_labels=np.asarray(point_labels), multimask_output=True)
                    i = int(np.argmax(quality))
                    return np.array(masks[i], copy=True), float(quality[i]), dict(backend='sam_prompt', queue_seconds=queued)
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


class PromptWorker:
    """Client of workers/sam_prompt_worker.py: one SAM decoder per job instead of one per process.

    Same interface and outputs as SharedPrompter: (mask, quality, meta). Images that are not LEVIR
    cached pairs are sent as the exact pixel arrays the session holds.
    """
    def __init__(self, url, timeout=600):
        self.url, self.timeout = url.rstrip('/'), timeout

    def for_image(self, split, name, rgb=None):
        def call(time, box=None, points=None, point_labels=None):
            payload = dict(time=time, box=None if box is None else [float(v) for v in box],
                           points=None if points is None else [[float(v) for v in p] for p in points],
                           point_labels=None if point_labels is None else [int(v) for v in point_labels])
            if rgb is not None:
                image = np.ascontiguousarray(rgb[0 if time == 't1' else 1])
                payload.update(mode='image', image=dict(dtype=str(image.dtype), shape=list(image.shape),
                               data=base64.b64encode(image.tobytes()).decode('ascii')))
            else:
                payload.update(mode='cached', split=split, name=name)
            result = post(self.url, payload, self.timeout)
            m = result['mask']
            bits = np.frombuffer(base64.b64decode(m['bits']), np.uint8)
            mask = np.unpackbits(bits, count=int(np.prod(m['shape']))).reshape(m['shape']).astype(bool)
            return mask, float(result['quality']), dict(backend='sam_prompt', worker=True,
                queue_seconds=result.get('queue_seconds'), worker_seconds=result.get('seconds'))
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
CORRESPONDENCE = verifier_prompts()['CORRESPONDENCE']


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

    def __init__(self, client=None, guard=None, model='qwen3.7-plus', max_attempts=12, correspondence=False, session=None):
        self.session = session
        self.client, self.guard, self.model, self.max_attempts = client, guard, model, max_attempts
        self.correspondence = correspondence

    def one(self, images, question, number, level='L1', *, question_variant=None, record=None, seed=0):
        from mllm import parse_json
        if question_variant not in (None, 'xbd-jds-v1'):
            raise ValueError('Unsupported verifier question variant')
        events = []
        def save_event(event):
            events.append({k: v for k, v in event.items() if k != 'request'})
            if record is not None:
                record(event)
        # Region crops are reproducible from the mask, so the archive keeps the reply and not the images.
        planner = Planner(seed=seed, client=self.client, guard=self.guard, max_attempts=self.max_attempts,
            record=save_event,
            parameters=dict(model=self.model, logprobs=True, top_logprobs=5, response_format={'type': 'json_object'},
                            parallel_tool_calls=False, max_tokens=256))
        state_question = self.session is not None and self.session.ask_variant == 't2_state'
        if state_question:
            from .map_input import T2_STATE_SYSTEM, T2_STATE_QUESTION
            if not getattr(self.session, 'old_map', None):
                raise ValueError('t2_state requires map input')
            images = [images[i] for i in (1, 3)] if len(images) == 4 else [images[1]]
        words = ('T2 close-up', 'T2 wider view') if state_question else CROP_WORDS
        content = list(example_blocks()) if level == 'L2' and not state_question else []
        for word, image in zip(words, images):
            content += [dict(type='text', text=word), dict(type='image_url',
                        image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
        settings = session_prompts(self.session) if self.session else verifier_prompts()
        if state_question:
            settings = dict(settings, SYSTEM=T2_STATE_SYSTEM)
        if question_variant == 'xbd-jds-v1':
            from .analysis.xbd_damage import JDS_QUESTION
            if len(images) != 2 or self.session is not None or self.correspondence or level != 'L1':
                raise ValueError('xbd-jds-v1 requires a standalone L1 pre/post pair')
            settings = dict(settings, SYSTEM='You assess building damage from satellite imagery.')
            question = JDS_QUESTION
        content.append(dict(type='text', text=question if question_variant else T2_STATE_QUESTION if state_question else settings['CORRESPONDENCE'] if self.correspondence else (CONTEXT_NOTE if len(images) > 2 else '') + question))
        try:
            with ASK_SLOTS:
                result = planner.complete([dict(role='system', content=settings['SYSTEM']), dict(role='user', content=content)], [], number)
        except Exception as exc:
            exc.partial = dict(attempts=events, tokens=planner.tokens, cost=planner.cost)
            raise
        raw = result['message'].get('content') or ''
        parsed = parse_json(raw)
        parsed = parsed if isinstance(parsed, dict) else {}
        answer = dict(answer=parsed.get('building' if state_question else 'change'), type=parsed.get('type'), reason=parsed.get('reason'), level=level,
                      p_yes=yes_probability(result['raw']['choices'][0]), raw=raw, usage=result['usage'])
        if question_variant == 'xbd-jds-v1':
            from .analysis.xbd_damage import parse_answer
            answer['answer'] = parse_answer(raw)
        if state_question and answer['p_yes'] is None:
            answer['p_yes'] = parsed.get('p_yes')
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
