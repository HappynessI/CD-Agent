"""Single-date visual exemplar interpretation; no task metadata or labels."""
import base64
from io import BytesIO
import json
import os
import numpy as np
from PIL import Image
from .planner import Planner, request_cost
from .backends import encode_image
from .view import ask_images
from mllm import parse_json
from .analysis.io_retry import read_text, write_json, exists
from .analysis.t3_routes_match import bbox

CLASSES = {1:'low vegetation',2:'non-vegetated ground',3:'tree',4:'water',5:'building',6:'playground'}
MODEL = 'qwen3.7-flash'
VOCAB = tuple(CLASSES.values()) + ('unknown',)
PARSE_SYSTEM = 'Describe the land cover inside the red contour in this single-date remote sensing image. The two views show the same region. Use visual evidence only.'
PARSE_QUESTION = ('Choose one class from ' + ', '.join(VOCAB) + '. '
    'Use unknown when the region cannot be identified reliably. Return JSON '
    '{"class":"one allowed class", "evidence":"visible texture, shape, colour and context supporting the choice"}.')


def canonical(value):
    value = ' '.join(str(value or '').lower().split())
    return value if value in VOCAB else 'unknown'


def single_date_views(image, exemplar):
    views = ask_images([image, image], exemplar, bbox(exemplar), True)
    return [('close', views[0]), ('wide', views[2])]


def request(folder, system, question, views, seed=1016):
    """Persistent response and exact rendered inputs; bounded transport retries."""
    answer = folder / 'answer.json'
    if exists(answer):
        previous = json.loads(read_text(folder / 'input.json'))
        if any(previous.get(key) != value for key,value in dict(system=system, question=question,
                model=MODEL, seed=seed, max_tokens=768, temperature=0).items()):
            raise ValueError('Cached semantic request parameters differ')
        return json.loads(read_text(answer))
    folder.mkdir(parents=True, exist_ok=True)
    content, paths = [], []
    for i, (label, image) in enumerate(views):
        path = folder / f'{i:02d}.png'
        encoded = encode_image(image)
        Image.open(BytesIO(base64.b64decode(encoded))).save(path)
        paths.append(dict(label=label, path=str(path)))
        content.extend([dict(type='text', text=label), dict(type='image_url',
            image_url=dict(url='data:image/jpeg;base64,' + encoded))])
    content.append(dict(type='text', text=question))
    write_json(folder / 'input.json', dict(system=system, question=question, images=paths,
        model=MODEL, seed=seed, max_tokens=768, temperature=0))
    # Keep all previous attempts, including failures across resumed jobs.
    offset = len(list(folder.glob('attempt_*.json')))
    def record(event):
        raw = event.get('raw') or {}
        message = ((raw.get('choices') or [{}])[0].get('message') or {})
        error_body = event.get('error_body')
        if error_body and os.environ.get('DASHSCOPE_API_KEY'):
            error_body = error_body.replace(os.environ['DASHSCOPE_API_KEY'], '[credential removed]')
        write_json(folder / f"attempt_{offset + event['total_attempt_id']:03d}.json",
            dict(attempt=event['total_attempt_id'], raw=message.get('content'),
                usage=event.get('usage'), http_status=event.get('http_status'),
                requested_model=event['request']['model'], error_body=error_body,
                error_type=event.get('error_type'), retryable=event.get('retryable'),
                seconds=event.get('seconds')))
        if (event.get('http_status') in (401,402,403,404) or event.get('quota_error')) and os.environ.get('AGENT_STOP_FILE'):
            write_json(os.environ['AGENT_STOP_FILE'], dict(reason='semantic API rejected request',
                http_status=event.get('http_status'), evidence=str(folder)))
    planner = Planner(seed=seed, record=record, max_attempts=3,
        parameters=dict(model=MODEL, max_tokens=768, temperature=0,
                        response_format={'type':'json_object'}))
    try:
        result = planner.complete([dict(role='system', content=system),
            dict(role='user', content=content)], [], 1)
    except Exception as exc:
        write_json(folder / f'failure_{offset:03d}.json', dict(error_type=type(exc).__name__,
            tokens=planner.tokens, cost_cny=request_cost(MODEL, planner.tokens)))
        raise
    raw = result['message'].get('content') or ''
    try:
        parsed = parse_json(raw)
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    tokens = dict(planner.tokens)
    record = dict(raw=raw, parsed=parsed, tokens=tokens, cost_cny=request_cost(MODEL, tokens))
    write_json(answer, record)
    return record


def interpret(images, exemplar, folder, seed):
    records = [request(folder / time, PARSE_SYSTEM, PARSE_QUESTION,
        single_date_views(image, exemplar), seed) for time, image in zip(('t1','t2'), images)]
    names = [canonical(r['parsed'].get('class')) for r in records]
    ids = [next((i for i, name in CLASSES.items() if name == n), None) for n in names]
    return dict(classes=names, pair=ids, evidence=[r['parsed'].get('evidence') for r in records],
        cost_cny=sum(r['cost_cny'] for r in records))

