"""Bounded urllib OpenAI-compatible transport; credentials never enter records."""
import base64
import copy
import http.client
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

PRICE = dict(input=2.0, cached=0.4, output=8.0)
DEFAULTS = dict(model='qwen3.7-plus', temperature=0, enable_thinking=False,
                parallel_tool_calls=True, max_tokens=1500)


def clean_messages(messages, encode=False):
    """Local paths are archive references only; the API receives image bytes."""
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message.get('content'), list):
            for block in message['content']:
                block.pop('_view', None)
                if encode and block.get('type') == 'image_url':
                    url = block['image_url']['url']
                    if not url.startswith('data:'):
                        block['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(Path(url).read_bytes()).decode('ascii')
    return messages


def usage_tokens(usage):
    usage = usage or {}
    total = int(usage.get('prompt_tokens', 0))
    cached = int((usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0))
    return dict(input=total, cached=min(total, cached), output=int(usage.get('completion_tokens', 0)))


def token_cost(tokens):
    return ((tokens['input'] - tokens['cached']) * PRICE['input'] + tokens['cached'] * PRICE['cached']
            + tokens['output'] * PRICE['output']) / 1_000_000


class PlannerError(RuntimeError):
    pass


class Planner:
    def __init__(self, seed=0, client=None, record=None, sleep=time.sleep, jitter=random.random,
                 max_attempts=180, parameters=None):
        self.parameters = dict(DEFAULTS, seed=seed, **(parameters or {}))
        self.client = client or self._http
        self.record = record or (lambda event: None)
        self.sleep, self.jitter = sleep, jitter
        self.attempts = 0
        self.max_attempts = max_attempts
        self.tokens = dict(input=0, cached=0, output=0)

    def _http(self, payload):
        key = os.environ.get('DASHSCOPE_API_KEY')
        if not key:
            raise PlannerError('Planner credentials are not configured.')
        base = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')
        request = urllib.request.Request(base + '/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def complete(self, messages, tools, round_id, tool_choice=None, seed_bump=0):
        params = dict(self.parameters)
        if tool_choice is not None:
            params['tool_choice'] = tool_choice
        if seed_bump and 'seed' in params:
            params['seed'] += seed_bump
        archived = dict(params, messages=clean_messages(messages), tools=tools)
        payload = dict(params, messages=clean_messages(messages, encode=True), tools=tools)
        if not tools:
            for body in (archived, payload):
                body.pop('tools')
                body.pop('parallel_tool_calls', None)
        for attempt in range(1, 7):
            if self.attempts >= self.max_attempts:
                raise PlannerError('Total planner attempt budget exhausted.')
            self.attempts += 1
            start = time.monotonic()
            event = dict(round_id=round_id, attempt_id=attempt, total_attempt_id=self.attempts,
                         requested_at=time.time(), request=archived)
            error = None
            try:
                raw = self.client(payload)
                usage = raw.get('usage') or {}
                event.update(raw=raw, request_id=raw.get('id', raw.get('request_id')), usage=usage)
                for key, n in usage_tokens(usage).items():
                    self.tokens[key] += n
                message = raw['choices'][0]['message']
                if message.get('role', 'assistant') != 'assistant':
                    raise PlannerError('Malformed planner response.')
            except Exception as exc:
                error = exc
                if isinstance(exc, urllib.error.HTTPError):
                    retry = exc.code == 429 or 500 <= exc.code < 600
                    # Server error text explains rejections (rate limits, content inspection); it holds no credentials.
                    event['error_body'] = exc.read(2000).decode('utf-8', 'replace')
                else:
                    retry = isinstance(exc, (OSError, http.client.HTTPException, json.JSONDecodeError))
                event.update(error_type=type(exc).__name__, http_status=getattr(exc, 'code', None), retryable=retry)
            event['seconds'] = time.monotonic() - start
            # Record outside the network try: logging failures must not cause a duplicate request.
            self.record(event)
            if error is None:
                return dict(message=message, raw=raw, usage=usage, request_id=event['request_id'],
                            seconds=event['seconds'], attempts=attempt)
            if not event['retryable'] or attempt == 6:
                raise PlannerError('Planner request failed; see recorded attempt status.') from error
            delay = min(32, 2 ** (attempt - 1)) + self.jitter()
            if isinstance(error, urllib.error.HTTPError):
                try:
                    delay = max(delay, min(60, float(error.headers.get('Retry-After', 0))))
                except (TypeError, ValueError):
                    pass
            self.sleep(delay)
        raise PlannerError('Planner attempts exhausted.')
