"""Bounded urllib OpenAI-compatible transport; credentials never enter records."""
import base64
import copy
import fcntl
import http.client
import json
import os
import random
import socket
import threading
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


# Third-party listing of the Beijing tiers by input length of one request; the console bill is authoritative.
# The cached price of qwen3.7-flash is not known, so cached tokens are charged in full and the estimate errs high.
FLASH_TIERS = ((32_000, .2, .8), (256_000, .6, 2.4), (float('inf'), 1.2, 4.8))


def request_cost(model, tokens):
    """Estimated cost of one request; models without their own table use the qwen3.7-plus price."""
    if model == 'qwen3.7-flash':
        _, price_in, price_out = next(tier for tier in FLASH_TIERS if tokens['input'] <= tier[0])
        return (tokens['input'] * price_in + tokens['output'] * price_out) / 1_000_000
    return token_cost(tokens)


def token_cost(tokens):
    return ((tokens['input'] - tokens['cached']) * PRICE['input'] + tokens['cached'] * PRICE['cached']
            + tokens['output'] * PRICE['output']) / 1_000_000


class PlannerError(RuntimeError):
    pass


class RunStopped(RuntimeError):
    """The provider quota or the batch budget is gone: every session stops and leaves no final.json, so the run can resume."""


QUOTA_WORDS = ('quota', 'arrearage', 'good standing', 'access denied', 'access to model denied')


def is_quota_error(code, body):
    return code == 403 or (code in (400, 402, 429) and any(word in body.lower() for word in QUOTA_WORDS))


class RunGuard:
    """Stop switch shared by all parallel jobs through one file, plus an optional budget ledger with its own batch stop.

    Each process writes only its own ledger file, replaced atomically, and the batch total is the sum of the files.
    A single shared file is unsafe here: jobs on different nodes append through a network filesystem with stale
    file sizes, which leaves zero-filled holes. A stale peer value only delays the stop by a moment. If this
    process cannot keep its ledger, the batch stops resumably, so the budget cap never silently lapses.
    """
    CHECK_SECONDS, MAX_WRITE_FAILURES = 2., 5

    def __init__(self, stop_file, budget_file=None, max_cost=None):
        self.stop_file, self.max_cost = Path(stop_file), max_cost
        self.directory = Path(budget_file).with_suffix('') if budget_file else None
        self.lock = threading.Lock()
        self.spent, self.peers, self.checked, self.write_failures = 0., {}, 0., 0
        self.own = None
        if self.directory:
            self.own = self.directory / f'{socket.gethostname()}-{os.getpid()}.txt'
            try:
                self.spent = float(self.own.read_text())
            except (OSError, ValueError):
                pass

    def active_stop(self):
        """The file that stops this guard: quota errors stop everything, a spent budget stops only its own batch."""
        if self.stop_file.exists():
            return self.stop_file
        batch = self.directory and self.directory / 'STOP.json'
        return batch if batch and batch.exists() else None

    def stopped(self):
        return self.active_stop() is not None

    def check(self):
        path = self.active_stop()
        if path:
            raise RunStopped(path.read_text())

    def trip(self, reason, batch_only=False):
        target = self.directory / 'STOP.json' if batch_only else self.stop_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(reason=reason, at=time.time()), ensure_ascii=False))
        raise RunStopped(reason)

    def total(self):
        """Own spending plus each peer's last readable value."""
        for path in self.directory.glob('*.txt'):
            if path != self.own:
                try:
                    self.peers[path.name] = float(path.read_text())
                except (OSError, ValueError):
                    pass
        return self.spent + sum(self.peers.values())

    def charge(self, cost):
        if self.directory is None:
            return
        with self.lock:
            self.spent += cost
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.own.with_suffix('.tmp')
                temporary.write_text(repr(self.spent))
                temporary.replace(self.own)
                self.write_failures = 0
            except OSError:
                self.write_failures += 1
            if self.write_failures >= self.MAX_WRITE_FAILURES:
                self.trip('budget ledger cannot be written')
            if self.max_cost is None or time.monotonic() - self.checked < self.CHECK_SECONDS and self.spent <= self.max_cost:
                return
            self.checked = time.monotonic()
            total = self.total()
            if total > self.max_cost:
                self.trip(f'budget: {total:.2f} > {self.max_cost:.2f} CNY', batch_only=True)


class Planner:
    def __init__(self, seed=0, client=None, record=None, sleep=time.sleep, jitter=random.random,
                 max_attempts=180, parameters=None, guard=None):
        self.guard = guard
        self.parameters = dict(DEFAULTS, seed=seed, **(parameters or {}))
        self.client = client or self._http
        self.record = record or (lambda event: None)
        self.sleep, self.jitter = sleep, jitter
        self.attempts = 0
        self.max_attempts = max_attempts
        self.tokens = dict(input=0, cached=0, output=0)
        self.cost = 0.

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
            if self.guard:
                self.guard.check()
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
                    # Server error text explains rejections (rate limits, content inspection); it holds no credentials.
                    event['error_body'] = exc.read(2000).decode('utf-8', 'replace')
                    quota = is_quota_error(exc.code, event['error_body'])
                    retry = not quota and (exc.code == 429 or 500 <= exc.code < 600)
                    event['quota_error'] = quota
                else:
                    retry = isinstance(exc, (OSError, http.client.HTTPException, json.JSONDecodeError))
                event.update(error_type=type(exc).__name__, http_status=getattr(exc, 'code', None), retryable=retry)
            event['seconds'] = time.monotonic() - start
            # Record outside the network try: logging failures must not cause a duplicate request.
            self.record(event)
            if error is None:
                cost = request_cost(self.parameters.get('model'), usage_tokens(usage))
                self.cost += cost
                if self.guard:
                    self.guard.charge(cost)
            if error is not None and event.get('quota_error') and self.guard:
                self.guard.trip(f'quota: HTTP {event.get("http_status")}')
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
