"""Slurm runtime for routes A/B. All result/cache writes stay under output-dir."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import threading
import urllib.error
import numpy as np
from agent_baseline.planner import Planner, request_cost, usage_tokens
from agent_baseline.tools_t13 import Features
from mllm import parse_json
from .io_retry import read_image, read_text, write_json, retry_io, exists
from .t3_routes import load_dev, evaluate_a, pair_id, group, pooled, pixel_counts, GROUPS, GRID, CLASSES
from .t3_routes_cache import CachedVision, read_npz, write_npz
from .t3_routes_match import ranked_indices, build_messages, filtered_union

# Eight 384px images and fixed short text; reserve far above the ledger estimate.
# Actual provider usage is still authoritative and checked after every response.
# 512 (was 256): the JSON reply with a free-text reason was cut off at 256 tokens (2026-11-12 run, unparsed).
MAX_TOKENS = 512
REQUEST_RESERVE = request_cost('qwen3.7-flash', dict(input=32000, cached=0, output=MAX_TOKENS))


class Stopped(RuntimeError):
    pass


def bind(path, policy, resume):
    if exists(path):
        if not resume:
            raise ValueError('Existing progress; pass --resume or use a fresh output directory')
        if json.loads(read_text(path)) != policy:
            raise ValueError('Resume policy differs; use a fresh output directory')
    else:
        write_json(path, policy)


class SemanticPool:
    def __init__(self, config, folder):
        self.config, self.folder = config, folder
        self.process = self.log = None

    def for_row(self, row):
        def classify(operation, time, classes):
            if operation != 'classify':
                raise ValueError('Semantic worker supports classify only')
            path = self.folder / f"{row['id']}_{time}.npz"
            if not exists(path):
                if self.process is None:
                    retry_io(lambda: self.folder.mkdir(parents=True, exist_ok=True))
                    self.log = retry_io(lambda: (self.folder / 'semantic.stderr.log').open('a'))
                    runtime = self.folder / 'runtime'
                    retry_io(lambda: runtime.mkdir(exist_ok=True))
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(runtime),
                        MPLCONFIGDIR=str(runtime), HF_HOME=str(runtime), TORCH_HOME=str(runtime),
                        PYTHONPATH=str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
                    self.process = subprocess.Popen([self.config['python'], '-B',
                        str(Path(__file__).resolve().parents[2] / 't3_routes_bootstrap.py'),
                        'agent_baseline.analysis.t3_routes_semantic_worker'], env=env,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
                request = dict(config=self.config, cache=str(self.folder), image=row['image_t1' if time == 't1' else 'image_t2'],
                               classes=classes, output=str(path))
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                line = self.process.stdout.readline()
                if not line or not json.loads(line)['ok']:
                    raise RuntimeError('SegEarth worker failed; see semantic.stderr.log')
            return read_npz(path)['data']
        return classify

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait()
            self.log.close()


def route_a(rows, run, out, resume):
    config = json.loads(read_text(run / 't3_diag/report.json'))['settings']['vision_config']['dynamic']
    folder = out / 'route_a'
    policy = dict(version=1, task_ids=[r['id'] for r in rows], config=config,
        classes=list(CLASSES.values()), route='six-class mean exemplar probability; directed dense argmax intersection')
    bind(folder / 'settings.json', policy, resume)
    progress = folder / 'progress.json'
    tasks = json.loads(read_text(progress))['tasks'] if exists(progress) else []
    done = {t['id'] for t in tasks}
    pool = SemanticPool(config, folder / 'semantic')
    try:
        for row in rows:
            if row['id'] in done:
                continue
            tasks.append(evaluate_a(row, pool.for_row(row)))
            write_json(progress, dict(policy=policy, tasks=tasks))
            print('route-a', row['id'], flush=True)
    finally:
        pool.close()


def prepare_b(rows, run, out, resume):
    from .probe_t3 import task_scores
    folder = out / 'route_b'
    cache = CachedVision(run / 't3_diag/cache/vision')
    config = json.loads(read_text(run / 't3_diag/report.json'))['settings']['vision_config']['dynamic']
    policy = dict(version=1, task_ids=[r['id'] for r in rows], grid=list(GRID),
        cache='t3_diag/cache/vision', config=config, ranking='descending cosine; proposal index ties; no threshold')
    bind(folder / 'prepared_settings.json', policy, resume)
    progress_path = folder / 'prepared.json'
    tasks = json.loads(read_text(progress_path))['tasks'] if exists(progress_path) else []
    done = {t['id'] for t in tasks}
    for row in rows:
        if row['id'] in done:
            continue
        backend = cache.for_row(row)
        exemplar = read_image(row['exemplar_mask']) > 0
        masks = np.concatenate([backend('segment_everything', time=t) for t in ('t1', 't2')])
        features = [Features(backend('extract_features', time=t), exemplar.shape, t) for t in ('t1', 't2')]
        task = dict(instances=[np.flatnonzero(exemplar).tolist()], exemplar_index=0)
        similarities = task_scores(masks, *features, task) if len(masks) else np.zeros(0)
        indices = ranked_indices(similarities, max(GRID))
        # Store only top32 masks. Proposal indices remain stable across k.
        write_npz(folder / 'candidates' / (row['id'] + '.npz'),
            masks=np.packbits(masks[indices].reshape(len(indices), exemplar.size), axis=1),
            shape=exemplar.shape, indices=np.asarray(indices, dtype=np.int64), scores=similarities[indices])
        tasks.append(dict(id=row['id'], proposals=len(masks), indices=indices,
                          similarities=similarities[indices].tolist()))
        write_json(progress_path, dict(policy=policy, tasks=tasks))
        print('prepare-b', row['id'], flush=True)


def candidates(folder, task_id):
    saved = read_npz(folder / 'candidates' / (task_id + '.npz'))
    shape = tuple(map(int, saved['shape']))
    masks = np.unpackbits(saved['masks'], axis=1, count=int(np.prod(shape))).reshape(-1, *shape).astype(bool)
    return saved['indices'].tolist(), masks


class Budget:
    """One process, bounded concurrent requests. Reserve before dispatch; persist bills.

    Completed answers and billed failures survive restarts. STOP must be reviewed
    and removed explicitly before resuming, even if the user raises the budget.
    """
    def __init__(self, folder, max_cost):
        if max_cost <= 0:
            raise ValueError('max-cost-cny must be positive')
        self.folder, self.maximum = folder, max_cost
        self.lock = threading.RLock()
        self.reserved = 0.
        files = retry_io(lambda: list((folder / 'answers').glob('*.json')) + list((folder / 'failures').glob('*.json')))
        self.spent = sum(json.loads(read_text(p)).get('cost_cny', 0.) for p in files)

    def check(self):
        if exists(self.folder / 'STOP.json'):
            raise Stopped('STOP exists; review it before resuming')

    def stop(self, reason):
        # Call while locked when modifying billing state. No provider body/URL is archived.
        path = self.folder / 'STOP.json'
        if exists(path):
            reason = json.loads(read_text(path))['reason']
        write_json(path, dict(reason=reason, spent_cny=self.spent,
                   in_flight_reserved_cny=self.reserved, max_cost_cny=self.maximum))

    def reserve(self):
        with self.lock:
            self.check()
            if self.spent + self.reserved + REQUEST_RESERVE > self.maximum + 1e-12:
                self.stop('budget: insufficient room for another request reservation')
                raise Stopped('Budget reached')
            self.reserved += REQUEST_RESERVE

    def finish(self, path, record):
        with self.lock:
            write_json(path, record)
            self.spent += record['cost_cny']
            self.reserved = max(0., self.reserved - REQUEST_RESERVE)
            if self.spent > self.maximum or record['cost_cny'] > REQUEST_RESERVE:
                self.stop('budget: actual provider usage exceeds reservation or total cap')


def provider_is_fatal(code, body):
    return code in (402, 403, 429) or any(word in body.lower() for word in
        ('arrearage', 'insufficient', 'quota', 'good standing', '欠费', '余额不足'))


def ask_one(messages, budget, client=None):
    """Existing Planner transport and price table; no API retries for this probe."""
    transport = client or Planner()._http
    def guarded(payload):
        budget.check()
        try:
            response = transport(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read(2000).decode('utf-8', 'replace')
            with budget.lock:
                budget.stop(f'provider HTTP {exc.code}' if provider_is_fatal(exc.code, body) else f'request failed HTTP {exc.code}')
            raise Stopped(f'provider HTTP {exc.code}') from None
        except Exception:
            with budget.lock:
                budget.stop('transport failure; request status requires review')
            raise Stopped('Transport failed') from None
        if response.get('error'):
            with budget.lock:
                budget.stop('provider returned an error response')
            raise Stopped('Provider returned error')
        return response
    planner = Planner(client=guarded, max_attempts=1, parameters=dict(model='qwen3.7-flash',
        logprobs=True, top_logprobs=5, max_tokens=MAX_TOKENS, response_format={'type':'json_object'}, parallel_tool_calls=False))
    try:
        result = planner.complete(messages, [], 1)
        content = result['message'].get('content') or ''
        tokens = usage_tokens(result['usage'])
        try:
            parsed = parse_json(content)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict) or parsed.get('change') not in ('yes', 'no'):
            # The provider answered and billed, but not in the format (e.g. cut off at max_tokens).
            # Same rule as the harness ask: unparsed is not no, so the candidate is kept. Raw text kept for review.
            return dict(answer='unparsed', raw=content[:4000], tokens=tokens,
                        cost_cny=request_cost('qwen3.7-flash', tokens))
        return dict(answer=parsed['change'], description={k: parsed.get(k) for k in
            ('example_t1', 'example_t2', 'candidate_t1', 'candidate_t2', 'reason')},
            tokens=tokens, cost_cny=request_cost('qwen3.7-flash', tokens))
    except Exception as exc:
        # Include usage even if the response was malformed before Planner charged it.
        exc.billed_cost = request_cost('qwen3.7-flash', planner.tokens)
        exc.billed_tokens = planner.tokens
        raise


def verify_jobs(jobs, folder, concurrency, max_cost, resume, policy, request, client=None):
    """jobs=(uid, metadata); request(metadata) builds messages. Fake-client testable."""
    if not 1 <= concurrency <= 16:
        raise ValueError('concurrency must be in 1..16')
    bind(folder / 'verify_settings.json', policy, resume)
    budget = Budget(folder, max_cost)
    budget.check()
    pending = iter((uid, meta) for uid, meta in jobs if not exists(folder / 'answers' / (uid + '.json')))
    failure = None
    def work(uid, meta):
        # Build images before reserving, so a local I/O failure never consumes budget.
        messages = request(meta)
        budget.reserve()
        try:
            record = ask_one(messages, budget, client=client)
            budget.finish(folder / 'answers' / (uid + '.json'), dict(uid=uid, **record))
        except Exception as exc:
            # Unique attempt file: repeated resumes must not overwrite a prior bill.
            with budget.lock:
                prior = retry_io(lambda: list((folder / 'failures').glob(uid + '_*.json')))
                budget.finish(folder / 'failures' / f'{uid}_{len(prior):03d}.json',
                    dict(uid=uid, cost_cny=getattr(exc, 'billed_cost', 0.),
                         tokens=getattr(exc, 'billed_tokens', {}), error=type(exc).__name__))
                budget.stop('request incomplete; review failed request before resuming')
            raise
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = set()
        def fill():
            while len(active) < concurrency and not exists(budget.folder / 'STOP.json'):
                item = next(pending, None)
                if item is None:
                    break
                active.add(pool.submit(work, *item))
        fill()
        while active:
            completed, active = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                try:
                    future.result()
                except Exception as exc:
                    failure = failure or exc
                    with budget.lock:
                        budget.stop('batch stopped after request failure')
            if failure is None:
                fill()
    if failure is not None:
        raise Stopped('Verification stopped; see STOP.json and failures') from failure
    budget.check()
    return budget.spent


def verify(rows, out, concurrency, max_cost, resume, ask_variant):
    if ask_variant != 't3_transition':
        raise ValueError('Route B requires explicit --ask-variant t3_transition')
    folder = out / 'route_b'
    prepared = json.loads(read_text(folder / 'prepared.json'))
    if {t['id'] for t in prepared['tasks']} != {r['id'] for r in rows}:
        raise ValueError('prepare-b is incomplete')
    by_id = {r['id']: r for r in rows}
    jobs = [(t['id'] + f'__{i}', dict(task_id=t['id'], index=i)) for t in prepared['tasks'] for i in t['indices']]
    policy = dict(preparation=prepared['policy'], ask_variant=ask_variant, model='qwen3.7-flash',
                  request_version=2, views='example four + candidate four', max_tokens=MAX_TOKENS)
    # Keep only a bounded number of image/mask arrays in memory, one job per worker.
    def request(meta):
        row = by_id[meta['task_id']]
        indices, masks = candidates(folder, row['id'])
        rgb = [read_image(row[k], rgb=True) for k in ('image_t1', 'image_t2')]
        exemplar = read_image(row['exemplar_mask']) > 0
        return build_messages(rgb, exemplar, masks[indices.index(meta['index'])], ask_variant)
    spent = verify_jobs(jobs, folder, concurrency, max_cost, resume, policy, request)
    write_json(folder / 'verification.json', dict(complete=True, requests=len(jobs), cost_cny=spent))


def score_b(rows, out):
    folder = out / 'route_b'
    scores = {str(k): [] for k in GRID}
    for row in rows:
        indices, masks = candidates(folder, row['id'])
        answers = {j: json.loads(read_text(folder / 'answers' / f"{row['id']}__{i}.json")) for j, i in enumerate(indices)}
        exemplar = read_image(row['exemplar_mask']) > 0
        truth = read_image(row['gt_mask']) > 0
        for k in GRID:
            pred = filtered_union(masks, range(min(k, len(indices))), answers, exemplar)
            scores[str(k)].append(dict(id=row['id'], counts=pixel_counts(pred, truth, exemplar)))
    summary = {}
    for k, tasks in scores.items():
        summary[k] = {}
        by_id = {t['id']: t for t in tasks}
        for g in ('全部', *GROUPS):
            selected = [by_id[r['id']] for r in rows if g == '全部' or group(r) == g]
            summary[k][g] = dict(tasks=len(selected), score=pooled(selected))
    selected = max(GRID, key=lambda k: (round(summary[str(k)]['全部']['score']['f1_c'], 10), -k))
    write_json(folder / 'scores.json', dict(task_ids=[r['id'] for r in rows], scores=summary, selected_k=selected, per_task=scores))


def run_command(a):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit runtime commands through sbatch; metadata report and fake tests are CPU-only')
    rows = load_dev(a.run_root)
    if a.command == 'route-a':
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('route-a requires an allocated GPU')
        route_a(rows, a.run_root, a.output_dir, a.resume)
    elif a.command == 'prepare-b':
        prepare_b(rows, a.run_root, a.output_dir, a.resume)
    elif a.command == 'verify':
        verify(rows, a.output_dir, a.concurrency, a.max_cost_cny, a.resume, a.ask_variant)
    else:
        score_b(rows, a.output_dir)
