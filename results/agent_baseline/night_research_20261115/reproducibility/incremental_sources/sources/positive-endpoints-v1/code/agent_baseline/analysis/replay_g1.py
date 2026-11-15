"""Offline request regression against a G1 trace; never creates an HTTP/model client.

Planner requests were archived with image paths: restore the exact JPEG bytes as base64.
G1 did NOT archive verifier request bodies. Those are reconstructed independently using
code_g, archived masks, source RGB and the recorded options. Report this weaker provenance
explicitly: a reconstructed verifier request is not an original wire capture.
"""
import argparse
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.error
from unittest.mock import patch
import numpy as np
from PIL import Image
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline

from common import load_mask_png, read_jsonl
from agent_baseline import instances
from agent_baseline.backends import SegmentBackend, ApiAsk
from agent_baseline.loop_json import RUN, run_session
from agent_baseline.planner import Planner, PlannerError
from agent_baseline.recorder import Recorder
from agent_baseline.state import Mask, ToolError
from agent_baseline.tools import Toolbox, Result


class ReplayMismatch(BaseException):
    """Escape the online loop's recoverable-error handlers on the first difference."""


def first_difference(expected, actual, path='$'):
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        if list(expected) != list(actual):
            return path + '.keys', list(expected), list(actual)
        for key in expected:
            diff = first_difference(expected[key], actual[key], path + '.' + str(key))
            if diff:
                return diff
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return path + '.length', len(expected), len(actual)
        for i, (a, b) in enumerate(zip(expected, actual)):
            diff = first_difference(a, b, f'{path}[{i}]')
            if diff:
                return diff
    elif expected != actual:
        return path, expected, actual
    return None


def compare(expected, actual, label):
    diff = first_difference(expected, actual)
    if diff:
        path, a, b = diff
        if isinstance(a, str) and isinstance(b, str):
            offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            # Base64 is compared in full, but show only the bytes around the first mismatch.
            detail = dict(offset=offset, expected_length=len(a), actual_length=len(b),
                          expected=a[max(0, offset-80):offset+160], actual=b[max(0, offset-80):offset+160])
        else:
            detail = dict(expected=a, actual=b)
        raise ReplayMismatch(json.dumps(dict(request=label, location=path, **detail), ensure_ascii=False))
    # Compare serialization too; neither key order nor whitespace inside string fields is normalized.
    if json.dumps(expected).encode() != json.dumps(actual).encode():
        raise ReplayMismatch(f'{label}: request serialization differs')


def reference_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_reference(root):
    """Load the frozen builders under a private package; do not run their entry points."""
    root = Path(root)
    if not (root / 'agent_baseline' / 'backends.py').is_file():
        raise ValueError('G1 reference code directory is missing; use --reference-code')
    reference_module('_g1_replay', root / 'agent_baseline' / '__init__.py', True)
    reference_module('_g1_replay.state', root / 'agent_baseline' / 'state.py')
    reference_module('_g1_replay.instances', root / 'agent_baseline' / 'instances.py')
    reference_module('_g1_replay.planner', root / 'agent_baseline' / 'planner.py')
    prompts = reference_module('_g1_prompts', root / 'ovcd_v0' / 'prompts.py')
    render = reference_module('_g1_render', root / 'ovcd_v0' / 'render.py')
    view = reference_module('_g1_replay.view', root / 'agent_baseline' / 'view.py')
    view.crop_box, view.render_pair = render.crop_box, render.render_pair
    backend = reference_module('_g1_replay.backends', root / 'agent_baseline' / 'backends.py')
    backend.SYSTEM, backend.VERIFY, backend.PSEUDO_TYPES = prompts.SYSTEM, prompts.VERIFY, prompts.PSEUDO_TYPES
    return backend, view


class Trace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.info = json.loads((self.directory / 'input.json').read_text())
        events = read_jsonl(self.directory / 'messages.jsonl')
        self.planner_events = [e for e in events if e.get('backend') != 'api_ask']
        self.ask_events = [e for e in events if e.get('backend') == 'api_ask']
        # A harness report outside any tool call adds a note row with the same step number; it holds no calls.
        self.steps = {r['step']: r for r in read_jsonl(self.directory / 'steps.jsonl') if not r.get('note')}
        self.planner_count = self.ask_count = 0
        self.cache_calls = self.recorded_calls = 0

    def asset(self, path):
        original = Path(path)
        # Supports a moved session directory; the image filename is unique in a session.
        local = self.directory / 'images' / original.name
        return local if local.exists() else original

    def request(self, body):
        body = copy.deepcopy(body)
        for message in body['messages']:
            if isinstance(message.get('content'), list):
                for block in message['content']:
                    block.pop('_view', None)
                    if block.get('type') == 'image_url':
                        url = block['image_url']['url']
                        if not url.startswith('data:'):
                            block['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(self.asset(url).read_bytes()).decode('ascii')
        return body

    def mask(self, key):
        return load_mask_png(self.directory / 'masks' / ('mask_' + key.split('#')[1] + '.png'))

    def inst(self, key):
        with np.load(self.directory / 'insts' / ('inst_' + key.split('#')[1] + '.npz')) as z:
            h, w = map(int, z['shape'])
            dense = np.unpackbits(z['masks'], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
            rows = instances.from_dense(dense, z['score'])
            for row, id in zip(rows, z['ids']):
                row['id'] = int(id)
            return rows

    def reply(self, event):
        if 'raw' in event:
            return event['raw']
        if event.get('http_status'):
            from io import BytesIO
            raise urllib.error.HTTPError('offline', event['http_status'], 'recorded error', {},
                                         BytesIO(event.get('error_body', '').encode()))
        if event.get('retryable'):
            raise OSError('recorded transport failure')
        raise PlannerError('recorded planner failure')


class ReplayAsk:
    context = True

    def __init__(self, trace, box, rgb, frozen, view):
        self.trace, self.box, self.rgb, self.frozen, self.view = trace, box, rgb, frozen, view
        self.session = box.session
        self.options = trace.info['backends']
        self.compared = 0

    def __call__(self, pairs, question):
        # Expanded components are kept in the exact order used by the real Toolbox.verify.
        expected_pairs = []
        for key, item, comp in self.box.expanded:
            archived = self.trace.mask(key)
            # PNG assets do not retain component IDs after remove_region. Match by bbox.
            from common import components
            labels, comps = components(archived)
            match = next((c for c in comps if list(c['bbox']) == list(comp['bbox'])), None)
            if match is None:
                raise ReplayMismatch(f'verifier: no archived component for {key} bbox={list(comp["bbox"])}')
            expected_pairs.append(self.view.ask_images(self.rgb, labels == match['id'], match['bbox'], True))
        compare(len(expected_pairs), len(pairs), 'verifier region count')
        results, attempts = [], []
        total = dict(input=0, cached=0, output=0)
        cost = 0.
        failure = None
        for number, (old_images, images) in enumerate(zip(expected_pairs, pairs), 1):
            # The frozen one() builds the original payload independently. Its client only captures it.
            captured = []
            dummy = dict(id='offline-reference', choices=[dict(message=dict(role='assistant', content='{}'))], usage={})
            reference = self.frozen.ApiAsk(client=lambda payload: captured.append(copy.deepcopy(payload)) or dummy,
                model=self.options['ask_model'], correspondence=self.options.get('ask_correspondence', False))
            reference.one(old_images, self.frozen.VERIFY, number)
            expected_request = captured[0]
            def client(payload):
                if self.trace.ask_count >= len(self.trace.ask_events):
                    raise ReplayMismatch('Unexpected verifier request after end of archive')
                event = self.trace.ask_events[self.trace.ask_count]
                compare(number, event['round_id'], 'verifier component order')
                expected = self.trace.request(event['request']) if 'request' in event else expected_request
                compare(expected, payload, f'verifier[{self.trace.ask_count}] (reconstructed)' if 'request' not in event else f'verifier[{self.trace.ask_count}]')
                self.trace.ask_count += 1
                self.compared += 1
                return self.trace.reply(event)
            live = ApiAsk(client=client, model=self.options['ask_model'],
                correspondence=self.options.get('ask_correspondence', False), session=self.session)
            try:
                answer, events, tokens, spent = live.one(images, question, number)
                results.append(answer)
            except ReplayMismatch:
                raise
            except Exception as exc:
                failure = failure or exc
                part = getattr(exc, 'partial', dict(attempts=[], tokens=dict(input=0, cached=0, output=0), cost=0.))
                events, tokens, spent = part['attempts'], part['tokens'], part['cost']
            attempts.extend(events)
            cost += spent
            for key, value in tokens.items():
                total[key] += value
        meta = dict(attempts=attempts, tokens=total, cost_cny=cost)
        if failure:
            failure.backend_meta = meta
            raise failure
        return results, meta


class ReplayBox(Toolbox):
    def __init__(self, *args, trace, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        self.active = None
        self.expanded = []
        self.round_id = 0
        self.used = set()

    def execute(self, name, args, allowed=None):
        rows = self.trace.steps.get(self.round_id, {}).get('calls', [])
        self.active = next((row for i, row in enumerate(rows) if i not in self.used and row['tool'] == name and row['args'] == args), None)
        if self.active:
            self.used.add(rows.index(self.active))
        # Reproduce recorded external backend failures without reaching a worker.
        if (self.active and not self.active['ok'] and name in ('segment_text', 'segment_prompt', 'refine_region')
                and self.active.get('error_code') in ('backend_error', 'backend_unavailable')):
            self.trace.recorded_calls += 1
            return Result(json.loads(self.active['result_text']))
        if self.active and self.active['ok'] and name == 'refine_region':
            # G1 saved only the final refined mask, not the raw SAM decoder mask.
            # Reapply that recorded backend outcome; geometry edits without SAM still execute normally.
            data = self.trace.mask(json.loads(self.active['result_text'])['mask'])
            self._mask(data, (args['mask'],), args.get('set_current', True))
            self.trace.recorded_calls += 1
            return Result(json.loads(self.active['result_text']), internal=dict(handles_created=self.active['handles_created']))
        return super().execute(name, args, allowed)

    def verify(self, expanded, source):
        self.expanded = expanded
        return super().verify(expanded, source)

    def recorded_segment(self, time, text, tile, score_thr):
        row = self.active
        backend = SegmentBackend(self.split, self.name, pixel_thr=self.trace.info['backends']['pixel_thr'])
        try:
            result = backend(time, text, tile, score_thr)
            self.trace.cache_calls += 1
            return result
        except (ToolError, FileNotFoundError):
            if row is None:
                raise ReplayMismatch('Missing recorded SAM3 result')
            self.trace.recorded_calls += 1
            key = json.loads(row['result_text'])['inst']
            return self.trace.inst(key), dict(backend='recorded_sam3', cache_hit=False)

    def recorded_prompt(self, time, box=None, points=None, point_labels=None):
        row = self.active
        if row is None:
            raise ReplayMismatch('Missing recorded SAM prompt result')
        result = json.loads(row['result_text'])
        self.trace.recorded_calls += 1
        return self.trace.mask(result['mask']), result['sam_quality'], dict(backend='recorded_sam_prompt')


class ReplayRecorder(Recorder):
    def step(self, session, number, planner, calls):
        expected = self.trace.steps.get(number, {}).get('calls', [])
        compare(len(expected), len(calls), f'step[{number}].calls')
        for n, (old, new) in enumerate(zip(expected, calls)):
            compare(old['result_text'], new['result_text'], f'step[{number}].calls[{n}].result_text')
            compare(old.get('handles_created', []), new.get('handles_created', []), f'step[{number}].calls[{n}].handles')
            for key in new['handles_created']:
                item = session.get(key)
                if isinstance(item, Mask) and not np.array_equal(self.trace.mask(key), item.data):
                    raise ReplayMismatch(f'step[{number}] mask differs: {key}')
        super().step(session, number, planner, calls)


def replay(directory, output, reference_code=None):
    trace = Trace(directory)
    info = trace.info
    backend, view = frozen_reference(reference_code or RUN / 'code_g')
    rgb = [np.asarray(Image.open(path).convert('RGB')) for path in info['source_images']]
    init = trace.mask(info['initial_handle']) if info.get('initial_handle') else None
    output = Path(output)
    box = ReplayBox(rgb, output, trace=trace, init_mask=init)
    box.split, box.name = trace.directory.parent.name, trace.directory.name + Path(info['source_images'][0]).suffix
    box.segment_backend, box.prompt_backend = box.recorded_segment, box.recorded_prompt
    ask = ReplayAsk(trace, box, rgb, backend, view)
    box.ask_backend = ask
    def client(payload):
        if trace.planner_count >= len(trace.planner_events):
            raise ReplayMismatch('Unexpected planner request after end of archive')
        event = trace.planner_events[trace.planner_count]
        compare(trace.request(event['request']), payload, f'planner[{trace.planner_count}]')
        box.round_id = event['round_id']
        box.used = set()
        trace.planner_count += 1
        return trace.reply(event)
    parameters = dict(info['model_parameters'])
    seed = parameters.pop('seed')
    planner = Planner(seed=seed, client=client, parameters=parameters, sleep=lambda _: None,
                      max_attempts=info.get('total_attempt_limit', 180))
    recorder = ReplayRecorder(output, 'offline', 'image.png', gt=np.zeros(rgb[0].shape[:2], bool))
    recorder.trace = trace
    options = info['backends']
    # Fallbacks must never reach HTTP, even if a later edit accidentally creates another client.
    def no_network(*args, **kwargs):
        raise ReplayMismatch('Offline replay attempted a network request')
    with patch('urllib.request.urlopen', no_network), patch.object(Planner, '_http', no_network), \
         patch('agent_baseline.backends.Planner', lambda **kw: Planner(sleep=lambda _: None, **kw)):
        final = run_session(box, planner, recorder, info['task_id'], budget=info['budget'],
            prompt_version=info['system_prompt_version'], review=options.get('review', False),
            status=options.get('status', False), verify=options.get('verify_stage', False),
            plan=options.get('plan', False), skills=options.get('skills', ()))
    compare(len(trace.planner_events), trace.planner_count, 'planner request count')
    compare(len(trace.ask_events), trace.ask_count, 'verifier request count')
    old_final = json.loads((trace.directory / 'final.json').read_text())
    for key in ('final_handle', 'summary', 'submitted_by', 'termination_reason'):
        compare(old_final[key], final[key], 'final.' + key)
    if not np.array_equal(trace.mask(old_final['final_handle']), box.session.get(final['final_handle'], Mask).data):
        raise ReplayMismatch('final mask differs')
    return dict(planner_requests=trace.planner_count, verifier_requests=trace.ask_count,
        sam_cache_calls=trace.cache_calls, recorded_backend_calls=trace.recorded_calls,
        verifier_request_source='reconstructed from frozen code, archived masks and source RGB; not original wire capture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', type=Path, help='Session directory or messages.jsonl')
    parser.add_argument('--config')
    parser.add_argument('--task', choices=['T1.1xC0', 'T5.1'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--reference-code', type=Path, default=RUN / 'code_g')
    parser.add_argument('--output-dir', type=Path, required=True, help='Scratch/report directory outside the archived run')
    args = parser.parse_args()
    if args.session:
        sessions = [args.session.parent if args.session.name == 'messages.jsonl' else args.session]
    elif args.config and args.task:
        sessions = sorted(p.parent for p in (args.run_root / args.config / args.task / args.split).glob('*/messages.jsonl'))
    else:
        parser.error('Supply a session, or --config and --task')
    if args.limit is not None:
        sessions = sessions[:args.limit]
    if not sessions:
        parser.error('No archived sessions found')
    if any(args.output_dir.resolve().is_relative_to(session.resolve()) for session in sessions):
        parser.error('--output-dir must be outside archived session directories')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed, failed = [], []
    for session in sessions:
        # Temp directories are removed even on the first mismatch; archives are read-only.
        try:
            with tempfile.TemporaryDirectory(prefix='replay_', dir=args.output_dir) as scratch:
                report = replay(session, scratch, args.reference_code)
            passed.append(dict(session=str(session), **report))
            print('PASS', session, flush=True)
        except (ReplayMismatch, Exception) as exc:
            failed.append(dict(session=str(session), difference=str(exc)))
            print('FAIL', session, str(exc), flush=True)
    report = dict(passed=len(passed), total=len(sessions), successes=passed, failures=failed)
    (args.output_dir / 'replay_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=len(passed), total=len(sessions), failures=failed), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
