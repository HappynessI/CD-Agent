"""One-way trace sink. This is the only online module that reads reference labels."""
import copy
import base64
import json
import time
from pathlib import Path
import numpy as np
from common import load_gt, confusion, image_f1, object_counts, scores, save_mask_png, write_json
from .tools_t13 import Features
from .state import Mask


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


class Recorder:
    def __init__(self, directory, split, name, gt=None, gt_path=None, gt_ignore_path=None):
        self.gt_ignore_path = str(gt_ignore_path) if gt_ignore_path else None
        self.valid = None
        self.gt_path = str(gt_path) if gt_path is not None else None
        self.directory = Path(directory)
        for folder in ('images', 'masks', 'insts'):
            (self.directory / folder).mkdir(parents=True, exist_ok=True)
        # Exactly one read, before any execution, never passed back into Session.
        if gt_path is not None:
            from .evaluation import load_reference
            gt, self.valid = load_reference(gt_path, gt_ignore_path)
        self.gt = np.array(load_gt(split, name) if gt is None else gt, dtype=bool, copy=True)
        self.gt.flags.writeable = False
        if self.valid is None:
            self.valid = np.ones_like(self.gt, bool)
        self.valid.flags.writeable = False
        self.cache, self.saved = {}, set()
        self.previous = None
        self.initial_eval = None
        self.initial_objects = None
        self.started = time.monotonic()
        self.steps = []
        self.ask_tokens = dict(input=0, cached=0, output=0)
        self.ask_cost = 0.
        self.ask_attempts = 0
        self.asset_number = 0
        for filename in ('steps.jsonl', 'messages.jsonl'):
            (self.directory / filename).write_text('')

    def append(self, filename, row):
        with (self.directory / filename).open('a') as file:
            file.write(json.dumps(row, ensure_ascii=False, default=json_default) + '\n')

    def request(self, event):
        self.append('messages.jsonl', self.archive_images(event))

    def archive_images(self, value):
        """Keep exact transmitted JPEG bytes; replace inline image bodies by archive paths."""
        if isinstance(value, dict):
            return {k: self.archive_images(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.archive_images(v) for v in value]
        if isinstance(value, str) and value.startswith('data:image/jpeg;base64,'):
            self.asset_number += 1
            path = self.directory / 'images' / f'backend_{self.asset_number}.jpg'
            path.write_bytes(base64.b64decode(value.split(',', 1)[1]))
            return str(path)
        return value

    def start(self, session, metadata):
        if self.gt_path is not None:
            metadata = dict(metadata, gt_mask_path=self.gt_path, gt_ignore_mask_path=self.gt_ignore_path, ignored_pixels=int((~self.valid).sum()))
        write_json(self.directory / 'input.json', metadata)
        self.assets(session)
        if session.initial_mask is not None:
            key = session.initial_mask
            self.initial_eval = self.metric(session, key)
            self.initial_objects = object_counts(session.get(key, Mask).data & self.valid, self.gt)
        self.append('steps.jsonl', dict(step=0, planner=None, calls=[], current_mask=session.current_mask,
            eval=self.edge(session, session.current_mask, None), initial_objects=self.initial_objects))
        self.previous = session.current_mask

    def input_failure(self, task, metadata, error):
        """Archive an empty prediction if source imagery cannot be read; no Session is created."""
        empty = np.zeros_like(self.gt)
        save_mask_png(self.directory / 'masks' / 'mask_0.png', empty)
        c = confusion(empty, self.gt)
        metric = dict(c, f1=image_f1(empty, self.gt), iou=0. if self.gt.any() else 1.)
        if self.gt_path is not None:
            metadata = dict(metadata, gt_mask_path=self.gt_path, gt_ignore_mask_path=self.gt_ignore_path, ignored_pixels=int((~self.valid).sum()))
        write_json(self.directory / 'input.json', dict(metadata, task_id=task, input_error=error))
        row = dict(task_id=task, final_handle='mask#0', mask_path='masks/mask_0.png',
            summary='系统兜底：输入影像无法读取；使用空掩码。', summary_source='system',
            submitted_by='error', termination_reason='input_error', fallback_kind='empty',
            change_component_count=0, change_area_px=0, eval=metric, initial_eval=None,
            objects=object_counts(empty, self.gt), initial_objects=None,
            precision=0., recall=0., planning_rounds=0, api_attempts=0,
            planner_api_attempts=0, ask_api_attempts=0, tool_calls=0,
            tokens=dict(input=0, cached=0, output=0), cost_cny=0.,
            elapsed_seconds=time.monotonic()-self.started,
            identical_to_init=None, added_px=None, removed_px=None, view_calls=0, ask_calls=0,
            initial_images_present=False, error=error)
        write_json(self.directory / 'final.json', row)
        return row

    def assets(self, session):
        for key, value in session.handles.items():
            if key in self.saved:
                continue
            number = key.split('#')[1]
            from .examples_t3x import archive_example
            if archive_example(self.directory, key, value):
                self.saved.add(key)
                continue
            if isinstance(value, Mask):
                save_mask_png(self.directory / 'masks' / f'{key.replace("#", "_")}.png', value.data)
            elif isinstance(value, Features):
                directory = self.directory / 'features'
                directory.mkdir(exist_ok=True)
                np.savez(directory / f'feat_{number}.npz', data=value.data, shape=value.shape, time=value.time)
            else:
                # Pack one instance at a time; avoid a large dense instance stack.
                packed = []
                for row in value.items:
                    flat = np.zeros(int(np.prod(session.shape)), bool)
                    flat[row['idx']] = True
                    packed.append(np.packbits(flat))
                nbytes = (int(np.prod(session.shape)) + 7) // 8
                np.savez_compressed(self.directory / 'insts' / f'inst_{number}.npz',
                    masks=np.asarray(packed, dtype=np.uint8).reshape(len(packed), nbytes), shape=session.shape,
                    score=np.array([r.get('score', np.nan) for r in value.items], float),
                    ids=np.array([r['id'] for r in value.items], int), time=value.time, parents=np.array(value.parents, dtype=str))
            self.saved.add(key)

    def metric(self, session, key):
        if key is None:
            return None
        if key not in self.cache:
            mask = session.get(key, Mask).data & self.valid
            c = confusion(mask, self.gt)
            union = c['tp'] + c['fp'] + c['fn']
            self.cache[key] = dict(**c, f1=image_f1(mask, self.gt), iou=c['tp'] / union if union else 1.)
        return dict(self.cache[key])

    def edge(self, session, after, before):
        metric = self.metric(session, after)
        if metric is None:
            return None
        prior = self.metric(session, before)
        metric['delta_f1'] = (0. if after == before else metric['f1'] - prior['f1']) if prior else None
        return metric

    def ask_spending(self, meta, tool_call_id):
        for k, n in meta.get('tokens', {}).items():
            self.ask_tokens[k] += n
        self.ask_cost += meta.get('cost_cny', 0.)
        self.ask_attempts += len(meta.get('attempts', []))
        for event in meta.get('attempts', []):
            self.append('messages.jsonl', dict(event, backend='api_ask', tool_call_id=tool_call_id))

    def note(self, number, result):
        """A report the harness sent on its own, outside any tool call."""
        self.append('steps.jsonl', dict(step=number, note='report', report=result.public['report'], calls=[], planner=None))

    def step(self, session, number, planner, calls):
        self.assets(session)
        calls = self.archive_images(copy.deepcopy(calls))
        for call in calls:
            call['eval'] = self.edge(session, call.get('current_after'), call.get('current_before'))
            # Candidates a model keeps out of current (set_current=false) still get scored here, for analysis only.
            scored = []
            for key in call.get('handles_created', []):
                value = session.handles.get(key)
                if isinstance(value, Mask):
                    parent = next((p for p in value.parents if isinstance(session.handles.get(p), Mask)), None)
                    row = dict(handle=key, parent=parent, **self.metric(session, key))
                    if parent is not None:
                        row['delta_f1_from_parent'] = row['f1'] - self.metric(session, parent)['f1']
                    scored.append(row)
            call['candidate_eval'] = scored
            self.ask_spending(call.get('provenance', {}), call['tool_call_id'])
        row = dict(step=number, planner=planner, calls=calls, current_mask=session.current_mask,
                   eval=self.edge(session, session.current_mask, self.previous))
        self.previous = session.current_mask
        self.steps.append(row)
        self.append('steps.jsonl', row)

    def final(self, session, task, submitted_by, reason, fallback, rounds, planner, error=None):
        self.assets(session)
        key = session.submitted
        mask = session.get(key, Mask)
        metric = self.metric(session, key)
        obj = object_counts(mask.data & self.valid, self.gt)
        tokens = {k: planner.tokens[k] + self.ask_tokens[k] for k in planner.tokens}
        initial = session.get(session.initial_mask, Mask).data if session.initial_mask is not None else None
        calls = [c for row in self.steps for c in row['calls']]
        row = dict(task_id=task, final_handle=key, mask_path=f'masks/{key.replace("#", "_")}.png',
            summary=session.summary, summary_source='model' if submitted_by == 'model' else 'system',
            submitted_by=submitted_by, termination_reason=reason, fallback_kind=fallback,
            change_component_count=len(mask.components), change_area_px=int(mask.data.sum()),
            eval=metric, initial_eval=self.initial_eval, objects=obj, initial_objects=self.initial_objects,
            precision=scores(metric)['precision'], recall=scores(metric)['recall'],
            planning_rounds=rounds, api_attempts=planner.attempts + self.ask_attempts,
            planner_api_attempts=planner.attempts, ask_api_attempts=self.ask_attempts,
            tool_calls=len(calls), tokens=tokens, planner_tokens=planner.tokens, ask_tokens=self.ask_tokens,
            cost_cny=planner.cost + self.ask_cost, planner_cost_cny=planner.cost, ask_cost_cny=self.ask_cost, planner_model=planner.parameters.get('model'), elapsed_seconds=time.monotonic()-self.started,
            identical_to_init=bool(np.array_equal(mask.data, initial)) if initial is not None else None,
            added_px=int((mask.data & ~initial).sum()) if initial is not None else None,
            removed_px=int((initial & ~mask.data).sum()) if initial is not None else None,
            view_calls=sum(c['tool'] == 'view' and c['ok'] for c in calls),
            ask_calls=sum(c['tool'] == 'ask' and c['ok'] for c in calls), initial_images_present=True,
            final_delta=self.edge(session, key, self.previous), error=error)
        if hasattr(session, 'g4_submission'):
            row.update(session.g4_submission)
        write_json(self.directory / 'final.json', row)
        return row
