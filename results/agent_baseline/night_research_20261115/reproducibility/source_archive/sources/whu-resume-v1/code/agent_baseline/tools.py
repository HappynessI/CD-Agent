"""Atomic BF tools, public schemas and a single execution boundary.

Toolbox backends are injected callables:
 segment(time, text, tile, score_thr) -> (items, metadata)
 prompt(time, box=None, points=None, point_labels=None) -> (mask, quality, metadata)
 ask(pairs, question) -> (answers, metadata)
Only public results reach the planner; metadata/tracebacks are recorder-only.
"""
import ast
import inspect as pyinspect
import json
import operator
import time as clock
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
from scipy import ndimage as ndi
from prompts import VERIFY
from .scope_t13 import verifier_prompts, session_prompts, session_card
from .tools_t13 import ProposalTools, tool_schemas, enabled_names, extend_schemas, NAMES as T13_NAMES
from .semantic_bridge_tools import SemanticBridgeTools, schemas as bridge_schemas, NAMES as BRIDGE_NAMES
from .find_missed import find_missed_schema, offer as offer_missed
from . import instances
from common import components
from .state import Session, Mask, Instances, ToolError, chosen_ids
from .view import render_view, box_pixels, ask_images

MATCH_BASIS = 'raw'
MAX_ASK = 64
BIG_BOX_SHARE = .10
# Opposite direction only; rebuilt and pseudo changes have no single direction.
OPPOSITE_TYPES = {'appeared': {'demolished', 'removed_tree'},
                  'disappeared': {'new_building', 'new_tree'}}
CURRENT = 'By default the returned mask becomes your current answer; pass set_current=false for intermediate candidates.'


def arg(kind, **kw):
    return dict(type=kind, **kw)


S = arg('string')
M = arg('string', description='An existing mask#k handle.')
I = arg('string', description='An existing inst#k handle.')
IDS = arg('array', items=arg('integer', minimum=1), uniqueItems=True)
BOX = arg('array', items=arg('number'), minItems=4, maxItems=4,
          description='Full-image [x0,y0,x1,y1], exclusive end.')
POINTS = arg('array', minItems=1, items=arg('array', items=arg('number'), minItems=2, maxItems=2))
TIME = arg('string', enum=['t1', 't2'])
SET_CURRENT = arg('boolean', default=True)
PROMPT_ARGS = dict(time=TIME, box=BOX, points=POINTS,
                   point_labels=arg('array', items=arg('integer', enum=[0, 1])))


def schema(name, description, properties, required=()):
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties, required=list(required), additionalProperties=False)))


SCHEMAS = [
    schema('view', 'Look at T1 and T2 with full-image coordinate ticks. Optional thin red contours of a mask or inst set; component IDs can be labelled on masks only.',
           dict(region=BOX, overlay=arg('string', description='An existing mask#k or inst#k handle.'), label_components=arg('boolean', default=False), ids=IDS)),
    schema('inspect', 'List the components of a mask (id, area, bbox) or the instances of an inst set (id, area, bbox, score); sorted by decreasing area then ID.',
           dict(mask=arg('string', description='An existing mask#k or inst#k handle.'), offset=arg('integer', minimum=0, default=0), limit=arg('integer', minimum=1, maximum=50, default=50)), ['mask']),
    schema('segment_text', 'Segment a named object class at one time. Returns an instance set and SAM3 score statistics; does not change current answer.',
           dict(time=TIME, text=S, tile=arg('integer', enum=[0, 512], default=512),
                score_thr=arg('number', minimum=0, maximum=1, default=0.5)), ['time', 'text']),
    schema('segment_prompt', 'Segment using exactly one of box or points (with point_labels). Returns a mask and sam_quality; does not change current answer.', PROMPT_ARGS, ['time']),
    schema('match_instances', 'Directional overlap presence: each instance is present if any single opposite-time instance covers at least overlap_thr of its own area. Returns appeared, disappeared, present_t2 instance handles.',
           dict(inst_t1=I, inst_t2=I, overlap_thr=arg('number', minimum=0, maximum=1, default=0.25)), ['inst_t1', 'inst_t2']),
    schema('ask', 'Ask the independent verifier about building change in mask components. With no arguments it checks the components of your current answer that have no answer yet, largest first, up to 64 per call; remaining_unverified says how many are left. For a mask without ids, the same on that mask. With ids it asks about exactly those components, up to 64 in total. Returns answer, type, reason, p_yes (relative yes/no reading, may be null), status. Unparsed is not no.',
           dict(regions=arg('array', items=dict(type='object', properties=dict(mask=M, ids=IDS),
                required=['mask'], additionalProperties=False)))),
    schema('select', 'Select instances using intersected conditions. IDs are retained. where allows score/area comparisons, numbers, and/or, parentheses; unavailable fields are errors.',
           dict(set=I, ids=IDS, where=S, area_min=arg('number', minimum=0), area_max=arg('number', minimum=0)), ['set']),
    schema('to_mask', 'Union one or more instance sets into a binary mask. ' + CURRENT,
           dict(sets=arg('array', items=I, minItems=1), set_current=SET_CURRENT), ['sets']),
    schema('combine', 'Combine masks; subtract means a minus b. ' + CURRENT,
           dict(op=arg('string', enum=['union', 'intersect', 'subtract']), a=M, b=M, set_current=SET_CURRENT), ['op', 'a', 'b']),
    schema('morph', 'Open/close with a disk of positive integer radius in original pixels; fill_holes uses no radius. ' + CURRENT,
           dict(mask=M, op=arg('string', enum=['open', 'close', 'fill_holes']), radius=arg('integer', minimum=1), set_current=SET_CURRENT), ['mask', 'op']),
    schema('remove_region', 'Remove full components identified on mask. The remaining components keep their IDs; other edits renumber components. ' + CURRENT,
           dict(mask=M, ids=IDS, set_current=SET_CURRENT), ['mask', 'ids']),
    schema('add_region', 'Add selected regions from other: mask component IDs or inst instance IDs; omitted IDs mean all. ' + CURRENT,
           dict(mask=M, other=S, ids=IDS, set_current=SET_CURRENT), ['mask', 'other']),
    schema('refine_region', 'Replace only the target component using SAM at the required time, clipped to the editing ROI. Without a prompt use its box; default ROI expands that box by 10%, at least 4 pixels. An explicit box is the ROI. ' + CURRENT,
           dict(mask=M, id=arg('integer', minimum=1), **PROMPT_ARGS, set_current=SET_CURRENT), ['mask', 'id', 'time']),
    schema('submit', 'Finish with any existing mask and a short summary. Pass mask="empty" to answer that nothing changed. Subsequent calls in this reply are skipped.',
           dict(mask=arg('string', description='An existing mask#k handle, or "empty".'), summary=S), ['mask', 'summary']),
]
# Offered only when the status harness is on, so runs without it keep their exact tool list.
PLAN_SCHEMA = schema('plan', 'Write or replace your own short to-do list for this image. Each item has a text and a status. The list is only your notes: it runs nothing and adds no evidence.',
    dict(items=arg('array', minItems=1, maxItems=12, items=dict(type='object', properties=dict(
        text=dict(type='string'), status=dict(type='string', enum=['todo', 'doing', 'done', 'dropped'])),
        required=['text', 'status'], additionalProperties=False))), ['items'])
# Offered only when skills are on, so runs without them keep their exact tool list.
SKILL_SCHEMA = schema('load_skill', 'Read the full text of one skill from the index in the system prompt.', dict(name=S), ['name'])
PYTHON_SCHEMA = schema('python', 'Run array calculations in an isolated Python subprocess with np and ndi. Inputs: masks[handle] (H,W bool), instances[handle] (N,H,W bool, only for handles written out in the code), images["t1"/"t2"] (H,W,3 uint8), all read-only. Call save_mask(array) for each output mask. No filesystem, network or submission access. New handles become usable in the next reply; submit them with submit. ' + CURRENT,
    dict(code=S, set_current=SET_CURRENT), ['code'])
ANSWER_SCHEMA = schema('set_answer', 'Make an existing mask#k your current answer, including restoring an earlier candidate. Does not edit pixels or component IDs, verify, or submit. Use inspect/ask to evaluate it and submit when ready.', dict(mask=M), ['mask'])
PAIR_SCHEMA = schema('segment_text_pair', 'Segment the same named class at both T1 and T2 in one planner tool call. Returns separate inst handles for each time; does not change current answer. Runs two backend requests sequentially, reusing their existing caches; this does not imply less GPU computation.',
    dict(text=S, tile=arg('integer', enum=[0, 512], default=512),
         score_thr=arg('number', minimum=0, maximum=1, default=0.5)), ['text'])
SCHEMA_BY_NAME = {s['function']['name']: s['function']['parameters'] for s in [*SCHEMAS, ANSWER_SCHEMA, PAIR_SCHEMA, PLAN_SCHEMA, SKILL_SCHEMA, PYTHON_SCHEMA, *tool_schemas()]}
SCHEMA_BY_NAME.update({s['function']['name']:s['function']['parameters'] for s in bridge_schemas()})
SCHEMA_BY_NAME['find_missed'] = find_missed_schema()['function']['parameters']


def area_select_schema(function):
    """Extend a session-local copy only; the frozen no-area interface stays intact."""
    function['description'] += (' For tasks with an area constraint, set may also be a binary mask: '
        'select whole 8-connected change regions by component IDs and area in pixels (area_min/area_max are inclusive). '
        'Use the same component IDs as inspect/remove_region; retained components keep their IDs. '
        'where on masks accepts only area comparisons. A mask input returns a new mask. ' + CURRENT +
        ' set_current applies only to mask inputs; instance selection never changes the current answer.')
    properties = function['parameters']['properties']
    properties['set'] = arg('string', description='An existing inst#k handle, or mask#k for connected-region selection by area in pixels.')
    properties['set_current'] = SET_CURRENT.copy()
    return function


def session_schemas(session, python=False, tools_t13=False, tool_hints='off', gate5=False, answer_state=False, segment_pair=False, t3_semantic_bridge=False, missed_candidates=False):
    import copy
    schemas = copy.deepcopy(SCHEMAS)
    if t3_semantic_bridge:
        schemas.extend(bridge_schemas())
    if missed_candidates:
        schemas.append(find_missed_schema())
    if answer_state:
        schemas.append(copy.deepcopy(ANSWER_SCHEMA))
    if segment_pair:
        schemas.append(copy.deepcopy(PAIR_SCHEMA))
    if session.area_min_px is not None:
        area_select_schema(next(s['function'] for s in schemas if s['function']['name'] == 'select'))
    if (session.target, session.change_direction) != ('building', 'any'):
        ask = next(s['function'] for s in schemas if s['function']['name'] == 'ask')
        label = 'any land-cover change' if session.target == 'any change' else 'changes involving ' + ', '.join(session.target) if isinstance(session.target,(list,tuple)) else f'{session.target} change'
        ask['description'] = ask['description'].replace('building change', label)
        if session.change_direction != 'any':
            ask['description'] += f' Count only the {session.change_direction} direction.'
    if session.scope:
        from .scope_t13 import scope_card
        card = session_card(session)
        ask = next(s['function'] for s in schemas if s['function']['name'] == 'ask')
        ask['description'] += ' Scope: ' + '; '.join(card['counts'] + card['not_counts'])
    if python:
        schemas.append(copy.deepcopy(PYTHON_SCHEMA))
    from .examples_t3x import extend_schema
    schemas = extend_schema(extend_schemas(schemas, tools_t13, tool_hints), session)
    if gate5:
        from .gate5 import SEGMENT_TOOLS, region_spec
        for entry in schemas:
            function = entry['function']
            if function['name'] in SEGMENT_TOOLS:
                function['parameters'] = region_spec(function['parameters'])
    if getattr(session, 'old_map', None):
        from .map_input import schemas_for_map
        schemas = schemas_for_map(schemas)
    return schemas
def disclosed_schemas(schemas, session):
    """Delay mask-edit schemas only until a usable mask exists; core stays visible."""
    if any(isinstance(item, Mask) for item in session.handles.values()):
        return schemas
    edits = {'combine', 'morph', 'remove_region', 'add_region', 'refine_region'}
    return [s for s in schemas if s['function']['name'] not in edits]


SKILLS = Path(__file__).parent / 'skills'


def skill_index(names):
    """One line per skill for the system prompt: its name and when to use it."""
    lines = []
    for name in names:
        head = (SKILLS / name / 'SKILL.md').read_text().split('---')[1]
        when = next(l.split(':', 1)[1].strip() for l in head.splitlines() if l.startswith('when:'))
        lines.append(f'- {name}: {when}')
    return '\n'.join(lines)


def validate(value, spec, path='arguments'):
    kind = spec['type']
    good = {'object': isinstance(value, dict), 'array': isinstance(value, list),
            'string': isinstance(value, str), 'boolean': type(value) is bool,
            'integer': type(value) is int,
            'number': type(value) in (int, float) and np.isfinite(value)}[kind]
    if not good:
        raise ToolError('bad_args', f'{path} must be {kind}.')
    if 'enum' in spec and value not in spec['enum']:
        raise ToolError('bad_args', f'{path} must be one of {spec["enum"]}.')
    if kind == 'object':
        for key in spec.get('required', []):
            if key not in value:
                raise ToolError('bad_args', f'{path}.{key} is required.')
        for key, item in value.items():
            if key not in spec['properties']:
                raise ToolError('bad_args', f'{path}.{key} is not supported.')
            validate(item, spec['properties'][key], f'{path}.{key}')
    if kind == 'array':
        if len(value) < spec.get('minItems', 0) or len(value) > spec.get('maxItems', float('inf')):
            raise ToolError('bad_args', f'{path} has an invalid length.')
        for j, item in enumerate(value):
            validate(item, spec['items'], f'{path}[{j}]')
        if spec.get('uniqueItems') and len(set(value)) != len(value):
            raise ToolError('bad_args', f'{path} must contain distinct IDs.')
    if kind in ('integer', 'number'):
        if value < spec.get('minimum', -float('inf')) or value > spec.get('maximum', float('inf')):
            raise ToolError('bad_args', f'{path} is outside the allowed range.')


COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def predicate(text, rows, allowed_fields=('score', 'area')):
    try:
        tree = ast.parse(text, mode='eval').body
    except SyntaxError:
        raise ToolError('bad_args', 'where must be a comparison expression.') from None
    fields = set()
    def check(node):
        if isinstance(node, ast.Name) and node.id in allowed_fields:
            fields.add(node.id)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            pass
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
            check(node.operand)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for v in node.values:
                check(v)
        elif isinstance(node, ast.Compare) and all(type(op) in COMPARE for op in node.ops):
            for v in [node.left] + node.comparators:
                check(v)
        else:
            raise ToolError('bad_args', 'where allows only score/area, comparisons, numbers and and/or.')
    check(tree)
    if not isinstance(tree, (ast.Compare, ast.BoolOp)):
        raise ToolError('bad_args', 'where must compare fields and numbers.')
    if any(any(f not in row or row[f] is None for f in fields) for row in rows):
        raise ToolError('bad_args', 'where references a field missing from this instance set.')
    def read(node, row):
        if isinstance(node, ast.Name):
            return row[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return (-1 if isinstance(node.op, ast.USub) else 1) * read(node.operand, row)
        if isinstance(node, ast.BoolOp):
            vals = [bool(read(v, row)) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        vals = [read(v, row) for v in [node.left] + node.comparators]
        return all(COMPARE[type(op)](a, b) for op, a, b in zip(node.ops, vals, vals[1:]))
    return lambda row: bool(read(tree, row))


@dataclass
class Result:
    public: dict
    images: list = field(default_factory=list)
    internal: dict = field(default_factory=dict)

    @property
    def text(self):
        return json.dumps(self.public, ensure_ascii=False, separators=(',', ':'))


def failure(code, message):
    return Result(dict(ok=False, error_code=code, message=message))


class Toolbox(ProposalTools, SemanticBridgeTools):
    def __init__(self, rgb, directory, segment=None, prompt=None, ask=None, session=None, init_mask=None, python=False, tools_t13=False, vision=None, gate3=False, big_box_share=BIG_BOX_SHARE, gate4=False, map_input=False, gate5=False, gate6=False, answer_state=False, segment_pair=False):
        self.rgb = tuple(np.array(im, copy=True) for im in rgb)
        self.session = session or Session(self.rgb[0].shape[:2], init_mask)
        if map_input:
            from .map_input import attach_map
            attach_map(self.session, self.rgb[0])
        self.directory = Path(directory)
        self.segment_backend, self.prompt_backend, self.ask_backend = segment, prompt, ask
        self.python_enabled = python
        self.tools_t13 = enabled_names(tools_t13)
        self.vision_backend = vision
        self.allowed = None
        self.meta = {}
        # Verifier answers by exact component outline, so they survive renumbering and lapse when an outline changes.
        # Each entry keeps every answer, because a later answer may contradict an earlier one.
        self.asked = {}
        self.proposal = None
        self.todo = []
        self.skills = ()
        if not 0 < big_box_share <= 1:
            raise ValueError('big_box_share must be in (0, 1]')
        self.gate3, self.big_box_share = gate3 or gate5, big_box_share
        self.gate4 = gate4 or gate5
        self.gate5 = gate5
        self.gate6 = gate6
        self.answer_state = answer_state
        self.segment_pair = segment_pair
        self.missed_provider = None
        self.t3_semantic_bridge = False
        self.bridge_seed = 1016
        self.recent_calls = []
        self.observed_views = []

    def get(self, key, kind=None):
        return self.session.get(key, kind, self.allowed)

    def execute(self, name, args, allowed=None):
        before = self.session.current_mask
        keys = set(self.session.handles)
        start = clock.monotonic()
        started_at = clock.time()
        self.allowed, self.meta = allowed, {}
        try:
            if name == 'find_missed' and self.missed_provider is None:
                raise ToolError('bad_args', 'Unknown tool name.')
            if name in BRIDGE_NAMES and not self.t3_semantic_bridge:
                raise ToolError('bad_args', 'Unknown tool name.')
            if (name == 'segment_text_pair' and not self.segment_pair) or (name == 'set_answer' and not self.answer_state) or name not in SCHEMA_BY_NAME or (name == 'python' and not self.python_enabled) or (name in T13_NAMES and name not in self.tools_t13):
                raise ToolError('bad_args', 'Unknown tool name.')
            from .examples_t3x import external_spec
            spec = external_spec(self.session, name, SCHEMA_BY_NAME[name])
            if name == 'select' and self.session.area_min_px is not None:
                import copy
                spec = area_select_schema(dict(description='', parameters=copy.deepcopy(spec)))['parameters']
            if self.gate5:
                from .gate5 import SEGMENT_TOOLS, region_spec
                if name in SEGMENT_TOOLS:
                    spec = region_spec(spec)
            validate(args, spec)
            pyinspect.signature(getattr(self, name)).bind(**args)
            if getattr(self.session, 'old_map', None):
                from .map_input import check_tool
                check_tool(name, args)
            result = getattr(self, name)(**args)
            if not isinstance(result, Result):
                result = Result(dict(ok=True, **result))
        except ToolError as exc:
            result = failure(exc.code, str(exc))
        except Exception as exc:
            result = failure('backend_error', 'Tool backend failed; inspect other evidence or retry later.')
            self.meta.update(getattr(exc, 'backend_meta', {}))
            self.meta['traceback'] = traceback.format_exc()
        finally:
            self.allowed = None
        if name == 'segment_text_pair' and not result.public['ok']:
            result.public['completed_results'] = self.meta.get('completed_results', {})
        result.internal.update(self.meta)
        result.internal.update(current_before=before, current_after=self.session.current_mask,
            handles_created=[k for k in self.session.handles if k not in keys],
            seconds=clock.monotonic() - start, started_at=started_at, finished_at=clock.time())
        if self.answer_state:
            self.recent_calls.append(dict(tool=name, ok=result.public['ok'], current=self.session.current_mask))
            self.recent_calls = self.recent_calls[-6:]
            if not result.public['ok'] and result.public.get('error_code') in ('bad_args', 'bad_handle', 'bad_id'):
                result.public['recovery_state'] = self.state_summary()
        return result

    def state_summary(self):
        """Bounded task-local index, never image pixels, GT, or the full trace."""
        handles = list(self.session.handles)
        selected = handles[-12:]
        if self.session.current_mask and self.session.current_mask not in selected:
            selected.insert(0, self.session.current_mask)
        rows = []
        for key in selected:
            item = self.session.handles[key]
            row = dict(handle=key, type=key.split('#', 1)[0], value_type=type(item).__name__,
                       created_order=handles.index(key), time=getattr(item, 'time', None))
            if isinstance(item, Mask):
                row.update(component_ids=[c['id'] for c in item.components][:64],
                           component_count=len(item.components))
                if key == self.session.current_mask:
                    verified = self.records(item)
                    visible = set(row['component_ids'])
                    row['verification'] = {k: len(v) for k, v in verified.items()}
                    row['verification_ids'] = {k: [c['id'] for c in v if c['id'] in visible]
                                               for k, v in verified.items()}
                    row['verification_ids_truncated'] = len(item.components) > len(visible)
            rows.append(row)
        return dict(current_mask=self.session.current_mask, handles=rows,
                    omitted_handles=max(0, len(handles)-len(selected)),
                    recent_calls=self.recent_calls, observed_views=self.observed_views[-8:],
                    total_views=self.session.view_count)

    def set_answer(self, mask):
        self.get(mask, Mask)
        if not mask.startswith('mask#'):
            raise ToolError('bad_handle', 'Expected an answer mask#k handle.')
        previous = self.session.current_mask
        self.session.current_mask = mask
        return dict(mask=mask, current_mask=mask, previous_mask=previous)

    def _backend(self, backend, *args, **kwargs):
        if backend is None:
            raise ToolError('backend_unavailable', 'This backend is unavailable in the current run.')
        start = clock.monotonic()
        try:
            return backend(*args, **kwargs)
        finally:
            self.meta['backend_seconds'] = clock.monotonic() - start

    def _mask(self, data, parents=(), set_current=True, previous=None, labeled=None, **extra):
        key = self.session.add_mask(data, parents, labeled)
        item = self.session.get(key, Mask)
        out = dict(mask=key, area=int(item.data.sum()), component_count=len(item.components), **extra)
        if self.answer_state:
            out.update(valid_component_ids=[c['id'] for c in item.components][:64],
                       ids_truncated=len(item.components) > 64,
                       component_identity='preserved' if labeled is not None else 'renumbered; use IDs on this mask')
        if previous is not None:
            out.update(added_px=int((item.data & ~previous).sum()), removed_px=int((previous & ~item.data).sum()))
        if set_current:
            before = self.session.current_mask
            self.session.current_mask = key
            out['current_mask'] = dict(before=before, after=key)
        return out

    def view(self, region=None, overlay=None, label_components=False, ids=None):
        item = self.get(overlay) if overlay is not None else None
        if isinstance(item, Instances):
            if label_components or ids is not None:
                raise ToolError('bad_args', 'Component labels need a mask handle; call to_mask first.')
            union = instances.to_mask(item.items, self.session.shape)
            labels, comps = components(union)
            item = Mask(union, labels, tuple(comps))
        num = self.session.view_count
        info, paths = render_view(self.rgb, self.directory / 'images', num, region, item,
                                  overlay, label_components, ids)
        if self.gate6:
            h, w = self.session.shape
            info.update(original_size=[w, h], coordinate_system="All coordinates are original-image pixels; image_size is the rendered image size.")
        self.session.view_count += 1
        if self.answer_state:
            self.observed_views.append(dict(index=num, region=region, overlay=overlay))
        return Result(dict(ok=True, **info), paths)

    def inspect(self, mask, offset=0, limit=50):
        item = self.get(mask)
        keys = ('id', 'area', 'bbox') if isinstance(item, Mask) else ('id', 'area', 'bbox', 'score')
        rows = sorted(item.components if isinstance(item, Mask) else item.items, key=lambda c: (-c['area'], c['id']))
        selected = [{k: r[k] for k in keys if k in r} for r in rows[offset:offset + limit]]
        return dict(mask=mask, total_components=len(rows), returned=len(selected),
                    has_more=offset + len(selected) < len(rows), components=selected)

    def segment_text(self, time, text, tile=512, score_thr=0.5, region=None):
        from .gate5 import resolve_region, clip_instances, clip_result
        region_data = resolve_region(self, region)
        text = ' '.join(text.lower().split())
        if not text:
            raise ToolError('bad_args', 'text must name an object class.')
        rows, meta = self._backend(self.segment_backend, time, text, tile, score_thr)
        self.meta.update(meta)
        if region is not None:
            rows, stats = clip_instances(rows, region, region_data)
        key = self.session.add_instances(rows, time, (region,) if region is not None else ())
        scores = [r['score'] for r in rows if r.get('score') is not None]
        result = dict(inst=key, n=len(rows), area=int(instances.to_mask(rows, self.session.shape).sum()),
                    score=dict(min=float(min(scores)), median=float(np.median(scores)), max=float(max(scores))) if scores else None)
        return clip_result(self, result, stats) if region is not None else result

    def segment_text_pair(self, text, tile=512, score_thr=0.5):
        # SAM3 worker serializes mutable processor state. Reuse the existing
        # per-time adapter/cache without introducing unsafe shared-model threads.
        results, provenance = {}, {}
        try:
            for time in ('t1', 't2'):
                self.meta = {}
                try:
                    results[time] = self.segment_text(time, text, tile, score_thr)
                finally:
                    provenance[time] = dict(self.meta)
        finally:
            self.meta = dict(temporal_backends=provenance, completed_times=list(results), completed_results=results,
                             backend_seconds=sum(m.get('backend_seconds', 0.) for m in provenance.values()),
                             cache_hit=len(provenance) == 2 and all(m.get('cache_hit', False) for m in provenance.values()))
        return dict(results=results, backend_invocations=2, execution='sequential', current_unchanged=True)

    def _prompt(self, time, box=None, points=None, point_labels=None):
        if (box is None) == (points is None):
            raise ToolError('bad_args', 'Supply exactly one of box or points.')
        if box is not None:
            box = box_pixels(box, self.session.shape)
            if point_labels is not None:
                raise ToolError('bad_args', 'point_labels requires points.')
        else:
            if point_labels is None or len(points) != len(point_labels):
                raise ToolError('bad_args', 'point_labels must have one 0/1 label per point.')
            h, w = self.session.shape
            if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
                raise ToolError('bad_args', 'points must lie inside the full image.')
        data, quality, meta = self._backend(self.prompt_backend, time, box=box, points=points, point_labels=point_labels)
        self.meta.update(meta)
        return np.asarray(data, bool), float(quality)

    def segment_prompt(self, time, box=None, points=None, point_labels=None):
        if self.gate3 and box is not None and points is None:
            x0, y0, x1, y1 = box_pixels(box, self.session.shape)
            share = (x1-x0) * (y1-y0) / np.prod(self.session.shape)
            if share > self.big_box_share:
                raise ToolError('box_too_large', 'A box prompt segments one object. '
                    f'This box covers {100*share:.2f}% of the image. For several objects or a whole area, '
                    'use segment_text, then intersect with a region if needed.')
        data, quality = self._prompt(time, box, points, point_labels)
        return self._mask(data, set_current=False, bbox=bbox(data), sam_quality=quality)

    def match_instances(self, inst_t1, inst_t2, overlap_thr=0.25):
        a, b = self.get(inst_t1, Instances), self.get(inst_t2, Instances)
        if a.time != 't1' or b.time != 't2':
            raise ToolError('bad_args', 'inst_t1 and inst_t2 must have their respective times.')
        aa, bb = a.items, b.items
        if MATCH_BASIS == 'union':
            aa, bb = (instances.union_components(rows, self.session.shape) for rows in (aa, bb))
        elif MATCH_BASIS != 'raw':
            raise RuntimeError('Unknown matching basis')
        matched = instances.match(aa, bb, overlap_thr)
        out = {}
        for name, rows, time in zip(('appeared', 'disappeared', 'present_t2'), matched, ('t2', 't1', 't2')):
            out[name] = self.session.add_instances(rows, time, (inst_t1, inst_t2))
            out[name + '_n'] = len(rows)
        self.meta['match_basis'] = MATCH_BASIS
        return out

    def ask(self, regions=None):
        if regions is None:
            if self.session.current_mask is None:
                raise ToolError('bad_args', 'There is no current answer. Call ask(regions=[{"mask":"mask#k"}]) with an existing mask; inst handles need to_mask first. Repeating ask() without setting current cannot work.')
            regions = [dict(mask=self.session.current_mask)]
        expanded, left = [], 0
        for entry in regions:
            item = self.get(entry['mask'], Mask)
            if entry.get('ids') is None:
                # The default is everything still unanswered, largest first; the cap applies per call.
                todo = self.pending(item)
                room = max(0, MAX_ASK - len(expanded))
                expanded.extend((entry['mask'], item, c) for c in todo[:room])
                left += len(todo[room:])
            else:
                ids = chosen_ids(item.components, entry['ids'])
                expanded.extend((entry['mask'], item, c) for c in item.components if c['id'] in ids)
        if len(expanded) > MAX_ASK:
            raise ToolError('bad_args', f'regions expands to more than {MAX_ASK} components.')
        if not expanded:
            result = dict(results=[], remaining_unverified=left)
            if self.answer_state:
                cached = []
                for region in regions:
                    item = self.get(region['mask'], Mask)
                    for answer, comps in self.records(item).items():
                        if answer == 'none':
                            continue
                        for c in comps:
                            if region.get('ids') is not None and c['id'] not in region['ids']:
                                continue
                            cached.append(dict(mask=region['mask'], id=c['id'], bbox=list(c['bbox']),
                                               answer=answer if answer in ('yes', 'no') else None,
                                               status='unparsed' if answer == 'unparsed' else 'parsed', cached=True))
                result.update(cached_verification=cached[:MAX_ASK],
                              cached_verification_total=len(cached),
                              cached_verification_truncated=len(cached) > MAX_ASK)
            return result
        return dict(results=self.verify(expanded, 'ask'), remaining_unverified=left)

    def find_missed(self, current=None, limit=20):
        return offer_missed(self, self.missed_provider, current, limit)

    def verify(self, expanded, source):
        context = getattr(self.ask_backend, 'context', False)
        renderer = getattr(self.ask_backend, 'render', None)
        pairs = [(renderer(self.rgb, item.labels == c['id'], c['bbox']) if renderer else
                  ask_images(self.rgb, item.labels == c['id'], c['bbox'], context)) for _, item, c in expanded]
        question = session_prompts(self.session)['VERIFY']
        if hasattr(self.ask_backend, 'session'):
            self.ask_backend.session = self.session
        rows, meta = self._backend(self.ask_backend, pairs, question)
        if self.session.ask_variant == 't2_state':
            from .map_input import t2_state_change
            if not getattr(self.session, 'old_map', None):
                raise ValueError('t2_state requires map input')
            old = self.session.get(self.session.old_map, Mask).data
            rows = [t2_state_change(row, old, item.labels == c['id'])
                    for (_, item, c), row in zip(expanded, rows)]
        self.meta.update(meta)
        self.meta['answers'] = rows
        if len(rows) != len(expanded):
            raise ValueError('backend returned the wrong answer count')
        results = []
        for (key, item, c), row in zip(expanded, rows):
            answer = row.get('answer')
            valid = answer in ('yes', 'no')
            outline = np.flatnonzero(item.labels == c['id'])
            entry = self.asked.setdefault(outline.tobytes(), dict(outline=outline, area=c['area'], bbox=list(c['bbox']), records=[]))
            entry['records'].append(dict(answer=answer if valid else None, type=row.get('type'), reason=row.get('reason'),
                                         p_yes=row.get('p_yes'), level=row.get('level', 'L1'), source=source, mask=key, id=c['id']))
            results.append(dict(mask=key, id=c['id'], p_yes=row.get('p_yes'), answer=answer if valid else None,
                type=row.get('type'), reason=row.get('reason'), status='parsed' if valid else 'unparsed'))
            if self.session.ask_variant == 't2_state':
                for field in ('p_change', 'state_answer', 'state_p_yes', 'old_building', 'old_building_fraction'):
                    results[-1][field] = row.get(field)
                    entry['records'][-1][field] = row.get(field)
        return results

    def select(self, set, ids=None, where=None, area_min=None, area_max=None, set_current=True):
        if self.session.area_min_px is not None:
            item = self.get(set)
            if isinstance(item, Mask):
                wanted = chosen_ids(item.components, ids)
                pred = predicate(where, item.components, ('area',)) if where is not None else lambda r: True
                if area_min is not None and area_max is not None and area_min > area_max:
                    raise ToolError('bad_args', 'area_min must not exceed area_max.')
                rows = [r for r in item.components if r['id'] in wanted and pred(r)
                        and (area_min is None or r['area'] >= area_min) and (area_max is None or r['area'] <= area_max)]
                kept = [r['id'] for r in rows]
                labels = np.where(np.isin(item.labels, kept), item.labels, 0)
                return self._mask(labels > 0, (set,), set_current, item.data, (labels, rows))
        item = self.get(set, Instances)
        wanted = chosen_ids(item.items, ids)
        pred = predicate(where, item.items) if where is not None else lambda r: True
        if area_min is not None and area_max is not None and area_min > area_max:
            raise ToolError('bad_args', 'area_min must not exceed area_max.')
        rows = [r for r in item.items if r['id'] in wanted and pred(r)
                and (area_min is None or r['area'] >= area_min) and (area_max is None or r['area'] <= area_max)]
        if set.startswith('props#'):
            result = self._props(rows, item.time, (set,))
            result.update(ids=[r['id'] for r in rows], n_in=len(item.items), n_out=len(rows))
            return result
        key = self.session.add_instances(rows, item.time, (set,))
        return dict(inst=key, ids=[r['id'] for r in rows], n_in=len(item.items), n_out=len(rows))

    def to_mask(self, sets, set_current=True):
        rows = [r for key in sets for r in self.get(key, Instances).items]
        return self._mask(instances.to_mask(rows, self.session.shape), sets, set_current)

    def combine(self, op, a, b, set_current=True):
        aa, bb = self.get(a, Mask).data, self.get(b, Mask).data
        data = {'union': lambda: aa | bb, 'intersect': lambda: aa & bb, 'subtract': lambda: aa & ~bb}[op]()
        return self._mask(data, (a, b), set_current, aa)

    def morph(self, mask, op, radius=None, set_current=True):
        data = self.get(mask, Mask).data
        if op == 'fill_holes':
            if radius is not None:
                raise ToolError('bad_args', 'fill_holes does not accept radius.')
            new = ndi.binary_fill_holes(data)
        else:
            if radius is None:
                raise ToolError('bad_args', 'radius is required for open/close.')
            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = x*x + y*y <= radius*radius
            new = (ndi.binary_opening if op == 'open' else ndi.binary_closing)(data, structure=disk)
        return self._mask(new, (mask,), set_current, data)

    def remove_region(self, mask, ids, set_current=True):
        item = self.get(mask, Mask)
        selected = chosen_ids(item.components, ids)
        gone = np.isin(item.labels, list(selected))
        # Whole components leave, so the others stay intact and keep their IDs across batched removals.
        labeled = (np.where(gone, 0, item.labels), [c for c in item.components if c['id'] not in selected])
        return self._mask(item.data & ~gone, (mask,), set_current, item.data, labeled)

    def add_region(self, mask, other, ids=None, set_current=True):
        target, source = self.get(mask, Mask), self.get(other)
        if isinstance(source, Mask):
            selected = chosen_ids(source.components, ids)
            add = np.isin(source.labels, list(selected))
        else:
            selected = chosen_ids(source.items, ids)
            add = instances.to_mask([r for r in source.items if r['id'] in selected], self.session.shape)
        return self._mask(target.data | add, (mask, other), set_current, target.data)

    def refine_region(self, mask, id, time, box=None, points=None, point_labels=None, set_current=True):
        item = self.get(mask, Mask)
        chosen_ids(item.components, [id])
        comp = next(c for c in item.components if c['id'] == id)
        x0, y0, x1, y1 = comp['bbox']
        dx, dy = max(4, int(np.ceil((x1-x0)*0.1))), max(4, int(np.ceil((y1-y0)*0.1)))
        h, w = self.session.shape
        roi = box_pixels(box, self.session.shape) if box is not None else [max(0,x0-dx), max(0,y0-dy), min(w,x1+dx), min(h,y1+dy)]
        if box is None and points is None:
            box = list(comp['bbox'])
        data, quality = self._prompt(time, box, points, point_labels)
        local = np.zeros(self.session.shape, bool)
        l, t, r, b = roi
        local[t:b, l:r] = data[t:b, l:r]
        new = (item.data & (item.labels != id)) | local
        return self._mask(new, (mask,), set_current, item.data, old_bbox=list(comp['bbox']),
            new_bbox=bbox(local), old_area=comp['area'], new_area=int(local.sum()), roi=roi, sam_quality=quality)

    def python(self, code, set_current=True):
        from .python_sandbox import calculate
        masks, insts = {}, {}
        parents = []
        # A dense instance set takes about 1 MB per instance, so only the sets the code names are sent.
        named = set(re.findall(r'inst#\d+', code))
        for key, item in self.session.handles.items():
            if self.allowed is not None and key not in self.allowed:
                continue
            parents.append(key)
            if isinstance(item, Mask):
                masks[key] = item.data
            else:
                if key not in named:
                    continue
                dense = np.zeros((len(item.items), *self.session.shape), bool)
                for row, data in zip(item.items, dense):
                    data.ravel()[row['idx']] = True
                insts[key] = dense
        arrays, stdout = calculate(code, masks, insts, self.rgb, self.session.shape)
        results = [self._mask(data, parents, set_current) for data in arrays]
        return dict(masks=results, stdout=stdout)

    def load_skill(self, name):
        if name not in self.skills:
            raise ToolError('bad_args', f'Unknown skill; available: {list(self.skills)}.')
        return dict(name=name, text=(SKILLS / name / 'SKILL.md').read_text().split('---', 2)[2].strip())

    def plan(self, items):
        self.todo = [dict(item) for item in items]
        return dict(items=len(self.todo), done=sum(item['status'] == 'done' for item in self.todo))

    def entry(self, item, c):
        return self.asked.get(np.flatnonzero(item.labels == c['id']).tobytes())

    def latest(self, entry):
        answers = [r['answer'] for r in (entry or dict(records=[]))['records'] if r['answer'] is not None]
        return answers[-1] if answers else None

    def records(self, item):
        """Components of a mask by latest verifier answer: yes, no, unparsed, or none for an outline nobody asked about."""
        answered = dict(yes=[], no=[], unparsed=[], none=[])
        for c in sorted(item.components, key=lambda c: (-c['area'], c['id'])):
            entry = self.entry(item, c)
            answered[self.latest(entry) or ('unparsed' if entry else 'none')].append(c)
        return answered

    def feedback_limited(self):
        return self.gate3 and (self.session.feedback_points is not None or self.session.feedback_cells is not None)

    def relevant(self, item, comparison=None):
        """Public point disks/cells plus changed outlines relative to mask#0.

        For an original component, comparison is the proposed mask; any removal
        counts as a change, including a complete deletion. No ground truth is used.
        """
        from .feedback import cell_mask
        s = self.session
        region = np.zeros(s.shape, bool)
        yy, xx = np.ogrid[:s.shape[0], :s.shape[1]]
        for x, y in s.feedback_points or ():
            region |= (xx-x)**2 + (yy-y)**2 <= 32**2
        for where in s.feedback_cells or ():
            region |= cell_mask(where, s.shape)
        start = s.get(s.initial_mask, Mask) if s.initial_mask else None
        original = {np.flatnonzero(start.labels == c['id']).tobytes() for c in start.components} if start else set()
        selected = set()
        for c in item.components:
            pixels = item.labels == c['id']
            changed = (not comparison.data[pixels].all() if comparison is not None
                       else np.flatnonzero(pixels).tobytes() not in original)
            if changed or region[pixels].any():
                selected.add(c['id'])
        return selected

    def pending(self, item, comparison=None):
        """Components the verifier still owes an answer: never asked, or unparsed once and allowed one retry."""
        rows = sorted(item.components, key=lambda c: (-c['area'], c['id']))
        if self.feedback_limited():
            relevant = self.relevant(item, comparison)
            rows = [c for c in rows if c['id'] in relevant]
        out = []
        for c in rows:
            entry = self.entry(item, c)
            if entry is None or (self.latest(entry) is None and len(entry['records']) < 2):
                out.append(c)
        return out

    def owed(self, item):
        """What the stage would check for this mask: its pending components and given components it dropped unasked."""
        rows = self.pending(item)
        if self.session.initial_mask is not None:
            start = self.session.get(self.session.initial_mask, Mask)
            rows = rows + self.dropped_pending(start, item)
        return rows

    def dropped_pending(self, start, item):
        if self.feedback_limited():
            return [c for c in self.pending(start, comparison=item)
                    if not item.data[start.labels == c['id']].all()]
        return [c for c in self.pending(start) if item.data[start.labels == c['id']].mean() < .5]

    def stage(self, key):
        """The fixed verification stage: every component of this mask still without an answer. Never edits a mask."""
        item = self.session.get(key, Mask)
        todo = [(key, item, c) for c in self.pending(item)]
        if self.session.initial_mask is not None:
            # Given components the planner dropped unasked: the report must be able to say what they were.
            start = self.session.get(self.session.initial_mask, Mask)
            todo += [(self.session.initial_mask, start, c) for c in self.dropped_pending(start, item)]
        self.meta = {}
        failed = False
        if todo:
            try:
                self.verify(todo, 'stage')
            except Exception as exc:
                failed = True
                self.meta.update(getattr(exc, 'backend_meta', {}))
                self.meta['traceback'] = traceback.format_exc()
        return len(todo), failed, dict(self.meta)

    def status(self, replies_left):
        """One short state line for each reply, from the session's own records."""
        key = self.session.current_mask
        lines = [f'Status: {replies_left} replies left.']
        if key is None:
            lines.append('No current answer yet; a run that ends without a current answer returns an empty mask.')
        else:
            item = self.session.get(key, Mask)
            answered = self.records(item)
            lines.append(f"Current answer {key}: {int(item.data.sum())} pixels in {len(item.components)} components; "
                         f"verifier answers: {len(answered['yes'])} yes, {len(answered['no'])} no, {len(answered['unparsed'])} unparsed, {len(answered['none'])} none.")
        if self.proposal is not None and self.proposal != key:
            lines.append(f'Last proposed: {self.proposal}.')
        if self.todo:
            done = sum(row['status'] in ('done', 'dropped') for row in self.todo)
            following = next((row['text'] for row in self.todo if row['status'] in ('doing', 'todo')), None)
            lines.append(f'Plan: {done}/{len(self.todo)} closed' + (f'; next: {following}' if following else '') + '.')
        return ' '.join(lines)

    def pre_submit(self, key, replies_left, stage=False):
        """The response to a first submit: the proposal becomes the current answer, then the report. No edits."""
        meta = {}
        checked, failed = None, False
        if stage:
            checked, failed, meta = self.stage(key)
        previous, self.proposal = self.proposal, key
        self.session.current_mask = key
        return Result(dict(ok=True, submitted=False, report=self.report(key, replies_left, checked, failed, previous)),
                      internal=dict(meta, status='report'))

    def report(self, key, replies_left, checked=None, failed=False, previous=None):
        """What the session's own records say about a proposed mask, including what it no longer contains."""
        item = self.session.get(key, Mask)
        answered = self.records(item)
        row = lambda c: f"id={c['id']} area={c['area']} bbox={list(c['bbox'])}"
        def answer_row(c):
            last = next(r for r in reversed(self.entry(item, c)['records']) if r['answer'] is not None)
            return row(c) + f" type={last['type']} p_yes={None if last['p_yes'] is None else round(last['p_yes'], 2)} reason={str(last['reason'])[:120]}"
        lines = [f"Report before submission. Every run gets this report as the first response to submit; it does not judge the mask, and nothing has to change. {key} is now your current answer: {int(item.data.sum())} pixels in {len(item.components)} components."]
        if checked is not None:
            if failed:
                lines.append('The verifier could not answer now; the answers below are the earlier ones.')
            elif checked:
                lines.append(f'The verifier has just checked the {checked} components that had no answer.')
        lines.append(f"Verifier answers on {key}: {len(answered['yes'])} yes, {len(answered['no'])} no, {len(answered['unparsed'])} unparsed, {len(answered['none'])} with no answer. An answer applies only to the exact outline it was about; a view is not an answer.")
        if answered['yes']:
            lines.append(f"  IDs answered yes: {[c['id'] for c in answered['yes']]}")
        if answered['no']:
            lines.append(f"  IDs answered no: {[c['id'] for c in answered['no']]}")
            lines += [f"  answered no: {answer_row(c)}" for c in answered['no'][:30]]
            if len(answered['no']) > 30:
                lines.append(f"  and {len(answered['no']) - 30} more answered no.")
        mixed = [c for c in item.components if len({r['answer'] for r in (self.entry(item, c) or dict(records=[]))['records'] if r['answer']}) == 2]
        if mixed:
            lines.append(f"  answers disagree across repeats for IDs: {[c['id'] for c in mixed]}")
        for name, label in (('none', 'no answer'), ('unparsed', 'unparsed')):
            lines += [f"  {label}: {row(c)}" for c in answered[name][:20]]
            if len(answered[name]) > 20:
                lines.append(f"  and {len(answered[name]) - 20} more with {label}.")
        present = item.data.ravel()
        gone = [e for e in self.asked.values() if self.latest(e) == 'yes' and present[e['outline']].mean() < .5]
        if gone:
            gone.sort(key=lambda e: -e['area'])
            lines.append(f"Answered yes and not in {key}: {len(gone)} regions, {sum(e['area'] for e in gone)} pixels.")
            lines += [f"  area={e['area']} bbox={e['bbox']} reason={str(e['records'][-1]['reason'])[:120]}" for e in gone[:10]]
        if self.session.initial_mask is not None:
            start = self.session.get(self.session.initial_mask, Mask)
            outside = item.data & ~start.data
            ids = sorted(set(int(v) for v in item.labels[outside]))
            lost = [c for c in start.components if item.data[start.labels == c['id']].mean() < .5]
            verdicts = [self.latest(self.entry(start, c)) for c in lost]
            lines.append(f"Compared with the given mask: {int(outside.sum())} pixels added, in components {ids[:20]}; {int((start.data & ~item.data).sum())} pixels removed; "
                         f"{len(lost)} given components are gone: {verdicts.count('no')} answered no, {verdicts.count('yes')} answered yes, {verdicts.count(None)} with no answer. Removed pixels are not the same as corrected errors.")
            lines += [f"  gone with no answer: {row(c)}" for c in sorted(lost, key=lambda c: -c['area']) if self.latest(self.entry(start, c)) is None][:10]
            back = [c['id'] for c, v in zip(lost, verdicts) if v == 'yes']
            if back:
                lines.append(f"  gone and answered yes, IDs on {self.session.initial_mask}: {back}. add_region with other={self.session.initial_mask} and these ids brings any of them back.")
        if previous is not None and previous != key:
            before = self.session.get(previous, Mask).data
            lines.append(f"Compared with your previous proposal {previous}: {int((item.data & ~before).sum())} pixels added, {int((before & ~item.data).sum())} pixels removed.")
        if self.gate3:
            lines.extend(self.constraint_report(item))
        lines.append(f"A verifier answer is evidence, not an order, and it can be wrong in both directions. No answer is not an error. To finish with this mask, call submit with {key} again. To change it, edit and submit the new handle; components the verifier has not seen will be reported again. In the summary, say what you changed, and why you went against any verifier answer. You have {replies_left} replies left.")
        return '\n'.join(lines)

    def constraint_report(self, item):
        lines = []
        if self.session.region_mask is not None:
            region = self.session.get(self.session.region_mask, Mask).data
            outside = item.data & ~region
            if outside.any():
                ids = sorted(int(i) for i in np.unique(item.labels[outside]))
                lines.append(f'Outside region#0: {int(outside.sum())} pixels in component IDs {ids}.')
        types = set(session_prompts(self.session)['PSEUDO_TYPES'])
        opposite = OPPOSITE_TYPES.get(self.session.change_direction, set()) & types
        wrong = []
        for c in item.components:
            records = (self.entry(item, c) or dict(records=[]))['records']
            last = next((r for r in reversed(records) if r.get('type') in types), None)
            if last and last['type'] in opposite:
                wrong.append(c['id'])
        if wrong:
            lines.append(f'Opposite-direction verifier types for {self.session.change_direction}: component IDs {sorted(wrong)} (types {sorted(opposite)}).')
        if self.session.area_min_px is not None:
            # Same mechanism as the region and direction reports: driven by the task list, reported, never removed.
            small = [c for c in item.components if c['area'] <= self.session.area_min_px]
            if small:
                lines.append(f'Area constraint (regions must be larger than {self.session.area_min_m2:g} m^2): {len(small)} connected regions are not larger, '
                             f'{sum(c["area"] for c in small)} pixels in total, component IDs {sorted(c["id"] for c in small)[:30]}.')
                lines.append('Regions are defined by connectivity in the change mask; do not merge regions to meet the area constraint.')
        return lines

    def ledger(self, final_key):
        """Per outline: every verifier answer and what the planner finally did with it. For the trace only."""
        present = self.session.get(final_key, Mask).data.ravel() if final_key else None
        rows = []
        for entry in self.asked.values():
            kept = bool(present[entry['outline']].mean() >= .5) if present is not None else None
            latest = self.latest(entry)
            rows.append(dict(area=entry['area'], bbox=entry['bbox'], runs=runs(entry['outline']), records=entry['records'], latest=latest,
                             kept=kept, differs_from_rule=None if latest is None or kept is None else (kept != (latest == 'yes'))))
        return rows

    def submit(self, mask, summary):
        item = self.get(mask, Mask)
        if not summary.strip():
            raise ToolError('bad_args', 'summary must be a nonempty explanation.')
        from .gate4 import clip_submission
        mask, report = clip_submission(self, mask)
        item = self.session.get(mask, Mask)
        before = self.session.current_mask
        self.session.submitted = self.session.current_mask = mask
        self.session.summary = summary
        return dict(mask=mask, area=int(item.data.sum()), component_count=len(item.components),
                    **({'report': report} if report else {}),
                    current_mask=dict(before=before, after=mask), submitted=True)


def runs(outline):
    """Flat pixel indices as [start, length] runs."""
    breaks = np.flatnonzero(np.diff(outline) != 1) + 1
    return [[int(part[0]), int(part.size)] for part in np.split(outline, breaks) if part.size]


def bbox(data):
    y, x = np.nonzero(data)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None
