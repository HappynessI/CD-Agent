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
import numpy as np
from scipy import ndimage as ndi
from render import crop_box, render_pair
from prompts import VERIFY
from . import instances
from common import components
from .state import Session, Mask, Instances, ToolError, chosen_ids
from .view import render_view, box_pixels

MATCH_BASIS = 'raw'
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
    schema('ask', 'Verify building change in up to 16 mask components with a fixed question. Returns answer, type, reason, p_yes (relative yes/no reading), status. Unparsed is not no.',
           dict(regions=arg('array', items=dict(type='object', properties=dict(mask=M, ids=IDS),
                required=['mask', 'ids'], additionalProperties=False))), ['regions']),
    schema('select', 'Select instances using intersected conditions. IDs are retained. where allows score/area comparisons, numbers, and/or, parentheses; unavailable fields are errors.',
           dict(set=I, ids=IDS, where=S, area_min=arg('number', minimum=0), area_max=arg('number', minimum=0)), ['set']),
    schema('to_mask', 'Union one or more instance sets into a binary mask. ' + CURRENT,
           dict(sets=arg('array', items=I, minItems=1), set_current=SET_CURRENT), ['sets']),
    schema('combine', 'Combine masks; subtract means a minus b. ' + CURRENT,
           dict(op=arg('string', enum=['union', 'intersect', 'subtract']), a=M, b=M, set_current=SET_CURRENT), ['op', 'a', 'b']),
    schema('morph', 'Open/close with a disk of positive integer radius in original pixels; fill_holes uses no radius. ' + CURRENT,
           dict(mask=M, op=arg('string', enum=['open', 'close', 'fill_holes']), radius=arg('integer', minimum=1), set_current=SET_CURRENT), ['mask', 'op']),
    schema('remove_region', 'Remove full components identified on mask. ' + CURRENT,
           dict(mask=M, ids=IDS, set_current=SET_CURRENT), ['mask', 'ids']),
    schema('add_region', 'Add selected regions from other: mask component IDs or inst instance IDs; omitted IDs mean all. ' + CURRENT,
           dict(mask=M, other=S, ids=IDS, set_current=SET_CURRENT), ['mask', 'other']),
    schema('refine_region', 'Replace only the target component using SAM at the required time, clipped to the editing ROI. Without a prompt use its box; default ROI expands that box by 10%, at least 4 pixels. An explicit box is the ROI. ' + CURRENT,
           dict(mask=M, id=arg('integer', minimum=1), **PROMPT_ARGS, set_current=SET_CURRENT), ['mask', 'id', 'time']),
    schema('submit', 'Finish with any existing mask (including an old or empty mask) and a short summary. Subsequent calls in this reply are skipped.',
           dict(mask=M, summary=S), ['mask', 'summary']),
]
SCHEMA_BY_NAME = {s['function']['name']: s['function']['parameters'] for s in SCHEMAS}


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


def predicate(text, rows):
    try:
        tree = ast.parse(text, mode='eval').body
    except SyntaxError:
        raise ToolError('bad_args', 'where must be a comparison expression.') from None
    fields = set()
    def check(node):
        if isinstance(node, ast.Name) and node.id in ('score', 'area'):
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


class Toolbox:
    def __init__(self, rgb, directory, segment=None, prompt=None, ask=None, session=None, init_mask=None):
        self.rgb = tuple(np.array(im, copy=True) for im in rgb)
        self.session = session or Session(self.rgb[0].shape[:2], init_mask)
        self.directory = Path(directory)
        self.segment_backend, self.prompt_backend, self.ask_backend = segment, prompt, ask
        self.allowed = None
        self.meta = {}

    def get(self, key, kind=None):
        return self.session.get(key, kind, self.allowed)

    def execute(self, name, args, allowed=None):
        before = self.session.current_mask
        keys = set(self.session.handles)
        start = clock.monotonic()
        started_at = clock.time()
        self.allowed, self.meta = allowed, {}
        try:
            if name not in SCHEMA_BY_NAME:
                raise ToolError('bad_args', 'Unknown tool name.')
            validate(args, SCHEMA_BY_NAME[name])
            pyinspect.signature(getattr(self, name)).bind(**args)
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
        result.internal.update(self.meta)
        result.internal.update(current_before=before, current_after=self.session.current_mask,
            handles_created=[k for k in self.session.handles if k not in keys],
            seconds=clock.monotonic() - start, started_at=started_at, finished_at=clock.time())
        return result

    def _backend(self, backend, *args, **kwargs):
        if backend is None:
            raise ToolError('backend_unavailable', 'This backend is unavailable in the current run.')
        start = clock.monotonic()
        try:
            return backend(*args, **kwargs)
        finally:
            self.meta['backend_seconds'] = clock.monotonic() - start

    def _mask(self, data, parents=(), set_current=True, previous=None, **extra):
        key = self.session.add_mask(data, parents)
        item = self.session.get(key, Mask)
        out = dict(mask=key, area=int(item.data.sum()), component_count=len(item.components), **extra)
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
        self.session.view_count += 1
        return Result(dict(ok=True, **info), paths)

    def inspect(self, mask, offset=0, limit=50):
        item = self.get(mask)
        keys = ('id', 'area', 'bbox') if isinstance(item, Mask) else ('id', 'area', 'bbox', 'score')
        rows = sorted(item.components if isinstance(item, Mask) else item.items, key=lambda c: (-c['area'], c['id']))
        selected = [{k: r[k] for k in keys if k in r} for r in rows[offset:offset + limit]]
        return dict(mask=mask, total_components=len(rows), returned=len(selected),
                    has_more=offset + len(selected) < len(rows), components=selected)

    def segment_text(self, time, text, tile=512, score_thr=0.5):
        text = ' '.join(text.lower().split())
        if not text:
            raise ToolError('bad_args', 'text must name an object class.')
        rows, meta = self._backend(self.segment_backend, time, text, tile, score_thr)
        self.meta.update(meta)
        key = self.session.add_instances(rows, time)
        scores = [r['score'] for r in rows if r.get('score') is not None]
        return dict(inst=key, n=len(rows), area=int(instances.to_mask(rows, self.session.shape).sum()),
                    score=dict(min=float(min(scores)), median=float(np.median(scores)), max=float(max(scores))) if scores else None)

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

    def ask(self, regions):
        expanded = []
        for entry in regions:
            item = self.get(entry['mask'], Mask)
            ids = chosen_ids(item.components, entry['ids'])
            expanded.extend((entry['mask'], item, c) for c in item.components if c['id'] in ids)
        if len(expanded) > 16:
            raise ToolError('bad_args', 'regions expands to more than 16 components.')
        if not expanded:
            return dict(results=[])
        pairs = [render_pair(self.rgb, item.labels == c['id'], crop_box(c['bbox'], self.session.shape))
                 for _, item, c in expanded]
        rows, meta = self._backend(self.ask_backend, pairs, VERIFY)
        self.meta.update(meta)
        self.meta['answers'] = rows
        if len(rows) != len(expanded):
            raise ValueError('backend returned the wrong answer count')
        results = []
        for (key, _, c), row in zip(expanded, rows):
            answer = row.get('answer')
            valid = answer in ('yes', 'no')
            results.append(dict(mask=key, id=c['id'], p_yes=row.get('p_yes'), answer=answer if valid else None,
                type=row.get('type'), reason=row.get('reason'), status='parsed' if valid else 'unparsed'))
        return dict(results=results)

    def select(self, set, ids=None, where=None, area_min=None, area_max=None):
        item = self.get(set, Instances)
        wanted = chosen_ids(item.items, ids)
        pred = predicate(where, item.items) if where is not None else lambda r: True
        if area_min is not None and area_max is not None and area_min > area_max:
            raise ToolError('bad_args', 'area_min must not exceed area_max.')
        rows = [r for r in item.items if r['id'] in wanted and pred(r)
                and (area_min is None or r['area'] >= area_min) and (area_max is None or r['area'] <= area_max)]
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
        return self._mask(item.data & ~np.isin(item.labels, list(selected)), (mask,), set_current, item.data)

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

    def submit(self, mask, summary):
        item = self.get(mask, Mask)
        if not summary.strip():
            raise ToolError('bad_args', 'summary must be a nonempty explanation.')
        before = self.session.current_mask
        self.session.submitted = self.session.current_mask = mask
        self.session.summary = summary
        return dict(mask=mask, area=int(item.data.sum()), component_count=len(item.components),
                    current_mask=dict(before=before, after=mask), submitted=True)


def bbox(data):
    y, x = np.nonzero(data)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None
