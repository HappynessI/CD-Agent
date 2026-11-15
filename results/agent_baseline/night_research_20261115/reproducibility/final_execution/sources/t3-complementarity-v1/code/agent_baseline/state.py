"""Per-image state; no metrics or label access belongs in this module."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from common import components


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Mask:
    data: np.ndarray
    labels: np.ndarray
    components: tuple
    parents: tuple = ()


@dataclass(frozen=True)
class Instances:
    items: tuple
    time: str
    parents: tuple = ()


class Session:
    def __init__(self, shape, init_mask=None, target="building", change_direction="any", instruction=None, constraint=None, scope=None, region_mask=None, targets=None, feedback_points=None, feedback_cells=None, source_classes=None, ask_variant=None, area_min_m2=None, gsd_m=None):
        from .scope_t13 import scope_card
        target = tuple(targets) if targets is not None else target
        scope_card(target, change_direction, scope)
        if source_classes is not None and (target != 'building' or change_direction != 'appeared'
                or list(source_classes) != ['low vegetation', 'non-vegetated ground surface']):
            raise ValueError('Source classes require the declared ground/vegetation to building conversion')
        from .scope_t13 import ASK_CANDIDATES
        if ask_variant is not None and not ((target == 'any change' and ask_variant in ASK_CANDIDATES)
                                            or (target == 'building' and ask_variant == 't2_state')):
            raise ValueError('ask_variant requires any change and a declared candidate')
        self.source_classes = tuple(source_classes) if source_classes is not None else None
        self.ask_variant = ask_variant
        self.scope = scope
        self.target, self.change_direction = target, change_direction
        self.instruction, self.constraint = instruction, constraint
        self.shape = tuple(shape)
        self.feedback_points = None if feedback_points is None else tuple(tuple(p) for p in feedback_points)
        self.feedback_cells = None if feedback_cells is None else tuple(feedback_cells)
        self.handles = {}
        self.next_id = 0
        self.current_mask = None
        self.initial_mask = None
        self.region_mask = None
        self.region_bbox = None
        # Area constraint declared by the task list (T4.1): regions must be larger than area_min_m2 at gsd_m metres per pixel.
        self.area_min_m2 = area_min_m2
        self.area_min_px = area_min_m2 / (gsd_m * gsd_m) if area_min_m2 is not None and gsd_m else None
        self.submitted = None
        self.summary = None
        self.view_count = 0
        if init_mask is not None:
            self.initial_mask = self.add_mask(init_mask)
            self.current_mask = self.initial_mask
        if region_mask is not None:
            self.region_mask = self.add_mask(region_mask, handle='region#0')
            yy, xx = np.nonzero(self.get(self.region_mask, Mask).data)
            if len(xx):
                self.region_bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]

    def _put(self, prefix, value):
        key = f'{prefix}#{self.next_id}'
        self.next_id += 1
        self.handles[key] = value
        return key

    def add_mask(self, data, parents=(), labeled=None, handle=None):
        data = np.array(data, dtype=bool, copy=True)
        if data.shape != self.shape:
            raise ValueError('backend mask shape differs from image shape')
        # labeled keeps existing component IDs when an edit removes whole components.
        labels, comps = labeled if labeled is not None else components(data)
        data.flags.writeable = labels.flags.writeable = False
        comps = tuple(MappingProxyType(dict(c, bbox=tuple(c['bbox']))) for c in comps)
        value = Mask(data, labels, comps, tuple(parents))
        if handle is not None:
            self.handles[handle] = value
            return handle
        return self._put('mask', value)

    def add_instances(self, items, time, parents=()):
        copied = []
        for i, item in enumerate(items, 1):
            row = dict(item)
            row['idx'] = np.array(row['idx'], dtype=np.int32, copy=True)
            row['idx'].flags.writeable = False
            row['bbox'] = tuple(row['bbox'])
            row.setdefault('id', i)
            copied.append(MappingProxyType(row))
        return self._put('inst', Instances(tuple(copied), time, tuple(parents)))

    def get(self, key, kind=None, allowed=None):
        if not isinstance(key, str) or key not in self.handles or (allowed is not None and key not in allowed):
            raise ToolError('bad_handle', 'Unknown handle; wait until the next reply to use a newly created handle.')
        value = self.handles[key]
        if kind is not None and not isinstance(value, kind):
            expected = {Mask: 'mask', Instances: 'inst'}.get(kind, kind.__name__)
            actual = key.split('#', 1)[0]
            hint = (' Convert instances with to_mask(sets=[inst_handle]).' if kind is Mask and isinstance(value, Instances)
                    else ' An existing mask can be passed directly to ask/submit.' if isinstance(value, Mask) else '')
            raise ToolError('bad_handle', f'{key} is a {actual} handle ({type(value).__name__}); expected {expected}.{hint}')
        return value


def chosen_ids(rows, ids):
    valid = {r['id'] for r in rows}
    if ids is None:
        return valid
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ToolError('bad_args', 'ids must be a list of integers.')
    if not set(ids) <= valid:
        raise ToolError('bad_id', f'IDs {sorted(set(ids) - valid)[:10]} do not exist on this handle (valid IDs: {sorted(valid)[:64]}; total {len(valid)}). remove_region keeps the IDs of the remaining components; any other edit renumbers them, so inspect the new handle first.')
    return set(ids)
