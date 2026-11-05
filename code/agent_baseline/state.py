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
    def __init__(self, shape, init_mask=None, target="building", change_direction="any", instruction=None, constraint=None, scope=None):
        from prompts import scope_card
        scope_card(target, change_direction, scope)
        self.scope = scope
        self.target, self.change_direction = target, change_direction
        self.instruction, self.constraint = instruction, constraint
        self.shape = tuple(shape)
        self.handles = {}
        self.next_id = 0
        self.current_mask = None
        self.initial_mask = None
        self.submitted = None
        self.summary = None
        self.view_count = 0
        if init_mask is not None:
            self.initial_mask = self.add_mask(init_mask)
            self.current_mask = self.initial_mask

    def _put(self, prefix, value):
        key = f'{prefix}#{self.next_id}'
        self.next_id += 1
        self.handles[key] = value
        return key

    def add_mask(self, data, parents=(), labeled=None):
        data = np.array(data, dtype=bool, copy=True)
        if data.shape != self.shape:
            raise ValueError('backend mask shape differs from image shape')
        # labeled keeps existing component IDs when an edit removes whole components.
        labels, comps = labeled if labeled is not None else components(data)
        data.flags.writeable = labels.flags.writeable = False
        comps = tuple(MappingProxyType(dict(c, bbox=tuple(c['bbox']))) for c in comps)
        return self._put('mask', Mask(data, labels, comps, tuple(parents)))

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
            raise ToolError('bad_handle', f'Expected a {"mask" if kind is Mask else "inst"} handle.')
        return value


def chosen_ids(rows, ids):
    valid = {r['id'] for r in rows}
    if ids is None:
        return valid
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ToolError('bad_args', 'ids must be a list of integers.')
    if not set(ids) <= valid:
        raise ToolError('bad_id', f'IDs {sorted(set(ids) - valid)[:10]} do not exist on this handle ({len(valid)} valid IDs). remove_region keeps the IDs of the remaining components; any other edit renumbers them, so inspect the new handle first.')
    return set(ids)
