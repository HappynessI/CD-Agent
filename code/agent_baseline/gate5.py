"""Opt-in region clipping after full-image segmentation. No backend changes."""
import copy
from . import instances
from .state import Mask, ToolError


SEGMENT_TOOLS = ('segment_text', 'segment_everything')


def region_spec(spec):
    """Extend a segmentation parameter spec without changing shared schemas."""
    spec = copy.deepcopy(spec)
    spec['properties']['region'] = dict(type='string', description=(
        'Optional same-image mask handle (region#k, mask#k or map#k) to clip full-image segmentation results to.'))
    return spec


def resolve_region(toolbox, region):
    if region is None:
        return None
    try:
        return toolbox.get(region, Mask).data
    except ToolError as exc:
        raise ToolError('bad_handle', f'Invalid region {region!r}: use an existing same-image region#k or mask#k handle. {exc}') from None


def clip_instances(rows, region, data):
    """Keep nonempty intersections, as instances.from_dense does (min area 1).

    Preserve IDs and scores. Recompute geometry without splitting an instance.
    Pixel counts are unique union pixels, matching the tools' public area.
    Never edit the full-image rows supplied by a backend/cache.
    """
    clipped = []
    flat = data.ravel()
    width = data.shape[1]
    for row in rows:
        idx = row['idx'][flat[row['idx']]]
        if not len(idx):
            continue
        yy, xx = idx // width, idx % width
        clipped.append(dict(row, idx=idx, area=int(idx.size),
                            bbox=[int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]))
    before = int(instances.to_mask(rows, data.shape).sum())
    after = int(instances.to_mask(clipped, data.shape).sum())
    stats = dict(region=region, instances_before=len(rows), instances_after=len(clipped),
                 dropped_instances=len(rows) - len(clipped), pixels_before=before,
                 pixels_after=after, clipped_px=before - after, min_area_px=1)
    return clipped, stats


def clip_result(toolbox, result, stats):
    """Attach statistics to recorder provenance and a concise planner result."""
    toolbox.meta['region_clip'] = stats
    result['report'] = (f'Results clipped to {stats["region"]}: removed {stats["dropped_instances"]} instances / '
                        f'{stats["clipped_px"]} pixels.')
    return result
