"""G4 helpers. Read only public session state; never consume reference labels."""
from .state import Mask


TOOL_HINTS = {
    'a': {
        'compare_features': "Class-agnostic change candidates for the whole image: segment all regions, then score how much each region's features changed between the two dates.",
        'segment_semantic': 'A per-class mask for one date. Comparing the masks of the two dates shows where a class appeared or disappeared.',
        'match_exemplar': 'Ranks candidate regions by similarity to the example change. Use it to find changes like the example.',
    },
    'b': {
        'compare_features': 'Use class-agnostic whole-image proposals and temporal feature differences to find change candidates.',
        'segment_semantic': 'Use per-class masks from both dates to find where a class appeared or disappeared.',
        'match_exemplar': 'Use example similarity to rank candidate regions and find changes like the example.',
    },
}


def latest_generated_mask(session, calls):
    """Return (handle, tool) for the last mask newly created by a tool call.

    Scan calls and their handles_created in execution order, regardless of
    set_current. Only mask# handles containing Mask count. Exclude the given
    initial mask and region#0; a tool-created mask#0 without an initial mask counts.
    Merely referencing an existing handle does not make it the latest mask.
    This function does not mutate session or calls.
    """
    latest = None
    for call in calls:
        for key in call.get('handles_created', ()):
            if (key.startswith('mask#') and key != session.initial_mask
                    and key != session.region_mask and isinstance(session.handles.get(key), Mask)):
                latest = (key, call['tool'])
    return latest


def current_answer_reminder(session, calls):
    latest = latest_generated_mask(session, calls)
    if latest is None or latest[0] == session.current_mask:
        return ''
    key, tool = latest
    current = session.current_mask
    answer = (f'Your current answer is {current} ({int(session.get(current, Mask).data.sum())} pixels).'
              if current is not None else 'Your current answer is empty.')
    return (f'{answer} Your most recent mask is {key} ({int(session.get(key, Mask).data.sum())} pixels), '
            f'made by {tool}. Submit the one you mean.')


def remind(result, session, calls):
    """Append to the actual report so both the planner and recorder receive it."""
    text = current_answer_reminder(session, calls)
    if text:
        result.public['report'] += '\n' + text
    return result


def clip_submission(toolbox, key):
    """Apply the user's region once at actual submission, including fallback."""
    session = toolbox.session
    if not toolbox.gate4 or session.region_mask is None:
        return key, ''
    before = session.get(key, Mask).data
    region = session.get(session.region_mask, Mask).data
    clipped = before & region
    removed = int((before & ~region).sum())
    after = session.add_mask(clipped, parents=(key, session.region_mask))
    report = f'Removed {removed} pixels outside region#0 at submission' if removed else ''
    session.g4_submission = dict(pre_region_handle=key,
        pre_region_mask_path=f'masks/{key.replace("#", "_")}.png',
        region_clipped_px=removed, submission_report=report)
    toolbox.meta.update(session.g4_submission)
    return after, report
