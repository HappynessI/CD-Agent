"""Scope card (口径卡 v0, fixed) and the T3 verify / T5 box prompts."""

import json
from pathlib import Path

CLASS = 'building'
SCOPES = Path(__file__).resolve().parents[1] / 'agent_baseline' / 'scopes'
DIRECTIONS = ('any', 'appeared', 'disappeared')


def scope_card(target='building', change_direction='any', scope=None):
    if change_direction not in DIRECTIONS:
        raise ValueError(f'Unknown change_direction: {change_direction}')
    # A category must have an explicit scope; never silently borrow building's rules.
    path = SCOPES / ((scope or target).removesuffix('.json') + '.json')
    if path.parent != SCOPES or not path.is_file():
        raise ValueError(f'Missing scope card for target {target!r}')
    card = json.loads(path.read_text())
    if card['target'] != target:
        raise ValueError('Scope card target differs from the session target')
    if change_direction != 'any':
        card['counts'] = card['direction_counts'][change_direction]
        excluded = 'disappeared' if change_direction == 'appeared' else 'appeared'
        card['not_counts'] = [*card['not_counts'], f'changes in the {excluded} direction; replacement alone is not evidence of the requested direction']
    return card


def verifier_prompts(target='building', change_direction='any', scope=None):
    card = scope_card(target, change_direction, scope)
    scope = {key: card[key] for key in ('counts', 'not_counts')}
    types = card['pseudo_types']
    system = (
        f'You verify changes of the class "{target}" between two co-registered remote sensing images '
        'of the same place: T1 (earlier) and T2 (later).\n'
        f'Scope card for "{target}".\n'
        f'COUNTS as a {target} change:\n' + ''.join(f'- {x}\n' for x in scope['counts']) +
        'Does NOT count:\n' + ''.join(f'- {x}\n' for x in scope['not_counts']) +
        'Quote the scope card when you decide.')
    suffix = ('{"change": "yes" or "no", "type": one of ' + str(types).replace("'", '"') +
              ', "reason": one short sentence}')
    verify = (
        f'The red contour marks the same region in both close-ups. Is there a {target} change inside the '
        'red contour according to the scope card? Reply with JSON only: ' + suffix)
    correspondence = card['correspondence'] + suffix
    if change_direction != 'any':
        note = f' Only count changes in the {change_direction} direction according to the scope card.'
        verify = verify.replace(' Reply with JSON only:', note + ' Reply with JSON only:')
        correspondence = correspondence.replace(' Reply with JSON only:', note + ' Reply with JSON only:')
    return dict(SCOPE_CARD=scope, PSEUDO_TYPES=types, SYSTEM=system, VERIFY=verify,
                CORRESPONDENCE=correspondence)


# Existing ovcd_v0 callers continue to get the frozen building / any prompts.
_DEFAULT = verifier_prompts()
SCOPE_CARD, PSEUDO_TYPES, SYSTEM, VERIFY = (_DEFAULT[key] for key in
    ('SCOPE_CARD', 'PSEUDO_TYPES', 'SYSTEM', 'VERIFY'))
FRAMING = ('T1 close-up (earlier):', 'T2 close-up (later):')

PREFIX = '{"change": "'

BOX = (
    'The red contour marks a suspicious region in both close-ups. Draw one tight bounding box around '
    'the building that changed between T1 and T2 at this location (the building present in T2 if it '
    'is new, in T1 if it was demolished). Use coordinates of the close-up normalized to 0-1000. '
    'Reply with JSON only: {"bbox_2d": [x1, y1, x2, y2]}. If there is no changed building, reply '
    '{"bbox_2d": []}.')
BOX_PREFIX = '{"bbox_2d": ['


def messages(t1, t2, question):
    content = []
    for label, im in zip(FRAMING, (t1, t2)):
        content += [{'type': 'text', 'text': label}, {'type': 'image', 'image': im}]
    content.append({'type': 'text', 'text': question})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': content}]
