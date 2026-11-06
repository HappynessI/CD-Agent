"""Scope extensions; delegate the frozen single-class path byte for byte."""
import json
from pathlib import Path
from prompts import scope_card as original_card, verifier_prompts as original_prompts


def scope_card(target='building', change_direction='any', scope=None):
    if target != 'any change' and not isinstance(target, (tuple,list)):
        return original_card(target, change_direction, scope)
    if change_direction != 'any':
        raise ValueError('Unrestricted and multiclass scopes require change_direction=any')
    card = json.loads((Path(__file__).parent/'scopes'/'any_change.json').read_text())
    if isinstance(target,(tuple,list)):
        if not target or any(not isinstance(x,str) or not x.strip() for x in target):
            raise ValueError('targets must be a nonempty list of class names')
        text = ', '.join(target)
        card.update(target=text, plural=text)
        card['counts'] = [f'a land-cover change involving at least one of these classes: {text}']
        card['not_counts'].append('changes involving none of the requested classes')
    return card


def verifier_prompts(target='building', change_direction='any', scope=None):
    if target != 'any change' and not isinstance(target,(tuple,list)):
        return original_prompts(target,change_direction,scope)
    card = scope_card(target,change_direction,scope)
    label = 'any land-cover change, without a category restriction' if target == 'any change' else 'a change involving any of: ' + ', '.join(target)
    suffix = '{"change": "yes" or "no", "type": one of ' + json.dumps(card['pseudo_types']) + ', "reason": one short sentence}'
    system = 'You verify ' + label + ' between T1 (earlier) and T2 (later).\nCOUNTS:\n' + '\n'.join('- '+x for x in card['counts']) + '\nDoes NOT count:\n' + '\n'.join('- '+x for x in card['not_counts']) + '\nQuote the scope card when you decide.'
    question = 'The red contour marks the same region in both close-ups. Is there ' + label + ' inside it according to the scope card? Reply with JSON only: ' + suffix
    return dict(SCOPE_CARD={k:card[k] for k in ('counts','not_counts')}, PSEUDO_TYPES=card['pseudo_types'], SYSTEM=system, VERIFY=question, CORRESPONDENCE='The first two images are close-ups and the last two show the same place with more surroundings. '+question)
