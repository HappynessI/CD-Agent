"""Reusable exemplar matching: rank -> top k -> compare example and candidate.

No truth or class-transition metadata enters these functions. New prompting is
explicit opt-in; the default remains the existing correspondence question.
"""
import numpy as np
from agent_baseline.backends import CROP_WORDS, CORRESPONDENCE, encode_image
from agent_baseline.scope_t13 import verifier_prompts
from agent_baseline.view import ask_images

T3_SYSTEM = ('Compare land-cover changes in two regions of remote sensing image pairs. '
    'Example and Candidate are separate regions. Each has T1/T2 close-ups and wider views. '
    'The red contour marks the region. Infer land cover before and after in each region. '
    'Reply JSON only. Reverse transitions and unchanged regions are not matches.')
T3_QUESTION = ('Does the Candidate show the same directed land-cover transition as the Example? '
    'Identify what each region was at T1 and becomes at T2. '
    'Reply {"change":"yes" or "no", "example_t1":"class", "example_t2":"class", '
    '"candidate_t1":"class", "candidate_t2":"class", "reason":"short explanation"}.')


def ranked_indices(similarities, k):
    scores = np.asarray(similarities, float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or k <= 0:
        raise ValueError('Require finite similarity vector and positive k')
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Cannot render an empty region')
    return [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]


def build_messages(rgb, exemplar, candidate, ask_variant='default'):
    """Same four-view renderer and encoding as ApiAsk/ask_eval; eight for T3."""
    if ask_variant not in ('default', 't3_transition'):
        raise ValueError('Unknown ask variant')
    content = []
    regions = [('Example', exemplar), ('Candidate', candidate)] if ask_variant == 't3_transition' else [('', candidate)]
    for name, mask in regions:
        for word, image in zip(CROP_WORDS, ask_images(rgb, mask, bbox(mask), True)):
            content += [dict(type='text', text=f'{name} {word}'.strip()),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
    content.append(dict(type='text', text=T3_QUESTION if ask_variant == 't3_transition' else CORRESPONDENCE))
    system = T3_SYSTEM if ask_variant == 't3_transition' else verifier_prompts()['SYSTEM']
    return [dict(role='system', content=system), dict(role='user', content=content)]


def filtered_union(masks, indices, answers, exemplar):
    """Delete explicit no. A returned but unparsable reply is kept (unparsed is not no);
    a missing answer means the run is incomplete."""
    pred = np.zeros(exemplar.shape, bool)
    for i in indices:
        answer = answers[i]['answer']
        if answer not in ('yes', 'no', 'unparsed'):
            raise ValueError('Candidate answer is incomplete')
        if answer != 'no':
            pred |= masks[i]
    return pred & ~exemplar


def match_ranked(masks, similarities, exemplar, rgb, k, verifier, ask_variant='default'):
    """Tool-ready synchronous primitive; verifier accepts messages, returns answer.

    Offline batching reuses these functions and adds persistence/budget control.
    """
    indices = ranked_indices(similarities, k)
    answers = {i: verifier(build_messages(rgb, exemplar, masks[i], ask_variant)) for i in indices}
    return dict(indices=indices, answers=answers,
                mask=filtered_union(masks, indices, answers, exemplar))
