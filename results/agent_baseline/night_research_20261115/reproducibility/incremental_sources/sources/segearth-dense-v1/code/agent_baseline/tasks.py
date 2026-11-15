"""Task manifest I/O. File paths and metadata stay on the harness side."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .scope_t13 import scope_card
from .map_input import INSTRUCTION as MAP_INSTRUCTION

INSTRUCTIONS = {
    'T1.2xC0': 'Find changes involving buildings or low vegetation between T1 and T2. Output a binary change mask.',
    'T1.1xC1t': 'Find buildings newly built on low vegetation or bare ground between T1 and T2. Output a binary change mask.',
    'T1.3': 'Find all land-cover changes between T1 and T2. Output a binary change mask.',
    'T1.2': 'Find changes involving the requested land-cover classes between T1 and T2.',
    'T3.1': 'Find all changes similar to the marked example between T1 and T2.',
    'T1.1xC0': 'Find all buildings that changed between T1 and T2 (newly built or demolished). Output a binary change mask.',
    'T5.1': 'Here is a building change detection mask (mask#0). Check it against T1 (earlier) and T2 (later), and correct it where needed. Submit the final binary change mask and briefly summarize your decision.'}
C1_INSTRUCTIONS = {
    'T1.1xC1': {
        'appeared': 'Find the buildings that were newly built between T1 and T2.',
        'disappeared': 'Find the buildings that were demolished between T1 and T2.'},
    'T5.1xC1': {direction: 'Here is a building change detection mask (mask#0) that marks both newly built and demolished buildings. Check it against T1 (earlier) and T2 (later), correct it where needed, and keep only the buildings that were ' + verb + '. Submit the final binary change mask and briefly summarize your decision.'
        for direction, verb in (('appeared', 'newly built'), ('disappeared', 'demolished'))}}
INSTRUCTIONS.update({task: values['appeared'] for task, values in C1_INSTRUCTIONS.items()})
BOX_INSTRUCTIONS = {
    'T1.1xC2': 'Find the building changes inside the region region#0 (box {box}).',
    'T1.1xC4': 'Find the buildings that were newly built inside the region region#0 (box {box}).'}
BOX_DIRECTIONS = {'T1.1xC2': 'any', 'T1.1xC4': 'appeared'}
INSTRUCTIONS.update(BOX_INSTRUCTIONS)
from .examples_t3x import INSTRUCTIONS as EXTERNAL_INSTRUCTIONS, resolve_examples
INSTRUCTIONS.update(EXTERNAL_INSTRUCTIONS)
INSTRUCTIONS['T5.4'] = MAP_INSTRUCTION
# T4.1 size attribute: the manifest carries the full instruction with its area threshold (analysis/make_t41_tasks.py).
INSTRUCTIONS['T4.1'] = 'Find building changes whose footprint is larger than a given area between T1 and T2. Output a binary change mask.'


def t52_instruction(feedback):
    """Only point coordinates and user error types enter the planner instruction."""
    phrases = {'missed': 'a building change is missed here.',
               'false_positive': 'this is not a building change.'}
    clicks = []
    for item in feedback:
        x, y = item['point']
        clicks.append(f'The user clicked ({x}, {y}): ' + phrases[item['type']])
    review = ' '.join(clicks) if clicks else 'The user reviewed the mask and pointed out no errors.'
    requirement = ('Correct only the places the user pointed out and keep the rest of mask#0 unchanged.'
                   if clicks else 'Keep mask#0 unchanged.')
    return INSTRUCTIONS['T5.1'] + ' User feedback: ' + review + ' ' + requirement


INSTRUCTIONS['T5.2'] = t52_instruction([])


def t53_instruction(feedback):
    from .feedback import CELLS
    if len(feedback) > 1:
        raise ValueError('T5.3 allows at most one feedback sentence')
    phrases = {'missed': 'A building in the {where} was missed.',
               'false_positive': 'One of the marked changes in the {where} is not a building change.'}
    for item in feedback:
        if item['where'] not in CELLS:
            raise ValueError('Unknown T5.3 cell')
    review = (phrases[feedback[0]['type']].format(where=feedback[0]['where']) if feedback
              else 'The user reviewed the mask and pointed out no errors.')
    requirement = ('Correct only what the user described and keep the rest of mask#0 unchanged.'
                   if feedback else 'Keep mask#0 unchanged.')
    return INSTRUCTIONS['T5.1'] + ' User feedback: ' + review + ' ' + requirement


INSTRUCTIONS['T5.3'] = t53_instruction([])


def needs_init(task):
    return task in ('T5.1', 'T5.1xC1', 'T5.2', 'T5.3')


FIELDS = {'id', 'image_t1', 'image_t2', 'instruction', 'task_type', 'constraint', 'target',
          'change_direction', 'init_mask', 'gt_mask', 'meta'}


def instruction_for(task, session):
    if task == 'T1.2xC0' and (session.target != ('building', 'low vegetation') or session.change_direction != 'any'):
        raise ValueError('T1.2xC0 requires building and low vegetation in both directions')
    if task == 'T1.3' and (session.target != 'any change' or session.change_direction != 'any'):
        raise ValueError('T1.3 requires any change in both directions')
    if task == 'T1.1xC1t' and session.source_classes is None:
        raise ValueError('T1.1xC1t requires source_classes')
    if task in BOX_INSTRUCTIONS:
        if (session.change_direction != BOX_DIRECTIONS[task] or session.target != 'building'
                or session.scope != 'building_rebuild_excluded' or session.region_bbox is None):
            raise ValueError('C2/C4 requires a nonempty region, building scope and the task direction')
    if task in C1_INSTRUCTIONS and session.change_direction not in ('appeared', 'disappeared'):
        raise ValueError('C1 requires appeared or disappeared')
    if session.instruction is not None:
        text = session.instruction
    elif task in BOX_INSTRUCTIONS:
        text = BOX_INSTRUCTIONS[task].format(box=session.region_bbox)
    elif task in C1_INSTRUCTIONS:
        text = C1_INSTRUCTIONS[task][session.change_direction]
    elif task in ('T1.2xC0', 'T1.1xC1t'):
        text = INSTRUCTIONS[task]
    else:
        card = scope_card(session.target, session.change_direction, session.scope)
        text = INSTRUCTIONS[task].replace('buildings', card['plural']).replace('building', card['target'])
        if session.change_direction != 'any':
            text = text.replace('(newly built or demolished)', f'({session.change_direction} only)')
            text += f' Count only changes in the {session.change_direction} direction.'
        elif session.target != 'building':
            text = text.replace('(newly built or demolished)', '(appeared or disappeared)')
    if session.constraint:
        constraint = session.constraint if isinstance(session.constraint, str) else json.dumps(session.constraint, ensure_ascii=False)
        text += '\nConstraint: ' + constraint
    return text


def seed_for(name, offset=0):
    return (sum((i + 1) * ord(c) for i, c in enumerate(name)) + 1000 * offset) % (2**31)


def read_tasks(filename, task=None):
    path = Path(filename).resolve()
    rows, seen = [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = FIELDS - row.keys()
        if missing:
            raise ValueError(f'Task line {n} missing fields: {sorted(missing)}')
        key = (row['task_type'], row['change_direction'], row['id'])
        if key in seen:
            raise ValueError(f'Duplicate task: {key}')
        seen.add(key)
        # IDs are output directory names and may be image filenames, never directory paths.
        if not row['id'] or Path(row['id']).name != row['id'] or row['id'] in ('.', '..'):
            raise ValueError(f'Task line {n}: id must be a filename or stem')
        if row['task_type'] not in INSTRUCTIONS:
            raise ValueError(f'Unsupported task_type: {row["task_type"]}')
        row = resolve_examples(row, path.parent)
        row = dict(row, target=tuple(row['targets']) if row.get('targets') else row['target'] or 'building', change_direction=row['change_direction'] or 'any')
        scope_card(row['target'], row['change_direction'], row.get('scope'))
        if row['task_type'] == 'T5.4':
            from .map_input import check_task
            check_task(row)
        if row.get('ask_variant') is not None and row['task_type'] != 'T1.3':
            raise ValueError('ask_variant is only used by T1.3')
        if row['task_type'] in ('T1.2xC0', 'T1.1xC1t', 'T1.3'):
            from .state import Session
            session = Session((1, 1), target=row['target'], change_direction=row['change_direction'],
                scope=row.get('scope'), source_classes=row.get('source_classes'), ask_variant=row.get('ask_variant'))
            instruction_for(row['task_type'], session)
        if row['task_type'] in C1_INSTRUCTIONS and row['change_direction'] not in ('appeared', 'disappeared'):
            raise ValueError('C1 requires appeared or disappeared')
        if row['task_type'] in BOX_INSTRUCTIONS:
            if (not row.get('region_mask') or row['change_direction'] != BOX_DIRECTIONS[row['task_type']]
                    or row['target'] != 'building' or row.get('scope') != 'building_rebuild_excluded'):
                raise ValueError('C2/C4 requires region_mask, building scope and the task direction')
        for field in ('image_t1', 'image_t2', 'init_mask', 'gt_mask', 'gt_ignore_mask', 'region_mask', *(['exemplar_mask'] if row.get('exemplar_mask') else [])):
            if row.get(field) is not None:
                row[field] = str((path.parent / row[field]).resolve())
        if not row['gt_mask']:
            raise ValueError('gt_mask is required for recorder scoring')
        if needs_init(row['task_type']) and not row['init_mask']:
            raise ValueError('T5.1 requires init_mask')
        if row['task_type'] == 'T5.2':
            feedback = row['meta']['feedback']
            if row['target'] != 'building' or row['change_direction'] != 'any':
                raise ValueError('T5.2 uses LEVIR building changes in both directions')
            if row['instruction'] != t52_instruction(feedback):
                raise ValueError('T5.2 instruction must match the point feedback')
        if row['task_type'] == 'T5.3':
            if row['target'] != 'building' or row['change_direction'] != 'any':
                raise ValueError('T5.3 uses LEVIR building changes in both directions')
            if row['instruction'] != t53_instruction(row['meta']['feedback']):
                raise ValueError('T5.3 instruction must match the language feedback')
        if task is None or row['task_type'] == task:
            rows.append(row)
    return rows


def image_name(row):
    """Resolve this ID to its source image filename; preserve the legacy filename seed."""
    return Path(row['image_t1']).name


def levir_cache_name(row, split):
    """Check both source paths before using the dataset split/name cache namespace."""
    from common import LEVIR
    name = image_name(row)
    expected = [LEVIR / f'{split}_original' / part / name for part in ('A', 'B')]
    if [Path(row[key]).resolve() for key in ('image_t1', 'image_t2')] != [p.resolve() for p in expected]:
        raise ValueError('T5.2 source pair must match the LEVIR split/name cache')
    return name


def load_pair(row):
    rgb = [np.asarray(Image.open(row[key]).convert('RGB')) for key in ('image_t1', 'image_t2')]
    if rgb[0].shape != rgb[1].shape:
        raise ValueError('T1 and T2 image sizes differ')
    return rgb
