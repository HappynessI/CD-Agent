"""Task manifest I/O. File paths and metadata stay on the harness side."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from prompts import scope_card

INSTRUCTIONS = {
    'T1.1xC0': 'Find all buildings that changed between T1 and T2 (newly built or demolished). Output a binary change mask.',
    'T5.1': 'Here is a building change detection mask (mask#0). Check it against T1 (earlier) and T2 (later), and correct it where needed. Submit the final binary change mask and briefly summarize your decision.'}
C1_INSTRUCTIONS = {
    'T1.1xC1': {
        'appeared': 'Find the buildings that were newly built between T1 and T2.',
        'disappeared': 'Find the buildings that were demolished between T1 and T2.'},
    'T5.1xC1': {direction: 'Here is a building change detection mask (mask#0) that marks both newly built and demolished buildings. Check it against T1 (earlier) and T2 (later), correct it where needed, and keep only the buildings that were ' + verb + '. Submit the final binary change mask and briefly summarize your decision.'
        for direction, verb in (('appeared', 'newly built'), ('disappeared', 'demolished'))}}
INSTRUCTIONS.update({task: values['appeared'] for task, values in C1_INSTRUCTIONS.items()})


def needs_init(task):
    return task in ('T5.1', 'T5.1xC1')


FIELDS = {'id', 'image_t1', 'image_t2', 'instruction', 'task_type', 'constraint', 'target',
          'change_direction', 'init_mask', 'gt_mask', 'meta'}


def instruction_for(task, session):
    if task in C1_INSTRUCTIONS and session.change_direction not in ('appeared', 'disappeared'):
        raise ValueError('C1 requires appeared or disappeared')
    if session.instruction is not None:
        text = session.instruction
    elif task in C1_INSTRUCTIONS:
        text = C1_INSTRUCTIONS[task][session.change_direction]
    else:
        card = scope_card(session.target, session.change_direction, session.scope)
        text = INSTRUCTIONS[task].replace('buildings', card['plural']).replace('building', session.target)
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
        row = dict(row, target=row['target'] or 'building', change_direction=row['change_direction'] or 'any')
        scope_card(row['target'], row['change_direction'], row.get('scope'))
        if row['task_type'] in C1_INSTRUCTIONS and row['change_direction'] not in ('appeared', 'disappeared'):
            raise ValueError('C1 requires appeared or disappeared')
        for field in ('image_t1', 'image_t2', 'init_mask', 'gt_mask'):
            if row[field] is not None:
                row[field] = str((path.parent / row[field]).resolve())
        if not row['gt_mask']:
            raise ValueError('gt_mask is required for recorder scoring')
        if needs_init(row['task_type']) and not row['init_mask']:
            raise ValueError('T5.1 requires init_mask')
        if task is None or row['task_type'] == task:
            rows.append(row)
    return rows


def image_name(row):
    """Resolve this ID to its source image filename; preserve the legacy filename seed."""
    return Path(row['image_t1']).name


def load_pair(row):
    rgb = [np.asarray(Image.open(row[key]).convert('RGB')) for key in ('image_t1', 'image_t2')]
    if rgb[0].shape != rgb[1].shape:
        raise ValueError('T1 and T2 image sizes differ')
    return rgb
