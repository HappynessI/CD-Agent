"""Opt-in external examples. No evaluation labels or dataset metadata enter Session."""
from dataclasses import dataclass, field
from pathlib import Path
import copy
import numpy as np
from PIL import Image
from render import edge, SIZE
from .state import ToolError
from .tools_t13 import Features, pool_features, pooled, cosine

INSTRUCTIONS = {
    'T3.2': 'Find changes where targets like the positive external example appeared in T2. Output a binary change mask.',
    'T3.3': 'Find all changes with the same directed transition as the positive external before/after example. Output a binary change mask.',
    'T3.4': 'Find all changes like the positive external before/after example, excluding changes like the negative example. Output a binary change mask.',
}


@dataclass(frozen=True)
class Example:
    images: dict  # t1 and/or t2 -> unannotated RGB crop, native crop resolution
    mask: object = None
    role: str = 'positive'
    backend: object = None
    features: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ExampleMask:
    data: np.ndarray
    example: str


def resolve_examples(row, parent):
    """Resolve only declared input files; transition/source metadata stays in the manifest."""
    examples = row.get('examples')
    if row['task_type'] in INSTRUCTIONS:
        roles = ['positive', 'negative'] if row['task_type'] == 'T3.4' else ['positive']
        if not examples or [e.get('role') for e in examples] != roles:
            raise ValueError('External task requires ordered positive/negative examples')
        expected = ('image_t2',) if row['task_type'] == 'T3.2' else ('image_t1', 'image_t2')
        if any(any(not e.get(k) for k in expected) for e in examples):
            raise ValueError('External example is missing its required date')
    if examples is not None:
        row['examples'] = [dict(e, **{k: str((parent / e[k]).resolve())
            for k in ('image_t1', 'image_t2', 'mask') if e.get(k)}) for e in examples]
    return row


def load_examples(specs, pool=None):
    """Whitelist RGB, example mask and role; never load the evaluation truth."""
    out = []
    for spec in specs:
        images = {t: np.asarray(Image.open(spec['image_' + t]).convert('RGB'))
                  for t in ('t1', 't2') if spec.get('image_' + t)}
        mask = np.asarray(Image.open(spec['mask'])) > 0 if spec.get('mask') else None
        paths = [spec.get('image_t1') or spec['image_t2'], spec.get('image_t2') or spec['image_t1']]
        out.append(Example(images, mask, spec['role'], pool.for_pair(paths) if pool else None))
    return out


def attach_examples(session, examples=None):
    """No examples leaves all state and requests unchanged. Input masks have crop coordinates."""
    if not examples:
        return
    handles = []
    for value in examples:
        if value.role not in ('positive', 'negative') or not value.images or not set(value.images) <= {'t1', 't2'}:
            raise ValueError('Example requires dates and a positive/negative role')
        images = {t: np.array(im, dtype=np.uint8, copy=True) for t, im in value.images.items()}
        shape = next(iter(images.values())).shape[:2]
        if any(im.ndim != 3 or im.shape != (*shape, 3) for im in images.values()):
            raise ValueError('Example dates must have matching RGB crop dimensions')
        mask = None if value.mask is None else np.array(value.mask, dtype=bool, copy=True)
        if mask is not None and (mask.shape != shape or not mask.any()):
            raise ValueError('Example mask must be nonempty and match its own crop')
        for im in images.values():
            im.flags.writeable = False
        if mask is not None:
            mask.flags.writeable = False
        key = session._put('example', Example(images, mask, value.role, value.backend))
        mask_key = session._put('exmask', ExampleMask(mask, key)) if mask is not None else None
        handles.append((key, mask_key))
    session.external_examples = tuple(handles)


def example_blocks(session, directory):
    """Appended after the evaluation T1/T2 view. Positive then negative; T1 then T2."""
    blocks = []
    directory = Path(directory) / 'examples'
    for key, mask_key in getattr(session, 'external_examples', ()):
        value = session.get(key, Example)
        dates = ', '.join(t.upper() for t in ('t1', 't2') if t in value.images)
        mode = 'change' if len(value.images) == 2 else next(iter(value.images))
        text = (f'External {value.role} example: {key}; dates: {dates}. '
                f'Use match_exemplar(exemplar={{"example":"{key}"}}, mode="{mode}"). ')
        text += (f'Example mask: {mask_key}, shown by the red contour. ' if mask_key else 'Use the whole crop. ')
        text += 'Example coordinates belong only to this crop; submit a mask for the evaluation pair.'
        blocks.append(dict(type='text', text=text))
        for t in ('t1', 't2'):
            if t not in value.images:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            image = np.array(Image.fromarray(value.images[t]).resize((SIZE, SIZE), Image.Resampling.BICUBIC))
            if value.mask is not None:
                image[edge(value.mask)] = (255, 0, 0)
            path = directory / f'{key.replace("#", "_")}_{t}.jpg'
            Image.fromarray(image).save(path, quality=92)
            blocks.extend([dict(type='text', text=f'{key} {value.role} {t.upper()}'),
                           dict(type='image_url', image_url=dict(url=str(path)))])
    if len(getattr(session, 'external_examples', ())) == 2:
        blocks.append(dict(type='text', text='For positive/negative matching, pass negative={"example":"'
                           + session.external_examples[1][0] + '"}; score = cosine(positive) - cosine(negative).'))
    return blocks


def extend_schema(schemas, session):
    if not getattr(session, 'external_examples', ()):
        return schemas
    schemas = copy.deepcopy(schemas)
    for entry in schemas:
        f = entry['function']
        if f['name'] == 'match_exemplar':
            region = f['parameters']['properties']['exemplar']
            region['properties']['example'] = dict(type='string')
            f['parameters']['properties']['negative'] = copy.deepcopy(region)
            f['description'] += (' External exemplar accepts {"example":"example#k"} or {"mask":"exmask#k"}. '
                'Use its own crop features; a single T2 example requires mode=t2. Optional negative uses '
                'the same mode; score=positive cosine minus negative cosine, range [-2,2].')
    return schemas


def external_spec(session, name, default):
    if name != 'match_exemplar' or not getattr(session, 'external_examples', ()):
        return default
    return extend_schema([dict(function=dict(name=name, description='', parameters=default))], session)[0]['function']['parameters']


def example_vector(value, mode):
    dates = ('t1', 't2') if mode == 'change' else (mode,)
    if any(t not in value.images for t in dates):
        raise ToolError('bad_args', 'Example does not contain the dates required by mode.')
    if value.backend is None:
        raise ToolError('bad_args', 'External example feature backend is required.')
    shape = next(iter(value.images.values())).shape[:2]
    idx = np.flatnonzero(value.mask) if value.mask is not None else np.arange(np.prod(shape))
    vectors = []
    for t in dates:
        if t not in value.features:
            value.features[t] = Features(np.array(value.backend('extract_features', time=t), copy=True), shape, t)
        vectors.append(pool_features(value.features[t], idx))
    return vectors[1] - vectors[0] if mode == 'change' else vectors[0]


def combine_scores(positive, negative=None):
    return np.asarray(positive) if negative is None else np.asarray(positive) - np.asarray(negative)


def match_external(toolbox, props, exemplar, mode, negative=None):
    from .state import Instances
    item = toolbox.get(props, Instances)
    parents = [props]
    def reference(spec):
        if set(spec) == {'example'}:
            key = spec['example']
            value = toolbox.get(key, Example)
        elif set(spec) == {'mask'}:
            m = toolbox.get(spec['mask'], ExampleMask)
            key = m.example
            value = toolbox.get(key, Example)
            parents.append(spec['mask'])
        else:
            raise ToolError('bad_args', 'External matching requires one example or exmask handle.')
        parents.append(key)
        return example_vector(value, mode)
    positive = reference(exemplar)
    neg = reference(negative) if negative is not None else None
    vectors = []
    for t in (('t1', 't2') if mode == 'change' else (mode,)):
        key, feature = toolbox._feature(t)
        parents.append(key)
        vectors.append(pooled(feature, item.items))
    candidates = vectors[1] - vectors[0] if mode == 'change' else vectors[0]
    scores = combine_scores(cosine(candidates, positive), cosine(candidates, neg) if neg is not None else None)
    rows = [dict(r, score=float(s)) for r, s in zip(item.items, scores)]
    result = toolbox._props(rows, item.time, tuple(parents))
    result.update(mode=mode, similarities=[dict(id=r['id'], score=r['score'])
        for r in sorted(rows, key=lambda r: (-r['score'], r['id']))])
    return result


def archive_example(directory, key, value):
    """Recorder-only assets; return False for legacy handle types."""
    if not isinstance(value, (Example, ExampleMask)):
        return False
    directory = Path(directory) / 'examples'
    directory.mkdir(exist_ok=True)
    stem = key.replace('#', '_')
    if isinstance(value, ExampleMask):
        Image.fromarray(value.data.astype(np.uint8) * 255).save(directory / f'{stem}.png')
    else:
        for t, image in value.images.items():
            Image.fromarray(image).save(directory / f'{stem}_{t}.png')
    return True
