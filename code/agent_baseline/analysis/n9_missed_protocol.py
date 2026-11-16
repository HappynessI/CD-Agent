"""N9 task-only profiles and frozen sparse search space (no GT dependency)."""
import random

SEED = 20261116
SOURCES = ('S1_02', 'S2', 'S3', 'S4', 'S5')
PROFILE_GRIDS = {
    'any': dict(semantic=(.2, .4), dino=(.4, .6), alpha=(0.,), variants=('V2',), k=(1, 3)),
    'build': dict(semantic=(.05, .2), dino=(.2,), alpha=(0., .5), variants=('V0', 'V3', 'V4'), k=(5, 10)),
    'demolish': dict(semantic=(0., .05, .2), dino=(.2,), alpha=(0.,), variants=('V0', 'V4'), k=(10, 15)),
}


def profile_for(direction):
    return {'any': 'any', 'appeared': 'build', 'build': 'build',
            'disappeared': 'demolish', 'demolish': 'demolish'}[direction]


def source_set(profile):
    return ('S1_02', 'S2', 'S5') if profile == 'demolish' else SOURCES


def split_names(names, seed=SEED):
    names = sorted(names)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate physical images')
    random.Random(seed).shuffle(names)
    return {name: 'tune' if i < len(names)//2 else 'confirm' for i, name in enumerate(names)}


def online_names(split):
    names = sorted(k for k, v in split.items() if v == 'confirm')
    random.Random(SEED + 1).shuffle(names)
    if len(names) < 10:
        raise ValueError('Need ten confirmation images before scoring')
    return names[:10]


def configurations(profile):
    from itertools import product
    g = PROFILE_GRIDS[profile]
    keys = ('semantic', 'dino', 'alpha', 'variants', 'k')
    source_options = ('all', 'S1_S2_S5') if profile == 'any' else ('S1_S2_S5',) if profile == 'demolish' else ('all',)
    return [dict(profile=profile, semantic=t, dino=d, alpha=a, verifier=v, k=k, sources=s)
            for t, d, a, v, k in product(*(g[key] for key in keys)) for s in source_options]


def candidate_indices(candidates, config):
    # Dataset/group/task IDs are intentionally absent from this API.
    enabled = SOURCES if config.get('sources', 'all') == 'all' else ('S1_02', 'S2', 'S5')
    accepted = [c for c in candidates if (set(c.get('sources', SOURCES)) & set(enabled))
                and c['semantic_delta'] >= config['semantic']
                and c['dino'] >= config['dino']]
    accepted.sort(key=lambda c: (-(max(c['semantic_delta'], 0.) * max(c['dino'], 0.)
                                  * c['area'] ** config['alpha']), c['index']))
    return [c['index'] for c in accepted[:config['k']]]


def choose_config(rows):
    """Rows each contain one config and separate dataset-line tune summaries."""
    if any(r['split'] != 'tune' for r in rows):
        raise ValueError('Selection only consumes tune summaries')
    expected = set(rows[0]['lines'])
    if len(expected) != 2 or any(set(r['lines']) != expected for r in rows):
        raise ValueError('Each profile requires both independently scored tune lines')
    def key(r):
        delta = sum(v['delta_f1'] for v in r['lines'].values()) / len(expected)
        calls = sum(v['calls'] for v in r['lines'].values())
        return (delta, -calls, -r['config']['k'])
    return max(rows, key=key)  # final ties retain predeclared grid order
