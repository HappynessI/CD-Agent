"""Section 10.12, class 1: tasks derived from SECOND labels (R1 purpose, R2 C3 exclusion, R3 relation statistics).

Labels only (slabel1/slabel2: 1 low vegetation, 2 non-vegetated ground surface, 3 tree, 4 water, 5 building,
6 playground; 0 = unchanged and unlabelled). No model output is read. Seed 1016.
Images: report = the 128 SECOND report images (second_val128.txt, drawn from the 297 validation tiles);
development = 64 train tiles outside the 297 validation tiles (so outside second_dev and the T3 example pool).
R1: one purpose per image (least used so far among purposes with nonempty truth, seeded ties); about 20% of
images get a purpose whose truth is empty, to check that the Agent does not report changes that are not there.
R2: T1.3 truth minus pixels where both dates are in {low vegetation, tree}.
R3: new buildings (T1 != 5, T2 = 5), 8-connected, kept whole when within D pixels of a new playground
(T2 = 6, T1 != 6) or new water (T2 = 4, T1 != 4); statistics over all train tiles.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

DATASET = Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/SECOND')
RUN = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
VALIDATION = Path('/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/exp6/second_cv/verifier_val.txt')
SEED, EMPTY_SHARE, D = 1016, 0.2, 20
EIGHT = np.ones((3, 3), bool)
VEG, BUILT = {1, 3}, {5, 6}
# Draft mapping (Ethan reviews and may edit in the review page before any Agent run).
PURPOSES = {
    'ecology': dict(t1=VEG, t2_not=VEG, text='For an ecological protection check, find places where vegetation (grass, crops or trees) was destroyed between T1 and T2.'),
    'urban': dict(t1_not=BUILT, t2=BUILT, text='For urban expansion monitoring, find newly added built-up land (buildings or sports grounds) between T1 and T2.'),
    'water_loss': dict(t1={4}, t2_not={4}, text='To check encroachment on rivers and lakes, find places where water surface was occupied or lost between T1 and T2.'),
    'water_gain': dict(t1_not={4}, t2={4}, text='Find newly appeared water surfaces between T1 and T2, such as newly dug ponds or new reservoirs.'),
    'bare_soil': dict(t1=VEG, t2={2}, text='For dust control, find vegetated land that became bare, non-vegetated ground between T1 and T2.'),
    'greening': dict(t1={2, 5, 6}, t2=VEG, text='To assess greening work, find newly added green space (grass, crops or trees) between T1 and T2.'),
}
SUFFIX = ' Output a binary change mask.'
C3_TEXT = 'Find all land-cover changes between T1 and T2, except changes between low vegetation and trees. Output a binary change mask.'


def labels(name, dataset=DATASET):
    out = []
    for part in ('slabel1', 'slabel2'):
        with Image.open(Path(dataset) / part / name) as im:
            out.append(np.asarray(im).astype(np.int16))
    return out


def changed(a, b):
    return (a != 0) & (b != 0) & (a != b)


def member(x, spec, key):
    if key in spec:
        return np.isin(x, list(spec[key]))
    if key + '_not' in spec:
        return ~np.isin(x, list(spec[key + '_not']))
    return np.ones_like(x, bool)


def purpose_truth(a, b, purpose):
    spec = PURPOSES[purpose]
    return changed(a, b) & member(a, spec, 't1') & member(b, spec, 't2')


def c3_truth(a, b):
    return changed(a, b) & ~(np.isin(a, list(VEG)) & np.isin(b, list(VEG)))


def r3_truth(a, b, d=D):
    new_building = (a != 5) & (b == 5) & (a != 0)
    refs = {'playground': (a != 6) & (b == 6) & (a != 0), 'water': (a != 4) & (b == 4) & (a != 0)}
    lab, n = ndi.label(new_building, EIGHT)
    out = {}
    for key, ref in refs.items():
        if not ref.any() or n == 0:
            out[key] = np.zeros_like(new_building)
            continue
        near = ndi.distance_transform_edt(~ref) <= d
        keep = np.unique(lab[near & new_building])
        out[key] = np.isin(lab, keep[keep > 0])
    return out


def save(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, np.uint8) * 255).save(path)


def assign_purposes(names, truths, seed=SEED):
    rng = random.Random(seed)
    empty_slots = set(rng.sample(sorted(names), round(EMPTY_SHARE * len(names))))
    # Empty-truth slots are balanced with their own counter, so they spread over all purposes instead of
    # piling up on rare purposes (water) that are empty on most images.
    used, used_empty, out = Counter(), Counter(), {}
    for name in sorted(names):
        nonempty = [p for p in PURPOSES if truths[name][p]]
        empty = [p for p in PURPOSES if not truths[name][p]]
        if (name in empty_slots and empty) or not nonempty:
            low = min(used_empty[p] for p in empty)
            choice = rng.choice([p for p in empty if used_empty[p] == low])
            used_empty[choice] += 1
        else:
            low = min(used[p] for p in nonempty)
            choice = rng.choice([p for p in nonempty if used[p] == low])
        used[choice] += 1
        out[name] = choice
    return out, used


def r1_rows(names, phase, run, dataset):
    truths, sizes = {}, {}
    for name in names:
        a, b = labels(name, dataset)
        masks = {p: purpose_truth(a, b, p) for p in PURPOSES}
        truths[name] = {p: bool(m.any()) for p, m in masks.items()}
        sizes[name] = {p: int(m.sum()) for p, m in masks.items()}
    chosen, used = assign_purposes(names, truths)
    rows = []
    for name in sorted(names):
        p = chosen[name]
        a, b = labels(name, dataset)
        gt = run / 'construct/r1/gt' / phase / name
        save(gt, purpose_truth(a, b, p))
        rows.append(dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
            task_type='T4.3', instruction=PURPOSES[p]['text'] + SUFFIX, constraint=None, target='any change',
            change_direction='any', scope=None, init_mask=None, gt_mask=str(gt),
            meta=dict(purpose=p, truth_px=sizes[name][p], empty_truth=sizes[name][p] == 0)))
    return rows, dict(used), {p: sum(1 for n in names if truths[n][p]) for p in PURPOSES}


R3_TEXT = {'playground': 'Find newly built buildings within {d} pixels of a newly built sports ground (playground) between T1 and T2.',
           'water': 'Find newly built buildings within {d} pixels of a newly appeared water surface between T1 and T2.'}


def r3_tasks(run, dataset, r3_names, validation, excluded, report_size=128, dev_size=64):
    """Report: every validation tile with nonempty R3 truth, topped up from other train tiles; development
    from the remaining nonempty train tiles. One reference per image (least used, seeded ties)."""
    rng = random.Random(SEED)
    nonempty = sorted(set(r3_names['playground']) | set(r3_names['water']))
    in_val = [n for n in nonempty if n in validation]
    rest = [n for n in nonempty if n not in validation and n not in excluded]
    extra = sorted(rng.sample(rest, min(len(rest), report_size - len(in_val))))
    report = sorted(in_val + extra)
    dev = sorted(rng.sample([n for n in rest if n not in set(extra)], min(dev_size, len(rest) - len(extra))))
    out, info = run / 'construct/r3', {}
    for phase, names in (('report', report), ('dev', dev)):
        used, rows = Counter(), []
        for name in names:
            a, b = labels(name, dataset)
            t = r3_truth(a, b)
            pool = [k for k in ('playground', 'water') if t[k].any()]
            low = min(used[k] for k in pool)
            ref = rng.choice([k for k in pool if used[k] == low])
            used[ref] += 1
            gt = out / 'gt' / phase / name
            save(gt, t[ref])
            rows.append(dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
                task_type='T4.4', instruction=R3_TEXT[ref].format(d=D) + SUFFIX, constraint=None, target='building',
                change_direction='appeared', scope=None, init_mask=None, gt_mask=str(gt),
                meta=dict(reference=ref, d_pixels=D, truth_px=int(t[ref].sum()), from_validation=name in validation)))
        write(out / f'tasks/T4.4_{phase}.jsonl', rows)
        info[f'{phase}_tasks'] = len(rows); info[f'{phase}_references'] = dict(used)
    info['report_from_validation'] = len(in_val)
    return info


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def generate(run=RUN, dataset=DATASET):
    run, dataset = Path(run), Path(dataset)
    train = sorted(p.name for p in (dataset / 'img1').glob('train_*.png'))
    validation = set(VALIDATION.read_text().split())
    report = (run / 'second/second_val128.txt').read_text().split()
    second_dev = (run / 'second_dev/names.txt').read_text().split()
    pool = [n for n in train if n not in validation]
    dev = sorted(random.Random(SEED).sample(pool, 64))
    assert not set(dev) & (set(report) | set(second_dev) | validation)
    out = run / 'construct'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'names_report.txt').write_text('\n'.join(report) + '\n')
    (out / 'names_dev64.txt').write_text('\n'.join(dev) + '\n')
    stats = dict(seed=SEED, report=len(report), dev=len(dev), mapping={p: {k: sorted(v) if isinstance(v, set) else v for k, v in s.items()} for p, s in PURPOSES.items()})
    for phase, names in (('report', report), ('dev', dev)):
        rows, used, available = r1_rows(names, phase, run, dataset)
        write(out / f'r1/tasks/T4.3_{phase}.jsonl', rows)
        stats[f'r1_{phase}'] = dict(tasks=len(rows), purposes=used, images_with_nonempty=available,
                                    empty_truth_tasks=sum(r['meta']['empty_truth'] for r in rows))
    for phase, names in (('report', report), ('dev', second_dev)):
        rows = []
        for name in names:
            a, b = labels(name, dataset)
            gt = out / 'r2/gt' / phase / name
            m = c3_truth(a, b)
            save(gt, m)
            rows.append(dict(id=name, image_t1=str(dataset / 'img1' / name), image_t2=str(dataset / 'img2' / name),
                task_type='T1.3xC3', instruction=C3_TEXT, constraint=None, target='any change', change_direction='any',
                scope=None, init_mask=None, gt_mask=str(gt), meta=dict(truth_px=int(m.sum()))))
        write(out / f'r2/tasks/T1.3xC3_{phase}.jsonl', rows)
        stats[f'r2_{phase}'] = dict(tasks=len(rows), nonempty=sum(r['meta']['truth_px'] > 0 for r in rows))
    r3 = Counter(); r3_px = Counter(); r3_names = {'playground': [], 'water': []}
    for name in train:
        a, b = labels(name, dataset)
        t = r3_truth(a, b)
        for k, m in t.items():
            if m.any():
                r3[k] += 1; r3_px[k] += int(m.sum()); r3_names[k].append(name)
        if any(m.any() for m in t.values()):
            r3['either'] += 1
            if name in validation:
                r3['either_in_validation'] += 1
    stats['r3'] = dict(d_pixels=D, train_tiles=len(train), nonempty_images=dict(r3), truth_pixels=dict(r3_px),
                       decision='do' if r3['either'] >= 40 else 'skip (fewer than 40 nonempty images; use A3)')
    if r3['either'] >= 40:
        stats['r3'].update(r3_tasks(run, dataset, r3_names, validation, set(dev) | set(second_dev)))
    (out / 'r3_names.json').write_text(json.dumps(r3_names))
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=RUN)
    print(json.dumps(generate(p.parse_args().run_root), indent=1))


if __name__ == '__main__':
    main()
