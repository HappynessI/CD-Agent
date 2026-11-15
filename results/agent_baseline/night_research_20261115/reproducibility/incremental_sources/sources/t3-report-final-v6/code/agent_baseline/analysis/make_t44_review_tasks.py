"""SECOND spatial-relation candidate construction; every result awaits human review.

Reads only the historical R3 manifest scene pool. Never overwrites that pool.
Proposed geometric definitions are documented in README_ai section 10.19.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), dtype=bool)
FOUR = ndi.generate_binary_structure(2, 1)


def relation_truth(candidate_mask, reference_mask, relation_type, relation_params):
    """Recompute whole-component GT from binary masks on their shared pixel grid."""
    candidate = np.asarray(candidate_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    if candidate.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError('candidate/reference must share a two-dimensional grid')
    lab, count = ndi.label(candidate, EIGHT)
    keep = []
    if reference.any():
        ry, rx = np.where(reference)
        if relation_type == 'near':
            distance = ndi.distance_transform_edt(~reference)
            selected = distance <= relation_params['distance_pixels']
        elif relation_type == 'contained_by':
            selected = ndi.binary_fill_holes(reference, structure=FOUR) & ~reference
        elif relation_type not in ('direction', 'surrounds'):
            raise ValueError(f'unknown relation: {relation_type}')
        for component_id, bounds in enumerate(ndi.find_objects(lab), 1):
            component = lab[bounds] == component_id
            y, x = bounds
            if relation_type == 'near':
                matches = bool((component & selected[bounds]).any())
            elif relation_type == 'contained_by':
                matches = bool(selected[bounds][component].all())
            elif relation_type == 'surrounds':
                inside_bbox = (ry.min() >= y.start and ry.max() < y.stop and
                               rx.min() >= x.start and rx.max() < x.stop)
                if inside_bbox:
                    holes = ndi.binary_fill_holes(component, structure=FOUR) & ~component
                    matches = bool(holes[reference[bounds]].all())
                else:
                    matches = False
            else:
                matches = {'north': y.stop - 1 < ry.min(), 'south': y.start > ry.max(),
                           'west': x.stop - 1 < rx.min(), 'east': x.start > rx.max()}[relation_params['direction']]
            if matches:
                keep.append(component_id)
    return dict(gt_mask=np.isin(lab, keep), target_component_ids=keep,
                candidate_count=int(count), filtered_count=int(count - len(keep)))


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def choices(a, b):
    candidate = (a != 0) & (a != 5) & (b == 5)
    for reference_name, class_id in [('water', 4), ('playground', 6)]:
        all_ref = (a != 0) & (a != class_id) & (b == class_id)
        ref_lab, count = ndi.label(all_ref, EIGHT)
        if not count:
            continue
        sizes = np.bincount(ref_lab.ravel()); sizes[0] = 0
        largest = ref_lab == int(np.argmax(sizes))
        specs = [('near', {'distance_pixels': 20}, all_ref, 'all_new_class')]
        specs += [('direction', {'direction': d}, largest, 'largest_component')
                  for d in ['north', 'south', 'west', 'east']]
        specs += [(r, {}, largest, 'largest_component') for r in ['contained_by', 'surrounds']]
        for relation, params, ref, selection in specs:
            params = dict(params, foreground_connectivity=8, background_connectivity=4,
                          coordinate_system='image_x_right_y_down', unit='pixel',
                          reference_selection=selection)
            truth = relation_truth(candidate, ref, relation, params)
            if truth['candidate_count'] >= 2 and truth['target_component_ids'] and truth['filtered_count']:
                yield reference_name, class_id, relation, params, candidate, ref, truth


def instruction(reference, relation, params):
    noun = 'water surface' if reference == 'water' else 'sports ground'
    if relation == 'near':
        clause = f'whose minimum pixel-center Euclidean distance to any newly appeared {noun} is at most 20 pixels'
    else:
        ref = f'the largest connected newly appeared {noun} region (pixel area; ties use top-to-bottom, left-to-right component order)'
        if relation == 'direction':
            clause = f'lying entirely {params["direction"]} of the bounding box of {ref} in image coordinates (up is north)'
        elif relation == 'contained_by':
            clause = f'lying entirely inside an enclosed background hole of {ref}'
        else:
            clause = f'whose enclosed background hole contains all of {ref}'
    return f'Find whole connected newly built building regions {clause}. Output a binary change mask.'


def generate(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(args.legacy_root)
    progress_dir = output / '.progress'
    progress_dir.mkdir(exist_ok=True)
    config = dict(data_version=args.data_version, legacy_root=str(source.resolve()), limit=args.limit)
    config_path = progress_dir / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('resume configuration differs; use a new output/version')
    else:
        config_path.write_text(json.dumps(config))
    pool = []
    seen = set()
    for split in ['dev', 'report']:
        path = source / f'T4.4_{split}.jsonl'
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] in seen:
                raise ValueError('duplicate scene across historical splits')
            seen.add(row['id']); pool.append((split, row, path))
    eligible = []
    available = Counter()
    for split, old, manifest in pool:
        name = old['id']
        dataset = Path(old['image_t1']).parent.parent
        annotations = [dataset / p / name for p in ['slabel1', 'slabel2']]
        cache = progress_dir / f'{split}-{Path(name).stem}.json'
        if cache.exists():
            options = json.loads(cache.read_text())
        else:
            a, b = [np.asarray(Image.open(p)) for p in annotations]
            options = [list(option[:4]) for option in choices(a, b)]
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(options))
            temporary.replace(cache)
        for relation in set(option[2] for option in options):
            available[relation] += 1
        if options:
            eligible.append((split, old, manifest, annotations, options))
    # Allocate scarce relation families first, one task per scene. No fallback empty GT.
    selected = []; used_scenes = set(); used = Counter()
    families = sorted(['near', 'direction', 'contained_by', 'surrounds'], key=lambda r: (available[r], r))
    while len(selected) < args.limit:
        progress = False
        for family in families:
            if len(selected) >= args.limit:
                break
            for index, item in enumerate(eligible):
                if index in used_scenes:
                    continue
                options = [x for x in item[-1] if x[2] == family]
                if options:
                    choice = min(options, key=lambda x: (used[(x[0], x[2], x[3].get('direction'))], x[0], x[3].get('direction', '')))
                    selected.append((item, choice)); used_scenes.add(index)
                    used[(choice[0], choice[2], choice[3].get('direction'))] += 1
                    progress = True; break
        if not progress:
            break
    rows = []
    for (split, old, manifest, annotations, _), choice in selected:
        scene = Path(old['id']).stem
        record = progress_dir / f'task-{scene}.json'
        if record.exists():
            rows.append(json.loads(record.read_text()))
            continue
        refname, class_id, relation, params = choice
        a, b = [np.asarray(Image.open(p)) for p in annotations]
        candidate = (a != 0) & (a != 5) & (b == 5)
        reference = (a != 0) & (a != class_id) & (b == class_id)
        if params['reference_selection'] == 'largest_component':
            lab, _ = ndi.label(reference, EIGHT)
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            reference = lab == int(np.argmax(sizes))
        truth = relation_truth(candidate, reference, relation, params)
        review_id = f'SECOND-T4.4-{scene}'
        base = output / 'masks' / split / scene
        paths = {key: base / f'{key}.png' for key in ['gt', 'candidate', 'reference']}
        for key, mask in [('gt', truth['gt_mask']), ('candidate', candidate), ('reference', reference)]:
            save_mask(paths[key], mask)
        row = dict(old)
        row.update(id=review_id, gt_mask=str(paths['gt']), instruction=instruction(refname, relation, params), review_status='pending_review')
        row['meta'] = dict(review_id=review_id, data_version=args.data_version, dataset='SECOND', split=split,
            scene_id=scene, source_annotations=[dict(path=str(p), feature_ids=[], version='local_SECOND_slabel', raster_class_ids=[5, class_id]) for p in annotations],
            reference_source='SECOND newly appeared labeled class; never class 0', reference_class_id=class_id,
            reference=refname, reference_mask=str(paths['reference']), candidate_mask=str(paths['candidate']),
            relation_type=relation, relation_params=params, target_component_ids=truth['target_component_ids'],
            candidate_count=truth['candidate_count'], filtered_count=truth['filtered_count'], truth_px=int(truth['gt_mask'].sum()),
            review_status='pending_review', rule_status='proposed_pending_Ethan_review',
            legacy_manifest=str(manifest), legacy_id=old['id'], original_split=split,
            from_validation=old.get('meta', {}).get('from_validation'), visual_reference_verified=False)
        rows.append(row)
        # Each completed task is recoverable even if later scenes fail or time out.
        temporary = record.with_suffix('.tmp')
        temporary.write_text(json.dumps(row))
        temporary.replace(record)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    stats = dict(target=args.limit, actual=len(rows), gap=max(0, args.limit-len(rows)), historical_scenes=len(pool),
                 eligible_scenes=len(eligible), available_scenes_by_relation=dict(available),
                 selected_by_relation=dict(Counter(r['meta']['relation_type'] for r in rows)),
                 selected_by_split=dict(Counter(r['meta']['split'] for r in rows)),
                 approved=0, review_status='pending_review', api_calls=0)
    (output / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-version', required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--resume', action='store_true', help='resume this unchanged data version from per-scene progress')
    generate(parser.parse_args())


def verify_task(task):
    """Check SECOND source labels independently of stored derived masks."""
    meta = task['meta']
    issues = []
    def issue(code, message):
        issues.append(dict(code=code, severity='hard_error', message=message))
    annotations = meta['source_annotations']
    with Image.open(annotations[0]['path']) as image:
        a = np.asarray(image)
    with Image.open(annotations[1]['path']) as image:
        b = np.asarray(image)
    if a.shape != b.shape or a.ndim != 2:
        issue('source_grid', 'SECOND source label grids differ or are not two-dimensional')
        return issues
    if not np.isin(a, range(7)).all() or not np.isin(b, range(7)).all():
        issue('source_classes', 'SECOND source labels contain values outside 0..6')
    if task.get('target') != 'building' or task.get('change_direction') != 'appeared':
        issue('target_direction', 'T4.4 source rule requires newly appeared buildings')
    c = meta['reference_class_id']
    if c not in (4, 6):
        issue('reference_class', 'Only labeled newly appeared water/playground are supported')
        return issues
    candidate = (a != 0) & (a != 5) & (b == 5)
    reference = (a != 0) & (a != c) & (b == c)
    selection = meta['relation_params']['reference_selection']
    if selection == 'largest_component':
        lab, n = ndi.label(reference, EIGHT)
        if n:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            reference = lab == int(np.argmax(sizes))
    elif selection != 'all_new_class':
        issue('reference_selection', 'Unknown reference selection rule')
    if (meta['relation_type'] == 'near') != (selection == 'all_new_class'):
        issue('reference_selection_relation', 'Reference selection does not match proposed relation rule')
    if meta['relation_type'] == 'near' and meta['relation_params'].get('distance_pixels') != 20:
        issue('near_threshold', 'Proposed historical R3 rule uses exactly 20 pixels')
    for key, expected in [('candidate_mask', candidate), ('reference_mask', reference)]:
        with Image.open(meta[key]) as image:
            actual = np.asarray(image) != 0
        if not np.array_equal(actual, expected):
            issue('source_' + key, f'{key} differs from source-label reconstruction')
    return issues
