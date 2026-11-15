"""SpaceNet8 flood candidates only; run construction on a Slurm compute node.

Candidate definition v1 awaits human review: buildings retain source footprints;
roads use a 3 m UTM buffer, pixel-centre rasterization. Unknown flood labels are
ignored, never treated as negatives. No formal export or model calls occur here.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform

RULES = {
    'version': 'sn8-flood-candidate-v1', 'status': 'pending_review',
    'positive_flooded': ['yes', 'true', '1'],
    'negative_flooded': ['no', 'false', '0'],
    'unknown_policy': 'ignore source geometry; ignore overrides positives',
    'building_geometry': 'original Polygon/MultiPolygon footprint',
    'road_buffer_m': 3.0, 'road_buffer_crs': 'local UTM from geometry centroid',
    'road_buffer_resolution': 16, 'road_cap_style': 'round',
    'road_join_style': 'round', 'all_touched': False,
    'partial_flood_policy': 'retain entire source feature marked flooded',
    'grid': 'native pre-event grid; post-event bilinear reprojected to this grid',
    'claim': 'flooded infrastructure coverage, not traffic passability',
}


def flood_state(value):
    if value is None:
        return 'unknown'
    value = str(value).strip().lower()
    if value in RULES['positive_flooded']:
        return 'positive'
    if value in RULES['negative_flooded']:
        return 'negative'
    return 'unknown'


def target_of(properties):
    def present(value):
        return value is not None and str(value).strip().lower() not in ('', 'no', 'false', '0', 'null')
    if present(properties.get('building')):
        return 'building'
    if present(properties.get('highway')):
        return 'road'
    return None


def ground_geometry(feature, target, dst_crs, radius=3.0):
    geometry = shape(feature['geometry'])
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError('empty or invalid source geometry')
    if target == 'road':
        if geometry.geom_type not in ('LineString', 'MultiLineString'):
            raise ValueError('road geometry is not a line')
        x, y = geometry.centroid.x, geometry.centroid.y
        zone = min(60, max(1, int((x + 180) // 6) + 1))
        utm = f'EPSG:{32600 + zone if y >= 0 else 32700 + zone}'
        geometry = transform(Transformer.from_crs('EPSG:4326', utm, always_xy=True).transform, geometry)
        geometry = geometry.buffer(radius, resolution=16, cap_style=1, join_style=1)
        return transform(Transformer.from_crs(utm, dst_crs, always_xy=True).transform, geometry)
    if geometry.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('building geometry is not a polygon')
    return transform(Transformer.from_crs('EPSG:4326', dst_crs, always_xy=True).transform, geometry)


def make_masks(features, target, profile, radius=3.0):
    groups = {'positive': [], 'negative': [], 'unknown': []}
    ids = {key: [] for key in groups}
    for index, feature in enumerate(features):
        properties = feature.get('properties') or {}
        if target_of(properties) != target:
            continue
        state = flood_state(properties.get('flooded'))
        geometry = ground_geometry(feature, target, profile['crs'], radius)
        groups[state].append((mapping(geometry), 1))
        ids[state].append({'feature_id': properties.get('id', feature.get('id')),
                           'feature_index': index, 'flooded_raw': properties.get('flooded')})
    size = (profile['height'], profile['width'])
    def burn(items):
        return (rasterize(items, out_shape=size, transform=profile['transform'],
                          all_touched=False, dtype='uint8') if items else np.zeros(size, np.uint8))
    ignore = burn(groups['unknown'])
    positive = burn(groups['positive'])
    positive[ignore != 0] = 0
    return positive, ignore, ids


def save_mask(path, array):
    Image.fromarray((array != 0).astype('uint8') * 255).save(path)


def aligned_images(pre_path, post_path, destination):
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        profile = pre.profile.copy()
        if pre.count < 3 or post.count < 3:
            raise ValueError('requires three image bands')
        a = pre.read([1, 2, 3])
        b = np.zeros((3, pre.height, pre.width), dtype=post.dtypes[0])
        for band in range(3):
            reproject(rasterio.band(post, band + 1), b[band], src_transform=post.transform,
                      src_crs=post.crs, dst_transform=pre.transform, dst_crs=pre.crs,
                      resampling=Resampling.bilinear)
        valid = pre.dataset_mask() != 0
        post_valid = np.zeros((pre.height, pre.width), np.uint8)
        reproject(post.dataset_mask(), post_valid, src_transform=post.transform,
                  src_crs=post.crs, dst_transform=pre.transform, dst_crs=pre.crs,
                  resampling=Resampling.nearest)
        valid &= post_valid != 0
        if a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError('non-uint8 imagery requires an explicit reviewed rendering rule')
        for name, array in [('t1.png', a), ('t2.png', b)]:
            Image.fromarray(array.transpose(1, 2, 0)).save(destination / name)
        grid = {'crs': str(pre.crs), 'transform': list(pre.transform)[:6],
                'width': pre.width, 'height': pre.height, 'bounds': list(pre.bounds)}
        return profile, valid, grid


def build_scene(pre, post, annotation, out, aoi, version, targets):
    features = json.loads(annotation.read_text())['features']
    out.mkdir(parents=True, exist_ok=True)
    profile, valid, grid = aligned_images(pre, post, out)
    rows = []
    for target in targets:
        mask, ignore, ids = make_masks(features, target, profile)
        ignore[~valid] = 1
        mask[ignore != 0] = 0
        if not mask.any():
            continue
        gt_path = out / f'{target}_gt.png'
        ignore_path = out / f'{target}_ignore.png'
        save_mask(gt_path, mask)
        save_mask(ignore_path, ignore)
        sensitivity = None
        if target == 'road':
            sensitivity = {}
            for radius in (2.0, 4.0):
                alternative, alt_ignore, _ = make_masks(features, target, profile, radius)
                alternative[(alt_ignore != 0) | (~valid)] = 0
                sensitivity[str(radius)] = {'positive_pixels': int(alternative.sum()),
                    'changed_pixels_from_3m': int(np.count_nonzero(alternative != mask))}
        review_id = f't43-{aoi}-{pre.stem}-{target}'
        rows.append({'id': review_id, 'task_type': 'T4.3',
            'image_t1': str(out / 't1.png'), 'image_t2': str(out / 't2.png'),
            'instruction': f'Find the {"road segments" if target == "road" else "buildings"} flooded after the flood event. Output a binary mask.',
            'target': target, 'change_direction': 'any', 'constraint': None,
            'init_mask': None, 'gt_mask': str(gt_path),
            'meta': {'review_id': review_id, 'data_version': version,
                'review_status': 'pending_review', 'dataset': 'SpaceNet8',
                'split': 'review_pool', 'source_split': 'Training_Public',
                'scene_id': f'{aoi}/{pre.stem}', 'event_type': 'flood',
                'target_object': target, 'impact_state': 'flooded',
                'aoi': {'name': aoi, 'crs': grid['crs'], 'bounds': grid['bounds']},
                'source_annotations': [{'path': str(annotation), 'version': 'SpaceNet8 public training release',
                    'feature_ids': ids['positive'], 'unknown_features': ids['unknown'],
                    'negative_features': ids['negative']}],
                'source_images': {'pre': str(pre), 'post': str(post)},
                'source_license': 'CC BY-SA 4.0', 'gt_rules': RULES,
                'grid': grid, 'ignore_mask': str(ignore_path),
                'positive_pixels': int(mask.sum()), 'ignore_pixels': int(ignore.sum()),
                'road_sensitivity': sensitivity,
                'registration_status': 'unverified_visual_alignment',
                'definition_status': 'pending_review'}})
    return rows


def verify_task(task):
    """Rebuild coverage from original vectors and validity masks for offline QC."""
    meta = task['meta']
    issues = []
    def issue(code, severity, message):
        issues.append(dict(code=code, severity=severity, message=message))
    with rasterio.open(meta['source_images']['pre']) as pre, rasterio.open(meta['source_images']['post']) as post:
        features = json.loads(Path(meta['source_annotations'][0]['path']).read_text())['features']
        expected, ignore, ids = make_masks(features, meta['target_object'], pre.profile)
        valid = pre.dataset_mask() != 0
        post_valid = np.zeros(expected.shape, np.uint8)
        reproject(post.dataset_mask(), post_valid, src_transform=post.transform,
                  src_crs=post.crs, dst_transform=pre.transform, dst_crs=pre.crs,
                  resampling=Resampling.nearest)
        ignore[(~valid) | (post_valid == 0)] = 1
        expected[ignore != 0] = 0
        actual = np.asarray(Image.open(task['gt_mask'])) != 0
        actual_ignore = np.asarray(Image.open(meta['ignore_mask'])) != 0
        if not np.array_equal(expected != 0, actual):
            issue('t43_gt_rebuild', 'hard_error', 'GT differs from source vector reconstruction')
        if not np.array_equal(ignore != 0, actual_ignore):
            issue('t43_ignore_rebuild', 'hard_error', 'Unknown/nodata ignore differs from reconstruction')
        if ids['positive'] != meta['source_annotations'][0]['feature_ids']:
            issue('t43_feature_ids', 'hard_error', 'Source flooded feature references differ')
        if meta['gt_rules'] != RULES:
            issue('t43_gt_rules', 'hard_error', 'Recorded candidate rules differ from generator')
        if meta['target_object'] == 'road':
            for radius in (2.0, 4.0):
                alternative, alt_ignore, _ = make_masks(features, 'road', pre.profile, radius)
                alternative[(alt_ignore != 0) | (~valid) | (post_valid == 0)] = 0
                rebuilt = {'positive_pixels': int(alternative.sum()),
                    'changed_pixels_from_3m': int(np.count_nonzero(alternative != expected))}
                if meta.get('road_sensitivity', {}).get(str(radius)) != rebuilt:
                    issue('t43_road_sensitivity', 'hard_error', f'{radius} m sensitivity differs from reconstruction')
    issue('t43_visual_registration', 'unverified', 'Human review must assess post/pre visual registration')
    issue('t43_candidate_definition', 'unverified', 'Human review must approve candidate coverage definition')
    return issues


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def write_manifest(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    temporary.replace(path)


def restore(output, version, resume):
    """Scene checkpoints are committed only after all referenced assets are saved."""
    checkpoint_dir = output / 'scene_checkpoints'
    manifests = [output / 'manifest.jsonl', output / 'manifest.partial.jsonl']
    if not resume and (any(path.exists() for path in manifests) or checkpoint_dir.exists()):
        raise ValueError('Existing construction: use --resume, or a new revision directory')
    by_id, completed = {}, {}
    if resume:
        # Also import v1 smoke output, which predates per-scene checkpoints.
        for path in manifests:
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    by_id[row['id']] = row
                    completed.setdefault(row['meta']['scene_id'], set()).add(row['target'])
        if checkpoint_dir.exists():
            for path in sorted(checkpoint_dir.glob('*/*.json')):
                checkpoint = json.loads(path.read_text())
                completed.setdefault(checkpoint['scene_id'], set()).update(checkpoint['completed_targets'])
                for row in checkpoint['rows']:
                    by_id[row['id']] = row
    rows = list(by_id.values())
    for row in rows:
        if row['meta']['data_version'] != version or row['meta']['gt_rules'] != RULES:
            raise ValueError('Resume requires unchanged data_version and candidate GT rules')
        for path in (row['image_t1'], row['image_t2'], row['gt_mask'], row['meta']['ignore_mask']):
            if not Path(path).is_file():
                raise ValueError(f'Resume asset missing: {path}')
    return rows, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/guisongxia01/pangchao/wangyihan/datasets/SpaceNet8'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--version', default='t43-candidate-v1')
    parser.add_argument('--per-target', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, completed = restore(args.output, args.version, args.resume)
    config = {'dataset': str(args.dataset.absolute()), 'assets': str(args.assets.absolute()),
              'version': args.version, 'rules': RULES}
    config_path = args.output / 'construction_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Resume requires unchanged source, assets, version and rules')
    atomic_json(config_path, config)
    pools, inventory, failures = [], [], []
    for aoi in ('Germany_Training_Public', 'Louisiana-East_Training_Public'):
        directory = args.dataset / aoi
        entries = []
        with (directory / f'{aoi}_label_image_mapping.csv').open(newline='') as handle:
            for row in sorted(csv.DictReader(handle), key=lambda row: row['label']):
                annotation = directory / 'annotations' / row['label']
                features = json.loads(annotation.read_text())['features']
                counts = Counter((target_of(f.get('properties') or {}),
                                  flood_state((f.get('properties') or {}).get('flooded'))) for f in features)
                positive_targets = [t for t in ('road', 'building') if counts[t, 'positive']]
                inventory.append({'aoi': aoi, 'annotation': str(annotation),
                    'counts': {f'{t}:{s}': n for (t, s), n in counts.items()},
                    'post_2': row.get('post-event image 2') or None})
                if positive_targets:
                    entries.append((aoi, directory / 'PRE-event' / row['pre-event image'],
                                    directory / 'POST-event' / row['post-event image 1'], annotation, positive_targets))
        pools.append(entries)
    # Alternate AOIs deterministically, count each source scene/target once.
    candidates = [pool[index] for index in range(max(map(len, pools), default=0))
                  for pool in pools if index < len(pool)]
    (args.output / 'source_inventory.json').write_text(json.dumps(inventory, indent=2))
    used = Counter(row['target'] for row in rows)
    if any(count > args.per_target for count in used.values()):
        raise ValueError('Requested quota is smaller than existing candidates')
    for aoi, pre, post, annotation, targets in candidates:
        scene_id = f'{aoi}/{pre.stem}'
        targets = [t for t in targets if used[t] < args.per_target and t not in completed.get(scene_id, set())]
        if not targets:
            continue
        try:
            built = build_scene(pre, post, annotation, args.assets / aoi / pre.stem,
                                aoi, args.version, targets)
            rows.extend(built)
            used.update(row['target'] for row in built)
            completed.setdefault(scene_id, set()).update(targets)
            checkpoint_path = args.output / 'scene_checkpoints' / aoi / f'{pre.stem}.json'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(checkpoint_path, {'scene_id': scene_id,
                'completed_targets': sorted(completed[scene_id]),
                'rows': [row for row in rows if row['meta']['scene_id'] == scene_id]})
        except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
            failures.append({'scene': f'{aoi}/{pre.stem}', 'reason': str(exc)})
        with (args.output / 'construction_progress.jsonl').open('a') as handle:
            handle.write(json.dumps({'scene': f'{aoi}/{pre.stem}', 'counts': dict(used),
                'latest_failure': failures[-1] if failures and failures[-1]['scene'] == f'{aoi}/{pre.stem}' else None}) + '\n')
        write_manifest(args.output / 'manifest.partial.jsonl', rows)
    write_manifest(args.output / 'manifest.jsonl', rows)
    (args.output / 'manifest.partial.jsonl').unlink(missing_ok=True)
    (args.output / 'asset_audit.json').write_text(json.dumps({'scenes': inventory,
        'failures': failures, 'rules': RULES, 'post_choice': 'post-event image 1 only; image 2 retained in audit'}, indent=2))
    summary = {'count': len(rows), 'counts': dict(used),
        'shortfalls': {t: max(0, args.per_target - used[t]) for t in ('road', 'building')},
        'review_status': 'pending_review', 'formal_export_count': 0,
        'cost_cny': 0, 'source_scene_count': len(inventory), 'failed_scenes': len(failures)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
