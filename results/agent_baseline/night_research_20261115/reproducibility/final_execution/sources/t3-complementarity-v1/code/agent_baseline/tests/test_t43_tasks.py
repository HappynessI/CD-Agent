"""Offline synthetic geometry checks; run on Slurm, not the management node."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
try:
    from affine import Affine
    from agent_baseline.analysis.make_t43_tasks import flood_state, make_masks, target_of, restore, atomic_json, RULES
    GEO_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in {'affine', 'rasterio', 'pyproj', 'shapely'}:
        raise
    GEO_AVAILABLE = False


@unittest.skipUnless(GEO_AVAILABLE, 'SpaceNet8 construction requires its geospatial environment')
class FloodRulesTest(unittest.TestCase):
    def test_resume_checkpoints_restore_success_and_empty_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / 'asset.png'
            asset.write_bytes(b'fixture')
            row = {'id': 'scene-road', 'target': 'road', 'image_t1': str(asset),
                   'image_t2': str(asset), 'gt_mask': str(asset),
                   'meta': {'scene_id': 'AOI/scene', 'data_version': 'v1',
                            'gt_rules': RULES, 'ignore_mask': str(asset)}}
            checkpoint = root / 'scene_checkpoints' / 'AOI' / 'scene.json'
            checkpoint.parent.mkdir(parents=True)
            atomic_json(checkpoint, {'scene_id': 'AOI/scene',
                        'completed_targets': ['road', 'building'], 'rows': [row]})
            rows, completed = restore(root, 'v1', True)
            self.assertEqual(rows, [row])
            self.assertEqual(completed['AOI/scene'], {'road', 'building'})
            with self.assertRaises(ValueError):
                restore(root, 'v2', True)
            with self.assertRaises(ValueError):
                restore(root, 'v1', False)

    def test_resume_imports_partial_without_duplicate_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / 'asset.png'
            asset.write_bytes(b'fixture')
            row = {'id': 'scene-road', 'target': 'road', 'image_t1': str(asset),
                   'image_t2': str(asset), 'gt_mask': str(asset),
                   'meta': {'scene_id': 'AOI/scene', 'data_version': 'v1',
                            'gt_rules': RULES, 'ignore_mask': str(asset)}}
            for name in ('manifest.jsonl', 'manifest.partial.jsonl'):
                (root / name).write_text(json.dumps(row) + '\n')
            rows, completed = restore(root, 'v1', True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(completed['AOI/scene'], {'road'})

    def test_unknown_is_not_negative(self):
        for value in (None, '', 'partly', 'unknown'):
            self.assertEqual(flood_state(value), 'unknown')
        self.assertEqual(flood_state('yes'), 'positive')
        self.assertEqual(flood_state(False), 'negative')

    def test_null_target_properties(self):
        self.assertEqual(target_of({'building': None, 'highway': 'residential'}), 'road')
        self.assertIsNone(target_of({'building': None, 'highway': None}))

    def test_footprint_unknown_overrides_positive(self):
        geometry = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
        features = [{'geometry': geometry, 'properties': {'id': str(i), 'building': 'yes', 'flooded': value}}
                    for i, value in enumerate(('yes', None))]
        profile = {'crs': 'EPSG:4326', 'height': 4, 'width': 4, 'transform': Affine(1, 0, 0, 0, -1, 4)}
        gt, ignore, ids = make_masks(features, 'building', profile)
        self.assertEqual(int(gt.sum()), 0)
        self.assertEqual(int(ignore.sum()), 4)
        self.assertEqual(ids['positive'][0]['feature_id'], '0')
        self.assertEqual(ids['unknown'][0]['feature_id'], '1')

    def test_road_buffer_uses_metres(self):
        feature = {'geometry': {'type': 'LineString', 'coordinates': [[9, 0], [9, 0.0001]]},
                   'properties': {'id': 'road1', 'highway': 'residential', 'flooded': 'yes'}}
        profile = {'crs': 'EPSG:32632', 'height': 30, 'width': 30,
                   'transform': Affine(1, 0, 499985, 0, -1, 20)}
        a, _, _ = make_masks([feature], 'road', profile, 2)
        b, _, _ = make_masks([feature], 'road', profile, 3)
        self.assertGreater(int(b.sum()), int(a.sum()))
        self.assertTrue(np.all(a <= b))


if __name__ == '__main__':
    unittest.main()
