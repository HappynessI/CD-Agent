"""Synthetic geometry checks; execute on Slurm compute nodes."""
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from PIL import Image
import numpy as np
from agent_baseline.analysis.make_t44_review_tasks import relation_truth, verify_task, save_mask, generate
from agent_baseline.analysis.make_construct_tasks import r3_truth


class RelationTests(unittest.TestCase):
    def test_resume_reuses_completed_scene_and_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for part in ['slabel1', 'slabel2', 'img1', 'img2', 'legacy']:
                (root / part).mkdir()
            a = np.ones((80, 80), np.uint8); b = a.copy()
            b[5:8, 5:8] = 4; b[10:13, 10:40] = 5; b[70:74, 70:74] = 5
            for part, mask in [('slabel1', a), ('slabel2', b)]:
                Image.fromarray(mask).save(root / part / 'scene.png')
            row = dict(id='scene.png', image_t1=str(root / 'img1/scene.png'),
                       image_t2=str(root / 'img2/scene.png'), target='building',
                       change_direction='appeared', task_type='T4.4')
            (root / 'legacy/T4.4_dev.jsonl').write_text(json.dumps(row) + '\n')
            (root / 'legacy/T4.4_report.jsonl').write_text('')
            args = SimpleNamespace(output=str(root / 'out'), legacy_root=str(root / 'legacy'),
                                   data_version='test-v1', limit=1, resume=False)
            generate(args)
            original = (root / 'out/tasks.jsonl').read_text()
            (root / 'out/tasks.jsonl').unlink()
            args.resume = True
            with patch('agent_baseline.analysis.make_t44_review_tasks.choices', side_effect=AssertionError('scene recomputed')):
                generate(args)
            self.assertEqual((root / 'out/tasks.jsonl').read_text(), original)

    def test_source_reconstruction_detects_class_zero_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = np.ones((8, 8), np.uint8); b = a.copy()
            b[1, 1] = 4; a[2, 2] = 0; b[2, 2] = 4
            b[5, 5] = 5
            for name, mask in [('a', a), ('b', b)]:
                Image.fromarray(mask).save(root / f'{name}.png')
            save_mask(root / 'candidate.png', b == 5)
            save_mask(root / 'reference.png', b == 4)
            task = dict(target='building', change_direction='appeared', meta=dict(
                source_annotations=[dict(path=str(root / f'{n}.png')) for n in ['a', 'b']],
                reference_class_id=4, relation_type='near',
                relation_params=dict(reference_selection='all_new_class', distance_pixels=20),
                candidate_mask=str(root / 'candidate.png'), reference_mask=str(root / 'reference.png')))
            issues = verify_task(task)
            self.assertEqual([issue['code'] for issue in issues], ['source_reference_mask'])
            save_mask(root / 'reference.png', (a != 0) & (b == 4))
            self.assertEqual(verify_task(task), [])

    def test_near_keeps_whole_component_and_matches_legacy(self):
        a = np.ones((80, 80), dtype=np.uint8)
        b = a.copy(); b[5:8, 5:8] = 4
        b[10:13, 10:40] = 5; b[70:74, 70:74] = 5
        result = relation_truth(b == 5, b == 4, 'near', {'distance_pixels': 20})
        self.assertTrue(np.array_equal(result['gt_mask'], r3_truth(a, b)['water']))
        self.assertEqual(result['target_component_ids'], [1])
        self.assertEqual(result['filtered_count'], 1)
        self.assertTrue(result['gt_mask'][11, 39])

    def test_direction_strict_bbox_and_diagonal_connectivity(self):
        ref = np.zeros((12, 12), bool); ref[5:7, 5:7] = True
        c = np.zeros_like(ref); c[1, 1] = True; c[2, 2] = True
        c[5, 1] = True; c[9, 9] = True
        out = relation_truth(c, ref, 'direction', {'direction': 'north'})
        self.assertEqual(out['candidate_count'], 3)
        self.assertEqual(out['target_component_ids'], [1])
        self.assertFalse(out['gt_mask'][5, 1])

    def test_holes_not_bbox_inclusion(self):
        ring = np.zeros((15, 15), bool)
        ring[3:12, 3:12] = True; ring[4:11, 4:11] = False
        c = np.zeros_like(ring); c[6:8, 6:8] = True; c[0, 0] = True
        self.assertEqual(relation_truth(c, ring, 'contained_by', {})['target_component_ids'], [2])
        ring[3, 7] = False
        self.assertFalse(relation_truth(c, ring, 'contained_by', {})['gt_mask'].any())

    def test_surrounds_requires_all_reference(self):
        ring = np.zeros((15, 15), bool)
        ring[3:12, 3:12] = True; ring[4:11, 4:11] = False
        ring[0, 0] = True
        ref = np.zeros_like(ring); ref[6:8, 6:8] = True
        out = relation_truth(ring, ref, 'surrounds', {})
        self.assertEqual(out['target_component_ids'], [2])
        ref[14, 14] = True
        self.assertFalse(relation_truth(ring, ref, 'surrounds', {})['gt_mask'].any())


if __name__ == '__main__':
    unittest.main()
