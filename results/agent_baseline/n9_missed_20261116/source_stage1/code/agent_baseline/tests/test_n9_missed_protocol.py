import unittest
import numpy as np
from agent_baseline.analysis.n9_missed_protocol import (
    profile_for, source_set, split_names, online_names, candidate_indices, choose_config)
from agent_baseline.whu_gt import truth
from agent_baseline.evaluation import pixel_counts


class N9ProtocolTests(unittest.TestCase):
    def test_direction_and_fixed_image_split(self):
        self.assertEqual(profile_for('appeared'), 'build')
        self.assertEqual(profile_for('disappeared'), 'demolish')
        self.assertNotIn('S3', source_set(profile_for('disappeared')))
        split = split_names([str(i) for i in range(60)])
        self.assertEqual(list(split.values()).count('tune'), 30)
        self.assertEqual(split, split_names(list(reversed(list(split)))))
        self.assertEqual(len(online_names(split)), 10)
        self.assertTrue(all(split[n] == 'confirm' for n in online_names(split)))

    def test_gate_and_k_prefix_do_not_use_labels_or_group(self):
        config = dict(semantic=.2, dino=.4, alpha=0, k=3)
        candidates = [dict(index=0, semantic_delta=.8, dino=.5, area=40),
                      dict(index=1, semantic_delta=.6, dino=.6, area=900),
                      dict(index=2, semantic_delta=.8, dino=.1, area=900)]
        self.assertEqual(candidate_indices(candidates, config), [0, 1])
        self.assertEqual(candidate_indices(candidates, dict(config, k=1)), [0])
        noisy = [dict(c, dataset='arbitrary', gt_tp=999999) for c in candidates]
        self.assertEqual(candidate_indices(noisy, config), [0, 1])
        self.assertEqual(candidate_indices([], config), [])

    def test_equal_line_selection_and_no_confirm_access(self):
        rows = [dict(split='tune', config=dict(k=3), lines={
            'small': dict(delta_f1=4., calls=2), 'large': dict(delta_f1=-1., calls=2)}),
            dict(split='tune', config=dict(k=3), lines={
            'small': dict(delta_f1=0., calls=1), 'large': dict(delta_f1=2., calls=1)})]
        self.assertIs(choose_config(rows), rows[0])
        with self.assertRaises(ValueError): choose_config([dict(rows[0], split='confirm')])

    def test_whu_ignore_only_uncertain_official_positives(self):
        labels = truth(np.array([[0, 0, 1, 1]]), np.array([[0, 1, 0, 1]]), np.array([[0, 1, 1, 1]]))
        valid = ~labels['ignore']
        pred = np.ones((1,4), bool)
        count = pixel_counts(pred, labels['new'], valid)
        self.assertEqual(count, dict(tp=1, fp=2, fn=0))
        # The official negative at column0 remains an FP, never an ignored pixel.
        self.assertTrue(valid[0,0]); self.assertFalse(valid[0,3])

if __name__ == '__main__': unittest.main()
