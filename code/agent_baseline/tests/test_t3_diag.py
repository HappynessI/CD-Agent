"""T3 diagnostics with synthetic arrays and fake models only."""
import json
from pathlib import Path
import unittest
import numpy as np
from agent_baseline.analysis.t3_diag import (proposal_metrics, pixel_counts, directed_proposals,
                                           evaluate_task, aggregate, conclusions, select_rows)
from agent_baseline.analysis.io_retry import save_image
from agent_baseline.workers.t3_sam3_worker import prompt_state
from .helpers import tempdir


class DiagnosticTests(unittest.TestCase):
    def test_instance_iou_and_pixel_upper_exclude_exemplar(self):
        truth = np.zeros((12,12), bool)
        truth[1:3,1:3] = True; truth[5:7,5:7] = True; truth[9:11,9:11] = True
        exemplar = np.zeros_like(truth); exemplar[1:3,1:3] = True
        proposal = exemplar.copy(); proposal[5:7,5:7] = True
        result = proposal_metrics([proposal], truth, exemplar)
        self.assertEqual(result['gt_instances'], 2)
        self.assertEqual(result['aligned_instances'], 1)
        self.assertEqual(result['pixel_coverage_upper'], .5)
        self.assertAlmostEqual(result['pixel_f1_upper'], 200/3)
        self.assertEqual(pixel_counts(proposal, truth, exemplar), [4,0,4])
        empty = proposal_metrics([], truth, exemplar)
        self.assertEqual(empty['aligned_instances'], 0)
        self.assertEqual(empty['pixel_coverage_upper'], 0)
        # One broad proposal covers pixels but fails IoU: distinguish granularity.
        broad = proposal_metrics([np.ones_like(truth)], truth, exemplar)
        self.assertEqual(broad['aligned_instances'], 0)
        self.assertEqual(broad['pixel_coverage_upper'], 1)

    def test_directed_transition_rejects_reverse_and_unchanged(self):
        source1 = np.array([[[1,0,1,0]]], bool)
        source2 = np.array([[[0,1,1,0]]], bool)
        target1 = np.array([[[0,1,0,1]]], bool)
        target2 = np.array([[[1,0,0,1]]], bool)
        masks = directed_proposals(source1, source2, target1, target2, (1,4))
        np.testing.assert_array_equal(masks.any(axis=0), [[1,0,0,0]])
        np.testing.assert_array_equal(directed_proposals(source2, source1, target2, target1, (1,4)).any(axis=0), [[0,1,0,0]])
        self.assertEqual(directed_proposals([], [], [], [], (2,2)).shape, (0,2,2))

    def test_visual_box_uses_reference_query_mosaic_without_text(self):
        calls = []
        class Processor:
            def set_image(self, image):
                calls.append(('image', np.asarray(image)))
                return {'ready':True}
            def add_geometric_prompt(self, **kw):
                calls.append(('box', kw))
                return kw['state']
            def set_text_prompt(self, **kw):
                calls.append(('text', kw))
                return kw['state']
        query = np.full((10,20,3), 90, np.uint8)
        reference = np.full_like(query, 10)
        prompt_state(Processor(), query, reference=reference, box=[2,2,6,6])
        np.testing.assert_array_equal(calls[0][1][:,:20], reference)
        np.testing.assert_array_equal(calls[0][1][:,20:], query)
        self.assertEqual(calls[1][0], 'box')
        self.assertEqual(calls[1][1]['box'], [.1,.4,.1,.4])
        self.assertTrue(calls[1][1]['label'])
        calls.clear()
        prompt_state(Processor(), query, text='building')
        self.assertEqual(calls[1][0], 'text')
        self.assertEqual(calls[1][1]['prompt'], 'building')

    def test_end_to_end_fake_models_and_summary(self):
        with tempdir() as root:
            root = Path(root)
            exemplar = np.zeros((12,12), bool); exemplar[1:3,1:3] = True
            instance = np.zeros_like(exemplar); instance[7:9,7:9] = True
            union = exemplar | instance
            save_image(root/'ex.png', exemplar.astype('uint8')*255)
            save_image(root/'gt.png', union.astype('uint8')*255)
            row = dict(id='train_1_1to5', exemplar_mask=str(root/'ex.png'), gt_mask=str(root/'gt.png'))
            def dynamic(operation, time):
                if operation == 'segment_everything':
                    return np.asarray([exemplar, instance])
                return np.full((2,2,3), 1 if time == 't2' else 0, np.float32)
            calls = []
            def sam3(**kw):
                calls.append(kw)
                present = (kw['category'] == 'low vegetation' and kw['time'] == 't1') or (kw['category'] == 'building' and kw['time'] == 't2')
                return [union] if present else []
            report = evaluate_task(row, dynamic, sam3, [0.,.6,.95])
            self.assertEqual(len(calls), 8)
            for mode, result in report['methods'].items():
                key = '0.6' if mode == 'sam_dino' else 'default'
                self.assertEqual(result['scores'][key]['f1_c'], 100)
                self.assertEqual(result['proposals']['gt_instances'], 1)
                self.assertEqual(result['proposals']['instance_recall_upper'], 1)
            summary = aggregate([report])
            self.assertEqual(summary['sam_dino']['scores']['0.6']['f1_c'], 100)
            self.assertIn('没有运行 Agent', conclusions(summary))
            json.dumps(summary)

    def test_subset_selection_is_fixed_before_names_filter(self):
        rows = [dict(id=f'train_{i}_{transition}', image_t1=f'/fake/train_{i}.png')
                for i in range(3) for transition in ('1to5','5to1')]
        chosen = select_rows(rows)
        subset = select_rows(rows, ['train_1.png'])
        self.assertEqual(subset, [r for r in chosen if r['id'].startswith('train_1_')])
        self.assertEqual(len(chosen), 3)
        with self.assertRaises(ValueError):
            select_rows(rows, ['outside.png'])


if __name__ == '__main__':
    unittest.main()
