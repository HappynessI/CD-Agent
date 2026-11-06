"""Tiny T5.2 unit checks. No models, APIs, Slurm, or real dataset processing."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import LEVIR, components, load_mask_png, save_mask_png, write_json
from prompts import verifier_prompts
from agent_baseline.tasks import INSTRUCTIONS, instruction_for, needs_init, read_tasks, t52_instruction, levir_cache_name
from agent_baseline.state import Session
from agent_baseline.tools import session_schemas
from agent_baseline.loop_json import system_prompt
from agent_baseline.backends import SharedPrompter
from agent_baseline.analysis.make_t52_feedback import error_regions, farthest_point, select_feedback, generate
from agent_baseline.analysis.t52_fixed import correct
from agent_baseline.analysis.t52_metrics import repair_and_outside, summarize, counts, paired_bootstrap, evaluate_group, markdown


class T52Tests(unittest.TestCase):
    def test_half_coverage_is_not_an_error(self):
        gt = np.zeros((8, 8), bool)
        gt[1:3, 1:3] = True
        initial = np.zeros_like(gt)
        initial[1:3, 2:4] = True
        self.assertEqual(error_regions(gt, initial), [])
        initial[2, 2] = False
        errors = error_regions(gt, initial)
        self.assertEqual([(e['type'], e['area']) for e in errors], [('missed', 3), ('false_positive', 3)])
        # Prediction purity is now one quarter, below the false-positive threshold.
        gt[2, 1:3] = False
        initial[2, 2] = True
        self.assertIn('false_positive', [e['type'] for e in error_regions(gt, initial)])

    def test_missed_region_is_uncovered_part_and_8_connected(self):
        gt = np.eye(5, dtype=bool)
        initial = np.zeros_like(gt)
        initial[0, 0] = True
        errors = error_regions(gt, initial)
        self.assertEqual(len(errors), 1)
        item = errors[0]
        self.assertEqual((item['type'], item['component_id'], item['component_area'], item['area']), ('missed', 1, 5, 4))
        self.assertFalse(initial[item['point'][1], item['point'][0]])

    def test_farthest_point_ties_and_image_edges(self):
        region = np.zeros((8, 9), bool)
        region[1:5, 2:6] = True
        self.assertEqual(farthest_point(region), [3, 2])
        self.assertEqual(farthest_point(np.ones((3, 3), bool)), [1, 1])
        region[:] = False
        region[1, 7] = region[5, 1] = True
        self.assertEqual(farthest_point(region), [7, 1])
        with self.assertRaises(ValueError):
            farthest_point(np.zeros((3, 3), bool))

    def test_k_ranks_mixed_error_area_with_stable_ties(self):
        gt = np.zeros((12, 12), bool)
        gt[0:3, 0:3] = True  # missed area 9
        gt[0:2, 10:12] = True  # missed area 4
        initial = np.zeros_like(gt)
        initial[6:10, 6:10] = True  # false positive area 16
        initial[10:12, 0:2] = True  # false positive area 4
        errors = error_regions(gt, initial)
        self.assertEqual([e['area'] for e in select_feedback(errors, 3)], [16, 9, 4])
        self.assertEqual([e['type'] for e in select_feedback(errors, 3)], ['false_positive', 'missed', 'missed'])
        self.assertEqual(select_feedback(errors, 1), errors[:1])
        self.assertEqual(select_feedback(list(reversed(errors)), 3), errors[:3])

    def test_no_errors(self):
        for mask in (np.zeros((3, 3), bool), np.ones((3, 3), bool)):
            self.assertEqual(error_regions(mask, mask), [])
        self.assertIn('The user reviewed the mask and pointed out no errors.', t52_instruction([]))
        self.assertTrue(t52_instruction([]).endswith('Keep mask#0 unchanged.'))

    def test_instruction_coordinates_only_and_given_mask(self):
        feedback = [dict(type='missed', point=[23, 17], area=7654321, region_id='gt:999'),
                    dict(type='false_positive', point=[7, 8], component_id=123456)]
        text = t52_instruction(feedback)
        self.assertTrue(text.startswith(INSTRUCTIONS['T5.1']))
        self.assertIn('The user clicked (23, 17): a building change is missed here.', text)
        self.assertIn('The user clicked (7, 8): this is not a building change.', text)
        self.assertIn('Correct only the places the user pointed out and keep the rest of mask#0 unchanged.', text)
        for hidden in ('7654321', 'gt:999', '123456'):
            self.assertNotIn(hidden, text)
        session = Session((32, 32), np.zeros((32, 32), bool), instruction=text)
        self.assertTrue(needs_init('T5.2'))
        self.assertEqual(session.initial_mask, 'mask#0')
        self.assertEqual(instruction_for('T5.2', session), text)

    def test_default_g1_instructions_prompts_and_schemas_unchanged(self):
        frozen = json.loads((Path(__file__).parent / 'fixtures/g1_prompts.json').read_text())
        for task in ('T1.1xC0', 'T5.1'):
            session = Session((3, 3), np.zeros((3, 3), bool) if needs_init(task) else None)
            self.assertEqual(instruction_for(task, session), frozen[task]['instruction'])
            self.assertEqual(system_prompt('v4', 30, session)[2], frozen[task]['system_prompt'])
            self.assertEqual(session_schemas(session), frozen[task]['schemas'])
        self.assertEqual(verifier_prompts(), frozen['verifier'])

    def test_fixed_uses_t2_instance_and_original_component_deletion(self):
        initial = np.zeros((8, 8), bool)
        initial[1, 1] = initial[2, 2] = True  # same 8-connected component
        initial[7, 7] = True
        calls = []
        def prompter(time, **kwargs):
            calls.append((time, kwargs))
            mask = np.zeros_like(initial)
            x, y = kwargs['points'][0]
            mask[y, x] = True
            return mask, .9, dict(backend='unit_prompter')
        feedback = [dict(type='missed', point=[4, 4]), dict(type='false_positive', point=[1, 1]),
                    dict(type='missed', point=[5, 5])]
        pred, trace = correct(initial, feedback, prompter, [dict(idx=np.array([4 * 8 + 4]))])
        self.assertEqual([c[0] for c in calls], ['t2', 't1'])
        self.assertTrue(all(c[1]['point_labels'] == [1] for c in calls))
        self.assertFalse(pred[1, 1] or pred[2, 2])
        self.assertTrue(pred[7, 7] and pred[4, 4] and pred[5, 5])
        self.assertEqual(trace[1]['removed_component'], 1)
        np.testing.assert_array_equal(correct(initial, [], prompter, [])[0], initial)
        self.assertEqual(len(calls), 2)

    def test_fixed_composition_ignores_click_order(self):
        initial = np.zeros((6, 6), bool)
        initial[1:3, 1:3] = True
        addition = initial.copy()
        addition[3, 3] = True
        def prompter(*args, **kwargs):
            return addition, .8, {}
        feedback = [dict(type='missed', point=[3, 3]), dict(type='false_positive', point=[1, 1])]
        a = correct(initial, feedback, prompter, [])[0]
        b = correct(initial, list(reversed(feedback)), prompter, [])[0]
        np.testing.assert_array_equal(a, addition)
        np.testing.assert_array_equal(a, b)

    def test_shared_prompter_keeps_split_name_and_one_positive_point(self):
        calls = []
        class FakePrompter:
            def points(self, split, name, time, points, labels):
                calls.append((split, name, time, points, labels))
                return np.ones((3, 3), bool), .75
        prompt = SharedPrompter(factory=FakePrompter).for_image('val', 'val_17.png')
        correct(np.zeros((3, 3), bool), [dict(type='missed', point=[1, 1])], prompt,
                [dict(idx=np.array([4]))])
        self.assertEqual(calls, [('val', 'val_17.png', 1, [[1, 1]], [1])])
        row = dict(image_t1=str(LEVIR / 'val_original/A/val_17.png'),
                   image_t2=str(LEVIR / 'val_original/B/val_17.png'), id='different_output_id')
        self.assertEqual(levir_cache_name(row, 'val'), 'val_17.png')

    def test_repair_half_thresholds_and_full_missed_component(self):
        gt = np.zeros((12, 12), bool)
        gt[1:3, 1:3] = True
        initial = np.zeros_like(gt)
        initial[1, 1] = True
        initial[8:10, 8:10] = True
        feedback = error_regions(gt, initial)
        final = initial.copy()
        final[1, 2] = True  # total GT coverage reaches 50%, uncovered region coverage is <50%
        final[9, 8:10] = False  # retains exactly 50%: not yet repaired
        local = repair_and_outside(final, initial, gt, feedback)
        self.assertEqual(local['repaired'], dict(missed=1, false_positive=0))
        final[8, 8] = False
        local = repair_and_outside(final, initial, gt, feedback)
        s = summarize([counts(final, gt)], [local])
        self.assertEqual(s['repair']['all'], dict(repaired=2, pointed=2, rate=1.))

    def test_outside_disk32_add_remove_and_f1(self):
        gt = np.zeros((85, 85), bool)
        gt[40, 40] = gt[0, 0] = True
        initial = np.zeros_like(gt)
        initial[0, 0] = True
        feedback = error_regions(gt, initial)
        final = initial.copy()
        final[0, 0] = False  # outside: remove TP
        final[0, 1] = True  # outside: add FP
        final[40, 72] = True  # exactly radius 32, excluded
        final[40, 73] = True  # radius 33, included
        local = repair_and_outside(final, initial, gt, feedback)
        self.assertEqual((local['outside_added_px'], local['outside_removed_px']), (2, 1))
        self.assertEqual(local['outside_given'], [1, 0, 0])
        self.assertEqual(local['outside_final'], [0, 2, 1])
        s = summarize([counts(final, gt)], [local])
        self.assertAlmostEqual(s['outside']['delta_f1'], -100, places=4)
        same = repair_and_outside(initial, initial, gt, feedback)
        self.assertEqual(summarize([counts(initial, gt)], [same])['outside']['delta_f1'], 0)

    def test_no_feedback_outside_is_whole_image(self):
        initial = np.ones((3, 3), bool)
        final = initial.copy()
        final[0, 0] = False
        local = repair_and_outside(final, initial, initial, [])
        self.assertEqual((local['outside_pixels'], local['outside_removed_px']), (9, 1))
        s = summarize([counts(final, initial)], [local])
        self.assertIsNone(s['repair']['all']['rate'])

    def test_bootstrap_shared_samples_across_repeats(self):
        baseline = np.array([[3, 1, 2], [10, 2, 1]])
        agent = np.array([baseline, baseline, baseline])
        self.assertEqual(paired_bootstrap(agent, baseline, samples=8), [0., 0.])
        agent[0, 0, 0] += 1
        self.assertEqual(paired_bootstrap(agent, baseline, samples=8), paired_bootstrap(agent, baseline, samples=8))

    def test_generation_and_report_io_and_missing_sessions(self):
        # A 6x6 unit fixture verifies real JSON/PNG I/O; it is not an experiment result.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            dataset, run = root / 'dataset', root / 'run'
            gt = np.zeros((6, 6), bool)
            gt[1:3, 1:3] = True
            initial = np.zeros_like(gt)
            initial[4:6, 4:6] = True
            for part, data in (('A', gt), ('B', gt), ('label', gt)):
                save_mask_png(dataset / f'val_original/{part}/image.png', data)
            init = root / 'given'
            save_mask_png(init / 'image.png', initial)
            output = root / 'feedback.json'
            report = generate(dataset, init, run, output, expected_images=1)
            self.assertEqual(report['connectivity'], 8)
            self.assertEqual(len(report['images']['image.png']['k3']), 2)
            manifest = run / 't52/tasks/T5.2_k3.jsonl'
            rows = read_tasks(manifest)
            self.assertEqual(rows[0]['instruction'], t52_instruction(report['images']['image.png']['k3']))
            self.assertEqual(set(rows[0]['meta']), {'feedback', 'k'})
            for item in report['images']['image.png']['k3']:
                p = run / 't52/regions/k3/image' / (item['region_id'].replace(':', '_') + '.png')
                self.assertEqual(int(load_mask_png(p).sum()), item['area'])
            spec = dict(manifest=str(manifest), fixed_dir=str(root / 'fixed'),
                agent_runs=[str(root / f'agent{n}') for n in range(3)],
                g2_runs=[str(root / f'g2_{n}') for n in range(3)])
            save_mask_png(root / 'fixed/image.png', gt)
            for index, directory in enumerate(spec['agent_runs'] + spec['g2_runs']):
                d = Path(directory) / 'image'
                save_mask_png(d / 'mask.png', gt)
                write_json(d / 'final.json', dict(mask_path='mask.png', submitted_by='error' if index == 2 else 'model', error=None))
            result = evaluate_group(spec, samples=8)
            self.assertEqual(result['methods']['agent']['mean']['repair']['all']['rate'], 1)
            self.assertEqual(result['methods']['agent']['diagnostics'][2], dict(error_sessions=1, fallback_sessions=1))
            self.assertEqual(result['comparisons']['fixed']['ci95'], [0., 0.])
            self.assertIn('g2_no_feedback s3', markdown({'k3': result}))
            write_json(root / 'report.json', result)
            self.assertEqual(json.loads((root / 'report.json').read_text())['images'], 1)
            missing = Path(spec['agent_runs'][1]) / 'image/final.json'
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate_group(spec, samples=8)


if __name__ == '__main__':
    unittest.main()
