"""Small G3/T5.3 unit checks only: no model inference, network or real-data jobs."""
import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import components, load_mask_png, save_mask_png, write_json
from agent_baseline.state import Session, Mask
from agent_baseline.tools import Toolbox
from agent_baseline.feedback import CELLS, cell_at, cell_mask, public_feedback
from agent_baseline.tasks import INSTRUCTIONS, needs_init, t53_instruction, read_tasks
from agent_baseline.analysis.make_t52_feedback import error_regions
from agent_baseline.analysis import make_t53_feedback as generator, t53_fixed as fixed, t53_metrics as metrics
from agent_baseline.loop_json import run_session, main as loop_main
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from .helpers import tempdir, rows, call, response, ScriptClient, ask


def rectangle(shape, box):
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = True
    return out


class G3T53Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def box(self, shape=(180, 180), initial=None, gate3=True, **kwargs):
        rgb = [np.zeros((*shape, 3), np.uint8)] * 2
        return Toolbox(rgb, self.root, ask=ask, session=Session(shape, initial, **kwargs), gate3=gate3)

    def test_big_box_strict_boundary_override_and_disabled(self):
        box = self.box((100, 100))
        seen = []
        def prompter(time, **args):
            seen.append(args)
            return np.zeros((100, 100), bool), .9, {}
        box.prompt_backend = prompter
        self.assertTrue(box.execute('segment_prompt', dict(time='t1', box=[0, 0, 10, 100])).public['ok'])
        result = box.execute('segment_prompt', dict(time='t1', box=[0, 0, 11, 100])).public
        self.assertEqual(result['error_code'], 'box_too_large')
        self.assertIn('11.00%', result['message'])
        self.assertIn('segment_text', result['message'])
        self.assertEqual(len(seen), 1)
        box.big_box_share = .11
        self.assertTrue(box.execute('segment_prompt', dict(time='t1', box=[0, 0, 11, 100])).public['ok'])
        box.gate3 = False
        self.assertTrue(box.execute('segment_prompt', dict(time='t1', box=[0, 0, 100, 100])).public['ok'])
        box.gate3 = True
        self.assertTrue(box.execute('segment_prompt', dict(time='t1', points=[[90, 90]], point_labels=[1])).public['ok'])
        self.assertEqual(len(seen), 4)
        self.assertIsNone(seen[-1]['box'])

    def test_constraint_report_pixels_ids_and_no_edits(self):
        shape = (30, 30)
        initial = rectangle(shape, (1, 1, 5, 5)) | rectangle(shape, (20, 20, 24, 24))
        region = rectangle(shape, (0, 0, 3, 10))
        box = self.box(shape, initial, region_mask=region, change_direction='appeared')
        def verifier(pairs, question):
            return [dict(answer='no', type='demolished', p_yes=.1),
                    dict(answer='yes', type='new_building', p_yes=.9)], {}
        box.ask_backend = verifier
        report = box.pre_submit('mask#0', 2, stage=True).public['report']
        self.assertIn('Outside region#0: 24 pixels in component IDs [1, 2].', report)
        self.assertIn('Opposite-direction verifier types for appeared: component IDs [1]', report)
        np.testing.assert_array_equal(box.session.get('mask#0').data, initial)
        box.gate3 = False
        old_report = box.report('mask#0', 2)
        self.assertNotIn('Outside region#0', old_report)
        self.assertNotIn('Opposite-direction', old_report)

    def test_constraint_no_violation_adds_no_text_and_tree_mapping(self):
        shape = (20, 20)
        initial = rectangle(shape, (2, 2, 4, 4))
        for target, direction, bad_type in [('building', 'disappeared', 'new_building'),
                                            ('tree', 'disappeared', 'new_tree'),
                                            ('tree', 'appeared', 'removed_tree')]:
            box = self.box(shape, initial, target=target, region_mask=np.ones(shape, bool), change_direction=direction)
            box.execute('ask', {})
            entry = next(iter(box.asked.values()))
            entry['records'][-1]['type'] = 'no_change'
            self.assertEqual(box.constraint_report(box.session.get('mask#0')), [])
            new = box.report('mask#0', 2)
            box.gate3 = False
            self.assertEqual(new, box.report('mask#0', 2))
            box.gate3 = True
            entry['records'][-1]['type'] = bad_type
            entry['records'][-1]['answer'] = None  # type is usable even if change was unparsed
            self.assertIn('component IDs [1]', '\n'.join(box.constraint_report(box.session.get('mask#0'))))

    def test_cli_gate3_requires_stage_and_preserves_t52_error_text(self):
        for argv, expected in [(['--config', 'unused', '--gate3'], '--gate3 requires --verify-stage'),
                               (['--config', 'unused', '--task', 'T5.2'], 'T5.2 requires a feedback task manifest (--tasks)'),
                               (['--config', 'unused', '--big-box-share', '0'], '--big-box-share must be in (0, 1]')]:
            stderr = io.StringIO()
            with patch('sys.argv', ['loop_json.py', *argv]), patch('sys.stderr', stderr), self.assertRaises(SystemExit) as error:
                loop_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(expected, stderr.getvalue())

    def test_point_radius_and_default_ask_scope_explicit_ids_still_work(self):
        shape = (180, 180)
        initial = np.zeros(shape, bool)
        initial[10, [10, 42, 44, 100]] = True
        box = self.box(shape, initial, feedback_points=[[10, 10]])
        item = box.session.get('mask#0')
        self.assertEqual([c['id'] for c in box.pending(item)], [1, 2])
        result = box.execute('ask', {}).public
        self.assertEqual([r['id'] for r in result['results']], [1, 2])
        self.assertEqual(box.owed(item), [])
        self.assertEqual(box.stage('mask#0')[0], 0)
        self.assertEqual([r['id'] for r in box.execute('ask', dict(regions=[dict(mask='mask#0', ids=[4])])).public['results']], [4])
        box.gate3 = False
        self.assertEqual([c['id'] for c in box.pending(item)], [3])

    def test_feedback_changed_added_partial_deleted_components_and_stage(self):
        shape = (180, 180)
        initial = (rectangle(shape, (10, 10, 14, 14)) | rectangle(shape, (70, 10, 74, 14)) |
                   rectangle(shape, (140, 140, 144, 144)))
        box = self.box(shape, initial, feedback_cells=['upper left'])
        edited = initial.copy()
        edited[10, 70] = False  # <50% removed still counts as a modified component
        edited[140:144, 140:144] = False
        edited[80:84, 100:104] = True
        key = box.session.add_mask(edited)
        item = box.session.get(key)
        self.assertEqual(len(box.pending(item)), 3)  # cell, changed outline, addition
        self.assertEqual(len(box.owed(item)), 5)  # also original partial and deleted outlines
        checked, failed, _ = box.stage(key)
        self.assertEqual((checked, failed), (5, False))
        self.assertEqual(box.owed(item), [])
        np.testing.assert_array_equal(box.session.get(key).data, edited)
        # With no feedback, unchanged mask is not verified, but an outside edit is.
        empty = self.box(shape, initial, feedback_points=[])
        self.assertEqual(empty.pending(empty.session.get('mask#0')), [])
        self.assertEqual(len(empty.pending(empty.session.get(empty.session.add_mask(edited)))), 2)

    def test_g3_last_reply_gate_metadata_public_input_and_unchanged_schema(self):
        shape = (40, 40)
        initial = rectangle(shape, (2, 2, 5, 5)) | rectangle(shape, (30, 30, 34, 34))
        feedback = [dict(where='upper left', type='false_positive', region_id='SECRET_REGION', area=9876543, point=[3, 3])]
        row = dict(task_type='T5.3', meta=dict(feedback=feedback))
        self.assertEqual(public_feedback(row), dict(feedback_cells=['upper left']))
        box = self.box(shape, initial, instruction=t53_instruction(feedback), **public_feedback(row))
        box.big_box_share = .07
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), f's{n}')]) for n in (1, 2)]
        client = ScriptClient(script)
        recorder = Recorder(self.root, 'synthetic', 'image.png', gt=initial)
        final = run_session(box, Planner(client=client), recorder, task='T5.3', budget=1,
                            verify=True, gate3=True, prompt_version='v4', metadata=dict(task_meta=row['meta']))
        self.assertEqual(final['submitted_by'], 'model')
        self.assertEqual(final['backends'], dict(gate3=True, gate2=True, big_box_share=.07))
        self.assertEqual(json.loads((self.root / 'final.json').read_text())['backends'], final['backends'])
        self.assertEqual(json.loads((self.root / 'gate.json').read_text())['grant']['kind'], 'last_reply')
        self.assertEqual(len(box.asked), 1)
        text = json.dumps(client.requests)
        for hidden in ('SECRET_REGION', '9876543', 'region_id'):
            self.assertNotIn(hidden, text)
        self.assertNotIn('(3, 3)', t53_instruction(feedback))
        with self.assertRaisesRegex(ValueError, 'verify-stage'):
            run_session(box, Planner(client=client), recorder, gate3=True)

    def test_g3_late_report_contains_constraints_and_limits_feedback(self):
        shape = (90, 90)
        initial = rectangle(shape, (2, 2, 5, 5)) | rectangle(shape, (70, 70, 74, 74))
        box = self.box(shape, initial, region_mask=rectangle(shape, (0, 0, 30, 30)),
                       feedback_cells=['upper left'], instruction=t53_instruction([]))
        script = [response([call('inspect', dict(mask='mask#0'), 'i')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')])]
        client = ScriptClient(script)
        run_session(box, Planner(client=client), Recorder(self.root, 'synthetic', 'x', gt=initial),
                    task='T5.3', budget=4, verify=True, gate3=True)
        self.assertIn('Outside region#0: 16 pixels in component IDs [2]', json.dumps(client.requests[1]))
        self.assertEqual(len(box.asked), 1)

    def test_grid_partition_nondivisible_sizes_and_boundaries(self):
        for shape in ((9, 9), (10, 11)):
            total = np.zeros(shape, int)
            for where in CELLS:
                mask = cell_mask(where, shape)
                total += mask
                for y, x in zip(*np.nonzero(mask)):
                    self.assertEqual(cell_at([int(x), int(y)], shape), where)
            np.testing.assert_array_equal(total, np.ones(shape, int))
        self.assertEqual(cell_at([3, 3], (9, 9)), 'center')
        self.assertEqual(cell_at([2, 2], (9, 9)), 'upper left')
        self.assertEqual(cell_at([10, 9], (10, 11)), 'lower right')

    def test_unique_cell_selection_error_types_ranking_and_sentences(self):
        shape = (90, 90)
        errors = [dict(type='missed', point=[3, 3], area=100, component_id=1),
                  dict(type='false_positive', point=[20, 20], area=50, component_id=2),
                  dict(type='missed', point=[40, 3], area=20, component_id=3),
                  dict(type='false_positive', point=[80, 80], area=25, component_id=4)]
        selected = generator.select_feedback(errors, shape)
        self.assertEqual([(e['where'], e['area']) for e in selected], [('lower right', 25)])
        self.assertIn('One of the marked changes in the lower right is not a building change.', t53_instruction(selected))
        self.assertIn('A building in the top was missed.', t53_instruction(generator.select_feedback(errors[:3], shape)))
        self.assertEqual(generator.select_feedback(errors[:2], shape), [])
        self.assertTrue(t53_instruction([]).endswith('Keep mask#0 unchanged.'))
        self.assertTrue(t53_instruction(selected).startswith(INSTRUCTIONS['T5.1']))
        self.assertTrue(needs_init('T5.3'))

    def tiny_dataset(self):
        shape = (30, 30)
        truth = rectangle(shape, (1, 1, 4, 4))
        initial = rectangle(shape, (22, 22, 26, 26))
        for name in ('a.png', 'b.png'):
            for part in ('A', 'B'):
                path = self.root / 'data/val_original' / part / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(path)
            save_mask_png(self.root / 'data/val_original/label' / name, truth if name == 'a.png' else initial)
            save_mask_png(self.root / 'init' / name, initial)
        generator.generate(self.root / 'data', self.root / 'init', self.root / 'run', self.root / 'feedback.json', 2)
        return self.root / 'run/t53/tasks/T5.3.jsonl'

    def test_generator_real_png_manifest_and_hidden_feedback(self):
        manifest = self.tiny_dataset()
        task_rows = read_tasks(manifest, 'T5.3')
        self.assertEqual(len(task_rows), 2)
        e = task_rows[0]['meta']['feedback'][0]
        self.assertEqual((e['type'], e['where']), ('false_positive', 'lower right'))
        self.assertTrue(load_mask_png(self.root / 'run/t53/regions/a/given_1.png').sum() == 16)
        self.assertEqual(task_rows[1]['meta']['feedback'], [])
        self.assertEqual(public_feedback(task_rows[1]), dict(feedback_cells=[]))
        self.assertNotIn('point', json.dumps(public_feedback(task_rows[0])))
        report = json.loads((self.root / 'feedback.json').read_text())
        self.assertEqual(len(report['images']['a.png']['errors']), 2)
        task_rows[0]['instruction'] += ' Point (1,1).'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in task_rows))
        with self.assertRaisesRegex(ValueError, 'language feedback'):
            read_tasks(manifest)

    def test_two_date_candidates_coverage_cell_and_deduplication(self):
        shape = (30, 30)
        initial = rectangle(shape, (1, 1, 5, 3))
        half = rectangle(shape, (1, 1, 5, 5))  # exactly 50% covered; excluded
        t1 = rectangle(shape, (5, 5, 8, 8))
        t2 = rectangle(shape, (1, 6, 3, 8))
        outside = rectangle(shape, (21, 21, 26, 26))
        feedback = [dict(where='upper left', type='missed')]
        pool = fixed.candidates(initial, feedback, rows([half, t1]), rows([t1, t2, outside]))
        self.assertEqual([(c['time'], c['area']) for c in pool], [('t1', 9), ('t2', 4)])
        pred, trace = fixed.correct(initial, feedback, pool)
        np.testing.assert_array_equal(pred, initial | t1)
        self.assertEqual(trace['selected'], 't1:2')
        calls = []
        def verifier(c):
            calls.append(c['uid'])
            return dict(p_yes=.1 if c['time'] == 't1' else .9)
        pred, trace = fixed.correct(initial, feedback, pool, 'verifier', verifier)
        np.testing.assert_array_equal(pred, initial | t2)
        self.assertEqual(len(calls), 2)

    def test_false_positive_candidates_smallest_probability_and_empty_policy(self):
        shape = (30, 30)
        large, small = rectangle(shape, (0, 0, 4, 4)), rectangle(shape, (6, 6, 8, 8))
        other = rectangle(shape, (20, 20, 23, 23))
        initial = large | small | other
        feedback = [dict(type='false_positive', where='upper left')]
        pool = fixed.candidates(initial, feedback, [], [])
        np.testing.assert_array_equal(fixed.correct(initial, feedback, pool)[0], small | other)
        pred, _ = fixed.correct(initial, feedback, pool, 'verifier', lambda c: dict(p_yes=.8 if c['area'] == 16 else .1))
        np.testing.assert_array_equal(pred, large | other)
        pred, trace = fixed.correct(initial, feedback, pool, 'verifier', lambda c: dict(p_yes=None))
        np.testing.assert_array_equal(pred, initial)
        self.assertEqual(trace['reason'], 'no_scored_candidates')
        for fb in ([], feedback):
            np.testing.assert_array_equal(fixed.correct(initial, fb, [])[0], initial)

    def test_fixed_runner_resumes_per_candidate_after_interruption(self):
        manifest = self.tiny_dataset()
        calls = []
        def verifier(images, spec, guard, seed, session):
            calls.append(seed)
            self.assertEqual(spec['model'], 'qwen3.7-flash')
            return dict(answer='no', p_yes=.1, cost_cny=0)
        with patch.object(fixed, 'levir_cache_name', side_effect=lambda r, s: r['id']), \
             patch.object(fixed, 'api_answer', side_effect=verifier):
            fixed.run(manifest, self.root / 'fixed', 'largest')
            fixed.run(manifest, self.root / 'fixed', 'verifier', max_cost=1)
            self.assertEqual(len(calls), 1)
            # Simulate output interruption after the answer checkpoint.
            (self.root / 'fixed/verifier/masks/a.png').unlink()
            fixed.run(manifest, self.root / 'fixed', 'verifier', max_cost=1)
            self.assertEqual(len(calls), 1)
        self.assertTrue((self.root / 'fixed/largest/masks/b.png').exists())
        self.assertTrue((self.root / 'fixed/verifier/answers.jsonl').exists())
        with self.assertRaisesRegex(ValueError, 'max-cost'):
            fixed.run(manifest, self.root / 'bad', 'verifier')

    def test_localization_xor_components_target_hits_and_empty(self):
        shape = (90, 90)
        truth = rectangle(shape, (1, 1, 5, 5))
        initial = rectangle(shape, (70, 70, 74, 74))
        errors = error_regions(truth, initial)
        feedback = [next(e for e in errors if e['type'] == 'missed')]
        pred = initial | truth | rectangle(shape, (40, 40, 42, 42))
        loc = metrics.localization_summary([metrics.localization(pred, initial, truth, feedback)])
        self.assertEqual(loc['component_precision'], .5)
        self.assertEqual(loc['described_region_edit_rate'], 1)
        loc = metrics.localization_summary([metrics.localization(initial, initial, truth, feedback)])
        self.assertIsNone(loc['component_precision'])
        self.assertEqual(loc['described_region_edit_rate'], 0)
        self.assertIsNone(metrics.localization_summary([metrics.localization(initial, initial, truth, [])])['described_region_edit_rate'])

    def test_metrics_all_rows_paired_bootstrap_and_missing_masks_fail(self):
        manifest = self.tiny_dataset()
        task_rows = read_tasks(manifest)
        spec = dict(manifest=str(manifest), largest_dir=str(self.root / 'largest'), verifier_dir=str(self.root / 'verifier'))
        for key in ('agent_runs', 'g2_runs', 't52_runs'):
            spec[key] = [str(self.root / key / str(n)) for n in (1, 2, 3)]
        for row in task_rows:
            initial = load_mask_png(row['init_mask'])
            for directory in (spec['largest_dir'], spec['verifier_dir']):
                save_mask_png(Path(directory) / row['id'], initial)
            for key in ('agent_runs', 'g2_runs', 't52_runs'):
                for root in spec[key]:
                    directory = Path(root) / Path(row['id']).stem
                    save_mask_png(directory / 'mask.png', initial)
                    write_json(directory / 'final.json', dict(mask_path='mask.png', submitted_by='model'))
        report = metrics.evaluate(spec, samples=10)
        self.assertEqual(set(report['methods']), {'given', 'largest', 'verifier', 'g2_t51', 't52', 'agent'})
        self.assertTrue(all(c['ci95'] == [0., 0.] for c in report['comparisons'].values()))
        self.assertIn('定位组件比例', metrics.markdown(report))
        self.assertEqual(report['methods']['agent']['mean']['delta_f1_given'], 0)
        (Path(spec['verifier_dir']) / 'a.png').unlink()
        with self.assertRaises(FileNotFoundError):
            metrics.evaluate(spec, samples=10)

    def test_default_spec_detects_existing_t52_names_and_keeps_partial_runs_visible(self):
        self.assertNotIn('t52_runs', metrics.default_spec(self.root))
        (self.root / 'json_qwen37flash_g2_t52_k1_s2/T5.2/val').mkdir(parents=True)
        spec = metrics.default_spec(self.root)
        self.assertEqual(len(spec['t52_runs']), 3)
        self.assertTrue(spec['t52_runs'][0].endswith('json_qwen37flash_g2_t52_k1_s1/T5.2/val'))


if __name__ == '__main__':
    unittest.main()
