"""Tiny region-task contracts; no models, API, Slurm or production preprocessing."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from common import save_mask_png
from agent_baseline.analysis.make_second_boxes import make_boxes, region_mask, generate
from agent_baseline.analysis.second_box_baselines import generate as make_baselines
from agent_baseline.analysis.second_box_metrics import (outside_violation, counts, score,
    paired_bootstrap, evaluate_group, default_spec, markdown)
from agent_baseline.analysis.second_metrics import mask, direction_violation
from agent_baseline.state import Session, Mask, Instances, ToolError
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.tasks import read_tasks, instruction_for, BOX_INSTRUCTIONS
from agent_baseline.loop_json import latest_mask, referenced_handles, run_session, system_prompt
from agent_baseline.recorder import Recorder
from agent_baseline.planner import Planner
from .helpers import mask as rectangle, rgb, call, response, ScriptClient, SHAPE


class SecondBoxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self):
        run = self.root / 'run'
        names = ['train_1.png', 'train_2.png']
        (run / 'second/tasks').mkdir(parents=True)
        (run / 'second/second_val128.txt').write_text('\n'.join(names) + '\n')
        rows = []
        for name in names:
            new = rectangle((0, 0, 18, 24))
            demolished = rectangle((30, 0, 48, 32))
            for kind, data in [('new', new), ('demolished', demolished), ('c0', new | demolished)]:
                save_mask_png(run / 'second/gt' / kind / name, data)
            sources = [self.root / time / name for time in ('a', 'b')]
            for source, image in zip(sources, rgb()):
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(source)
            rows.append(dict(id=name, image_t1=str(sources[0]), image_t2=str(sources[1]), instruction=None,
                task_type='T1.1xC0', constraint=None, target='building', change_direction='any',
                scope='building_rebuild_excluded', init_mask=None, gt_mask=str(run / 'second/gt/c0' / name), meta={}))
        (run / 'second/tasks/T1.1xC0_any.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        boxes = self.root / 'second_boxes.json'
        document = generate(run, boxes, expected_count=2)
        return run, boxes, document

    def test_reproducible_boxes_size_and_half_intersection(self):
        names = [f'train_{n}.png' for n in range(8)]
        truth = rectangle((0, 0, 1, 1), (20, 30))
        loader = lambda name: dict(c0=truth, new=truth)
        a = make_boxes(names, loader)
        self.assertEqual(a, make_boxes(names[::-1], loader))
        self.assertNotEqual(a, make_boxes(names, loader, 1017))
        for kind in ('c2', 'c4'):
            self.assertEqual(sum(i[kind]['require_intersection'] for i in a['images'].values()), 4)
            for item in a['images'].values():
                row = item[kind]
                x0, y0, x1, y1 = row['box']
                self.assertTrue(12 <= x1 - x0 <= 21)
                self.assertTrue(8 <= y1 - y0 <= 14)
                region = region_mask((20, 30), row['box'])
                self.assertEqual(row['intersects_truth'], bool((truth & region).any()))
                if row['require_intersection']:
                    self.assertTrue(row['intersects_truth'])
        self.assertTrue(any(i[k]['attempts'] > 1 for i in a['images'].values() for k in ('c2', 'c4')))

    def test_empty_truth_waiver(self):
        truth = np.ones((10, 10), bool)
        a = make_boxes(['train_1.png', 'train_2.png'], lambda n: dict(c0=truth, new=~truth))
        for name, item in a['images'].items():
            selected = name in a['selected_for_intersection']
            self.assertEqual(item['c4']['empty_truth_waiver'], selected)
            self.assertFalse(item['c4']['require_intersection'])
            self.assertFalse(item['c4']['intersects_truth'])

    def test_generated_truth_manifest_instruction_and_conflict(self):
        run, output, document = self.prepare()
        before = output.read_bytes()
        generate(run, output, expected_count=2)
        self.assertEqual(before, output.read_bytes())
        for kind, task, direction, gt_kind in [('c2', 'T1.1xC2', 'any', 'c0'), ('c4', 'T1.1xC4', 'appeared', 'new')]:
            rows = read_tasks(run / 'second/tasks' / f'{task}_{direction}.jsonl')
            for row in rows:
                info = document['images'][row['id']][kind]
                region = mask(row['region_mask'])
                np.testing.assert_array_equal(region, region_mask(SHAPE, info['box']))
                np.testing.assert_array_equal(mask(row['gt_mask']), mask(run / 'second/gt' / gt_kind / row['id']) & region)
                session = Session(SHAPE, region_mask=region, change_direction=direction, scope=row['scope'])
                self.assertEqual(row['instruction'], instruction_for(task, session))
                self.assertEqual(row['instruction'], BOX_INSTRUCTIONS[task].format(box=info['box']))
                self.assertIsNone(row['init_mask'])
        with self.assertRaisesRegex(ValueError, 'differs'):
            generate(run, output, seed=23, expected_count=2)
        self.assertEqual(output.read_bytes(), before)
        region_path = run / 'second/regions/c2/train_1.png'
        save_mask_png(region_path, np.zeros(SHAPE, bool))
        with self.assertRaisesRegex(ValueError, 'differs'):
            generate(run, output, expected_count=2)

    def test_region_is_separate_from_answer_and_preserves_numbering(self):
        region = rectangle()
        session = Session(SHAPE, region_mask=region)
        self.assertEqual(session.region_mask, 'region#0')
        self.assertIsNone(session.current_mask)
        self.assertIsNone(session.initial_mask)
        self.assertIsNone(latest_mask(session))
        self.assertEqual(session.next_id, 0)
        self.assertFalse(session.get('region#0', Mask).data.flags.writeable)
        with self.assertRaises(ToolError):
            session.get('region#0', Instances)
        key = session.add_mask(~region)
        self.assertEqual(key, 'mask#0')
        self.assertEqual(latest_mask(session), key)
        with_init = Session(SHAPE, ~region, region_mask=region)
        self.assertEqual(with_init.initial_mask, 'mask#0')
        self.assertEqual(with_init.current_mask, 'mask#0')
        self.assertEqual(with_init.region_mask, 'region#0')
        self.assertEqual(with_init.next_id, 1)
        with self.assertRaises(ValueError):
            Session(SHAPE, region_mask=np.ones((1, 2)))

    def test_separate_readonly_source_and_output_root(self):
        source, source_boxes, document = self.prepare()
        snapshot = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
        destination = self.root / 'new_run'
        output = self.root / 'new_boxes.json'
        actual = generate(destination, output, expected_count=2, source_run=source)
        self.assertEqual(actual, document)
        self.assertEqual(output.read_bytes(), source_boxes.read_bytes())
        self.assertEqual(snapshot, {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()})
        for task, direction in [('T1.1xC2', 'any'), ('T1.1xC4', 'appeared')]:
            for row in read_tasks(destination / 'second/tasks' / f'{task}_{direction}.jsonl'):
                self.assertTrue(Path(row['region_mask']).is_relative_to(destination))
                self.assertTrue(Path(row['gt_mask']).is_relative_to(destination))
                self.assertTrue(Path(row['region_mask']).is_file())
                self.assertTrue(Path(row['gt_mask']).is_file())

    def test_combine_overlay_type_error_and_disabled_python(self):
        region = rectangle()
        session = Session(SHAPE, region_mask=region)
        answer = session.add_mask(np.ones(SHAPE, bool))
        box = Toolbox(rgb(), self.root, session=session)
        result = box.execute('combine', dict(a=answer, b='region#0', op='intersect'))
        self.assertTrue(result.public['ok'], result.public)
        np.testing.assert_array_equal(session.get(result.public['mask'], Mask).data, region)
        self.assertTrue(box.execute('view', dict(overlay='region#0')).public['ok'])
        error = box.execute('match_instances', dict(inst_t1='region#0', inst_t2='region#0'))
        self.assertEqual(error.public['error_code'], 'bad_handle')
        self.assertNotIn('python', [s['function']['name'] for s in session_schemas(session)])
        self.assertFalse(box.execute('python', dict(code='pass')).public['ok'])
        self.assertEqual(referenced_handles(dict(masks=[answer, 'region#0'])), [answer, 'region#0'])

    def run_region(self, script, verify=False, gate2=False, init=None, task='T1.1xC2'):
        region = rectangle((2, 3, 20, 22))
        session = Session(SHAPE, init, region_mask=region, change_direction='appeared' if task == 'T1.1xC4' else 'any',
                          scope='building_rebuild_excluded')
        asked = []
        def ask(pairs, question):
            asked.extend(pairs)
            return [dict(answer='yes', type='new_building', p_yes=.9) for _ in pairs], {}
        toolbox = Toolbox(rgb(), self.root, session=session, ask=ask)
        client = ScriptClient(script)
        final = run_session(toolbox, Planner(client=client), Recorder(self.root, 'synthetic', 'image.png', gt=region),
                            task=task, budget=len(script), verify=verify, gate2=gate2, prompt_version='v4')
        return session, final, client, asked

    def test_first_message_and_region_submission(self):
        session, final, client, _ = self.run_region([response([call('submit', dict(mask='region#0', summary='Use region.'))])], task='T1.1xC4')
        self.assertEqual(final['submitted_by'], 'model')
        self.assertEqual(final['final_handle'], 'region#0')
        self.assertEqual(final['mask_path'], 'masks/region_0.png')
        np.testing.assert_array_equal(mask(self.root / final['mask_path']), session.get('region#0', Mask).data)
        request = client.requests[0]
        text = request['messages'][1]['content'][0]['text']
        self.assertIn('Find the buildings that were newly built inside the region region#0 (box [2, 3, 20, 22]).', text)
        self.assertIn('region#0: area=342 pixels; bbox=[2, 3, 20, 22]', text)
        self.assertIn('exclusive right and bottom', text)
        self.assertNotIn('SECOND', json.dumps(request))
        self.assertNotIn(str(self.root), json.dumps(request))

    def test_region_only_never_becomes_automatic_candidate(self):
        session, final, _, asked = self.run_region([response() for _ in range(4)], verify=True, gate2=True)
        self.assertEqual(final['fallback_kind'], 'empty')
        self.assertEqual(final['change_area_px'], 0)
        self.assertEqual(asked, [])
        self.assertEqual(final['final_handle'], 'mask#0')
        self.assertFalse(np.array_equal(mask(self.root / 'masks/region_0.png'), mask(self.root / 'masks/mask_0.png')))

    def test_region_submission_uses_normal_verification(self):
        script = [response([call('submit', dict(mask='region#0', summary='Use region.'))]),
                  response([call('submit', dict(mask='region#0', summary='Confirmed region.'))])]
        _, final, _, asked = self.run_region(script, verify=True, gate2=True)
        self.assertEqual(final['submitted_by'], 'model')
        self.assertTrue(asked)

    def test_old_manifest_and_frozen_prompts_unchanged(self):
        run, _, _ = self.prepare()
        original = json.loads((run / 'second/tasks/T1.1xC0_any.jsonl').read_text().splitlines()[0])
        actual = read_tasks(run / 'second/tasks/T1.1xC0_any.jsonl')[0]
        self.assertEqual(actual, original)
        self.assertNotIn('region_mask', actual)
        fixture = json.loads((Path(__file__).parent / 'fixtures/g1_prompts.json').read_text())
        # Existing SECOND tests also check the frozen verifier body and scope card.
        from prompts import verifier_prompts
        self.assertEqual(verifier_prompts(), fixture['verifier'])
        for task in ('T1.1xC0', 'T5.1'):
            session = Session(SHAPE, rectangle() if task == 'T5.1' else None)
            self.assertEqual(session_schemas(session), fixture[task]['schemas'])
            self.assertEqual(system_prompt('v4', 30, session)[2], fixture[task]['system_prompt'])
            self.assertEqual(instruction_for(task, session), fixture[task]['instruction'])
        old = Session(SHAPE, rectangle())
        self.assertEqual(old.initial_mask, 'mask#0')
        self.assertEqual(old.next_id, 1)

    def test_task_policy_errors(self):
        run, _, _ = self.prepare()
        filename = run / 'second/tasks/T1.1xC4_appeared.jsonl'
        original = json.loads(filename.read_text().splitlines()[0])
        for key, value in [('region_mask', None), ('change_direction', 'any'), ('scope', None), ('target', 'tree')]:
            filename.write_text(json.dumps(dict(original, **{key: value})) + '\n')
            with self.assertRaises(ValueError):
                read_tasks(filename)
        with self.assertRaises(ValueError):
            instruction_for('T1.1xC4', Session(SHAPE, region_mask=rectangle(), scope='building_rebuild_excluded'))

    def test_violation_rates_and_pooled_counts(self):
        pred = np.array([[1, 1, 1], [0, 0, 0]], bool)
        truth = np.array([[1, 0, 0], [0, 0, 0]], bool)
        region = np.array([[1, 1, 0], [0, 0, 0]], bool)
        opposite = np.array([[0, 0, 1], [0, 0, 0]], bool)
        self.assertAlmostEqual(outside_violation(pred, region), 1/3)
        self.assertAlmostEqual(direction_violation(pred, opposite), 1/3)
        self.assertEqual(outside_violation(~np.ones_like(pred), region), 0)
        pooled = score([counts(pred, truth, region, opposite), counts(truth, truth, region, opposite)])
        self.assertEqual(pooled['f1_c'], 200/3)
        self.assertEqual(pooled['outside_violation'], .25)
        self.assertEqual(pooled['direction_violation'], .25)
        self.assertEqual(direction_violation(~np.ones_like(pred), opposite), 0)

    def test_paired_bootstrap_shared_repeats(self):
        base = np.array([[1, 1, 0, 0, 2, 0], [2, 0, 1, 0, 2, 0]])
        A = np.array([base, base * 2, base * 3])
        self.assertEqual(paired_bootstrap(A, A, samples=12), [0., 0.])
        self.assertEqual(paired_bootstrap(A, base, samples=12), [0., 0.])

    def test_real_file_baseline_and_metrics_pipeline_missing_session(self):
        run, _, _ = self.prepare()
        names = ['train_1.png', 'train_2.png']
        for kind, task in [('any', 'T1.1xC0'), ('appeared', 'T1.1xC1')]:
            for name in names:
                data = mask(run / 'second/gt' / ('c0' if kind == 'any' else 'new') / name)
                for part in ('fixed', 'rule'):
                    save_mask_png(run / 'second' / part / kind / Path(name).with_suffix('.png'), data)
                for n in (1, 2, 3):
                    directory = run / f'history_{kind}_s{n}' / task / 'val' / Path(name).stem
                    directory.mkdir(parents=True)
                    save_mask_png(directory / 'mask.png', data)
                    (directory / 'final.json').write_text(json.dumps(dict(mask_path='mask.png', submitted_by='error' if n == 1 else 'model')))
        dest = run / 'second/boxes_baselines'
        make_baselines(run, dest, run / 'second/fixed/any', run / 'second/fixed/appeared',
                       run / 'second/rule/any', run / 'second/rule/appeared', 'history')
        spec = default_spec(run, dest)
        reports = {}
        for kind, task, direction in [('c2', 'T1.1xC2', 'any'), ('c4', 'T1.1xC4', 'appeared')]:
            rows = read_tasks(run / 'second/tasks' / f'{task}_{direction}.jsonl')
            for n, directory in enumerate(spec['groups'][kind]['agent_runs'], 1):
                for row in rows:
                    session = Path(directory) / Path(row['id']).stem
                    session.mkdir(parents=True)
                    save_mask_png(session / 'mask.png', mask(row['gt_mask']))
                    (session / 'final.json').write_text(json.dumps(dict(mask_path='mask.png', submitted_by='budget' if n == 1 else 'model')))
            report = evaluate_group(spec['groups'][kind], samples=8)
            reports[kind] = report
            self.assertEqual(report['agent_mean']['f1_c'], 100.)
            self.assertEqual(report['agent_mean']['outside_violation'], 0.)
            self.assertEqual(report['agent_mean']['direction_violation'], 0.)
            self.assertEqual(report['comparisons']['posthoc']['ci95'], [0., 0.])
            self.assertEqual(report['run_diagnostics'][0]['fallback_sessions'], 2)
            self.assertEqual(report['posthoc_diagnostics']['posthoc_s1']['error_sessions'], 2)
            self.assertEqual(set(report['comparisons']), set(spec['groups'][kind]['baselines']))
        table = markdown(reports)
        self.assertIn('Agent s3', table)
        self.assertIn('posthoc s3', table)
        # The active Agent output is evaluated as saved, without automatic clipping.
        active = Path(spec['groups']['c4']['agent_runs'][2]) / 'train_1/mask.png'
        save_mask_png(active, np.ones(SHAPE, bool))
        uncut = evaluate_group(spec['groups']['c4'], samples=8)
        self.assertGreater(uncut['agent_per_run'][2]['outside_violation'], 0.)
        self.assertGreater(uncut['agent_per_run'][2]['direction_violation'], 0.)
        missing = Path(spec['groups']['c2']['agent_runs'][1]) / 'train_1/final.json'
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_group(spec['groups']['c2'], samples=8)
        historical = run / 'history_any_s2/T1.1xC0/val/train_1/final.json'
        historical.unlink()
        with self.assertRaises(FileNotFoundError):
            make_baselines(run, dest, run / 'second/fixed/any', run / 'second/fixed/appeared',
                           run / 'second/rule/any', run / 'second/rule/appeared', 'history')


if __name__ == '__main__':
    unittest.main()
