"""Tiny synthetic migration contracts; no model, API, data preprocessing or Slurm."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from prompts import scope_card, verifier_prompts
from common import components, save_mask_png
from agent_baseline.state import Session
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.backends import ApiAsk, SegmentBackend
from agent_baseline.fixed_routes import imc, run_manifest
from agent_baseline.tasks import read_tasks, instruction_for, C1_INSTRUCTIONS
from agent_baseline.loop_json import system_prompt
from agent_baseline.segment_cache import manifest_cache_path
from agent_baseline.analysis.second_labels import building_truth, ENCODING
from agent_baseline.analysis.make_second_tasks import sample_changed, generate, SUBTASKS
from agent_baseline.analysis.second_metrics import direction_violation, paired_bootstrap, evaluate_group
from agent_baseline.analysis.second_rule import apply_rule
from agent_baseline.analysis.second_t12_probe import annotated_counts


class SecondTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_label_conversion_including_unchanged_and_replacement(self):
        a = np.array([[0, 1, 5, 5, 2, 4]])
        b = np.array([[0, 5, 3, 5, 6, 5]])
        gt = building_truth(a, b)
        np.testing.assert_array_equal(gt['new'], [[0, 1, 0, 0, 0, 1]])
        np.testing.assert_array_equal(gt['demolished'], [[0, 0, 1, 0, 0, 0]])
        np.testing.assert_array_equal(gt['c0'], gt['new'] | gt['demolished'])
        for bad in ((a, b[:, :2]), (np.array([[7]]), np.array([[5]])),
                    (np.array([[0]]), np.array([[5]]))):
            with self.assertRaises(ValueError):
                building_truth(*bad)

    def test_reproducible_sampling_changed_only_independent_of_order(self):
        names = [f'train_{i}.png' for i in range(20)]
        loader = lambda n: {'c0': np.array([[int(Path(n).stem.split('_')[1]) % 2 == 1]])}
        a = sample_changed(names, loader, 5, 1016)
        self.assertEqual(a, sample_changed(names[::-1], loader, 5, 1016))
        self.assertTrue(all(loader(n)['c0'].any() for n in a))
        self.assertNotEqual(a, sample_changed(names, loader, 5, 2026))
        with self.assertRaises(ValueError):
            sample_changed(names, loader, 11)

    def test_fixed_route_direction_and_legacy_any_calls(self):
        class Box:
            def __init__(self):
                self.calls = []
                self.session = Session((2, 3))
            def execute(self, name, args):
                self.calls.append((name, args))
                if name == 'segment_text':
                    value = {'inst': args['time']}
                elif name == 'match_instances':
                    value = dict(appeared='a', disappeared='d')
                else:
                    m = np.zeros((2, 3), bool)
                    m[0, 0] = 'a' in args['sets']
                    m[1, 2] = 'd' in args['sets']
                    value = {'mask': self.session.add_mask(m)}
                return type('Result', (), {'public': dict(ok=True, **value)})()
        for direction, sets, area in [('any', ['a', 'd'], 2), ('appeared', ['a'], 1), ('disappeared', ['d'], 1)]:
            box = Box()
            self.assertEqual(imc(box, direction=direction).sum(), area)
            self.assertEqual(box.calls[-1], ('to_mask', {'sets': sets}))
            self.assertEqual(box.calls[0], ('segment_text', dict(time='t1', text='building', tile=512, score_thr=.5)))
            self.assertEqual(box.calls[2], ('match_instances', dict(inst_t1='t1', inst_t2='t2')))
        with self.assertRaises(ValueError):
            imc(Box(), direction='bad')

    def test_scope_zero_difference_and_rebuilt_reaches_all_prompts(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/g1_prompts.json').read_text())
        self.assertEqual(verifier_prompts(), fixture['verifier'])
        for direction in ('any', 'appeared', 'disappeared'):
            session = Session((4, 4), scope='building_rebuild_excluded', change_direction=direction)
            card = scope_card('building', direction, session.scope)
            self.assertNotIn('a building demolished and rebuilt (a different building in T2)', card['counts'])
            settings = verifier_prompts('building', direction, session.scope)
            texts = [settings['SYSTEM'], settings['CORRESPONDENCE'], system_prompt('v4', 30, session)[2],
                     next(s['function']['description'] for s in session_schemas(session) if s['function']['name'] == 'ask')]
            for text in texts:
                self.assertIn('rebuilt', text)
                # The clause lives in the scope card; the question itself stays the frozen wording.
                if text is not settings['CORRESPONDENCE']:
                    self.assertIn('answer no, type rebuilt', text)
                self.assertNotIn('SECOND', text)
            self.assertIn('rebuilt', settings['PSEUDO_TYPES'])
            if direction != 'any':
                self.assertEqual(len(card['counts']), 1)
                self.assertEqual(card['counts'], card['direction_counts'][direction])
            # API verifier must also use this scope, with no model request.
            requests = []
            reply = dict(choices=[dict(message=dict(content='{"change":"no","type":"rebuilt"}'))], usage={})
            api = ApiAsk(client=lambda p: requests.append(p) or reply, session=session, correspondence=True)
            result = api.one([], settings['VERIFY'], 1)[0]
            self.assertEqual(result['type'], 'rebuilt')
            self.assertEqual(requests[0]['messages'][0]['content'], settings['SYSTEM'])
            self.assertEqual(requests[0]['messages'][1]['content'][-1]['text'], settings['CORRESPONDENCE'])

    def test_generate_five_manifests_exact_instructions_and_fields(self):
        dataset = self.root / 'data'
        dataset.mkdir()
        (dataset / 'cls_gray.json').write_text(json.dumps(ENCODING))
        for part in ('img1', 'img2', 'slabel1', 'slabel2'):
            (dataset / part).mkdir()
        for i in range(3):
            for part in ('img1', 'img2'):
                Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(dataset / part / f'train_{i}.png')
            a = np.array([[0, 1], [5, 5]], np.uint8)
            b = np.array([[0, 5], [2, 5]], np.uint8) if i else a
            Image.fromarray(a).save(dataset / 'slabel1' / f'train_{i}.png')
            Image.fromarray(b).save(dataset / 'slabel2' / f'train_{i}.png')
        val = self.root / 'val.txt'
        val.write_text('train_0.png\ntrain_1.png\ntrain_2.png\n')
        run = self.root / 'run'
        names = self.root / 'names.txt'
        chosen = generate(dataset, val, run, names, count=2)
        self.assertEqual(chosen, ['train_1.png', 'train_2.png'])
        for task, direction in SUBTASKS:
            rows = read_tasks(run / 'second/tasks' / f'{task}_{direction}.jsonl')
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['scope'], 'building_rebuild_excluded')
                self.assertEqual(row['target'], 'building')
                self.assertEqual(row['change_direction'], direction)
                self.assertTrue(Path(row['gt_mask']).is_file())
                self.assertEqual(bool(row['init_mask']), task == 'T5.1xC1')
                session = Session((2,2), change_direction=direction, scope=row['scope'])
                self.assertEqual(instruction_for(task, session), row['instruction'])
                if task in C1_INSTRUCTIONS:
                    self.assertEqual(row['instruction'], C1_INSTRUCTIONS[task][direction])
        manifest = run / 'second/tasks/T5.1xC1_appeared.jsonl'
        row = json.loads(manifest.read_text().splitlines()[0])
        row['init_mask'] = None
        manifest.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'requires init_mask'):
            read_tasks(manifest)

    def test_manifest_cache_hit_without_worker(self):
        with patch.dict(os.environ, {'AGENT_RUN': str(self.root)}):
            source = self.root / 'image.png'
            path = manifest_cache_path(self.root, source, 'building', 512, .1, .85)
            path.parent.mkdir(parents=True)
            dense = np.array([[[1,0],[0,0]], [[0,0],[1,0]]], bool)
            np.savez(path, masks=np.packbits(dense.reshape(2,4), axis=1), scores=[.7,.4], shape=[2,2])
            backend = SegmentBackend('val', source.name, pixel_thr=.85, image_paths=[str(source), str(source)])
            rows, meta = backend('t1', 'building', 512, .5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['idx'].tolist(), [0])
            self.assertEqual(meta['backend'], 'sam3_manifest_cache')
            self.assertTrue(meta['cache_hit'])

    def test_fixed_route_reads_prepared_explicit_cache_and_writes_counts(self):
        with patch.dict(os.environ, {'AGENT_RUN': str(self.root)}):
            paths = [self.root / 'a.png', self.root / 'b.png']
            masks = [np.array([[1,0,0,0],[0,0,0,0]], bool), np.array([[0,0,0,0],[0,0,0,1]], bool)]
            for path, data in zip(paths, masks):
                Image.fromarray(np.zeros((2,4,3), np.uint8)).save(path)
                cache = manifest_cache_path(self.root, path, 'building', 512, .1, .85)
                cache.parent.mkdir(parents=True)
                np.savez(cache, masks=np.packbits(data.reshape(1,8),axis=1), scores=[.8], shape=[2,4])
            truth = self.root / 'truth.png'
            save_mask_png(truth, masks[1])
            row = dict(id='train_1.png', image_t1=str(paths[0]), image_t2=str(paths[1]),
                       target='building', change_direction='appeared', scope='building_rebuild_excluded', gt_mask=str(truth))
            output = self.root / 'fixed'
            report = run_manifest([row], output)
            self.assertEqual(report['per_image']['train_1'], dict(tp=1, fp=0, fn=0))
            np.testing.assert_array_equal(np.asarray(Image.open(output/'train_1.png')) > 0, masks[1])

    def test_c1_given_mask_loop_keeps_scope_and_hides_source_metadata(self):
        from .helpers import mask, rgb, ask, call, response, ScriptClient
        from agent_baseline.loop_json import run_session
        from agent_baseline.planner import Planner
        from agent_baseline.recorder import Recorder
        initial = mask()
        session = Session(initial.shape, initial, change_direction='appeared', scope='building_rebuild_excluded')
        directory = self.root / 'trace'
        box = Toolbox(rgb(), directory, ask=ask, session=session)
        client = ScriptClient([response([call('submit', dict(mask='mask#0', summary='Keep new buildings.'))])])
        recorder = Recorder(directory, 'synthetic', 'image.png', gt=initial)
        final = run_session(box, Planner(client=client), recorder, 'T5.1xC1', budget=1,
                            prompt_version='v4', metadata=dict(dataset='SECOND', gt_path='secret-label'))
        self.assertEqual(final['submitted_by'], 'model')
        self.assertTrue(final['identical_to_init'])
        request = json.dumps(client.requests)
        self.assertIn('answer no, type rebuilt', request)
        self.assertIn(C1_INSTRUCTIONS['T5.1xC1']['appeared'], request)
        self.assertNotIn('SECOND', request)
        self.assertNotIn('secret-label', request)

    def test_offline_rule_resumes_and_isolates_direction(self):
        from agent_baseline.analysis.second_rule import run
        mask_dir = self.root / 'input_masks'
        data = np.array([[1,0,1]], bool)
        save_mask_png(mask_dir / 'train_1.png', data)
        sources = [self.root/'a.png', self.root/'b.png']
        for path in sources:
            Image.fromarray(np.zeros((1,3,3), np.uint8)).save(path)
        row = dict(id='train_1.png', image_t1=str(sources[0]), image_t2=str(sources[1]),
                   target='building', change_direction='appeared', scope='building_rebuild_excluded')
        with patch('agent_baseline.analysis.second_rule.RunGuard'), patch('agent_baseline.analysis.second_rule.api_answer',
                return_value=dict(answer='no', type='rebuilt', cost_cny=.01)) as answer:
            out = self.root/'rules'
            report = run([row], mask_dir, out, 1, concurrency=1)
            self.assertEqual(report['rebuilt'], 2)
            self.assertEqual(answer.call_count, 2)
            self.assertFalse((np.asarray(Image.open(out/'appeared/masks/train_1.png')) > 0).any())
            run([row], mask_dir, out, 1, concurrency=1)
            self.assertEqual(answer.call_count, 2)
            self.assertFalse((out/'disappeared').exists())

    def test_direction_violation_and_empty(self):
        pred = np.array([[1, 1], [1, 0]], bool)
        opposite = np.array([[0, 1], [0, 1]], bool)
        self.assertAlmostEqual(direction_violation(pred, opposite), 1/3)
        self.assertEqual(direction_violation(np.zeros_like(pred), opposite), 0)

    def test_paired_bootstrap_zero_for_identical_masks(self):
        base = np.array([[1,2,3,0,3], [3,2,1,1,5]])
        self.assertEqual(paired_bootstrap(np.array([base]*3), base, samples=8), [0., 0.])

    def test_rule_deletes_only_no_and_keeps_unparsed(self):
        m = np.array([[1,0,1,0,1]], bool)
        labels, _ = components(m)
        result = apply_rule(m, labels, {1: {'answer':'no','type':'rebuilt'}, 2: {'answer':'yes'}, 3: {'answer':None}})
        np.testing.assert_array_equal(result, [[0,0,1,0,1]])

    def test_probe_excludes_unlabelled_pixels(self):
        label = np.array([[0, 5, 3]])
        pred = np.array([[1, 1, 1]], bool)
        self.assertEqual(annotated_counts(pred, label, 5), [1,1,0])

    def test_metrics_counts_errors_fallbacks_and_rebuilt(self):
        gt = np.array([[1,0],[0,0]], bool)
        opp = np.array([[0,1],[0,0]], bool)
        for part, data in [('new',gt), ('demolished',opp)]:
            save_mask_png(self.root / 'gt' / part / 'train_1.png', data)
        row = dict(id='train_1.png', image_t1='a.png', image_t2='b.png', instruction='Find newly built buildings.',
                   task_type='T5.1xC1', constraint=None, target='building', change_direction='appeared',
                   scope='building_rebuild_excluded', init_mask='init/train_1.png', gt_mask='gt/new/train_1.png', meta={})
        manifest = self.root / 'task.jsonl'
        manifest.write_text(json.dumps(row)+'\n')
        for part, data in [('fixed',gt), ('rule',gt), ('init',gt|opp)]:
            save_mask_png(self.root / part / 'train_1.png', data)
        runs = []
        for i in range(3):
            directory = self.root / f'run{i}' / 'train_1'
            directory.mkdir(parents=True)
            save_mask_png(directory / 'mask.png', gt)
            (directory / 'final.json').write_text(json.dumps(dict(submitted_by='error' if i==0 else 'model', mask_path='mask.png')))
            (directory / 'verification.json').write_text(json.dumps([dict(records=[dict(type='rebuilt')])]))
            runs.append(str(directory.parent))
        answers = self.root / 'answers.jsonl'
        answers.write_text(json.dumps(dict(uid='train_1:1', type='rebuilt'))+'\n')
        spec = dict(manifest=str(manifest), agent_runs=runs, fixed_dir=str(self.root/'fixed'),
                    rule_dir=str(self.root/'rule'), init_dir=str(self.root/'init'),
                    undirected_dir=str(self.root/'init'), rule_answers=str(answers))
        report = evaluate_group(spec, samples=8)
        self.assertEqual(report['agent_mean']['f1_c'], 100.)
        self.assertAlmostEqual(report['delta_f1_given'], 100-200/3)
        self.assertEqual(report['offline_rebuilt'], 1)
        self.assertEqual(sum(r['rebuilt_answers'] for r in report['run_diagnostics']), 3)
        self.assertEqual(report['run_diagnostics'][0]['error_sessions'], 1)
        self.assertEqual(report['run_diagnostics'][0]['fallback_sessions'], 1)
        self.assertEqual(report['comparisons']['fixed_direction']['ci95'], [0.,0.])
        (self.root/'run1/train_1/final.json').unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_group(spec, samples=8)


if __name__ == '__main__':
    unittest.main()
