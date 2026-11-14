"""Small T5.4 contracts with real raster I/O and fake planner/SAM3/verifier only."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import save_mask_png, write_json, load_mask_png
from agent_baseline.map_input import (INSTRUCTION, render_map, map_mask, attach_map,
                                      verifier_settings)
from agent_baseline.analysis.make_t54_tasks import building_truth, generate, ENCODING
from agent_baseline.analysis.t54_fixed import route, run as fixed_run, resolve_filter, SETTINGS
from agent_baseline.analysis import t54_metrics
from agent_baseline.state import Session, Mask
from agent_baseline.tasks import read_tasks, load_pair, instruction_for, needs_init
from agent_baseline.scope_t13 import session_prompts
from agent_baseline.tools import Toolbox, session_schemas, SCHEMAS
from .helpers import call, response, ScriptClient, rows as instance_rows


def legacy_requests(directory):
    """Capture actual encoded planner/verifier requests under either code snapshot."""
    import copy
    import json
    from pathlib import Path
    import numpy as np
    from agent_baseline.backends import ApiAsk
    from agent_baseline.planner import Planner, clean_messages
    from agent_baseline.recorder import Recorder
    from agent_baseline.state import Session
    from agent_baseline.tools import Toolbox
    from agent_baseline.loop_json import run_session
    captured = []
    for gate in (1, 2, 3):
        initial = np.zeros((8, 8), bool)
        initial[:2, :2] = True
        rgb = [np.zeros((8, 8, 3), np.uint8)] * 2
        session = Session(initial.shape, initial)
        def verifier(payload):
            captured.append(copy.deepcopy(payload))
            return dict(choices=[dict(message=dict(role='assistant', content='{"change":"yes","type":"new_building","reason":"New."}'))], usage={})
        box = Toolbox(rgb, Path(directory), session=session)
        box.ask_backend = ApiAsk(session=session, client=verifier, correspondence=True)
        def planner(payload):
            captured.append(copy.deepcopy(payload))
            tool = dict(id='s', type='function', function=dict(name='submit', arguments=json.dumps(dict(mask='mask#0', summary='Keep.'))))
            return dict(choices=[dict(message=dict(role='assistant', tool_calls=[tool]))], usage={})
        recorder = Recorder(Path(directory), 'fake', 'x.png', gt=initial)
        final = run_session(box, Planner(client=planner, seed=7), recorder, 'T5.1',
            budget=3, prompt_version='v4', verify=True, gate2=gate >= 2, gate3=gate == 3)
        if final['submitted_by'] != 'model':
            raise AssertionError(final)
    for payload in captured:
        payload['messages'] = clean_messages(payload['messages'], encode=True)
    return captured


class T54Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        dataset = self.root / 'data'
        dataset.mkdir()
        write_json(dataset / 'cls_gray.json', ENCODING)
        old = np.zeros((16, 24), np.uint8)
        old[1:4, 2:5] = 1
        old[10:13, 15:18] = 1
        new = old.copy()
        new[1:4, 2:5] = 0
        new[2:5, 10:13] = 1
        for name in ('train_0.png', 'train_1.png', 'train_2.png'):
            actual_new = old if name == 'train_1.png' else new
            for part, array in [('slabel1', old), ('slabel2', actual_new),
                                ('label_unch0chi1', old ^ actual_new),
                                ('img2', np.full((*old.shape, 3), 80, np.uint8))]:
                path = dataset / part / name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(array).save(path)
        val, dev = self.root / 'val.txt', self.root / 'dev.txt'
        val.write_text('train_0.png\ntrain_1.png\n')
        dev.write_text('train_2.png\n')
        report = generate(dataset, self.root / 'run', val, dev)
        manifest = self.root / 'run/t54/tasks/T5.4_val.jsonl'
        return dataset, manifest, report, old.astype(bool), new.astype(bool)

    def test_truth_new_demolished_unchanged_and_bad_encoding(self):
        a, b = np.array([[0, 0, 1, 1]]), np.array([[0, 1, 0, 1]])
        truth = building_truth(a, b)
        np.testing.assert_array_equal(truth['any'], [[0, 1, 1, 0]])
        np.testing.assert_array_equal(truth['new'], [[0, 1, 0, 0]])
        np.testing.assert_array_equal(truth['demolished'], [[0, 0, 1, 0]])
        np.testing.assert_array_equal(building_truth(b, a)['new'], truth['demolished'])
        self.assertFalse(building_truth(a, a)['any'].any())
        for old, new in ((a, b[:, :2]), (a, np.array([[0, 1, 255, 0]]))):
            with self.assertRaises(ValueError):
                building_truth(old, new)

    def test_render_map_white_buildings_black_background(self):
        old = np.array([[0, 1], [1, 0]], bool)
        raster = render_map(old)
        self.assertEqual(raster.dtype, np.uint8)
        self.assertEqual(raster.shape, (2, 2, 3))
        np.testing.assert_array_equal(raster[old], np.full((2, 3), 255))
        np.testing.assert_array_equal(raster[~old], np.zeros((2, 3)))
        np.testing.assert_array_equal(map_mask(raster), old)
        with self.assertRaises(ValueError):
            map_mask(np.array([[1, 0]]))
        with self.assertRaises(ValueError):
            map_mask(np.array([[[0, 0, 255]]]))

    def test_generation_keeps_empty_truth_and_is_deterministic_without_img1(self):
        dataset, manifest, report, old, new = self.data()
        self.assertFalse((dataset / 'img1').exists())
        first = manifest.read_bytes()
        generate(dataset, self.root / 'run', self.root / 'val.txt', self.root / 'dev.txt')
        self.assertEqual(first, manifest.read_bytes())
        self.assertEqual(report['splits']['val']['images'], 2)
        self.assertEqual(report['splits']['val']['nonempty_truth'], 1)
        self.assertEqual(report['splits']['val']['xor_agreement'], 1.)
        rows = read_tasks(manifest)
        self.assertEqual(rows[0]['instruction'], INSTRUCTION)
        self.assertEqual(rows[0]['input_type'], 'map_image')
        self.assertFalse(needs_init('T5.4'))
        np.testing.assert_array_equal(map_mask(load_pair(rows[0])[0]), old)
        np.testing.assert_array_equal(load_mask_png(rows[0]['gt_mask']), old ^ new)
        self.assertFalse(load_mask_png(rows[1]['gt_mask']).any())

    def test_audit_records_change_label_disagreement_but_gt_uses_semantics(self):
        dataset, manifest, _, old, new = self.data()
        Image.fromarray(np.zeros(old.shape, np.uint8)).save(dataset / 'label_unch0chi1/train_0.png')
        report = generate(dataset, self.root / 'run', self.root / 'val.txt', self.root / 'dev.txt')
        audit = report['splits']['val']
        self.assertEqual(audit['xor_mismatched_pixels'], int((old ^ new).sum()))
        self.assertLess(audit['xor_agreement'], 1.)
        np.testing.assert_array_equal(load_mask_png(read_tasks(manifest)[0]['gt_mask']), old ^ new)

    def test_lists_reject_overlap_duplicates_and_test_images(self):
        dataset, _, _, _, _ = self.data()
        for content in ('train_2.png\n', 'train_0.png\ntrain_0.png\n', 'test_0.png\n'):
            (self.root / 'val.txt').write_text(content)
            with self.assertRaises(ValueError):
                generate(dataset, self.root / 'bad', self.root / 'val.txt', self.root / 'dev.txt')

    def test_map_handle_is_immutable_and_works_with_combine_and_view(self):
        _, manifest, _, old, _ = self.data()
        row = read_tasks(manifest)[0]
        box = Toolbox(load_pair(row), self.root / 'tools', map_input=True)
        session = box.session
        self.assertEqual(session.old_map, 'map#0')
        self.assertIsNone(session.current_mask)
        self.assertIsNone(session.initial_mask)
        self.assertEqual(session.next_id, 0)
        self.assertFalse(session.get('map#0', Mask).data.flags.writeable)
        other = session.add_mask(~old)
        result = box.execute('combine', dict(op='union', a='map#0', b=other))
        self.assertTrue(result.public['ok'])
        self.assertTrue(session.get(result.public['mask'], Mask).data.all())
        view = box.execute('view', dict(overlay='map#0', label_components=True))
        self.assertTrue(view.public['ok'])
        self.assertEqual(len(view.images), 2)
        # The displayed T1 comes from the map; tools never load the absent 2012 photograph.
        self.assertTrue(set(np.unique(box.rgb[0])).issubset({0, 255}))

    def test_map_optical_tool_errors_and_t2_works(self):
        _, manifest, _, _, new = self.data()
        called = []
        def segment(time, text, tile, score_thr):
            called.append((time, text, tile, score_thr))
            return instance_rows([new]), {}
        box = Toolbox(load_pair(read_tasks(manifest)[0]), self.root / 'tools',
                      segment=segment, map_input=True, tools_t13=True)
        for name, args in [('segment_text', dict(time='t1', text='building')),
                           ('segment_prompt', dict(time='t1', box=[1, 1, 4, 4])),
                           ('segment_everything', dict(time='t1')),
                           ('segment_semantic', dict(time='t1', text='building')),
                           ('extract_features', dict(time='t1'))]:
            result = box.execute(name, args)
            self.assertEqual(result.public['error_code'], 'unsupported_input')
        self.assertFalse(called)
        self.assertTrue(box.execute('segment_text', dict(time='t2', text='building')).public['ok'])
        self.assertEqual(called, [('t2', 'building', 512, .5)])

    def test_map_is_recorded_as_input_and_never_as_auto_answer(self):
        from agent_baseline.loop_json import run_session, latest_mask, referenced_handles
        from agent_baseline.planner import Planner
        from agent_baseline.recorder import Recorder
        old = np.zeros((8, 8), bool)
        old[:2, :2] = True
        box = Toolbox([render_map(old), render_map(~old)], self.root / 'trace',
                      map_input=True, ask=lambda *args: self.fail('Input map must not be auto-verified'))
        self.assertIsNone(latest_mask(box.session))
        self.assertEqual(referenced_handles(dict(a='map#0', b='mask#0')), ['map#0', 'mask#0'])
        client = ScriptClient([response(), response(), response()])
        final = run_session(box, Planner(client=client),
            Recorder(self.root / 'trace', 'fake', 'x.png', gt=np.zeros_like(old)),
            'T5.4', budget=5, prompt_version='v4', verify=True, gate3=True)
        self.assertEqual(final['termination_reason'], 'no_submit')
        self.assertEqual(final['fallback_kind'], 'empty')
        self.assertEqual(final['change_area_px'], 0)
        candidate = box.session.add_mask(old)
        self.assertEqual(latest_mask(box.session), candidate)

    def test_map_schema_and_verifier_scope_do_not_mutate_default(self):
        frozen = copy.deepcopy(SCHEMAS)
        plain = Session((2, 2))
        before = session_schemas(plain)
        session = Session((2, 2))
        attach_map(session, render_map(np.zeros((2, 2), bool)))
        self.assertIn('map#0', json.dumps(session_schemas(session)))
        self.assertEqual(session_schemas(plain), before)
        self.assertEqual(SCHEMAS, frozen)
        settings = session_prompts(session)
        self.assertEqual(settings, verifier_settings())
        for key in ('VERIFY', 'CORRESPONDENCE'):
            self.assertIn('old building map', settings[key])
            self.assertIn('white buildings', settings[key])
        self.assertEqual(instruction_for('T5.4', session), INSTRUCTION)

    def test_agent_messages_and_t1_tools_exclude_2016_labels(self):
        from agent_baseline.loop_json import run_session
        from agent_baseline.planner import Planner, clean_messages
        from agent_baseline.recorder import Recorder
        dataset, manifest, _, old, _ = self.data()
        row = read_tasks(manifest)[0]
        # Once the task is prepared, online code needs neither semantic label file.
        (dataset / 'slabel1/train_0.png').unlink()
        (dataset / 'slabel2/train_0.png').unlink()
        (dataset / 'label_unch0chi1/train_0.png').unlink()
        box = Toolbox(load_pair(row), self.root / 'trace', map_input=True)
        client = ScriptClient([response([call('submit', dict(mask='empty', summary='No changes.'))])])
        final = run_session(box, Planner(client=client),
            Recorder(self.root / 'trace', 'fake', row['id'], gt_path=row['gt_mask']),
            'T5.4', budget=1, prompt_version='v4', metadata=dict(label_2016='SECRET_NEW_LABEL', gt_mask=row['gt_mask']))
        self.assertEqual(final['submitted_by'], 'model')
        request = json.dumps(client.requests)
        self.assertIn('T1 is the old building map', request)
        self.assertIn('map#0', request)
        for forbidden in ('SECRET_NEW_LABEL', 'slabel2', 'label_unch0chi1', row['gt_mask']):
            self.assertNotIn(forbidden, request)
        encoded = clean_messages(client.requests[0]['messages'], encode=True)
        self.assertEqual(sum(block['type'] == 'image_url' for msg in encoded if isinstance(msg['content'], list)
                             for block in msg['content']), 2)
        self.assertTrue((self.root / 'trace/masks/map_0.png').exists())
        np.testing.assert_array_equal(load_mask_png(self.root / 'trace/masks/map_0.png'), old)

    def test_map_flag_disabled_keeps_rgb_handles_and_requests_unchanged(self):
        rgb = [np.full((2, 3, 3), 17, np.uint8)] * 2
        box = Toolbox(rgb, self.root, map_input=False)
        self.assertFalse(hasattr(box.session, 'old_map'))
        self.assertEqual(box.session.handles, {})
        np.testing.assert_array_equal(box.rgb[0], rgb[0])
        reference = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_t')
        if not (reference / 'agent_baseline/loop_json.py').is_file():
            self.skipTest('Read-only code_t snapshot unavailable')
        source = inspect.getsource(legacy_requests) + '\nimport json\nprint(json.dumps(legacy_requests(' + repr(str(self.root / 'wire')) + ')))\n'
        env = dict(os.environ, PYTHONPATH=str(reference), OVCD_CODE=str(reference / 'ovcd_v0'))
        result = subprocess.run([sys.executable, '-B', '-c', source], env=env,
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.dumps(legacy_requests(self.root / 'wire')).encode(),
                         json.dumps(json.loads(result.stdout)).encode())

    def test_t54_requires_map_and_valid_manifest(self):
        from agent_baseline.loop_json import run_session, main
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            run_session(Toolbox([np.zeros((2, 2, 3), np.uint8)] * 2, self.root), None, None, 'T5.4')
        with patch.object(sys, 'argv', ['loop', '--task', 'T5.4', '--config', 'fake']), patch(
                'agent_baseline.loop_json.names', side_effect=AssertionError('No default dataset access')):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        _, manifest, _, _, _ = self.data()
        row = read_tasks(manifest)[0]
        row['input_type'] = 'image_image'
        manifest.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'input_type'):
            read_tasks(manifest)

    def test_fixed_xor_fake_sam3_uses_t2_only_and_scores_unxored_reference(self):
        _, manifest, _, old, new = self.data()
        row = read_tasks(manifest)[0]
        calls = []
        def segment(time, text, tile, score_thr):
            calls.append((time, text, tile, score_thr))
            return instance_rows([new]), {}
        box = Toolbox(load_pair(row), self.root, segment=segment, map_input=True)
        pred, raw = route(box)
        np.testing.assert_array_equal(pred, old ^ new)
        np.testing.assert_array_equal(raw, new)
        self.assertEqual(calls, [('t2', 'building', 512, .5)])
        filtered, raw_filtered = route(box, min_area=10)
        self.assertFalse(filtered.any())
        np.testing.assert_array_equal(raw_filtered, new)
        result = fixed_run([row], self.root / 'fixed', segment_factory=lambda row: segment)
        self.assertEqual(result['methods']['xor']['score']['f1_c'], 100.)
        self.assertLess(result['methods']['buildings']['score']['f1_c'], 100.)
        np.testing.assert_array_equal(load_mask_png(self.root / 'fixed/xor/train_0.png'), old ^ new)

    def test_filter_settings_chosen_on_development_only(self):
        rows = [dict(id='train_0.png')]
        self.assertEqual(resolve_filter(rows), 0)
        with self.assertRaisesRegex(ValueError, 'development'):
            resolve_filter(rows, 16)
        self.assertEqual(resolve_filter(rows, 16, dev_names={'train_0.png'}), 16)
        path = self.root / 'selection.json'
        write_json(path, dict(SETTINGS, min_area=16, selected_on='dev'))
        self.assertEqual(resolve_filter(rows, frozen=path), 16)
        write_json(path, dict(SETTINGS, min_area=16, selected_on='val'))
        with self.assertRaises(ValueError):
            resolve_filter(rows, frozen=path)

    def test_rule_map_question_deletes_no_keeps_unparsed_and_resumes(self):
        from agent_baseline.analysis.t54_rule import run
        _, manifest, _, old, new = self.data()
        row = read_tasks(manifest)[0]
        save_mask_png(self.root / 'xor/train_0.png', old ^ new)
        captured = []
        def fake(images, spec, guard, seed, session):
            captured.append((len(images), spec, session_prompts(session)))
            return dict(answer='no' if len(captured) == 1 else None, cost_cny=.01)
        with patch('agent_baseline.analysis.second_rule.RunGuard'), patch('agent_baseline.analysis.second_rule.api_answer', side_effect=fake):
            report = run([row], self.root / 'xor', self.root / 'rule', 1.)
            self.assertEqual(report['no'], 1)
            self.assertEqual(report['unparsed'], 1)
            run([row], self.root / 'xor', self.root / 'rule', 1.)
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(pair[0] == 4 and pair[1]['model'] == 'qwen3.7-flash' for pair in captured))
        self.assertIn('old building map', captured[0][2]['CORRESPONDENCE'])
        self.assertEqual(load_mask_png(self.root / 'rule/any/masks/train_0.png').sum(), 9)

    def test_metrics_pooled_counts_pairing_empty_errors_and_missing_fail(self):
        _, manifest, _, _, _ = self.data()
        rows = read_tasks(manifest)
        spec = dict(manifest=str(manifest), baselines={key: str(self.root / key)
                    for key in ('fixed_xor', 't2_buildings', 'rule')},
                    agent_runs=[str(self.root / f'agent{n}') for n in range(3)])
        for row in rows:
            truth, stem = load_mask_png(row['gt_mask']), Path(row['id']).stem
            for directory in spec['baselines'].values():
                save_mask_png(Path(directory) / (stem + '.png'), truth)
            for directory in spec['agent_runs']:
                save_mask_png(Path(directory) / stem / 'pred.png', truth)
                write_json(Path(directory) / stem / 'final.json', dict(mask_path='pred.png', submitted_by='model'))
        report = t54_metrics.evaluate(spec, samples=8)
        self.assertEqual(report['methods']['agent_mean']['f1_c'], 100.)
        self.assertEqual(report['comparisons']['agent_minus_fixed_xor']['ci95'], [0., 0.])
        self.assertEqual(report['methods']['fixed_xor']['tp'], 18)
        self.assertEqual(report['images'], 2)
        # Error session with no mask is an explicit empty fallback, retained in scoring.
        directory = Path(spec['agent_runs'][0]) / 'train_0'
        write_json(directory / 'final.json', dict(submitted_by='error', fallback_kind='empty'))
        report = t54_metrics.evaluate(spec, samples=8)
        self.assertEqual(report['run_diagnostics'][0]['error_sessions'], 1)
        self.assertLess(report['methods']['agent_mean']['f1_c'], 100.)
        (directory / 'final.json').unlink()
        with self.assertRaises(FileNotFoundError):
            t54_metrics.evaluate(spec, samples=8)
        del spec['agent_runs']
        self.assertEqual(t54_metrics.evaluate(spec, samples=8)['methods']['fixed_xor']['f1_c'], 100.)
        (Path(spec['baselines']['rule']) / 'train_0.png').unlink()
        with self.assertRaises(FileNotFoundError):
            t54_metrics.evaluate(spec, samples=8)


if __name__ == '__main__':
    unittest.main()
