"""G5 contracts with synthetic masks and scripted backends; no models or services."""
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import load_mask_png, read_jsonl
from agent_baseline import instances
from agent_baseline.analysis.g5_region_clip import read_clips, main as clips_main
from agent_baseline.loop_json import run_session, system_prompt, main as loop_main
from agent_baseline.make_demo import render
from agent_baseline.make_tools_page import build_catalog
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.state import Session, Instances
from agent_baseline.tasks import instruction_for
from agent_baseline.tools import Toolbox, SCHEMA_BY_NAME, session_schemas
from .helpers import tempdir, rgb, mask, rows, segment, prompt, ask, call, response, ScriptClient
from .test_g4 import compatibility_script
from .test_loop import assert_no_evidence


def schemas(session, **kwargs):
    return {s['function']['name']: s['function'] for s in session_schemas(session, **kwargs)}


class Gate5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        network = patch('urllib.request.urlopen', side_effect=AssertionError('No network in G5 tests'))
        network.start()
        self.addCleanup(network.stop)
        self.masks = np.asarray([mask((2, 2, 8, 8)), mask((20, 18, 28, 26)), mask((10, 10, 12, 12))])
        self.region = mask((0, 0, 5, 8))
        self.region[10, 10] = True
        self.full_rows = instances.from_dense(self.masks, [.9, .8, .7])
        self.backend_calls = []

    def segment(self, time, text, tile, score_thr):
        self.backend_calls.append(('segment_text', time, text, tile, score_thr))
        return self.full_rows, dict(cache_hit=True)

    def vision(self, operation, **kwargs):
        self.backend_calls.append((operation, kwargs))
        return self.masks

    def toolbox(self, gate5=True, region=None):
        session = Session(mask().shape, mask(), region_mask=self.region if region is None else region)
        return Toolbox(rgb(), self.root, self.segment, prompt, ask, session=session,
                       gate4=True, gate5=gate5, tools_t13=True, vision=self.vision)

    def execute(self, box, name, region='region#0', **kwargs):
        args = dict(time='t1')
        if name == 'segment_text':
            args['text'] = 'building'
        if region is not None:
            args['region'] = region
        return box.execute(name, dict(args, **kwargs))

    def run_script(self, directory, script, *, gate5=False, gate4=False, gate3=False, budget=5,
                   tools_t13=False, **kwargs):
        session = Session(mask().shape, mask(), region_mask=mask((3, 4, 6, 10)), change_direction='appeared')
        box = Toolbox(rgb(), directory, segment, prompt, ask, session=session,
                      tools_t13=tools_t13, vision=self.vision)
        client = ScriptClient(script)
        recorder = Recorder(directory, 'synthetic', 'x', gt=mask())
        final = run_session(box, Planner(client=client, seed=7, sleep=lambda _: None), recorder,
                            budget=budget, prompt_version='v4', verify=True,
                            gate5=gate5, gate4=gate4, gate3=gate3, **kwargs)
        self.assertIsNone(final['error'], final['error'])
        for request in client.requests:
            assert_no_evidence(self, request)
        return final, box, client, recorder

    def test_off_g1_schema_prompt_and_instruction_fixture_bytes(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/g1_prompts.json').read_text())
        session = Session(mask().shape)
        for task in ('T1.1xC0', 'T5.1'):
            expected = fixture[task]
            self.assertEqual(json.dumps(session_schemas(session, gate5=False)).encode(),
                             json.dumps(expected['schemas']).encode())
            self.assertEqual(system_prompt('v4', 30, session)[2].encode(), expected['system_prompt'].encode())
            self.assertEqual(instruction_for(task, session).encode(), expected['instruction'].encode())

    def test_off_g3_full_request_fixture_bytes(self):
        expected = json.loads((Path(__file__).parent / 'fixtures/g3_requests.json').read_text())
        final, _, client, _ = self.run_script(self.root, compatibility_script(), gate3=True)
        self.assertEqual(json.dumps(client.requests).encode(), json.dumps(expected).encode())
        self.assertNotIn('gate5', final['backends'])
        self.assertNotIn('pre_region_mask_path', final)

    def test_off_g4_full_request_fixture_bytes(self):
        # Captured with the supplied G4 tools.py and loop_json.py before G5 edits.
        expected = json.loads((Path(__file__).parent / 'fixtures/g4_requests.json').read_text())
        final, _, client, recorder = self.run_script(self.root / 'new', compatibility_script(), gate4=True)
        self.assertEqual(json.dumps(client.requests).encode(), json.dumps(expected).encode())
        self.assertEqual(final['region_clipped_px'], 18)
        self.assertNotIn('gate5', final['backends'])

    def test_on_without_region_matches_g4_requests_except_optional_schema(self):
        _, _, client, _ = self.run_script(self.root, compatibility_script(), gate5=True)
        expected = json.loads((Path(__file__).parent / 'fixtures/g4_requests.json').read_text())
        requests = copy.deepcopy(client.requests)
        for request in requests:
            for entry in request['tools']:
                if entry['function']['name'] == 'segment_text':
                    del entry['function']['parameters']['properties']['region']
        self.assertEqual(json.dumps(requests).encode(), json.dumps(expected).encode())

    def test_schema_changes_only_optional_region_and_does_not_mutate_shared_specs(self):
        box = self.toolbox()
        original_specs = copy.deepcopy(SCHEMA_BY_NAME)
        old = schemas(box.session, tools_t13=True)
        new = schemas(box.session, tools_t13=True, gate5=True)
        for name in ('segment_text', 'segment_everything'):
            params = new[name]['parameters']
            self.assertIn('region', params['properties'])
            self.assertNotIn('region', params['required'])
            self.assertIn('clip', params['properties']['region']['description'])
            del params['properties']['region']
        self.assertEqual(json.dumps(old).encode(), json.dumps(new).encode())
        self.assertEqual(SCHEMA_BY_NAME, original_specs)
        self.assertEqual(schemas(box.session, tools_t13=True), old)
        self.assertNotIn('segment_everything', schemas(box.session, gate5=True))

    def test_off_rejects_region_and_makes_no_backend_call(self):
        box = self.toolbox(gate5=False)
        for name in ('segment_text', 'segment_everything'):
            result = self.execute(box, name)
            self.assertEqual(result.public['error_code'], 'bad_args')
            self.assertIn('region is not supported', result.text)
        self.assertEqual(self.backend_calls, [])

    def test_clip_geometry_counts_ids_scores_and_one_pixel_instance(self):
        for name in ('segment_text', 'segment_everything'):
            with self.subTest(tool=name):
                box = self.toolbox()
                result = self.execute(box, name)
                self.assertTrue(result.public['ok'], result.text)
                key = result.public.get('inst', result.public.get('props'))
                item = box.session.get(key, Instances)
                self.assertEqual([r['id'] for r in item.items], [1, 3])
                self.assertEqual([r['area'] for r in item.items], [18, 1])
                self.assertEqual([r['bbox'] for r in item.items], [(2, 2, 5, 8), (10, 10, 11, 11)])
                self.assertEqual([r['score'] for r in item.items], [.9, .7] if name == 'segment_text' else [1., 1.])
                expected = instances.to_mask(self.full_rows, self.region.shape) & self.region
                np.testing.assert_array_equal(instances.to_mask(item.items, self.region.shape), expected)
                self.assertEqual(result.public['n'], 2)
                self.assertEqual(result.public['area'], 19)
                self.assertEqual(result.internal['region_clip'], dict(region='region#0', instances_before=3,
                    instances_after=2, dropped_instances=1, pixels_before=104, pixels_after=19, clipped_px=85, min_area_px=1))
                self.assertIn('Results clipped to region#0: removed 1 instances / 85 pixels.', result.text)
                self.assertEqual(item.parents, ('region#0',))
                self.assertEqual(box.session.current_mask, 'mask#0')

    def test_no_region_matches_gate4_and_does_not_add_clip_metadata(self):
        for name in ('segment_text', 'segment_everything'):
            old_box, new_box = self.toolbox(gate5=False), self.toolbox()
            old = self.execute(old_box, name, region=None)
            new = self.execute(new_box, name, region=None)
            self.assertEqual(old.text.encode(), new.text.encode())
            self.assertNotIn('region_clip', new.internal)
            key = new.public.get('inst', new.public.get('props'))
            for before, after in zip(old_box.session.get(key, Instances).items, new_box.session.get(key, Instances).items):
                np.testing.assert_array_equal(before['idx'], after['idx'])
                self.assertEqual(before['bbox'], after['bbox'])

    def test_cache_backend_arguments_and_source_masks_are_unchanged(self):
        for name in ('segment_text', 'segment_everything'):
            box = self.toolbox()
            first = self.execute(box, name)
            full = self.execute(box, name, region=None)
            self.assertTrue(first.public['ok'])
            self.assertEqual(full.public['n'], 3)
            self.assertEqual(full.public['area'], 104)
            self.assertEqual(self.backend_calls[-1], self.backend_calls[-2])
            if name == 'segment_text':
                self.assertTrue(first.internal['cache_hit'])
            for row, dense in zip(self.full_rows, self.masks):
                np.testing.assert_array_equal(row['idx'], np.flatnonzero(dense))
            self.assertEqual([r['area'] for r in self.full_rows], [36, 64, 4])

    def test_empty_and_full_regions_and_empty_backend_results(self):
        for name in ('segment_text', 'segment_everything'):
            for region, n, pixels in ((np.zeros_like(self.region), 0, 0), (np.ones_like(self.region), 3, 104)):
                result = self.execute(self.toolbox(region=region), name)
                self.assertTrue(result.public['ok'], result.text)
                self.assertEqual((result.public['n'], result.public['area']), (n, pixels))
                self.assertEqual(result.internal['region_clip']['clipped_px'], 104 - pixels)
            self.masks = np.zeros((0, *self.region.shape), bool)
            self.full_rows = []
            result = self.execute(self.toolbox(), name)
            self.assertTrue(result.public['ok'], result.text)
            self.assertEqual((result.public['n'], result.public['area']), (0, 0))
            self.assertEqual(result.internal['region_clip']['instances_before'], 0)
            # Restore the source for the next tool.
            self.masks = np.asarray([mask((2, 2, 8, 8)), mask((20, 18, 28, 26)), mask((10, 10, 12, 12))])
            self.full_rows = instances.from_dense(self.masks, [.9, .8, .7])

    def test_overlapping_instances_use_union_pixel_counts(self):
        self.masks = np.asarray([self.masks[0], self.masks[0]])
        self.full_rows = instances.from_dense(self.masks, [.9, .8])
        for name in ('segment_text', 'segment_everything'):
            result = self.execute(self.toolbox(), name)
            stats = result.internal['region_clip']
            self.assertEqual((stats['pixels_before'], stats['pixels_after'], stats['clipped_px']), (36, 18, 18))
            self.assertEqual(result.public['n'], 2)

    def test_other_same_image_region_mask_and_map_handles(self):
        for name in ('segment_text', 'segment_everything'):
            box = self.toolbox()
            other = mask((20, 18, 22, 20))
            handles = [box.session.add_mask(other), box.session.add_mask(other, handle='region#7'),
                       box.session.add_mask(other, handle='map#0')]
            for handle in handles:
                result = self.execute(box, name, region=handle)
                self.assertTrue(result.public['ok'], result.text)
                self.assertEqual(result.public['area'], 4)
                self.assertEqual(result.internal['region_clip']['region'], handle)

    def test_invalid_wrong_type_and_not_yet_visible_handles_fail_before_backend(self):
        for name in ('segment_text', 'segment_everything'):
            box = self.toolbox()
            inst = box.session.add_instances(self.full_rows, 't1')
            for handle in ('region#404', '', inst):
                result = self.execute(box, name, region=handle)
                self.assertEqual(result.public['error_code'], 'bad_handle')
                self.assertIn('Invalid region', result.public['message'])
            args = dict(time='t1', region='region#0', **({'text': 'building'} if name == 'segment_text' else {}))
            result = box.execute(name, args, allowed={'mask#0'})
            self.assertEqual(result.public['error_code'], 'bad_handle')
        self.assertEqual(self.backend_calls, [])

    def test_gate5_implies_reminder_and_clips_fallback_without_switching_answer(self):
        script = [response([call('remove_region', dict(mask='mask#0', ids=[1], set_current=False), 'r')]),
                  response(), response(), response()]
        final, _, client, _ = self.run_script(self.root, script, gate5=True, budget=4)
        self.assertIn('Your current answer is mask#0', json.dumps(client.requests))
        self.assertEqual(final['fallback_kind'], 'current')
        self.assertEqual(final['pre_region_handle'], 'mask#0')
        self.assertEqual(final['change_area_px'], 18)
        np.testing.assert_array_equal(load_mask_png(self.root / final['pre_region_mask_path']), mask())
        np.testing.assert_array_equal(load_mask_png(self.root / final['mask_path']), mask((3, 4, 6, 10)))
        for gate in ('gate2', 'gate3', 'gate4', 'gate5'):
            self.assertTrue(final['backends'][gate])

    def test_gate5_clips_explicit_and_empty_submissions(self):
        for handle in ('mask#0', 'empty'):
            script = [response([call('submit', dict(mask=handle, summary='Keep.'), 's')]),
                      response([call('submit', dict(mask=handle, summary='Keep.'), 't')])]
            final, _, _, _ = self.run_script(self.root / handle.replace('#', '_'), script, gate5=True)
            self.assertEqual(final['submitted_by'], 'model')
            self.assertIn('pre_region_mask_path', final)
            self.assertEqual(final['change_area_px'], 18 if handle == 'mask#0' else 0)

    def test_gate5_toolbox_constructor_enables_loop_gate4(self):
        box = self.toolbox()
        self.assertTrue(box.gate4)
        self.assertTrue(box.gate3)
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'))])] * 2
        final = run_session(box, Planner(client=ScriptClient(script)), Recorder(self.root, 'synthetic', 'x', gt=mask()),
                            budget=4, verify=True)
        self.assertIsNone(final['error'])
        self.assertTrue(final['backends']['gate5'])
        self.assertEqual(final['region_clipped_px'], int((mask() & ~self.region).sum()))

    def test_tool_hints_remain_independent(self):
        session = Session(mask().shape)
        off = schemas(session, tools_t13=True, gate5=True)
        for hint in ('a', 'b'):
            g4 = schemas(session, tools_t13=True, tool_hints=hint)
            g5 = schemas(session, tools_t13=True, tool_hints=hint, gate5=True)
            self.assertEqual(g4['compare_features'], g5['compare_features'])
            self.assertNotEqual(off['compare_features'], g5['compare_features'])

    def test_cli_gate5_requires_verification(self):
        stderr = io.StringIO()
        with patch('sys.argv', ['loop_json.py', '--config', 'unused', '--gate5']), \
             patch('sys.stderr', stderr), self.assertRaises(SystemExit):
            loop_main()
        self.assertIn('--gate5 requires --verify-stage', stderr.getvalue())

    def test_catalog_region_parameters_are_opt_in(self):
        for gate5 in (False, True):
            catalog = {r['name']: r for r in build_catalog(gate5=gate5)}
            for name in ('segment_text', 'segment_everything'):
                self.assertEqual('region' in catalog[name]['schema']['properties'], gate5)
                if gate5:
                    param = next(p for p in catalog[name]['parameters'] if p['name'] == 'region')
                    self.assertEqual(param['switch'], '--gate5')

    def test_recorder_replay_page_and_attribution_reader(self):
        script = [response([call('segment_text', dict(time='t1', text='building', region='region#0'), 'text')]),
                  response([call('segment_everything', dict(time='t1', region='region#0'), 'everything')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        source_paths = []
        for number, image in enumerate(rgb()):
            path = self.root / f't{number + 1}.png'
            Image.fromarray(image).save(path)
            source_paths.append(str(path))
        gt_path = self.root / 'gt.png'
        Image.fromarray(mask()).save(gt_path)
        _, _, _, recorder = self.run_script(self.root, script, gate5=True, tools_t13=True,
                                            metadata=dict(source_images=source_paths, gt_mask_path=str(gt_path)))
        records = list(read_clips(self.root))
        self.assertEqual([r['tool_call_id'] for r in records], ['text', 'everything'])
        calls = [c for step in read_jsonl(self.root / 'steps.jsonl') for c in step.get('calls', [])]
        for row, recorded, after in zip(records, calls, (18, 12)):
            self.assertEqual(recorded['args']['region'], 'region#0')
            self.assertEqual(row['pixels_after'], after)
            self.assertEqual(row['args'], recorded['args'])
            self.assertEqual(row['instances_after'], 1)
            self.assertEqual(row['clipped_px'], row['pixels_before'] - row['pixels_after'])
        self.assertEqual(recorder.steps[0]['calls'][0]['provenance']['region_clip']['pixels_before'], 36)
        output = self.root / 'clipping.jsonl'
        with patch('sys.argv', ['g5_region_clip', '--run', str(self.root), '--output', str(output)]):
            clips_main()
        self.assertEqual(read_jsonl(output), records)
        document = render(self.root, 'synthetic')
        for term in ('系统流程（G5）', 'G5 分割结果区域裁剪', '结果已裁到 region#0', '并集像素 36 → 18', 'G4 提交时区域求交'):
            self.assertIn(term, document)


if __name__ == '__main__':
    unittest.main()
