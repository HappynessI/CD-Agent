"""G4 unit contracts only; scripted backends, no model, API or dataset jobs."""
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from common import load_mask_png, read_jsonl
from agent_baseline.gate4 import TOOL_HINTS, latest_generated_mask, current_answer_reminder
from agent_baseline.loop_json import run_session, main as loop_main, system_prompt
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.tools_t13 import tool_schemas
from agent_baseline.state import Session, Mask
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.scope_t13 import session_prompts, scope_card, ASK_CANDIDATES
from agent_baseline.tasks import instruction_for
from agent_baseline.analysis.t13_ask_select import comparison_session, main as ask_main
from .helpers import tempdir, rgb, mask, segment, prompt, ask, call, response, ScriptClient
from .test_loop import assert_no_evidence


def compatibility_script():
    return [response([call('combine', dict(op='subtract', a='mask#0', b='region#0', set_current=False), 'c')]),
            response([call('submit', dict(mask='mask#0', summary='Keep.'), 's1')]),
            response([call('submit', dict(mask='mask#0', summary='Keep.'), 's2')])]


class G4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        network = patch('urllib.request.urlopen', side_effect=AssertionError('No network in G4 tests'))
        network.start()
        self.addCleanup(network.stop)

    def run_script(self, script, *, initial=True, region=None, budget=5, gate4=True, gate3=False, **kw):
        session = Session(mask().shape, mask() if initial else None, region_mask=region)
        box = Toolbox(rgb(), self.root, segment, prompt, ask, session=session)
        client = ScriptClient(script)
        recorder = Recorder(self.root, 'synthetic', 'x', gt=mask())
        final = run_session(box, Planner(client=client, seed=7, sleep=lambda _: None), recorder, budget=budget,
                            prompt_version='v4', verify=True, gate3=gate3, gate4=gate4, **kw)
        self.assertIsNone(final['error'], final['error'])
        for request in client.requests:
            assert_no_evidence(self, request)
        return final, box, client, recorder

    def test_off_g1_prompt_fixture_bytes(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/g1_prompts.json').read_text())
        session = Session(mask().shape)
        for task in ('T1.1xC0', 'T5.1'):
            expected = fixture[task]
            self.assertEqual(json.dumps(session_schemas(session, tool_hints='off')).encode(),
                             json.dumps(expected['schemas']).encode())
            self.assertEqual(system_prompt('v4', 30, session)[2].encode(), expected['system_prompt'].encode())
            self.assertEqual(instruction_for(task, session).encode(), expected['instruction'].encode())

    def test_off_g3_request_fixture_bytes(self):
        # Frozen code_t generated this complete planner-request fixture before G4.
        expected = json.loads((Path(__file__).parent/'fixtures/g3_requests.json').read_text())
        session = Session(mask().shape, mask(), region_mask=mask((3, 4, 6, 10)), change_direction='appeared')
        box = Toolbox(rgb(), self.root, segment, prompt, ask, session=session)
        client = ScriptClient(compatibility_script())
        recorder = Recorder(self.root, 'synthetic', 'x', gt=mask())
        final = run_session(box, Planner(client=client, seed=7), recorder, budget=5,
                            prompt_version='v4', verify=True, gate3=True, gate4=False, tool_hints='off')
        self.assertIsNone(final['error'])
        self.assertEqual(json.dumps(client.requests).encode(), json.dumps(expected).encode())
        self.assertNotIn('pre_region_mask_path', final)
        np.testing.assert_array_equal(load_mask_png(self.root/final['mask_path']), mask())

    def test_latest_mask_uses_call_order_excludes_given_and_region_and_is_pure(self):
        session = Session(mask().shape, mask(), region_mask=mask())
        a = session.add_mask(mask())
        b = session.add_mask(mask((15, 15, 18, 18)))
        calls = [dict(tool='setup', handles_created=['region#0', 'mask#0']),
                 dict(tool='combine', handles_created=[b], set_current=False),
                 dict(tool='remove_region', handles_created=[a], set_current=False),
                 dict(tool='view', input_handles=[b])]
        before = copy.deepcopy(calls)
        self.assertEqual(latest_generated_mask(session, calls), (a, 'remove_region'))
        self.assertEqual(calls, before)
        self.assertEqual(session.current_mask, 'mask#0')
        self.assertIsNone(latest_generated_mask(session, calls[:1]))

    def test_latest_counts_generated_mask_zero_without_given_mask(self):
        session = Session(mask().shape, region_mask=mask())
        key = session.add_mask(mask())
        self.assertEqual(key, 'mask#0')
        self.assertEqual(latest_generated_mask(session, [dict(tool='segment_prompt', handles_created=[key])]),
                         (key, 'segment_prompt'))

    def test_all_mask_tools_count_with_set_current_false(self):
        box = Toolbox(rgb(), self.root, segment, prompt, ask, init_mask=mask())
        calls = []
        actions = [('segment_text', dict(time='t1', text='building')),
                   ('to_mask', dict(sets=['inst#1'], set_current=False)),
                   ('combine', dict(op='union', a='mask#0', b='mask#2', set_current=False)),
                   ('remove_region', dict(mask='mask#3', ids=[1], set_current=False)),
                   ('add_region', dict(mask='mask#4', other='mask#0', set_current=False)),
                   ('segment_prompt', dict(time='t1', points=[[3, 4]], point_labels=[1])),
                   ('morph', dict(mask='mask#6', op='fill_holes', set_current=False)),
                   ('refine_region', dict(mask='mask#0', id=1, time='t1', set_current=False))]
        for name, args in actions:
            result = box.execute(name, args)
            self.assertTrue(result.public['ok'], result.internal)
            calls.append(dict(tool=name, handles_created=result.internal['handles_created']))
            if name != 'segment_text':
                self.assertEqual(latest_generated_mask(box.session, calls), (result.public['mask'], name))
            self.assertEqual(box.session.current_mask, 'mask#0')

    def test_reminder_conditions(self):
        session = Session(mask().shape, mask())
        self.assertEqual(current_answer_reminder(session, []), '')
        key = session.add_mask(mask((15, 15, 18, 18)))
        calls = [dict(tool='combine', handles_created=[key])]
        self.assertEqual(current_answer_reminder(session, calls),
            f'Your current answer is mask#0 (36 pixels). Your most recent mask is {key} (9 pixels), made by combine. Submit the one you mean.')
        session.current_mask = key
        self.assertEqual(current_answer_reminder(session, calls), '')
        session.current_mask = None
        self.assertTrue(current_answer_reminder(session, calls).startswith('Your current answer is empty.'))

    def test_late_report_reminder_trace_and_fallback_keeps_current(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c')]),
                  response([call('inspect', dict(mask='mask#1'), 'i')]), response(), response(), response()]
        final, box, client, _ = self.run_script(script)
        text = json.dumps(client.requests[2])
        self.assertIn('Your current answer is mask#0 (36 pixels).', text)
        self.assertIn('made by combine', text)
        self.assertEqual(final['final_handle'], 'mask#0')
        self.assertEqual(final['fallback_kind'], 'current')
        notes = [r for r in read_jsonl(self.root/'steps.jsonl') if r.get('note')]
        self.assertEqual(notes[0]['step'], 2)  # exactly REVIEW_REPLIES remain
        self.assertIn('made by combine', notes[0]['report'])

    def test_g2_candidate_report_still_sets_current_answer(self):
        # G4 keeps G2: with no current answer, the late report makes the latest mask the current answer.
        script = [response([call('segment_prompt', dict(time='t1', points=[[3, 4]], point_labels=[1]), 'p')]),
                  response(), response(), response()]
        final, _, client, _ = self.run_script(script, initial=False, budget=4)
        self.assertIn('You have no current answer yet', json.dumps(client.requests))
        self.assertNotIn('Your current answer is empty.', json.dumps(client.requests))
        self.assertNotEqual(final['fallback_kind'], 'empty')
        self.assertGreater(final['change_area_px'], 0)

    def test_early_submit_report_has_no_reminder(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        _, _, client, recorder = self.run_script(script, budget=8)
        report = recorder.steps[1]['calls'][0]['result_text']
        self.assertIn('Report before submission', report)
        self.assertNotIn('Your most recent mask', report)
        self.assertNotIn('Your most recent mask', json.dumps(client.requests))

    def test_last_reply_submit_report_reminds_and_grants_budget(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c'),
                            call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        final, _, client, recorder = self.run_script(script, budget=1)
        self.assertIn('made by combine', recorder.steps[0]['calls'][1]['result_text'])
        self.assertIn('made by combine', json.dumps(client.requests[1]))
        self.assertEqual(final['final_handle'], 'mask#0')
        self.assertEqual(json.loads((self.root/'gate.json').read_text())['grant']['kind'], 'last_reply')

    def test_budget_supplement_reminds(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')])]
        final, _, client, _ = self.run_script(script, budget=1)
        self.assertIn('made by combine', json.dumps(client.requests[1]))
        self.assertEqual(final['final_handle'], 'mask#0')
        self.assertEqual(json.loads((self.root/'gate.json').read_text())['grant']['kind'], 'budget')

    def test_no_reminder_when_latest_is_current(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0'), 'c')]),
                  response([call('submit', dict(mask='mask#1', summary='Keep.'), 's')])]
        _, _, client, _ = self.run_script(script, budget=4)
        self.assertNotIn('Your most recent mask', json.dumps(client.requests))

    def test_early_no_tools_report_does_not_add_late_reminder(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c')]),
                  response(), response(), response(),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')])]
        _, _, client, _ = self.run_script(script, budget=10)
        self.assertNotIn('Your most recent mask', json.dumps(client.requests))
        notes = [r for r in read_jsonl(self.root/'steps.jsonl') if r.get('note')]
        self.assertEqual(notes[0]['step'], 4)

    def test_region_submission_stores_both_masks_and_reports_removed_pixels(self):
        region = mask((3, 4, 6, 10))
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), 's1')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 's2')])]
        final, box, _, recorder = self.run_script(script, region=region, budget=4)
        np.testing.assert_array_equal(load_mask_png(self.root/final['pre_region_mask_path']), mask())
        np.testing.assert_array_equal(load_mask_png(self.root/final['mask_path']), mask() & region)
        self.assertEqual(final['region_clipped_px'], 18)
        self.assertEqual(final['pre_region_handle'], 'mask#0')
        self.assertEqual(final['submission_report'], 'Removed 18 pixels outside region#0 at submission')
        self.assertIn(final['submission_report'], recorder.steps[-1]['calls'][0]['result_text'])
        self.assertIn('Outside region#0', recorder.steps[0]['calls'][0]['result_text'])
        self.assertEqual(box.session.submitted, final['final_handle'])
        self.assertTrue(final['backends']['gate4'])
        self.assertTrue(final['backends']['gate3'])

    def test_region_fallback_clips_current_and_never_switches_to_candidate(self):
        region = mask((3, 4, 6, 10))
        script = [response([call('remove_region', dict(mask='mask#0', ids=[1], set_current=False), 'r')]),
                  response(), response(), response()]
        final, _, _, _ = self.run_script(script, region=region, budget=4)
        self.assertEqual(final['fallback_kind'], 'current')
        self.assertEqual(final['pre_region_handle'], 'mask#0')
        self.assertEqual(final['region_clipped_px'], 18)
        self.assertEqual(final['change_area_px'], 18)
        self.assertIn('Removed 18 pixels outside region#0 at submission', final['summary'])

    def test_zero_removed_adds_no_report_text(self):
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        final, _, _, recorder = self.run_script(script, region=mask(), budget=4)
        self.assertEqual(final['region_clipped_px'], 0)
        self.assertEqual(final['submission_report'], '')
        self.assertNotIn('Removed 0', json.dumps(recorder.steps))
        self.assertNotIn('report', json.loads(recorder.steps[-1]['calls'][0]['result_text']))
        self.assertTrue((self.root/final['pre_region_mask_path']).is_file())

    def test_no_region_submission_has_original_handle_and_no_clip_fields(self):
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        final, _, _, _ = self.run_script(script, budget=4)
        self.assertEqual(final['final_handle'], 'mask#0')
        self.assertNotIn('pre_region_mask_path', final)
        self.assertNotIn('region_clipped_px', final)

    def test_g4_retains_opposite_direction_pixels_and_report(self):
        session = Session(mask().shape, mask(), region_mask=mask(), change_direction='appeared')
        def opposite(pairs, question):
            return [dict(answer='no', type='demolished', reason='Removed.', p_yes=.1) for _ in pairs], {}
        box = Toolbox(rgb(), self.root, ask=opposite, session=session)
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                  response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
        recorder = Recorder(self.root, 'synthetic', 'x', gt=mask())
        final = run_session(box, Planner(client=ScriptClient(script)), recorder,
                            budget=4, verify=True, gate4=True)
        self.assertIsNone(final['error'])
        self.assertIn('Opposite-direction verifier types', recorder.steps[0]['calls'][0]['result_text'])
        self.assertEqual(final['change_area_px'], 36)
        self.assertEqual(final['region_clipped_px'], 0)

    def test_submit_empty_is_clipped_and_records_zero_removed(self):
        script = [response([call('submit', dict(mask='empty', summary='None.'), 's')]),
                  response([call('submit', dict(mask='empty', summary='None.'), 't')])]
        final, _, _, _ = self.run_script(script,
                                         region=mask(), budget=1)
        self.assertEqual(final['change_area_px'], 0)
        self.assertEqual(final['region_clipped_px'], 0)

    def test_submit_empty_report_treats_its_new_mask_as_latest(self):
        script = [response([call('combine', dict(op='union', a='mask#0', b='mask#0', set_current=False), 'c')]),
                  response([call('submit', dict(mask='empty', summary='None.'), 's')]),
                  response([call('submit', dict(mask='mask#2', summary='None.'), 't')])]
        final, _, _, recorder = self.run_script(script, budget=5)
        self.assertEqual(final['change_area_px'], 0)
        self.assertNotIn('Your most recent mask', recorder.steps[1]['calls'][0]['result_text'])
        self.assertEqual(recorder.steps[1]['calls'][0]['handles_created'], ['mask#2'])

    def test_hints_only_change_selected_descriptions(self):
        session = Session(mask().shape)
        for version in ('a', 'b'):
            self.assertEqual(session_schemas(session, tool_hints=version), session_schemas(session))
            base = session_schemas(session, tools_t13=True)
            hinted = session_schemas(session, tools_t13=True, tool_hints=version)
            for original, updated in zip(base, hinted):
                expected = copy.deepcopy(original)
                name = expected['function']['name']
                if name in TOOL_HINTS[version]:
                    expected['function']['description'] += ' ' + TOOL_HINTS[version][name]
                self.assertEqual(updated, expected)
            self.assertEqual([s['function']['parameters'] for s in tool_schemas(version)],
                             [s['function']['parameters'] for s in tool_schemas()])
        self.assertEqual(TOOL_HINTS['a']['compare_features'], "Class-agnostic change candidates for the whole image: segment all regions, then score how much each region's features changed between the two dates.")
        self.assertEqual(TOOL_HINTS['a']['segment_semantic'], 'A per-class mask for one date. Comparing the masks of the two dates shows where a class appeared or disappeared.')
        self.assertEqual(TOOL_HINTS['a']['match_exemplar'], 'Ranks candidate regions by similarity to the example change. Use it to find changes like the example.')
        for text in TOOL_HINTS['b'].values():
            self.assertTrue(text.startswith('Use '))
            self.assertLessEqual(len(text.split()), 20)

    def test_hints_change_only_request_schema_descriptions(self):
        from .test_tools_t13 import FakeVision, SHAPE
        def requests(version):
            initial = FakeVision().masks[0]
            box = Toolbox([np.zeros((*SHAPE, 3), np.uint8)]*2, self.root, ask=ask,
                          init_mask=initial, tools_t13=True)
            script = [response([call('submit', dict(mask='mask#0', summary='Keep.'), 's')]),
                      response([call('submit', dict(mask='mask#0', summary='Keep.'), 't')])]
            client = ScriptClient(script)
            final = run_session(box, Planner(client=client, seed=7), Recorder(self.root, 'x', 'x', gt=initial),
                                verify=True, gate3=True, budget=4, tool_hints=version)
            self.assertIsNone(final['error'])
            return client.requests
        base = requests('off')
        for version in ('a', 'b'):
            changed = requests(version)
            for expected, actual in zip(base, changed):
                for entry in actual['tools']:
                    name = entry['function']['name']
                    if name in TOOL_HINTS[version]:
                        entry['function']['description'] = entry['function']['description'].removesuffix(' ' + TOOL_HINTS[version][name])
                self.assertEqual(json.dumps(actual).encode(), json.dumps(expected).encode())

    def test_season_card_adds_only_requested_sentence_and_keeps_question(self):
        original, season = (scope_card('any change', scope=s) for s in ('any_change', 'any_change_season'))
        expected = copy.deepcopy(original)
        sentence = 'Seasonal color changes of vegetation, shadows and illumination differences are not changes.'
        expected['not_counts'].append(sentence)
        self.assertEqual(season, expected)
        baseline = session_prompts(comparison_session())
        changed = session_prompts(comparison_session('any_change_season'))
        self.assertIn(sentence, changed['SYSTEM'])
        self.assertNotIn(sentence, baseline['SYSTEM'])
        for key in ('VERIFY', 'CORRESPONDENCE', 'PSEUDO_TYPES'):
            self.assertEqual(changed[key], baseline[key])
        self.assertTrue(changed['VERIFY'].startswith(ASK_CANDIDATES['describe']))
        self.assertEqual(scope_card('any change'), original)

    def test_offline_explicit_scopes_run_describe_and_resume_separately(self):
        from PIL import Image
        from common import save_mask_png, write_json
        from agent_baseline.analysis import t13_ask_select as offline
        from agent_baseline.analysis.second_labels import ENCODING
        dataset = self.root/'dataset'
        write_json(dataset/'cls_gray.json', ENCODING)
        for part in ('slabel1', 'slabel2', 'img1', 'img2'):
            (dataset/part).mkdir()
            data = (np.full((1, 8), 1 if part == 'slabel1' else 5, np.uint8)
                    if part.startswith('slabel') else np.zeros((1, 8, 3), np.uint8))
            Image.fromarray(data).save(dataset/part/'a.png')
        candidates = self.root/'candidates'
        write_json(candidates/'complete.json', dict(phase='dev', names=['a.png'], threshold=.22,
                   backend_config={}, dataset=str(dataset.resolve())))
        save_mask_png(candidates/'masks/a.png', np.array([[0, 1, 1, 0, 0, 1, 0, 0]], bool))
        seen = []
        def verifier(images, spec, guard, seed, session):
            seen.append((session.ask_variant, session_prompts(session)))
            return dict(answer='yes', p_yes=.8, cost_cny=0)
        with patch.object(offline, 'inventories', return_value=dict(dev=['a.png'])):
            for scope in ('any_change', 'any_change_season'):
                output = self.root/scope
                result = offline.answer_questions(dataset, self.root, 'dev', candidates, output, 1,
                                                  answer_fn=verifier, scope=scope)
                self.assertEqual(list(result['reports']), ['describe'])
                self.assertEqual(result['policy']['scope'], scope)
                self.assertEqual(result['selected']['question'], 'describe')
                before = len(seen)
                offline.answer_questions(dataset, self.root, 'dev', candidates, output, 1,
                                         answer_fn=verifier, scope=scope)
                self.assertEqual(len(seen), before)
            with self.assertRaisesRegex(ValueError, 'Output settings differ'):
                offline.answer_questions(dataset, self.root, 'dev', candidates, self.root/'any_change', 1,
                                         answer_fn=verifier, scope='any_change_season')
        self.assertEqual([q for q, _ in seen], ['describe'] * 4)
        self.assertEqual(seen[0][1]['VERIFY'], seen[-1][1]['VERIFY'])
        self.assertNotEqual(seen[0][1]['SYSTEM'], seen[-1][1]['SYSTEM'])

    def test_cli_gate4_requires_stage(self):
        stderr = io.StringIO()
        with patch('sys.argv', ['loop_json.py', '--config', 'unused', '--gate4']), \
             patch('sys.stderr', stderr), self.assertRaises(SystemExit):
            loop_main()
        self.assertIn('--gate4 requires --verify-stage', stderr.getvalue())

    def test_offline_cli_scope_default_and_season(self):
        for scope in (None, 'any_change_season'):
            argv = ['t13_ask_select.py', 'answers', '--candidates-dir', str(self.root),
                    '--output-dir', str(self.root/'answers'), '--max-cost-cny', '1']
            if scope:
                argv += ['--scope', scope]
            with patch('sys.argv', argv), patch('agent_baseline.analysis.t13_ask_select.answer_questions') as run:
                ask_main()
            self.assertEqual(run.call_args.kwargs['scope'], scope or 'any_change')


if __name__ == '__main__':
    unittest.main()
