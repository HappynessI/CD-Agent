import tempfile
import unittest
from pathlib import Path
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.tests.helpers import rgb, mask, ScriptClient, response, call
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.loop_json import run_session


class AnswerStateTests(unittest.TestCase):
    def test_switch_preserves_mask_identity_and_verification(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, answer_state=True, init_mask=mask())
            candidate = box.session.add_mask(mask((20, 18, 28, 26)))
            original = box.session.get(candidate)
            box.asked['outline'] = ['existing verification']
            handles = dict(box.session.handles)
            result = box.execute('set_answer', {'mask': candidate})
            self.assertTrue(result.public['ok'])
            self.assertEqual(box.session.current_mask, candidate)
            self.assertIs(box.session.get(candidate), original)
            self.assertEqual(box.session.handles, handles)
            self.assertEqual(box.asked, {'outline': ['existing verification']})
            self.assertEqual(result.internal['handles_created'], [])
            self.assertTrue(box.execute('set_answer', {'mask': 'mask#0'}).public['ok'])
            self.assertEqual(box.session.current_mask, 'mask#0')

    def test_disabled_and_unavailable_handles_do_not_mutate(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, init_mask=mask())
            self.assertFalse(box.execute('set_answer', {'mask': 'mask#0'}).public['ok'])
            self.assertNotIn('set_answer', [s['function']['name'] for s in session_schemas(box.session)])
            box.answer_state = True
            candidate = box.session.add_mask(mask())
            for handle, allowed in [('mask#999', None), (candidate, {'mask#0'})]:
                self.assertFalse(box.execute('set_answer', {'mask': handle}, allowed).public['ok'])
                self.assertEqual(box.session.current_mask, 'mask#0')

    def test_loop_exposes_only_when_enabled_and_keeps_submit_report(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as d:
                box = Toolbox(rgb(), d, init_mask=mask())
                script = ([response([call('set_answer', {'mask': 'mask#0'})])] if enabled else [])
                script += [response([call('submit', {'mask': 'mask#0', 'summary': 'Keep.'})])] * 2
                client = ScriptClient(script)
                recorder = Recorder(Path(d), 'synthetic', 'image.png', gt=mask())
                final = run_session(box, Planner(client=client, sleep=lambda _: None), recorder,
                                    task='T5.1', review=True, answer_state=enabled)
                names = [s['function']['name'] for s in client.requests[0]['tools']]
                self.assertEqual('set_answer' in names, enabled)
                self.assertEqual(final['submitted_by'], 'model')
                self.assertEqual(recorder.steps[-2]['calls'][0]['status'], 'report')

    def test_no_current_and_wrong_type_have_recoverable_state(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, answer_state=True)
            key = box.session.add_mask(mask())
            result = box.execute('ask', {})
            self.assertFalse(result.public['ok'])
            self.assertEqual(result.public['recovery_state']['handles'][0]['handle'], key)
            self.assertIsNone(box.session.current_mask)
            result = box.execute('to_mask', {'sets': [key]})
            self.assertEqual(result.public['error_code'], 'bad_handle')
            self.assertIn('existing mask', result.public['message'])
            self.assertTrue(box.execute('set_answer', {'mask': key}).public['ok'])

    def test_deleted_ids_fail_with_current_ids_and_legal_delete_works(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, answer_state=True, init_mask=mask() | mask((30,20,35,25)))
            result = box.execute('remove_region', {'mask': 'mask#0', 'ids': [1]})
            self.assertTrue(result.public['ok'])
            self.assertEqual(result.public['valid_component_ids'], [2])
            key = result.public['mask']
            failed = box.execute('ask', {'regions': [{'mask': key, 'ids': [1]}]})
            self.assertEqual(failed.public['error_code'], 'bad_id')
            self.assertIn('valid IDs: [2]', failed.public['message'])
            self.assertTrue(box.execute('remove_region', {'mask': key, 'ids': [2]}).public['ok'])

    def test_summary_bounded_and_tracks_view_and_verification(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, answer_state=True, init_mask=mask())
            for _ in range(20):
                box.session.add_mask(mask())
            box.execute('view', {'overlay': 'mask#0'})
            state = box.state_summary()
            self.assertLessEqual(len(state['handles']), 13)
            self.assertEqual(state['observed_views'][0]['overlay'], 'mask#0')
            self.assertIn('verification', state['handles'][0])
            self.assertNotIn('gt', state)

    def test_identical_ask_runs_again_when_verification_progressed(self):
        from agent_baseline.tests.helpers import ask
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, ask=ask, init_mask=mask(), answer_state=True)
            client = ScriptClient([response([call('ask', {})]), response([call('ask', {})]),
                                   response([call('submit', {'mask': 'mask#0', 'summary': 'Checked.'})])])
            recorder = Recorder(Path(d), 'synthetic', 'image.png', gt=mask())
            run_session(box, Planner(client=client, sleep=lambda _: None), recorder, answer_state=True)
            self.assertEqual(recorder.steps[1]['calls'][0]['tool'], 'ask')

    def test_repeated_view_and_segment_are_not_unlocked_by_own_side_effects(self):
        from agent_baseline.tests.helpers import segment
        for tool, arguments in [('view', {}), ('segment_text', {'time': 't1', 'text': 'building'})]:
            with tempfile.TemporaryDirectory() as d:
                box = Toolbox(rgb(), d, segment=segment, init_mask=mask(), answer_state=True)
                action = response([call(tool, arguments)])
                client = ScriptClient([action, action, response([call('submit', {'mask': 'mask#0', 'summary': 'Keep.'})])])
                recorder = Recorder(Path(d), 'synthetic', 'image.png', gt=mask())
                run_session(box, Planner(client=client, sleep=lambda _: None), recorder, answer_state=True)
                self.assertEqual(len(recorder.steps[0]['calls']), 1)
                self.assertEqual(recorder.steps[1]['calls'], [])
                self.assertEqual(box.session.view_count, 2 if tool == 'view' else 1)
                self.assertEqual(len(box.session.handles), 2 if tool == 'segment_text' else 1)

    def test_same_ask_batch_runs_after_current_answer_switch(self):
        from agent_baseline.tests.helpers import ask
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, ask=ask, init_mask=mask(), answer_state=True)
            other = box.session.add_mask(mask((20,18,28,26)))
            batch = response([call('ask', {}, 'a'), call('set_answer', {'mask': other}, 's')])
            client = ScriptClient([batch, batch, response([call('submit', {'mask': other, 'summary': 'Checked new answer.'})])])
            recorder = Recorder(Path(d), 'synthetic', 'image.png', gt=mask())
            run_session(box, Planner(client=client, sleep=lambda _: None), recorder, answer_state=True)
            self.assertEqual(recorder.steps[1]['calls'][0]['tool'], 'ask')
            self.assertTrue(recorder.steps[1]['calls'][0]['ok'])
            self.assertIn(other, recorder.steps[1]['calls'][0]['result_text'])

    def test_deleted_components_keep_cached_verification_identity_without_backend(self):
        import copy
        from unittest.mock import Mock
        from agent_baseline.state import Mask
        backend = Mock(return_value=([
            dict(answer='no', type='no_change', reason='Unchanged.', p_yes=.1),
            dict(answer='no', type='no_change', reason='Unchanged.', p_yes=.1),
            dict(answer='yes', type='new_building', reason='New.', p_yes=.9)], {}))
        with tempfile.TemporaryDirectory() as d:
            initial = mask() | mask((20,4,26,10)) | mask((30,20,36,26))
            box = Toolbox(rgb(), d, ask=backend, init_mask=initial, answer_state=True)
            self.assertTrue(box.execute('ask', {}).public['ok'])
            edited = box.execute('remove_region', {'mask': 'mask#0', 'ids': [2]}).public['mask']
            item = box.session.get(edited, Mask)
            handles_before = dict(box.session.handles)
            records_before = copy.deepcopy(box.asked)
            backend.side_effect = AssertionError('Cached echo must not call the verifier')
            result = box.execute('ask', {})
            self.assertTrue(result.public['ok'])
            self.assertEqual(result.public['results'], [])
            self.assertEqual(result.public['remaining_unverified'], 0)
            cached = {r['id']: r for r in result.public['cached_verification']}
            self.assertEqual({k: r['answer'] for k, r in cached.items()}, {1: 'no', 3: 'yes'})
            self.assertEqual(cached[1]['bbox'], [3,4,9,10])
            self.assertEqual(cached[3]['bbox'], [30,20,36,26])
            self.assertTrue(all(r['mask'] == edited and r['cached'] for r in cached.values()))
            current = next(r for r in box.state_summary()['handles'] if r['handle'] == edited)
            self.assertEqual(current['verification_ids'], dict(yes=[3], no=[1], unparsed=[], none=[]))
            self.assertFalse(current['verification_ids_truncated'])
            self.assertEqual(backend.call_count, 1)
            self.assertIs(box.session.get(edited), item)
            self.assertEqual(box.session.handles, handles_before)
            self.assertEqual({k: v['records'] for k, v in box.asked.items()},
                             {k: v['records'] for k, v in records_before.items()})

    def test_cached_echo_default_off_preserves_old_response(self):
        from agent_baseline.tests.helpers import ask
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, ask=ask, init_mask=mask())
            box.execute('ask', {})
            result = box.execute('ask', {})
            self.assertEqual(result.public, dict(ok=True, results=[], remaining_unverified=0))

    def test_verification_id_summary_is_bounded(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            data = np.zeros((32,48), dtype=bool)
            data[::3, ::3] = True
            box = Toolbox(rgb(), d, init_mask=data, answer_state=True)
            row = box.state_summary()['handles'][0]
            self.assertEqual(sum(map(len, row['verification_ids'].values())), 64)
            self.assertTrue(row['verification_ids_truncated'])
