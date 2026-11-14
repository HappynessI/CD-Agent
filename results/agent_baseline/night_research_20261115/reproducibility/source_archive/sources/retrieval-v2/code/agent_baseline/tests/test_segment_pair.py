import tempfile
import unittest
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.state import Instances
from agent_baseline.tests.helpers import rgb, segment, mask


class SegmentPairTests(unittest.TestCase):
    def test_opt_in_schema_and_execution(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment=segment)
            self.assertNotIn('segment_text_pair', [s['function']['name'] for s in session_schemas(box.session)])
            self.assertFalse(box.execute('segment_text_pair', {'text': 'building'}).public['ok'])
            self.assertIn('segment_text_pair', [s['function']['name'] for s in session_schemas(box.session, segment_pair=True)])

    def test_reuses_single_backend_in_time_order_without_setting_answer(self):
        calls = []
        def backend(time, text, tile, threshold):
            calls.append(time)
            rows, meta = segment(time, text, tile, threshold)
            return rows, dict(meta, backend='test', time=time)
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment=backend, segment_pair=True, init_mask=mask())
            result = box.execute('segment_text_pair', {'text': 'building'})
            self.assertTrue(result.public['ok'])
            self.assertEqual(calls, ['t1', 't2'])
            self.assertEqual(box.session.current_mask, 'mask#0')
            self.assertEqual(result.public['backend_invocations'], 2)
            self.assertEqual(len(result.internal['handles_created']), 2)
            self.assertTrue(result.internal['cache_hit'])
            self.assertEqual(set(result.internal['temporal_backends']), {'t1', 't2'})
            for time, row in result.public['results'].items():
                item = box.session.get(row['inst'], Instances)
                expected, _ = segment(time, 'building', 512, .5)
                self.assertEqual(item.time, time)
                self.assertEqual(item.items[0]['idx'].tolist(), expected[0]['idx'].tolist())

    def test_backend_failure_keeps_completed_handle_traceable(self):
        def backend(time, *args):
            if time == 't2':
                raise RuntimeError('observed service failure')
            return segment(time, *args)
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment=backend, segment_pair=True)
            result = box.execute('segment_text_pair', {'text': 'building'})
            self.assertEqual(result.public['error_code'], 'backend_error')
            self.assertEqual(result.internal['completed_times'], ['t1'])
            self.assertIn('inst', result.public['completed_results']['t1'])
            self.assertEqual(len(result.internal['handles_created']), 1)
            self.assertIsNone(box.session.current_mask)
