"""Compute-node tests of the opt-in array subprocess and ordinary submit gate."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.loop_json import run_session
from agent_baseline.state import Mask
from agent_baseline.tools import Toolbox
from agent_baseline.summarize import summarize
from .helpers import tempdir, rgb, mask, rows, call, response, ScriptClient


class PythonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.box = Toolbox(rgb(), self.root / 'trace', init_mask=mask(), python=True)

    def test_only_named_instance_sets_are_sent(self):
        segment=lambda time,text,tile,score_thr:([dict(id=1,idx=np.array([0,1],np.int32),area=2,score=.9,bbox=[0,0,2,1])],{})
        self.box.segment_backend=segment
        kept=self.box.execute('segment_text',dict(time='t1',text='building')).public['inst']
        other=self.box.execute('segment_text',dict(time='t2',text='building')).public['inst']
        self.assertNotEqual(kept,other)
        code=f"assert list(instances)==[{kept!r}]\nprint('ok')"
        result=self.box.execute('python',dict(code=code))
        self.assertTrue(result.public['ok'],result.public)

    def test_readonly_arrays_and_save_mask(self):
        inst = self.box.session.add_instances(rows([mask()]), 't1')
        code = '''
for data in (masks['mask#0'], instances['inst#1'], images['t1']):
    assert not data.flags.writeable
    try:
        data.setflags(write=True)
        raise AssertionError('input was writable')
    except ValueError:
        pass
save_mask(masks['mask#0'] | instances['inst#1'][0])
print(images['t2'].shape)
'''
        self.assertEqual(inst, 'inst#1')
        result = self.box.execute('python', dict(code=code))
        self.assertTrue(result.public['ok'], result.public)
        key = result.public['masks'][0]['mask']
        self.assertEqual(key, 'mask#2')
        np.testing.assert_array_equal(self.box.session.get(key, Mask).data, mask())
        self.assertIsNone(self.box.session.submitted)
        self.assertIn('(32, 48, 3)', result.public['stdout'])

    def test_no_file_or_network_access_and_memory_limit(self):
        secret = self.root / 'private.txt'
        secret.write_text('private data')
        for code in (f'open({str(secret)!r}).read()',
                     f'import os\nos.remove({str(secret)!r})',
                     'import socket\nsocket.socket()',
                     'np.empty((3 * 1024**3,), dtype=np.uint8)'):
            with self.subTest(code=code):
                result = self.box.execute('python', dict(code=code))
                self.assertFalse(result.public['ok'], result.public)
                self.assertEqual(result.public['error_code'], 'python_failed')
                self.assertEqual(list(self.box.session.handles), ['mask#0'])
                self.assertNotIn('private data', result.text)
                self.assertTrue(secret.exists())

    def test_cpu_and_wall_limits(self):
        import agent_baseline.python_sandbox as sandbox
        result = self.box.execute('python', dict(code='while True: pass'))
        self.assertFalse(result.public['ok'])
        self.assertIn(result.public['error_code'], ('python_failed', 'python_timeout'))
        # A short test-specific wall timeout also covers subprocess cleanup.
        with patch.object(sandbox, 'WALL_SECONDS', .05):
            result = self.box.execute('python', dict(code='while True: pass'))
        self.assertEqual(result.public['error_code'], 'python_timeout')

    def test_new_handle_still_goes_through_verification_and_summary_counts(self):
        seen = []
        def verifier(pairs, question):
            seen.extend(pairs)
            return [dict(answer='yes', type='new_building', reason='Appeared.', p_yes=.8) for _ in pairs], {}
        self.box.ask_backend = verifier
        script = [response([call('python', dict(code='save_mask(masks["mask#0"])'))]),
                  response([call('submit', dict(mask='mask#1', summary='Keep.'))]),
                  response([call('submit', dict(mask='mask#1', summary='Keep.'))])]
        client = ScriptClient(script)
        recorder = Recorder(self.root / 'trace', 'synthetic', 'image.png', gt=mask())
        final = run_session(self.box, Planner(client=client), recorder, 'T5.1', budget=5,
                            prompt_version='v4', verify=True)
        self.assertEqual(final['submitted_by'], 'model')
        self.assertEqual(len(seen), 1)
        self.assertEqual(recorder.steps[1]['calls'][0]['status'], 'report')
        self.assertEqual(recorder.steps[2]['calls'][0]['status'], 'executed')
        self.assertIn('python', [s['function']['name'] for s in client.requests[0]['tools']])
        summary = summarize(self.root, 'synthetic', 'T5.1', gts={'trace.png': mask()})
        self.assertEqual(summary['python_sessions'], 1)
        self.assertEqual(summary['python_success_calls'], 1)
        self.assertEqual(summary['python_success_rate'], 1.)
        self.assertEqual(summary['python_session_success_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
