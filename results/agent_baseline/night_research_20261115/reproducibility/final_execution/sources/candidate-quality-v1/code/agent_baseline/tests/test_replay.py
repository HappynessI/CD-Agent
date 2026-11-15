"""Offline replay checks; synthetic sessions only, no API or model inference."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from agent_baseline.analysis.replay_g1 import compare, ReplayMismatch, Trace, replay
from agent_baseline.backends import ApiAsk
from agent_baseline.loop_json import RUN, run_session
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.tools import Toolbox
from .helpers import tempdir, rgb, mask, call, response, ScriptClient


class ReplayTests(unittest.TestCase):
    def test_first_difference_includes_base64_offset_and_both_sides(self):
        a = dict(messages=[dict(image_url=dict(url='data:image/jpeg;base64,AAAA'))])
        b = dict(messages=[dict(image_url=dict(url='data:image/jpeg;base64,AAAB'))])
        with self.assertRaises(ReplayMismatch) as error:
            compare(a, b, 'request[0]')
        detail = json.loads(str(error.exception))
        self.assertEqual(detail['location'], '$.messages[0].image_url.url')
        self.assertIn('AAAA', detail['expected'])
        self.assertIn('AAAB', detail['actual'])
        self.assertGreater(detail['offset'], 0)

    def make_trace(self, root):
        archive = root / 'T5.1' / 'val' / 'image'
        source = []
        for i, image in enumerate(rgb()):
            path = root / f't{i+1}.png'
            Image.fromarray(image).save(path)
            source.append(str(path))
        raw = dict(id='recorded-verifier', choices=[dict(message=dict(role='assistant',
            content='{"change":"yes","type":"new_building","reason":"Appeared."}'))], usage={})
        ask = ApiAsk(client=lambda _: raw, correspondence=True, model='qwen3.7-flash')
        box = Toolbox(rgb(), archive, ask=ask, init_mask=mask())
        submit = call('submit', dict(mask='mask#0', summary='Keep.'))
        client = ScriptClient([response([submit]), response([submit])])
        recorder = Recorder(archive, 'synthetic', 'image.png', gt=np.zeros_like(mask()))
        run_session(box, Planner(client=client, seed=7), recorder, 'T5.1', budget=5,
            prompt_version='v4', status=True, verify=True,
            metadata=dict(source_images=source, backends=dict(ask='api', ask_model='qwen3.7-flash',
                ask_correspondence=True, pixel_thr=.85, review=False, status=True, verify_stage=True)))
        return archive

    def test_synthetic_session_round_trip_and_prompt_drift(self):
        reference = RUN / 'code_g'
        if not (reference / 'agent_baseline/backends.py').is_file():
            self.skipTest('The frozen code_g source is required for verifier reconstruction')
        with tempdir() as tmp:
            root = Path(tmp)
            archive = self.make_trace(root)
            original = (archive / 'messages.jsonl').read_bytes()
            with patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
                result = replay(archive, root / 'replayed', reference)
            self.assertEqual(result['planner_requests'], 2)
            self.assertEqual(result['verifier_requests'], 1)
            self.assertEqual((archive / 'messages.jsonl').read_bytes(), original)
            import agent_baseline.loop_json as loop
            real = loop.system_prompt
            def changed(*args, **kwargs):
                path, template, text = real(*args, **kwargs)
                return path, template, text + '\nChanged prompt.'
            with patch.object(loop, 'system_prompt', changed), self.assertRaises(ReplayMismatch) as error:
                replay(archive, root / 'changed', reference)
            self.assertIn('$.messages[0].content', str(error.exception))

    def test_archived_jpeg_is_restored_instead_of_reencoded(self):
        with tempdir() as tmp:
            root = Path(tmp)
            archive = self.make_trace(root)
            trace = Trace(archive)
            request = trace.request(trace.planner_events[0]['request'])
            urls = [b['image_url']['url'] for m in request['messages'] if isinstance(m.get('content'), list)
                    for b in m['content'] if b.get('type') == 'image_url']
            self.assertTrue(urls)
            self.assertTrue(all(url.startswith('data:image/jpeg;base64,') for url in urls))
            self.assertEqual(trace.planner_count, 0)


if __name__ == '__main__':
    unittest.main()
