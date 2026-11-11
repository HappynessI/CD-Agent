"""Small CPU-only tests; no API, Slurm or model imports."""
from io import BytesIO
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
from PIL import Image
from common import components
from agent_baseline.backends import ApiAsk, encode_image
from agent_baseline.ask_eval import api_answer, VARIANTS
from agent_baseline.map_input import attach_map, render_map, t2_state_change, verifier_settings, T2_STATE_QUESTION
from agent_baseline.planner import Planner, RunGuard, RunStopped
from agent_baseline.state import Session
from agent_baseline.tools import Toolbox
from agent_baseline.analysis.io_retry import read_image, save_image, write_json, write_text, retry_io
from agent_baseline.analysis.make_t54_noisy import noisy_map, translate, generate
from agent_baseline.analysis.t54_ask_compare import run as compare_run
from agent_baseline.tasks import read_tasks
from .helpers import tempdir


def raw(answer='yes', field='building', probability=.8):
    return dict(choices=[dict(message=dict(role='assistant', content=json.dumps(
        {field: answer, 'reason': 'Visible state.', 'p_yes': probability})))], usage={})


def map_session(old, variant=None):
    session = Session(old.shape, ask_variant=variant)
    attach_map(session, render_map(old))
    return session


class StateQuestionTests(unittest.TestCase):
    def test_four_direction_cases_and_probability(self):
        region = np.ones((2, 2), bool)
        for old in (False, True):
            for yes in (False, True):
                result = t2_state_change(dict(answer='yes' if yes else 'no', p_yes=.8 if yes else .2),
                                         np.full(region.shape, old), region)
                self.assertEqual(result['answer'], 'yes' if yes != old else 'no')
                p_state = .8 if yes else .2
                self.assertAlmostEqual(result['p_change'], 1-p_state if old else p_state)
                self.assertEqual(result['state_answer'], 'yes' if yes else 'no')
        self.assertIsNone(t2_state_change(dict(answer=None, p_yes=None), region, region)['answer'])
        self.assertIsNone(t2_state_change(dict(answer='yes', p_yes='bad'), region, region)['p_change'])

    def test_mixed_component_majority_is_explicit(self):
        region = np.ones((2, 2), bool)
        result = t2_state_change(dict(answer='yes', p_yes=.8), np.eye(2, dtype=bool), region)
        self.assertEqual(result['old_building_fraction'], .5)
        self.assertTrue(result['old_building'])
        self.assertEqual(result['answer'], 'no')

    def test_api_new_body_only_contains_t2_crops(self):
        images = [Image.fromarray(np.full((8, 8, 3), x, np.uint8)) for x in (0, 50, 100, 150)]
        session = map_session(np.zeros((8, 8), bool), 't2_state')
        captured = []
        backend = ApiAsk(client=lambda payload: captured.append(payload) or raw(), session=session, correspondence=True)
        answer, *_ = backend.one(images, 'unused', 1)
        blocks = captured[0]['messages'][1]['content']
        urls = [b['image_url']['url'] for b in blocks if b['type'] == 'image_url']
        self.assertEqual(urls, ['data:image/jpeg;base64,'+encode_image(images[i]) for i in (1, 3)])
        self.assertEqual(blocks[-1]['text'], T2_STATE_QUESTION)
        self.assertEqual(answer['answer'], 'yes')
        self.assertEqual(answer['p_yes'], .8)
        # Offline helper must send the same two images and question.
        def fake_planner(**kw):
            return Planner(client=lambda payload: captured.append(payload) or raw(), **kw)
        with patch('agent_baseline.ask_eval.Planner', side_effect=fake_planner):
            result = api_answer(images, VARIANTS['f37_corr'], None, 0, session)
        self.assertEqual(result['p_yes'], .8)
        self.assertEqual(captured[0]['messages'], captured[1]['messages'])

    def test_default_body_matches_frozen_question_bytes(self):
        images = [Image.fromarray(np.full((8, 8, 3), x, np.uint8)) for x in (0, 50, 100, 150)]
        captured = []
        backend = ApiAsk(client=lambda p: captured.append(p) or raw(field='change'),
                         session=map_session(np.zeros((8, 8), bool)), correspondence=True, model='qwen3.7-flash')
        backend.one(images, 'unused', 1)
        content = []
        for word, image in zip(('T1 close-up', 'T2 close-up', 'T1 wider view', 'T2 wider view'), images):
            content += [dict(type='text', text=word), dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+encode_image(image)))]
        settings = verifier_settings()
        content.append(dict(type='text', text=settings['CORRESPONDENCE']))
        expected = dict(model='qwen3.7-flash', temperature=0, enable_thinking=False, max_tokens=256,
                        seed=0, logprobs=True, top_logprobs=5, response_format={'type':'json_object'},
                        messages=[dict(role='system', content=settings['SYSTEM']), dict(role='user', content=content)])
        # JSON object ordering is transport-independent; string/image bytes are exact.
        self.assertEqual(json.dumps(captured[0], sort_keys=True).encode(), json.dumps(expected, sort_keys=True).encode())

    def test_default_online_and_offline_wire_matches_readonly_snapshot(self):
        reference = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_v')
        if not (reference/'agent_baseline/map_input.py').is_file():
            self.skipTest('Frozen code_v unavailable')
        source = """
import json
import numpy as np
from PIL import Image
from unittest.mock import patch
from agent_baseline.backends import ApiAsk
from agent_baseline.ask_eval import api_answer, VARIANTS
from agent_baseline.map_input import attach_map, render_map
from agent_baseline.state import Session
from agent_baseline.planner import Planner
s = Session((8,8)); attach_map(s, render_map(np.zeros((8,8),bool)))
images = [Image.fromarray(np.full((8,8,3),x,np.uint8)) for x in (0,50,100,150)]
wire = []
def client(p):
    wire.append(p)
    return dict(choices=[dict(message=dict(role='assistant', content='{"change":"yes"}'))], usage={})
ApiAsk(client=client,session=s,correspondence=True,model='qwen3.7-flash').one(images, 'unused', 1)
def factory(**kw): return Planner(client=client, **kw)
with patch('agent_baseline.ask_eval.Planner', side_effect=factory):
    api_answer(images, VARIANTS['f37_corr'], None, 0, s)
print(json.dumps(wire))
"""
        def capture(root):
            env = dict(os.environ, PYTHONPATH=str(root), OVCD_CODE=str(root/'ovcd_v0'), PYTHONDONTWRITEBYTECODE='1')
            return subprocess.run([sys.executable, '-B', '-c', source], env=env, check=True, capture_output=True).stdout
        self.assertEqual(capture(Path(__file__).resolve().parents[2]), capture(reference))

    def test_tool_harness_converts_presence_to_change(self):
        with tempdir() as root:
            old = np.ones((8, 8), bool)
            session = Session(old.shape, ask_variant='t2_state')
            box = Toolbox([render_map(old), np.zeros((8, 8, 3), np.uint8)], root, session=session,
                          map_input=True, ask=ApiAsk(client=lambda p: raw('no', probability=.1)))
            handle = session.add_mask(old)
            answer = box.execute('ask', dict(regions=[dict(mask=handle)])).public['results'][0]
            self.assertEqual(answer['answer'], 'yes')
            self.assertAlmostEqual(answer['p_change'], .9)
            self.assertEqual(answer['state_answer'], 'no')

    def test_strict_stop_429_403_and_quota_no_retry(self):
        images = [Image.fromarray(np.zeros((4, 4, 3), np.uint8))] * 4
        for status, body in ((429, b'busy'), (403, b'denied'), (400, b'insufficient balance')):
            with self.subTest(status=status), tempdir() as root:
                guard = RunGuard(Path(root)/'STOP.json', Path(root)/'ledger.txt', 1.)
                calls = []
                def client(payload):
                    calls.append(payload)
                    raise urllib.error.HTTPError('unused', status, 'failed', {}, BytesIO(body))
                def factory(**kw):
                    return Planner(client=client, sleep=lambda _: self.fail('Must not retry'), **kw)
                with patch('agent_baseline.ask_eval.Planner', side_effect=factory), self.assertRaises(RunStopped):
                    api_answer(images, dict(VARIANTS['f37_corr'], strict_stop=True), guard, 0,
                               map_session(np.zeros((4, 4), bool)))
                self.assertEqual(len(calls), 1)
                self.assertTrue((Path(root)/'STOP.json').exists())


class NoisyAndComparisonTests(unittest.TestCase):
    def test_reproducible_shift_removal_and_no_wrap(self):
        old = np.zeros((48, 48), bool)
        old[8:40:6, 8:40:6] = True
        a, stat = noisy_map(old, 'train_0.png')
        b, again = noisy_map(old, 'train_0.png')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(stat, again)
        self.assertGreaterEqual(stat['displacement'], 2)
        self.assertLessEqual(stat['displacement'], 4)
        labels, comps = components(translate(old, stat['dx'], stat['dy']))
        self.assertEqual(stat['buildings_removed'], math.ceil(.1*len(comps)))
        self.assertEqual(len(components(a)[1]), len(comps)-stat['buildings_removed'])
        edge = np.zeros((8, 8), bool); edge[0, 0] = True
        self.assertFalse(translate(edge, -2, 0).any())
        self.assertFalse(noisy_map(np.zeros_like(old), 'empty.png')[0].any())
        offsets = {tuple(noisy_map(old, f'train_{i}.png')[1][k] for k in ('dx', 'dy')) for i in range(10)}
        self.assertGreater(len(offsets), 1)

    def make_data(self, root):
        root = Path(root)
        old = np.zeros((16, 16), bool); old[2:4, 2:4] = True
        new = old.copy(); new[10:12, 10:12] = True
        manifests = {}
        for split, name in (('dev', 'train_0.png'), ('val', 'train_1.png')):
            save_image(root/'clean/maps'/name, render_map(old))
            save_image(root/'dataset/img2'/name, np.full((16, 16, 3), 50, np.uint8))
            save_image(root/'dataset/slabel2'/name, new.astype('uint8'))
            save_image(root/'clean/gt'/name, (old ^ new).astype('uint8')*255)
            row = dict(id=name, task_type='T5.4', input_type='map_image', target='building',
                change_direction='any', scope=None, instruction='Update the map.', constraint=None, init_mask=None,
                image_t1=str(root/'clean/maps'/name), image_t2=str(root/'dataset/img2'/name),
                gt_mask=str(root/'clean/gt'/name), meta={})
            path = root/f'{split}.jsonl'; write_text(path, json.dumps(row)+'\n'); manifests[split] = path
        return manifests, new

    def test_noisy_manifests_and_truth_match_supplied_map(self):
        with tempdir() as root:
            root = Path(root)
            manifests, new = self.make_data(root)
            report = generate(manifests, root/'noisy')
            row = read_tasks(root/'noisy/tasks/T5.4n_dev.jsonl')[0]
            from agent_baseline.map_input import map_mask
            np.testing.assert_array_equal(read_image(row['gt_mask'])>0, new ^ map_mask(read_image(row['image_t1'])))
            before = (root/'noisy/tasks/T5.4n_dev.jsonl').read_bytes()
            generate(manifests, root/'noisy')
            self.assertEqual(before, (root/'noisy/tasks/T5.4n_dev.jsonl').read_bytes())
            self.assertEqual(report['splits']['dev']['images'], 1)
            self.assertEqual(row['meta']['variant'], 'T5.4n')

    def test_compare_fixed_pool_metrics_ledger_and_resume(self):
        with tempdir() as root:
            root = Path(root)
            manifests, _ = self.make_data(root)
            rows = read_tasks(manifests['dev'])
            candidate = np.zeros((16,16), bool)
            candidate[10:12,10:12] = True; candidate[6:8,6:8] = True
            save_image(root/'fixed/train_0.png', candidate.astype('uint8')*255)
            calls = []
            def fake(images, spec, guard, seed, session):
                calls.append((session.ask_variant, len(images)))
                guard.charge(.01)
                return dict(answer='yes', p_yes=.8, cost_cny=.01)
            report = compare_run(rows, root/'fixed', root/'compare', 1, answer_fn=fake)
            self.assertEqual(len(calls), 4)
            self.assertAlmostEqual(report['cost_cny'], .04)
            self.assertEqual(report['variants']['map_compare']['auc'], .5)
            self.assertAlmostEqual(report['variants']['map_compare']['score']['f1_c'], 200/3)
            second = compare_run(rows, root/'fixed', root/'compare', 1, answer_fn=lambda *a: self.fail('Already complete'))
            self.assertAlmostEqual(second['cost_cny'], .04)
            self.assertEqual(report['candidates'], second['candidates'])

    def test_compare_budget_stop_keeps_ledger_and_no_complete_report(self):
        with tempdir() as root:
            root = Path(root)
            manifests, _ = self.make_data(root)
            rows = read_tasks(manifests['dev'])
            save_image(root/'fixed/train_0.png', np.ones((16,16), np.uint8)*255)
            def expensive(images, spec, guard, seed, session):
                guard.charge(.1)
                return dict(answer='yes', cost_cny=.1)
            with self.assertRaises(RunStopped):
                compare_run(rows, root/'fixed', root/'compare', .01, answer_fn=expensive)
            self.assertTrue((root/'compare/STOP.json').exists())
            self.assertFalse((root/'compare/report.json').exists())
            report = json.loads((root/'compare/incomplete.json').read_text())
            self.assertAlmostEqual(report['cost_cny'], .1)
            with self.assertRaises(RunStopped):
                compare_run(rows, root/'fixed', root/'compare', 1, answer_fn=lambda *a: self.fail('Stopped'))

    def test_io_retry_only_three_attempts(self):
        call = unittest.mock.Mock(side_effect=[OSError('EIO'), OSError('EIO'), 7])
        with patch('agent_baseline.analysis.io_retry.time.sleep'):
            self.assertEqual(retry_io(call), 7)
        self.assertEqual(call.call_count, 3)


if __name__ == '__main__':
    unittest.main()
