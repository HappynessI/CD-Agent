"""Generalization contracts; run on a compute node, without model/API access."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from prompts import SCOPE_CARD, SYSTEM, VERIFY, PSEUDO_TYPES, scope_card, verifier_prompts
from agent_baseline.backends import ApiAsk, CORRESPONDENCE, SegmentBackend
from agent_baseline.loop_json import system_prompt, run_session
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.state import Session, Mask
from agent_baseline.tasks import read_tasks, image_name, seed_for, instruction_for, INSTRUCTIONS, load_pair
from agent_baseline.tools import Toolbox, SCHEMAS, session_schemas
from .helpers import tempdir, mask, rgb, call, response, ScriptClient, ask
from .test_loop import assert_no_evidence


class GeneralizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_frozen_g1_verifier_literals(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/g1_prompts.json').read_text())['verifier']
        self.assertEqual(verifier_prompts(), fixture)
        for name, value in dict(SCOPE_CARD=SCOPE_CARD, SYSTEM=SYSTEM, VERIFY=VERIFY,
                                PSEUDO_TYPES=PSEUDO_TYPES, CORRESPONDENCE=CORRESPONDENCE).items():
            self.assertEqual(value, fixture[name])

    def test_tree_and_both_directions_reach_every_prompt(self):
        for target in ('building', 'tree'):
            for direction in ('appeared', 'disappeared'):
                session = Session((32, 48), mask(), target=target, change_direction=direction)
                settings = verifier_prompts(target, direction)
                text = system_prompt('v4', 30, session)[2]
                self.assertIn(f'class "{target}"', text)
                self.assertIn(f'class "{target}"', settings['SYSTEM'])
                self.assertIn(direction, settings['VERIFY'])
                self.assertIn(direction, settings['CORRESPONDENCE'])
                self.assertIn(direction, instruction_for('T1.1xC0', session))
                schemas = session_schemas(session)
                question = next(s['function'] for s in schemas if s['function']['name'] == 'ask')
                self.assertIn(target + ' change', question['description'])
                self.assertIn(direction, question['description'])
                self.assertEqual(question['parameters'], SCHEMAS[5]['function']['parameters'])
        tree = verifier_prompts('tree')
        self.assertNotIn('building', json.dumps(tree))
        text = instruction_for('T1.1xC0', Session((32, 48), mask(), target='tree'))
        self.assertIn('(appeared or disappeared)', text)
        self.assertNotIn('built', text)
        with self.assertRaisesRegex(ValueError, 'Missing scope card'):
            Session((2, 2), target='category_without_a_card')
        with self.assertRaisesRegex(ValueError, 'change_direction'):
            Session((2, 2), change_direction='invalid')

    def test_ask_and_stage_read_the_session(self):
        for direction in ('appeared', 'disappeared'):
            session = Session((32, 48), mask(), target='tree', change_direction=direction)
            seen = []
            def verifier(pairs, question):
                seen.append(question)
                return ask(pairs, question)
            box = Toolbox(rgb(), self.root, ask=verifier, session=session)
            self.assertTrue(box.execute('ask', {}).public['ok'])
            box.asked.clear()
            box.stage(session.current_mask)
            self.assertEqual(seen, [verifier_prompts('tree', direction)['VERIFY']] * 2)
            requests = []
            raw = dict(choices=[dict(message=dict(role='assistant', content='{"change":"yes"}'))], usage={})
            backend = ApiAsk(client=lambda p: requests.append(p) or raw, session=session, correspondence=True)
            from agent_baseline.view import ask_images
            images = ask_images(rgb(), mask(), [3, 4, 9, 10], True)
            backend.one(images, seen[0], 1)
            self.assertEqual(requests[0]['messages'][0]['content'], verifier_prompts('tree', direction)['SYSTEM'])
            self.assertEqual(requests[0]['messages'][1]['content'][-1]['text'], verifier_prompts('tree', direction)['CORRESPONDENCE'])

    def test_cache_policy_keeps_building_thresholds(self):
        self.assertEqual(scope_card()['sam3_cache'], dict(soft_min_score=.3, tile_min_score=.1, full_min_score=.5))
        self.assertEqual(scope_card('tree')['sam3_cache'], {})
        building = SegmentBackend('val', 'image.png', pixel_thr=.85)
        with patch('agent_baseline.backends.read_soft', return_value=([])) as read:
            rows, meta = building('t1', 'building', 512, .3)
        self.assertEqual(rows, [])
        self.assertTrue(meta['cache_hit'])
        self.assertEqual(read.call_args.args[3:], (.85, .3))

    def make_manifest(self):
        for key in ('a', 'b', 'label', 'init'):
            Image.fromarray((mask() * 255).astype(np.uint8)).save(self.root / (key + '.png'))
        return dict(id='image_7', image_t1='a.png', image_t2='b.png', instruction='Find the changed trees.',
            task_type='T5.1', constraint='Only include appeared trees.', target='tree', change_direction='appeared',
            init_mask='init.png', gt_mask='label.png', meta={'dataset': 'private_dataset'})

    def test_manifest_seed_paths_and_agent_input(self):
        row = self.make_manifest()
        path = self.root / 'tasks.jsonl'
        path.write_text(json.dumps(row) + '\n')
        item = read_tasks(path)[0]
        self.assertEqual(image_name(item), 'a.png')
        self.assertEqual(seed_for(image_name(item), 2), (sum((i+1)*ord(c) for i,c in enumerate('a.png')) + 2000) % (2**31))
        self.assertEqual(load_pair(item)[0].shape, (32, 48, 3))
        other = dict(row, id='image_2')
        path.write_text(json.dumps(other) + '\n' + json.dumps(row) + '\n')
        self.assertEqual(seed_for(image_name(read_tasks(path)[1]), 2), seed_for(image_name(item), 2))
        session = Session((32,48), mask(), target=item['target'], change_direction=item['change_direction'],
            instruction=item['instruction'], constraint=item['constraint'])
        box = Toolbox(rgb(), self.root / 'trace', ask=ask, session=session)
        script = [response([call('submit', dict(mask='mask#0', summary='Keep.'))])]
        client = ScriptClient(script)
        recorder = Recorder(self.root / 'trace', 'synthetic', 'image.png', gt_path=item['gt_mask'])
        run_session(box, Planner(client=client), recorder, 'T5.1', budget=1, prompt_version='v4', verify=True,
            metadata=dict(task_meta=item['meta']))
        for request in client.requests:
            assert_no_evidence(self, request)
            text = json.dumps(request)
            self.assertNotIn('private_dataset', text)
            self.assertNotIn(str(path), text)
        self.assertIn('Constraint:', client.requests[0]['messages'][1]['content'][0]['text'])

    def test_generator_keeps_instruction_and_seed_contract(self):
        from agent_baseline.analysis.make_levir_tasks import generate
        data = self.root / 'dataset'
        for part in ('A', 'B', 'label'):
            (data / 'val_original' / part).mkdir(parents=True)
        Image.fromarray(rgb()[0]).save(data / 'val_original/A/val_7.png')
        output = self.root / 'generated.jsonl'
        generate(output, data, self.root / 'init')
        rows = read_tasks(output)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['instruction'], INSTRUCTIONS[row['task_type']])
            self.assertEqual(image_name(row), 'val_7.png')
            self.assertEqual(seed_for(image_name(row)), seed_for('val_7.png'))

    def test_python_off_is_absent_and_cannot_be_called(self):
        box = Toolbox(rgb(), self.root)
        self.assertNotIn('python', [s['function']['name'] for s in session_schemas(box.session)])
        result = box.execute('python', dict(code='save_mask(images["t1"][:,:,0] > 0)'))
        self.assertFalse(result.public['ok'])
        self.assertEqual(box.session.handles, {})


if __name__ == '__main__':
    unittest.main()
