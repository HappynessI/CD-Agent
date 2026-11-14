"""Tiny fake-backend contracts only. No models, API, Slurm or real dataset processing."""
from argparse import Namespace
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
from common import save_mask_png, write_json
from agent_baseline.examples_t3x import (Example, ExampleMask, attach_examples, load_examples,
    example_vector, combine_scores, INSTRUCTIONS)
from agent_baseline.state import Session, Mask
from agent_baseline.tools import Toolbox, session_schemas
from agent_baseline.tools_t13 import Features
from agent_baseline.tasks import read_tasks, load_pair
from agent_baseline.loop_json import run_session, referenced_handles, latest_mask
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.analysis import make_t3x_tasks as make, t3x_fixed as fixed, t3x_metrics as metrics
from agent_baseline.analysis.second_labels import ENCODING


def off_requests(directory):
    """Run the same encoded planner/verifier requests under the frozen and current checkout."""
    import copy
    import json
    from pathlib import Path
    import numpy as np
    from agent_baseline.state import Session
    from agent_baseline.tools import Toolbox
    from agent_baseline.backends import ApiAsk
    from agent_baseline.planner import Planner
    from agent_baseline.loop_json import run_session
    from agent_baseline.recorder import Recorder
    out = []
    for gate in (1, 2, 3):
        for tools in (False, True):
            shape = (8, 8)
            mask = np.zeros(shape, bool)
            mask[:2, :2] = True
            session = Session(shape, mask)
            box = Toolbox([np.zeros((*shape, 3), np.uint8)] * 2, directory, session=session, tools_t13=tools)
            def verifier(payload):
                out.append(copy.deepcopy(payload))
                return dict(choices=[dict(message=dict(role='assistant', content='{"change":"yes","type":"new_building","reason":"New."}'))], usage={})
            box.ask_backend = ApiAsk(session=session, client=verifier, correspondence=True)
            def client(payload):
                out.append(copy.deepcopy(payload))
                call = dict(id='s', type='function', function=dict(name='submit', arguments=json.dumps(dict(mask='mask#0', summary='Keep.'))))
                return dict(choices=[dict(message=dict(role='assistant', tool_calls=[call]))], usage={})
            result = run_session(box, Planner(client=client, seed=7), Recorder(Path(directory), 'fake', 'x.png', gt=mask),
                'T5.1', budget=3, prompt_version='v4', verify=True, gate2=gate >= 2, gate3=gate == 3)
            if result['submitted_by'] != 'model':
                raise AssertionError(result)
    return out


class FakeVision:
    def __init__(self, shape, reference=False, reverse=False):
        self.shape, self.reference, self.reverse = shape, reference, reverse
        self.calls = []
        self.masks = np.zeros((2, *shape), bool)
        self.masks[0, :2, :2] = True
        self.masks[1, -2:, -2:] = True

    def __call__(self, operation, **args):
        self.calls.append((operation, args))
        if operation == 'segment_everything':
            return self.masks.copy()
        if operation == 'extract_features':
            feature = np.zeros((*self.shape, 2), np.float32)
            if args['time'] == 't2':
                if self.reference:
                    feature[:] = [-1, 1] if self.reverse else [1, -1]
                else:
                    feature[:2, :2] = [1, -1]
                    feature[-2:, -2:] = [-1, 1]
            return feature
        raise AssertionError(operation)


class T3xTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(os.environ['TMPDIR']))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'data'
        self.dataset.mkdir()
        write_json(self.dataset / 'cls_gray.json', ENCODING)
        self.groups = dict(dev=['dev.png'], val=['report.png'], rest=['rest_a.png', 'rest_b.png'])
        self.a = np.zeros((24, 24), np.uint8)
        self.b = self.a.copy()
        for box, source, target in [((1, 1, 5, 5), 1, 5), ((9, 1, 13, 5), 2, 5), ((1, 10, 5, 14), 5, 1), ((22, 22, 23, 23), 1, 5)]:
            x0, y0, x1, y1 = box
            self.a[y0:y1, x0:x1], self.b[y0:y1, x0:x1] = source, target
        for name in sum(self.groups.values(), []):
            value = 43 if name.startswith('rest') else 211
            for part, data in [('slabel1', self.a), ('slabel2', self.b),
                    ('img1', np.full((24, 24, 3), value, np.uint8)), ('img2', np.full((24, 24, 3), value+1, np.uint8))]:
                path = self.dataset / part / name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(data).save(path)
        self.args = Namespace(dataset_root=self.dataset, run_root=self.root, output_root=self.root / 'tasks',
                              phase='both', seed=1016, min_area=16, max_area=4096)

    def generate(self):
        with patch.object(make, 'split_names', return_value=self.groups):
            return make.generate(self.args)

    def rows(self, phase='val'):
        return read_tasks(self.args.output_root / phase / 'tasks.jsonl')

    def test_split_names_drops_overlap_rejects_duplicates(self):
        d = self.root / 'second_dev'
        d.mkdir()
        rest = d / 'rest.txt'
        with patch.object(make, 'inventories', return_value={k: self.groups[k] for k in ('dev', 'val')}):
            for text in ('report.png dev.png', 'rest_a.png rest_a.png'):
                rest.write_text(text)
                with self.assertRaises(ValueError):
                    make.split_names(self.root)
            # rest.txt still lists the development images; they are dropped from the example pool.
            rest.write_text('rest_b.png dev.png rest_a.png report.png')
            self.assertEqual(make.split_names(self.root), self.groups)

    def test_generation_reproducible_and_rest_never_evaluated(self):
        first = self.generate()
        manifest = self.args.output_root / 'val/tasks.jsonl'
        raw = manifest.read_bytes()
        self.assertEqual(first, self.generate())
        self.assertEqual(raw, manifest.read_bytes())
        self.assertEqual(first['val']['totals_by_type'], {'T3.2': 2, 'T3.3': 3, 'T3.4': 3})
        self.assertEqual(first['val']['counts']['T3.3']['1->5'], 1)
        for row in self.rows():
            self.assertEqual(Path(row['image_t1']).name, 'report.png')
            for descriptor in row['meta']['example_provenance']:
                self.assertIn(descriptor['name'], self.groups['rest'])
                self.assertTrue(16 <= descriptor['area'] <= 4096)
            if row['task_type'] == 'T3.4':
                pos, neg = row['meta']['example_provenance']
                self.assertEqual(pos['name'], neg['name'])
                self.assertNotEqual((pos['source'], pos['target']), (neg['source'], neg['target']))

    def test_truth_and_crops_are_from_their_own_images(self):
        self.generate()
        for row in self.rows():
            if row['meta']['transition'] == '*->5':
                self.assertEqual(int(metrics.load(row['gt_mask']).sum()), 33)
            if row['task_type'] in ('T3.3', 'T3.4') and row['meta']['transition'] == '1->5':
                self.assertEqual(int(metrics.load(row['gt_mask']).sum()), 17)
            for spec, descriptor in zip(row['examples'], row['meta']['example_provenance']):
                x0, y0, x1, y1 = descriptor['crop_box']
                for t in ('t1', 't2'):
                    if spec.get('image_' + t):
                        image = np.asarray(Image.open(spec['image_' + t]))
                        self.assertTrue(np.all(image == (43 if t == 't1' else 44)))
                        self.assertEqual(image.shape[:2], (y1-y0, x1-x0))
                mask = metrics.load(spec['mask'])
                self.assertEqual(int(mask.sum()), descriptor['area'])
                self.assertEqual(mask.shape, (y1-y0, x1-x0))
        # Changing evaluation labels cannot alter the example pool or crop bytes.
        before = {str(p): p.read_bytes() for p in (self.args.output_root / 'val/examples').rglob('*.png')}
        a, b = self.a.copy(), self.b.copy()
        a[20, 0], b[20, 0] = 1, 5
        Image.fromarray(a).save(self.dataset / 'slabel1/report.png')
        Image.fromarray(b).save(self.dataset / 'slabel2/report.png')
        self.generate()
        self.assertEqual(before, {str(p): p.read_bytes() for p in (self.args.output_root / 'val/examples').rglob('*.png')})

    def test_component_area_bounds_and_missing_pool_fail(self):
        image = np.ones((4, 4), bool)
        self.assertEqual(len(list(make.components(image, 16, 16))), 1)
        self.assertEqual(list(make.components(image, 17, 20)), [])
        self.assertEqual(list(make.components(image, 1, 15)), [])
        pool = make.catalogue(['rest.png'], lambda _: (self.a, self.b))
        with self.assertRaises(ValueError):
            make.select_examples(pool, 'report.png', 'T3.3', 3, 4)
        forward = make.select_examples(pool, 'report.png', 'T3.4', 1, 5)
        reverse_pool = make.catalogue(['z.png', 'rest.png'], lambda _: (self.a, self.b))
        ordered_pool = make.catalogue(['rest.png', 'z.png'], lambda _: (self.a, self.b))
        self.assertEqual(make.select_examples(reverse_pool, 'report.png', 'T3.4', 1, 5),
                         make.select_examples(ordered_pool, 'report.png', 'T3.4', 1, 5))
        self.assertEqual(forward, make.select_examples(pool, 'report.png', 'T3.4', 1, 5))

    def external_box(self, masked=True, negative=False, single=False):
        shape = (6, 8)
        vision = FakeVision(shape)
        session = Session(shape, target='any change')
        images = {t: np.zeros((3, 5, 3), np.uint8) for t in (('t2',) if single else ('t1', 't2'))}
        mask = np.ones((3, 5), bool) if masked else None
        pos_backend = FakeVision((3, 5), reference=True)
        examples = [Example(images, mask, backend=pos_backend)]
        if negative:
            examples.append(Example(images, mask, 'negative', FakeVision((3, 5), reference=True, reverse=True)))
        attach_examples(session, examples)
        box = Toolbox([np.zeros((*shape, 3), np.uint8)] * 2, self.root / 'session',
                      session=session, tools_t13=True, vision=vision)
        props = box.execute('segment_everything', dict(time='t2')).public['props']
        return box, props, vision, pos_backend

    def test_external_handles_features_shapes_and_negative_scores(self):
        box, props, vision, backend = self.external_box(negative=True)
        self.assertIsNone(box.session.current_mask)
        self.assertIsNone(latest_mask(box.session))
        (pos, pm), (neg, nm) = box.session.external_examples
        self.assertIsInstance(box.session.get(pos), Example)
        self.assertIsInstance(box.session.get(pm), ExampleMask)
        self.assertNotIsInstance(box.session.get(pm), Mask)
        self.assertEqual(referenced_handles(dict(a=[pos, pm, neg, nm])), [pos, pm, neg, nm])
        args = dict(props=props, exemplar=dict(example=pos), negative=dict(mask=nm))
        out = box.execute('match_exemplar', args)
        self.assertTrue(out.public['ok'], out.internal)
        scores = [r['score'] for r in box.session.get(out.public['props']).items]
        np.testing.assert_allclose(scores, [2, -2], atol=1e-6)
        self.assertEqual([c[1]['time'] for c in backend.calls], ['t1', 't2'])
        self.assertEqual(len(vision.calls), 3)
        out_mask = box.execute('match_exemplar', dict(props=props, exemplar=dict(mask=pm)))
        np.testing.assert_allclose([r['score'] for r in box.session.get(out_mask.public['props']).items], [1, -1], atol=1e-6)
        self.assertEqual(len(backend.calls), 2)
        invalid = box.execute('match_exemplar', args, allowed={props})
        self.assertEqual(invalid.public['error_code'], 'bad_handle')
        bad = box.execute('submit', dict(mask=pm, summary='External mask'))
        self.assertFalse(bad.public['ok'])
        malformed = box.execute('match_exemplar', dict(props=props, exemplar=dict(example=pos, mask=pm)))
        self.assertEqual(malformed.public['error_code'], 'bad_args')

    def test_single_unmasked_example_uses_own_t2_features(self):
        box, props, _, backend = self.external_box(masked=False, single=True)
        key, mask_key = box.session.external_examples[0]
        self.assertIsNone(mask_key)
        out = box.execute('match_exemplar', dict(props=props, exemplar=dict(example=key), mode='t2'))
        self.assertTrue(out.public['ok'], out.internal)
        np.testing.assert_allclose([r['score'] for r in box.session.get(out.public['props']).items], [1, -1], atol=1e-6)
        self.assertEqual(backend.calls, [('extract_features', {'time': 't2'})])
        invalid = box.execute('match_exemplar', dict(props=props, exemplar=dict(example=key)))
        self.assertEqual(invalid.public['error_code'], 'bad_args')

    def test_off_attach_has_no_effect_and_rejects_new_arguments(self):
        s = Session((4, 4))
        before = dict(s.__dict__)
        attach_examples(s, None)
        self.assertEqual(before, s.__dict__)
        self.assertFalse(hasattr(s, 'external_examples'))
        schemas = session_schemas(s, tools_t13=True)
        match = next(e['function'] for e in schemas if e['function']['name'] == 'match_exemplar')
        self.assertNotIn('negative', match['parameters']['properties'])
        self.assertNotIn('example', match['parameters']['properties']['exemplar']['properties'])
        box = Toolbox([np.zeros((4, 4, 3), np.uint8)] * 2, self.root / 'off', session=s, tools_t13=True)
        result = box.execute('match_exemplar', dict(props='props#0', exemplar=dict(example='example#1')))
        self.assertEqual(result.public['error_code'], 'bad_args')

    def test_g1_g2_g3_off_requests_match_frozen_code_t_bytes(self):
        reference = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_t')
        self.assertTrue((reference / 'agent_baseline/loop_json.py').is_file())
        script = inspect.getsource(off_requests) + '\nprint(__import__("json").dumps(off_requests(' + repr(str(self.root / 'wire')) + ')))\n'
        env = dict(os.environ, PYTHONPATH=str(reference), OVCD_CODE=str(reference / 'ovcd_v0'))
        result = subprocess.run([sys.executable, '-B', '-c', script], env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.dumps(off_requests(self.root / 'wire')).encode(), json.dumps(json.loads(result.stdout)).encode())

    def test_cli_rejects_external_tasks_without_manifest_or_switch(self):
        from agent_baseline.loop_json import main
        self.generate()
        cases = [
            ['--task', 'T3.3'],
            ['--tasks', str(self.args.output_root / 'val/tasks.jsonl')],
            ['--external-examples', '--tasks', str(self.args.output_root / 'val/tasks.jsonl'), '--tools-t13', 'classify'],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch('sys.argv', ['loop', '--config', 'fake', *arguments]), \
                    patch('agent_baseline.loop_json.RunGuard.stopped', return_value=False), \
                    patch('agent_baseline.loop_json.ThreadPoolExecutor', side_effect=AssertionError('No sessions should start')):
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)

    def test_first_message_order_and_no_truth_or_provenance(self):
        self.generate()
        row = next(r for r in self.rows() if r['task_type'] == 'T3.4')
        def run(gt):
            session = Session((24, 24), target='any change', instruction=row['instruction'])
            attach_examples(session, load_examples(row['examples']))
            box = Toolbox(load_pair(row), self.root / 'wire', session=session, tools_t13=True)
            requests = []
            def client(payload):
                requests.append(copy.deepcopy(payload))
                call = dict(id='s', type='function', function=dict(name='submit', arguments=json.dumps(dict(mask='empty', summary='Empty.'))))
                return dict(choices=[dict(message=dict(role='assistant', tool_calls=[call]))], usage={})
            result = run_session(box, Planner(client=client), Recorder(self.root / 'wire', 'fake', 'x.png', gt=gt),
                                 'T3.4', budget=1, prompt_version='v4')
            self.assertEqual(result['submitted_by'], 'model', result)
            self.assertTrue((self.root / 'wire/examples/exmask_1.png').is_file())
            return requests
        first = run(np.zeros((24, 24), bool))
        self.assertEqual(json.dumps(first).encode(), json.dumps(run(np.ones((24, 24), bool))).encode())
        payload = json.dumps(first)
        for secret in ('gt_mask', row['gt_mask'], 'example_provenance', 'rest_a.png', 'rest_b.png', '1->5', 'SECOND'):
            self.assertNotIn(secret, payload)
        blocks = first[0]['messages'][1]['content']
        images = [b for b in blocks if b['type'] == 'image_url']
        self.assertEqual(len(images), 6)  # evaluation T1/T2, positive T1/T2, negative T1/T2
        labels = [b['text'] for b in blocks if b['type'] == 'text']
        self.assertEqual(labels[1:3], ['view#0 T1', 'view#0 T2'])
        self.assertEqual([v for v in labels if v.startswith('example#')],
            ['example#0 positive T1', 'example#0 positive T2', 'example#2 negative T1', 'example#2 negative T2'])

    def test_manifest_example_paths_and_required_dates(self):
        self.generate()
        row = next(r for r in self.rows() if r['task_type'] == 'T3.2')
        self.assertNotIn('image_t1', row['examples'][0])
        self.assertTrue(Path(row['examples'][0]['image_t2']).is_absolute())
        from agent_baseline.examples_t3x import resolve_examples
        with self.assertRaises(ValueError):
            resolve_examples(dict(row, examples=[]), self.root)
        with self.assertRaises(ValueError):
            resolve_examples(dict(row, task_type='T3.3'), self.root)

    def test_fixed_scoring_union_and_empty_proposals(self):
        box, props, vision, _ = self.external_box(negative=True)
        f = [Features(vision('extract_features', time=t), vision.shape, t) for t in ('t1', 't2')]
        examples = [box.session.get(k) for k, _ in box.session.external_examples]
        scores = fixed.proposal_scores(vision.masks, *f, examples, 'change')
        np.testing.assert_allclose(scores, [2, -2], atol=1e-6)
        np.testing.assert_array_equal(fixed.predict(vision.masks, scores, 1.9, vision.shape), vision.masks[0])
        self.assertEqual(fixed.task_counts(vision.masks, scores, vision.masks[0], [1.9])[1.9], [4, 0, 0])
        self.assertFalse(fixed.predict(vision.masks, scores, 2.1, vision.shape).any())
        empty = np.zeros((0, *vision.shape), bool)
        self.assertEqual(fixed.proposal_scores(empty, *f, examples, 'change').shape, (0,))
        self.assertFalse(fixed.predict(empty, [], .5, vision.shape).any())
        np.testing.assert_allclose(combine_scores([.8, .3], [.2, .9]), [.6, -.6])

    def test_fixed_entrypoint_freezes_dev_and_saves_report_masks(self):
        self.generate()
        config = self.root / 'vision.json'
        config.write_text('{}')
        args = Namespace(tasks=self.args.output_root / 'dev/tasks.jsonl', phase='dev', run_root=self.root,
            vision_config=config, cache=self.root / 'cache', settings=None, output=self.root / 'dev.json', mask_dir=None)
        class Pool:
            def __init__(this, *unused):
                pass
            def for_pair(this, paths):
                shape = np.asarray(Image.open(paths[1])).shape[:2]
                return FakeVision(shape, reference=Path(paths[1]).parent.name in ('0', '1'))
            def close(this):
                pass
        with patch.object(fixed, 'split_names', return_value=self.groups), patch.object(fixed, 'WorkerPool', Pool):
            dev = fixed.evaluate(args)
            args.phase, args.tasks = 'val', self.args.output_root / 'val/tasks.jsonl'
            with self.assertRaises(ValueError):
                fixed.evaluate(args)
            args.settings, args.output, args.mask_dir = self.root / 'dev.json', self.root / 'val.json', self.root / 'masks'
            val = fixed.evaluate(args)
            self.assertEqual(dev['selected'], val['selected'])
            self.assertEqual(len(list(args.mask_dir.glob('*.png'))), 8)
            for task in val['per_task'].values():
                self.assertEqual(len(task['counts']), 1)
            config.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                fixed.evaluate(args)

    def test_metrics_ignore_example_size_and_bootstrap_by_image(self):
        self.generate()
        rows = [r for r in self.rows() if r['task_type'] == 'T3.3']
        fixed_dir = self.root / 'fixed'
        runs = [self.root / f'agent{k}' for k in range(3)]
        for row in rows:
            truth = metrics.load(row['gt_mask'])
            save_mask_png(fixed_dir / (row['id'] + '.png'), truth)
            for run in runs:
                directory = run / row['task_type'] / 'val' / row['id']
                save_mask_png(directory / 'pred.png', truth)
                write_json(directory / 'final.json', dict(submitted_by='model', mask_path='pred.png', error=None))
        result = metrics.compare(rows, fixed_dir, runs, samples=10)
        self.assertEqual(result['per_type']['T3.3']['fixed']['f1_c'], 100)
        self.assertEqual(result['per_type']['T3.3']['images'], 1)
        self.assertEqual(result['per_type']['T3.3']['agent_minus_fixed']['ci95'], [0, 0])
        a = np.ones((3, 3, 3), np.int64)
        f = np.ones((3, 3), np.int64)
        grouped_a, grouped_f = metrics.grouped_counts(rows, a, f)
        self.assertEqual(grouped_a.shape, (3, 1, 3))
        np.testing.assert_array_equal(grouped_f, [[3, 3, 3]])


if __name__ == '__main__':
    unittest.main()
