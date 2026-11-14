"""T4.1 v3 contracts. All planners/backends are synthetic and offline."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from agent_baseline.analysis.io_retry import read_text, write_json, write_text
from agent_baseline.analysis.io_retry import save_image, retry_io
from agent_baseline.analysis.make_t41_tasks import INSTRUCTION_V2, INSTRUCTION_V3, generate_v3
from agent_baseline.analysis.t41_trace_stats import analyze_trace, summarize, markdown, area_request
from agent_baseline.loop_json import run_session
from agent_baseline.map_input import render_map
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.state import Session, Mask
from agent_baseline.tools import Toolbox, session_schemas
from .helpers import mask, rgb, rows, call, response, ScriptClient, ask, SHAPE


def legacy_requests(directory):
    """Actual first wire messages (including encoded image bytes), without path normalization.

    Frozen fixture was captured from this supplied code copy BEFORE changing tools.py.
    These six cases exercise the no-area routes named in TASK.md.
    """
    cases = [
        ('levir_c0', 'T1.1xC0', {}),
        ('second_c1', 'T1.1xC1', dict(change_direction='appeared', scope='building_rebuild_excluded')),
        ('whu', 'T1.1xC0', dict(instruction='Find building changes in this WHU image pair.')),
        ('t1x', 'T1.2xC0', dict(targets=['building', 'low vegetation'])),
        ('t3', 'T3.1', {}),
        ('t54', 'T5.4', {}),
    ]
    captured = {}
    for name, task, kwargs in cases:
        session = Session(SHAPE, **kwargs)
        images = rgb()
        if name == 't3':
            session.exemplar_handle = session.add_mask(mask())
        if name == 't54':
            images[0] = render_map(mask())
        box = Toolbox(images, Path(directory) / name, session=session, ask=ask, map_input=name == 't54')
        client = ScriptClient([response([call('submit', dict(mask='empty', summary='No changes.'))])])
        recorder = Recorder(box.directory, 'fake', 'image.png', gt=mask())
        run_session(box, Planner(client=client, seed=7), recorder, task, budget=1,
                    prompt_version='v4', status=True, verify=True, gate3=True)
        request = client.requests[0]
        # Reports are frozen as well as schema and the actual first messages.
        key = session.add_mask(mask())
        captured[name] = dict(schemas=session_schemas(session), messages=request['messages'],
                              report=box.pre_submit(key, 2, stage=True).public['report'])
    return captured


class T41V3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def toolbox(self, initial=None, area=True):
        session = Session(SHAPE, initial, **(dict(area_min_m2=1, gsd_m=.5) if area else {}))
        return Toolbox(rgb(), self.root, session=session, ask=ask, gate3=True)

    def three_components(self):
        # Two diagonal pixels form ONE 8-connected region; the next two have 4 and 9 pixels.
        data = mask(None)
        data[1, 1] = data[2, 2] = True
        data[5:7, 5:7] = True
        data[12:15, 12:15] = True
        return data

    def test_v3_instruction_and_copy_only_instruction(self):
        self.assertEqual(INSTRUCTION_V3, INSTRUCTION_V2.replace(
            'contiguous building change blocks', 'connected building change regions'))
        self.assertIn('contiguous building change blocks', INSTRUCTION_V2)
        originals = {}
        for suffix, count in (('_dev2', 10), ('', 64)):
            rows = [dict(id=f'x{i}.png', instruction=INSTRUCTION_V2.format(x=260),
                         meta=dict(version=2, min_m2=260), area_min_m2=260, gsd_m=.5,
                         gt_mask=f't41/gt/x{i}.png', arbitrary_field=['keep', i]) for i in range(count)]
            originals[suffix] = rows
            write_text(self.root / f't41/tasks_v2{suffix}/T4.1_any.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
        self.assertEqual(generate_v3(self.root, self.root / 'results'), dict(tasks_v3_dev2=10, tasks_v3=64))
        for suffix, rows in originals.items():
            generated = [json.loads(l) for l in read_text(self.root / f'results/tasks_v3{suffix}/T4.1_any.jsonl').splitlines()]
            for old, new in zip(rows, generated):
                self.assertEqual(new.pop('instruction'), INSTRUCTION_V3.format(x=260))
                self.assertEqual(new, {k: v for k, v in old.items() if k != 'instruction'})

    def test_schema_only_area_session_select_changes(self):
        ordinary = session_schemas(self.toolbox(area=False).session)
        area = session_schemas(self.toolbox().session)
        self.assertEqual(len(area), len(ordinary))
        for a, b in zip(area, ordinary):
            if a['function']['name'] == 'select':
                spec = a['function']
                self.assertIn('8-connected', spec['description'])
                self.assertIn('pixels', spec['description'])
                self.assertIn('mask#k', spec['parameters']['properties']['set']['description'])
                self.assertTrue(spec['parameters']['properties']['set_current']['default'])
                self.assertNotIn('set_current', b['function']['parameters']['properties'])
            else:
                self.assertEqual(a, b)
        # Area schema generation must not mutate shared constants or later no-area schemas.
        self.assertEqual(ordinary, session_schemas(self.toolbox(area=False).session))
        missing_gsd = Session(SHAPE, area_min_m2=1)
        self.assertEqual(ordinary, session_schemas(missing_gsd))

    def test_mask_select_inclusive_area_ids_current_and_provenance(self):
        box = self.toolbox(self.three_components())
        result = box.execute('select', dict(set='mask#0', area_min=4, area_max=9)).public
        self.assertTrue(result['ok'], result)
        self.assertEqual((result['area'], result['component_count'], result['added_px'], result['removed_px']), (13, 2, 0, 2))
        self.assertEqual(result['current_mask'], dict(before='mask#0', after=result['mask']))
        item = box.session.get(result['mask'], Mask)
        self.assertEqual(item.parents, ('mask#0',))
        self.assertEqual([r['id'] for r in item.components], [2, 3])
        inspected = box.execute('inspect', dict(mask=result['mask'])).public
        self.assertEqual([r['id'] for r in inspected['components']], [3, 2])
        next_result = box.execute('select', dict(set=result['mask'], ids=[3], where='area > 4', set_current=False)).public
        self.assertTrue(next_result['ok'], next_result)
        self.assertEqual(next_result['area'], 9)
        self.assertNotIn('current_mask', next_result)
        self.assertEqual(box.session.current_mask, result['mask'])
        self.assertEqual([r['id'] for r in box.session.get(next_result['mask']).components], [3])
        removed = box.execute('remove_region', dict(mask=result['mask'], ids=[2])).public
        np.testing.assert_array_equal(box.session.get(removed['mask']).data, box.session.get(next_result['mask']).data)

    def test_mask_select_where_empty_bad_ids_and_missing_fields(self):
        box = self.toolbox(self.three_components())
        for expression in ('score > .5', 'width > 3'):
            result = box.execute('select', dict(set='mask#0', where=expression)).public
            self.assertEqual(result['error_code'], 'bad_args')
        self.assertEqual(box.execute('select', dict(set='mask#0', ids=[99])).public['error_code'], 'bad_id')
        self.assertEqual(box.execute('select', dict(set='mask#0', area_min=10, area_max=2)).public['error_code'], 'bad_args')
        self.assertEqual(box.execute('select', dict(set='mask#0', area_min=-1)).public['error_code'], 'bad_args')
        result = box.execute('select', dict(set='mask#0', ids=[1, 2, 3], where='(area >= 4 and area < 9) or area == 2', area_max=4)).public
        self.assertEqual(result['area'], 6)
        empty = box.execute('select', dict(set='mask#0', ids=[])).public
        self.assertEqual((empty['area'], empty['component_count']), (0, 0))
        self.assertTrue(box.execute('select', dict(set=empty['mask'], where='area > 1')).public['ok'])
        self.assertEqual(box.execute('select', dict(set=empty['mask'], where='score > 0')).public['error_code'], 'bad_args')

    def test_mask_select_allowed_boundary_and_old_error_exact(self):
        for area in (False, True):
            box = self.toolbox(self.three_components(), area=area)
            result = box.execute('select', dict(set='mask#0', area_min=2), allowed=frozenset()).public
            self.assertEqual(result['error_code'], 'bad_handle')
        box = self.toolbox(self.three_components(), area=False)
        self.assertEqual(box.execute('select', dict(set='mask#0', area_min=2)).public,
                         dict(ok=False, error_code='bad_handle', message='Expected a inst handle.'))
        self.assertEqual(box.execute('select', dict(set='mask#0', set_current=False)).public['message'],
                         'arguments.set_current is not supported.')

    def test_instance_select_stays_instance_and_does_not_set_current(self):
        for area in (False, True):
            box = self.toolbox(self.three_components(), area=area)
            key = box.session.add_instances(rows([mask((1,1,3,3)), mask((8,8,11,11))], [.6, .9]), 't2')
            result = box.execute('select', dict(set=key, where='score > .7', area_min=4)).public
            self.assertTrue(result['ok'], result)
            self.assertEqual((result['ids'], result['n_in'], result['n_out']), ([2], 2, 1))
            self.assertIn('inst', result)
            self.assertNotIn('mask', result)
            self.assertEqual(box.session.current_mask, 'mask#0')

    def test_area_mask_select_in_actual_planner_loop(self):
        box = self.toolbox(self.three_components())
        client = ScriptClient([
            response([call('select', dict(set='mask#0', where='area > 4'))]),
            response([call('submit', dict(mask='mask#1', summary='Keep qualifying change regions.'))]),
        ])
        recorder = Recorder(self.root, 'fake', 'toy.png', gt=mask((12,12,15,15)))
        final = run_session(box, Planner(client=client, seed=7), recorder, 'T4.1', budget=2, prompt_version='v4')
        self.assertEqual(final['submitted_by'], 'model')
        self.assertEqual(final['change_area_px'], 9)
        self.assertEqual(final['eval']['f1'], 1.)
        offered = next(s['function'] for s in client.requests[0]['tools'] if s['function']['name'] == 'select')
        self.assertIn('mask#k', offered['parameters']['properties']['set']['description'])

    def test_trace_removal_uses_preserved_noncontiguous_ids(self):
        data = self.three_components()
        kept = data.copy()
        kept[1,1] = kept[2,2] = False
        final = kept & ~mask((12,12,15,15))
        directory = self.root / 'id_trace'
        for name, array in (('mask_0', data), ('mask_1', kept), ('mask_2', final)):
            save_image(directory / 'masks' / (name+'.png'), array.astype(np.uint8)*255)
        calls = [dict(step=1, calls=[dict(tool='remove_region', args=dict(mask='mask#0', ids=[1]),
                     ok=True, handles_created=['mask#1'], current_before='mask#0', current_after='mask#1')]),
                 dict(step=2, report='Area constraint (regions must be larger than 2.25 m^2): ...', calls=[]),
                 dict(step=3, calls=[dict(tool='remove_region', args=dict(mask='mask#1', ids=[3]),
                     ok=True, handles_created=['mask#2'], current_before='mask#1', current_after='mask#2')])]
        write_text(directory / 'steps.jsonl', ''.join(json.dumps(r)+'\n' for r in calls))
        # ID 3 becomes 2 if the PNG is simply relabelled. The trace must recover 3.
        trace = analyze_trace(directory, dict(area_min_m2=2.25, gsd_m=.5), self.root)
        self.assertEqual(len(trace['remove_small_after_report']), 1)
        self.assertEqual(trace['remove_small_after_report'][0]['ids'], [3])
        self.assertEqual(trace['remove_small_after_report'][0]['removed_small_px'], 9)

    def test_report_sentence_only_when_area_line_exists(self):
        sentence = 'Regions are defined by connectivity in the change mask; do not merge regions to meet the area constraint.'
        box = self.toolbox(self.three_components())
        lines = box.constraint_report(box.session.get('mask#0'))
        self.assertEqual(lines[-1], sentence)
        self.assertIn('2 connected regions', lines[-2])
        self.assertIn('component IDs [1, 2]', lines[-2])
        for initial, area in ((mask((1,1,5,5)), True), (self.three_components(), False), (mask(None), True)):
            box = self.toolbox(initial, area=area)
            self.assertEqual(box.constraint_report(box.session.get('mask#0')), [])

    def test_trace_stats_constructed_sessions(self):
        from scipy import ndimage as ndi
        root = self.root
        directory = root / 'run/T4.1/val/train_toy'
        initial = mask((3,3,7,7)) | mask((9,3,13,7)) | mask((25,20,27,22))
        kept = initial & ~mask((25,20,27,22))
        yy, xx = np.ogrid[-2:3, -2:3]
        merged = ndi.binary_closing(kept, structure=xx*xx + yy*yy <= 4)
        for h, data in (('mask_2', initial), ('mask_3', kept), ('mask_4', merged)):
            save_image(directory / 'masks' / (h + '.png'), data.astype(np.uint8)*255)
        instance_masks = np.array([mask((3,3,7,7)), mask((9,3,13,7)), mask((25,20,27,22))])
        for handle in ('inst_0', 'inst_1'):
            path = directory / 'insts' / (handle + '.npz')
            retry_io(lambda: path.parent.mkdir(parents=True, exist_ok=True))
            retry_io(lambda: np.savez(path, masks=np.packbits(instance_masks.reshape(3,-1), axis=1), shape=SHAPE))
        def event(tool, args, before=None, after=None, created=(), ok=True, text=''):
            return dict(tool=tool, args=args, current_before=before, current_after=after,
                        handles_created=list(created), ok=ok, result_text=text)
        steps = [dict(step=1, calls=[event('segment_text', dict(time='t1', text='building'), created=['inst#0'])]),
                 dict(step=2, calls=[event('segment_text', dict(time='t2', text='building'), created=['inst#1'])]),
                 dict(step=3, calls=[event('to_mask', dict(sets=['inst#1']), after='mask#2', created=['mask#2'])]),
                 dict(step=4, calls=[], note='report', report='Area constraint (regions must be larger than 1 m^2): ...'),
                 dict(step=5, calls=[event('remove_region', dict(mask='mask#2', ids=[3]), 'mask#2', 'mask#3', ['mask#3'])]),
                 dict(step=6, calls=[event('morph', dict(mask='mask#3', op='close', radius=2), 'mask#3', 'mask#4', ['mask#4'])]),
                 dict(step=7, calls=[event('select', dict(set='mask#2', area_max=4), 'mask#4', 'mask#4', ok=False),
                                     event('select', dict(set='inst#1', where='area > 4'), 'mask#4', 'mask#4')]),
                 dict(step=8, calls=[event('morph', dict(mask='mask#3', op='close', radius=2, set_current=False), 'mask#4', 'mask#4'),
                                     event('morph', dict(mask='mask#4', op='fill_holes'), 'mask#4', 'mask#4')])]
        write_text(directory / 'steps.jsonl', ''.join(json.dumps(r)+'\n' for r in steps))
        write_json(directory / 'input.json', dict(task_id='T4.1'))
        write_json(directory / 'final.json', dict(mask_path='masks/mask_4.png', submitted_by='model'))
        save_image(root / 'truth/train_toy.png', merged.astype(np.uint8)*255)
        save_image(root / 'fixed/train_toy.png', merged.astype(np.uint8)*255)
        task = dict(id='train_toy.png', image_t1='data/train_original/A/train_toy.png', gt_mask='truth/train_toy.png', area_min_m2=1, gsd_m=.5)
        trace = analyze_trace(directory, task, root)
        self.assertEqual(len(trace['mask_select']), 1)
        self.assertFalse(trace['mask_select'][0]['ok'])
        self.assertEqual(len(trace['instance_select']), 1)
        self.assertEqual(trace['remove_small_after_report'][0]['ids'], [3])
        self.assertEqual(trace['t2_as_change_candidates'][0]['step'], 3)
        self.assertTrue(trace['t2_as_change_candidates'][0]['majority_t1_overlap'])
        merge = trace['morph_merges'][0]
        self.assertEqual((merge['step'], merge['radius'], merge['components_before'], merge['components_after']), (6,2,2,1))
        self.assertEqual(len(trace['morph_merges']), 1)
        # A second, empty session tests no-filter sessions and per-image empty F1.
        empty_dir = root / 'run/T4.1/val/train_empty'
        save_image(empty_dir / 'masks/mask_0.png', np.zeros(SHAPE, np.uint8))
        write_json(empty_dir / 'input.json', dict(task_id='T4.1'))
        write_json(empty_dir / 'final.json', dict(mask_path='masks/mask_0.png', submitted_by='model'))
        write_text(empty_dir / 'steps.jsonl', '')
        save_image(root / 'truth/train_empty.png', np.zeros(SHAPE, np.uint8))
        save_image(root / 'fixed/train_empty.png', np.zeros(SHAPE, np.uint8))
        empty_task = dict(task, id='train_empty.png', gt_mask='truth/train_empty.png')
        write_text(root / 'tasks.jsonl', json.dumps(task)+'\n'+json.dumps(empty_task)+'\n')
        report = summarize(['run'], 'tasks.jsonl', root, 'fixed')
        self.assertEqual(report['sessions'], 2)
        self.assertEqual(report['filtering']['no_area_filter_sessions'], 1)
        self.assertEqual(report['filtering']['mask_select']['successful_calls'], 0)
        self.assertEqual(report['agent']['f1_c'], 100.)
        self.assertEqual(report['fixed_area']['f1_c'], 100.)
        self.assertEqual(report['missing_task_ids'], [])
        self.assertIn('组件 2 → 1', markdown(report))
        self.assertIn('步骤 3', markdown(report))
        self.assertTrue(area_request(dict(where='area <= 1040')))
        self.assertTrue(area_request(dict(area_max=10)))
        self.assertFalse(area_request(dict(ids=[1])))

    def test_no_area_requests_byte_identical(self):
        expected = json.loads(read_text(Path(__file__).parent / 'fixtures/t41_before_v3.json'))
        actual = legacy_requests(self.root)
        for name in expected:
            with self.subTest(case=name):
                for field in ('schemas', 'messages', 'report'):
                    self.assertEqual(json.dumps(actual[name][field], ensure_ascii=False).encode(),
                                     json.dumps(expected[name][field], ensure_ascii=False).encode())


if __name__ == '__main__':
    unittest.main()
