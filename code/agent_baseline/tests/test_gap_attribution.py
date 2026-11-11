"""Synthetic offline archives; no model/online module dependencies."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image
from agent_baseline.analysis.gap_attribution import (
    Trace, analyze_session, summarize, bootstrap_delta, parameter_bin, retry_read)


class GapAttributionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.d = self.root/'second_g3_appeared_s1/T1.1xC1/val/example'
        (self.d/'masks').mkdir(parents=True)
        (self.d/'insts').mkdir()
        self.shape, self.calls = (10, 10), []

    def tearDown(self):
        self.temp.cleanup()

    def mask(self, h, coords):
        a = np.zeros(self.shape, bool)
        for y, x in coords:
            a[y, x] = True
        Image.fromarray(a.astype('uint8')*255).save(self.d/'masks'/(h.replace('#', '_')+'.png'))
        return a

    def inst(self, h, coords, time='t2', ids=None):
        arrays = []
        for points in coords:
            a = np.zeros(self.shape, bool)
            for y, x in points:
                a[y, x] = True
            arrays.append(np.packbits(a.ravel()))
        np.savez(self.d/'insts'/(h.replace('#', '_')+'.npz'),
            masks=np.asarray(arrays, dtype=np.uint8).reshape(len(coords), 13), shape=self.shape,
            ids=ids if ids is not None else np.arange(1, len(coords)+1), time=time, parents=np.array([], dtype=str))

    def call(self, tool, args, outputs, ok=True):
        self.calls.append(dict(tool=tool, args=args, handles_created=outputs, ok=ok))
        (self.d/'steps.jsonl').write_text(''.join(json.dumps(dict(step=i+1, calls=[c]))+'\n' for i, c in enumerate(self.calls)))

    def trace(self, h):
        t = Trace(self.d, np.arange(100), self.shape)
        return t, t.origins(h)[1]

    def test_combine_add_region_and_earliest(self):
        self.mask('mask#0', [(1, 1), (1, 2)])
        self.call('segment_prompt', dict(time='t2', points=[[1, 1]], point_labels=[1]), ['mask#0'])
        self.inst('inst#1', [[(1, 1), (2, 2)], [(3, 3)]])
        self.call('segment_text', dict(time='t2', text='building'), ['inst#1'])
        self.mask('mask#2', [(1, 1), (1, 2), (3, 3)])
        self.call('add_region', dict(mask='mask#0', other='inst#1', ids=[2]), ['mask#2'])
        self.mask('mask#3', [(4, 4), (1, 1)])
        self.call('segment_prompt', dict(time='t2', box=[0, 0, 2, 2]), ['mask#3'])
        self.mask('mask#4', [(1, 1), (1, 2), (3, 3), (4, 4)])
        self.call('combine', dict(op='union', a='mask#2', b='mask#3'), ['mask#4'])
        t, origins = self.trace('mask#4')
        self.assertEqual(origins[[11, 12, 33, 44]].tolist(), [0, 0, 1, 3])
        self.assertTrue(t.ambiguity[11])
        self.assertEqual(origins[22], -1)

    def test_match_same_time_and_subtract(self):
        self.inst('inst#0', [[(2, 2)]], 't1')
        self.call('segment_text', dict(time='t1', text='building'), ['inst#0'])
        self.inst('inst#1', [[(2, 2), (3, 3)]], 't2')
        self.call('segment_text', dict(time='t2', text='house'), ['inst#1'])
        self.inst('inst#2', [[(2, 2), (3, 3)]], 't2')
        self.call('match_instances', dict(inst_t1='inst#0', inst_t2='inst#1'), ['inst#2'])
        self.mask('mask#3', [(2, 2), (3, 3)])
        self.call('to_mask', dict(sets=['inst#2']), ['mask#3'])
        _, origin = self.trace('mask#3')
        self.assertEqual(origin[[22, 33]].tolist(), [1, 1])
        self.mask('mask#4', [(3, 3)])
        self.call('segment_prompt', dict(time='t2', points=[[3, 3]]), ['mask#4'])
        self.mask('mask#5', [(2, 2)])
        self.call('combine', dict(op='subtract', a='mask#3', b='mask#4'), ['mask#5'])
        _, origin = self.trace('mask#5')
        self.assertEqual(origin[22], 1)
        self.assertEqual(origin[33], -1)

    def test_add_region_preserved_component_ids(self):
        self.mask('mask#0', [(1, 1), (5, 5)])
        self.call('segment_prompt', dict(time='t2', box=[0, 0, 9, 9]), ['mask#0'])
        self.mask('mask#1', [(5, 5)])
        self.call('remove_region', dict(mask='mask#0', ids=[1]), ['mask#1'])
        self.mask('mask#2', [])
        self.mask('mask#3', [(5, 5)])
        self.call('add_region', dict(mask='mask#2', other='mask#1', ids=[2]), ['mask#3'])
        _, origin = self.trace('mask#3')
        self.assertEqual(origin[55], 0)

    def test_proposal_archive_uses_inst_filename(self):
        self.inst('inst#0', [[(4, 4)]])
        self.call('segment_everything', dict(time='t2'), ['props#0'])
        self.mask('mask#1', [(4, 4)])
        self.call('to_mask', dict(sets=['props#0']), ['mask#1'])
        _, origin = self.trace('mask#1')
        self.assertEqual(origin[44], 0)

    def test_morph_refine_python_and_unknown(self):
        self.mask('mask#0', [(1, 1)])
        self.mask('mask#1', [(1, 1), (1, 2)])
        self.call('morph', dict(mask='mask#0', op='close', radius=1), ['mask#1'])
        self.mask('mask#2', [(1, 1), (1, 2), (2, 2)])
        self.call('refine_region', dict(mask='mask#1', id=1, time='t2'), ['mask#2'])
        self.mask('mask#3', [(1, 1), (1, 2), (2, 2), (3, 3)])
        self.call('python', dict(code='save_mask(masks["mask#2"])'), ['mask#3'])
        _, origin = self.trace('mask#3')
        self.assertEqual(origin[[11, 12, 22, 33]].tolist(), [-1, 0, 1, 2])

    def test_components_summary_and_bootstrap(self):
        self.mask('mask#0', [(1, 1), (1, 2), (2, 3), (8, 8)])
        self.call('segment_prompt', dict(time='t2', box=[0, 0, 5, 5]), ['mask#0'])
        self.mask('mask#8', [(1, 1)])
        self.mask('mask#9', [(1, 1), (1, 2), (8, 8), (9, 9)])
        (self.d/'input.json').write_text(json.dumps(dict(gt_mask_path=str(self.d/'masks/mask_9.png'))))
        (self.d/'final.json').write_text(json.dumps(dict(final_handle='mask#0', mask_path='masks/mask_0.png', eval=dict(tp=3, fp=1, fn=1))))
        row = analyze_session(self.d, self.d/'masks/mask_8.png', self.root)
        self.assertEqual(row['component_count'], 2)
        self.assertEqual((row['gap_tp_pixels'], row['gap_fp_pixels']), (2, 1))
        self.assertEqual(row['without_gap_counts'], [1, 0, 3])
        g = summarize([dict(row, seed=str(i)) for i in (1, 2, 3)])
        self.assertEqual(g['tool_bins'][0]['component_count'], 6)
        self.assertEqual(g['tool_bins'][0]['tp_pixels'], 6)
        self.assertAlmostEqual(g['gap_tp_fraction_of_agent_tp'], 2/3)
        self.assertAlmostEqual(g['f1']['delta_f1_pp'], 35.)
        self.assertEqual(g['f1']['ci95_pp'], [35., 35.])
        self.assertEqual(bootstrap_delta([[[0, 0, 0]]], [[[0, 0, 0]]])['delta_f1_pp'], 0.)

    def test_parameters_and_io_retry(self):
        for box, expected in [([0, 0, 1, 1], '<1%'), ([0, 0, 10, 10], '1–10%'), ([0, 0, 90, 90], '>10%')]:
            self.assertEqual(parameter_bin('segment_prompt', dict(box=box), (100, 100))['box_size'], expected)
        with patch('agent_baseline.analysis.gap_attribution.time.sleep'), patch('pathlib.Path.read_text', side_effect=[OSError('EIO'), OSError('EIO'), 'ok']):
            self.assertEqual(retry_read(self.root/'x', lambda p: p.read_text()), 'ok')


if __name__ == '__main__':
    unittest.main()
