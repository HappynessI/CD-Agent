"""Synthetic offline masks and steps.jsonl; all temporary files stay in workspace."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from agent_baseline.analysis import error_taxonomy as et


class PixelTest(unittest.TestCase):
    def test_partitions_and_diagonal_connectivity(self):
        gt = np.zeros((20, 20), bool)
        gt[1:3, 1:3] = True; gt[8:10, 8:10] = True
        pred = np.zeros_like(gt)
        pred[1, 1] = pred[0, 0] = pred[15, 15] = True
        c = et.pixel_taxonomy(pred, gt)
        self.assertEqual(int(c['粘连误检'].sum()), 1)
        self.assertEqual(int(c['整目标误检'].sum()), 1)
        self.assertEqual(int(c['整目标漏检'].sum()), 4)
        self.assertEqual(int(c['部分漏检或边界'].sum()), 3)
        self.assertEqual(sum(c[k].sum() for k in et.FP_KEYS), 2)
        self.assertEqual(sum(c[k].sum() for k in et.FN_KEYS[:2]), 7)
        self.assertEqual(int(c['小目标漏检'].sum()), 4)

    def test_direction_rebuilt_overlap(self):
        gt = np.zeros((10, 10), bool)
        pred = gt.copy(); pred[1:3, 1:3] = True
        t1 = np.zeros_like(gt, dtype=np.uint8); t1[1, 1:3] = 5
        t2 = t1.copy(); t2[1, 2] = 1
        c = et.pixel_taxonomy(pred, gt, pred, (t1 == 5) & (t2 == 5))
        self.assertEqual(int(c['整目标误检'].sum()), 4)
        self.assertEqual(int(c['方向错'].sum()), 4)
        self.assertEqual(int(c['原址重建'].sum()), 1)

    def test_strict_size_boundaries(self):
        gt = np.zeros((40, 40), bool)
        gt[0:8, 0:8] = True; gt[12:15, 0:3] = True
        gt[20:30, 10:26] = True
        c = et.pixel_taxonomy(np.zeros_like(gt), gt)
        self.assertEqual(int(c['小目标漏检'].sum()), 9)
        self.assertFalse(c['大目标漏检'].any())
        gt[30, 10] = True
        pred = np.zeros_like(gt); pred[20, 10] = True
        c = et.pixel_taxonomy(pred, gt)
        self.assertEqual(int(c['大目标漏检'].sum()), 160)
        self.assertEqual(int(c['部分漏检或边界'].sum()), 160)

    def test_empty_and_shape(self):
        z = np.zeros((4, 4), bool)
        self.assertTrue(all(not m.any() for m in et.pixel_taxonomy(z, z).values()))
        with self.assertRaises(ValueError):
            et.pixel_taxonomy(z, z[:2])


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = et.retry_io(lambda: tempfile.TemporaryDirectory(dir=et.ROOT))
        self.d = Path(self.temp.name)
        et.retry_io(lambda: (self.d/'masks').mkdir(exist_ok=True))
        self.shape, self.steps = (10, 10), []

    def tearDown(self):
        et.retry_io(self.temp.cleanup)

    def save(self, h, array):
        b = et.io.BytesIO()
        Image.fromarray(array.astype('uint8')*255).save(b, format='PNG')
        et.atomic_bytes(self.d/'masks'/(h.replace('#', '_')+'.png'), b.getvalue())

    def call(self, tool, args, outputs=(), public=None, ok=True, current=None, error=None):
        c = dict(tool=tool, args=args, handles_created=list(outputs), ok=ok,
                 result_text=json.dumps(public or {}), current_after=current, error_code=error)
        self.steps.append(dict(step=len(self.steps), calls=[c], current_mask=current))
        et.atomic_bytes(self.d/'steps.jsonl', ''.join(json.dumps(s)+'\n' for s in self.steps).encode())

    def analyze(self, pred, gt, final=None, meta=None, **kwargs):
        return et.trajectory_taxonomy(self.d, meta or {}, final or
            dict(submitted_by='model', final_handle='mask#2'), pred, gt, **kwargs)

    def test_fallback_latest_zero(self):
        a = np.zeros(self.shape, bool); a[0, 0] = True
        self.save('mask#0', a)
        self.call('segment_prompt', dict(points=[[0, 0]]), ['mask#0'])
        r, flags, d = self.analyze(np.zeros_like(a), a,
            final=dict(submitted_by='budget', final_handle='mask#0'))
        self.assertTrue(flags['兜底提交'])
        self.assertTrue(d['fallback_current_differs_latest'])
        self.assertEqual(d['fallback_latest'], 'mask#0')
        self.assertEqual(int(r['兜底提交'].sum()), 1)
        _, _, d = self.analyze(a, a, meta=dict(init_mask_path='given.png'),
            final=dict(submitted_by='budget', final_handle='mask#0'))
        self.assertIsNone(d['fallback_latest'])

    def test_bigbox_and_added_fp(self):
        a = np.zeros(self.shape, bool); a[1:6, 1:6] = True
        self.save('mask#1', a); self.save('mask#2', a)
        self.call('segment_prompt', dict(box=[0, 0, 5, 5]), ok=False, error='box_too_large')
        self.call('segment_prompt', dict(box=[0, 0, 5, 5]), ['mask#1'])
        self.call('morph', dict(mask='mask#1', op='fill_holes'), ['mask#2'])
        r, flags, d = self.analyze(a, np.zeros_like(a), fixed=np.zeros_like(a))
        self.assertTrue(flags['大框失败'])
        self.assertEqual((d['bigbox_calls'], d['bigbox_blocked'], d['bigbox_returned']), (2, 1, 1))
        self.assertEqual(int(r['大框失败'].sum()), 25)
        self.assertEqual(int(r['规划器补进 FP'].sum()), 25)

    def test_no_then_delete_stable_ids(self):
        gt = np.zeros(self.shape, bool); gt[1:3, 1:3] = True; gt[7:9, 7:9] = True
        after = gt.copy(); after[1:3, 1:3] = False
        for h, m in [('mask#0', gt), ('mask#1', after), ('mask#2', np.zeros_like(gt))]:
            self.save(h, m)
        self.call('ask', {}, public=dict(results=[dict(mask='mask#0', id=1, answer='no')]))
        self.call('remove_region', dict(mask='mask#0', ids=[1]), ['mask#1'])
        self.call('ask', {}, public=dict(results=[dict(mask='mask#1', id=2, answer='no')]))
        self.call('remove_region', dict(mask='mask#1', ids=[2]), ['mask#2'])
        r, flags, d = self.analyze(np.zeros_like(gt), gt)
        self.assertTrue(flags['验证器误删 TP'])
        self.assertEqual(d['verifier_removed_components'], 2)
        self.assertEqual(int(r['验证器误删 TP'].sum()), 8)

    def test_report_no_and_recovery(self):
        gt = np.zeros(self.shape, bool); gt[1:3, 1:3] = True
        self.save('mask#0', gt); self.save('mask#1', np.zeros_like(gt)); self.save('mask#2', gt)
        self.call('submit', dict(mask='mask#0'), public=dict(report='Verifier answers on mask#0: 0 yes, 1 no.\nIDs answered no: [1]'))
        self.call('remove_region', dict(mask='mask#0', ids=[1]), ['mask#1'])
        r, flags, _ = self.analyze(np.zeros_like(gt), gt, final=dict(submitted_by='model', final_handle='mask#1'))
        self.assertTrue(flags['验证器误删 TP'])
        self.assertEqual(int(r['验证器误删 TP'].sum()), 4)
        r, _, _ = self.analyze(gt, gt)
        self.assertFalse(r['验证器误删 TP'].any())

    def test_late_no_not_attributed(self):
        gt = np.zeros(self.shape, bool); gt[1:3, 1:3] = True
        self.save('mask#0', gt); self.save('mask#1', np.zeros_like(gt)); self.save('mask#2', np.zeros_like(gt))
        self.call('remove_region', dict(mask='mask#0', ids=[1]), ['mask#1'])
        self.call('ask', {}, public=dict(results=[dict(mask='mask#0', id=1, answer='no')]))
        _, flags, _ = self.analyze(np.zeros_like(gt), gt)
        self.assertFalse(flags['验证器误删 TP'])

    def test_route_retained_pixels_subtract(self):
        a = np.zeros(self.shape, bool); a[1, 1] = True
        b = np.zeros_like(a); b[5, 5] = True
        for h, m in [('mask#0', a), ('mask#1', b), ('mask#2', b)]:
            self.save(h, m)
        self.call('segment_semantic', dict(time='t1', text='tree'), ['mask#0'])
        self.call('segment_prompt', dict(points=[[5, 5]]), ['mask#1'])
        self.call('combine', dict(a='mask#1', b='mask#0', op='subtract'), ['mask#2'])
        _, flags, d = self.analyze(b, a, route='T1.2')
        self.assertTrue(flags['主路线选错'])
        self.assertEqual(d['route_reason'], '调用后未使用')
        self.call('combine', dict(a='mask#0', b='mask#1', op='union'), ['mask#3'])
        self.save('mask#3', a | b)
        _, flags, d = self.analyze(a | b, a, final=dict(final_handle='mask#3'), route='T1.2')
        self.assertFalse(flags['主路线选错'])
        self.assertEqual(d['retained_main_pixels'], 1)


class IOAndSummaryTest(unittest.TestCase):
    def test_retry_three(self):
        et.RETRIES.clear()
        with patch.object(et.time, 'sleep'):
            fn = Mock(side_effect=[OSError('EIO')]*3 + [7])
            self.assertEqual(et.retry_io(fn), 7)
            self.assertEqual(et.RETRIES['io_retries'], 3)

    def test_share_unique_images(self):
        row = dict(image='x', seed='1', counts=[2, 4, 2], details={}, categories={
            '粘连误检': dict(pixels=1, fp=1, fn=0, affected=True, denominator='FP', repaired_counts=[2, 3, 2], f1_repair_pp=1)})
        s = et.summarize({'d': {'a': [row, dict(row, seed='2')]}})['d']['a']
        self.assertEqual(s['sessions'], 2)
        self.assertEqual(s['categories']['粘连误检']['images'], 1)
        self.assertEqual(s['categories']['粘连误检']['pixel_share'], .25)

    def test_examples_maximum_and_seed_are_reproducible(self):
        rows = [dict(image=str(i), session='run/'+str(i), categories={'x': dict(affected=True, pixels=i)}) for i in range(8)]
        chosen = et.pick_examples(rows, 'x')
        self.assertEqual(chosen[0]['image'], '7')
        self.assertEqual(chosen, et.pick_examples(rows, 'x'))
        self.assertEqual(len({r['image'] for r in chosen}), 3)

    def test_merge_recomputes_weights_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory(dir=et.ROOT) as tmp:
            root = Path(tmp)
            def part(name, fp, attached):
                row = dict(image=name, session=None, seed=None, counts=[2, fp, 1], details={},
                    categories={'粘连误检': dict(pixels=attached, fp=attached, fn=0, affected=True,
                        denominator='FP', repaired_counts=[2, fp-attached, 1], f1_repair_pp=1)})
                return dict(records={'d': {'固定路线': [row]}}, coverage=[], missing=[],
                    r4_comparison=dict(older_fp=0, older_attached_fp=0, missing=[]),
                    elapsed_seconds=1, io_retries=0)
            one, two = root/'one.json', root/'two.json'
            et.write_json(one, part('a', 10, 2)); et.write_json(two, part('b', 30, 18))
            with patch.object(et, 'markdown', return_value='test'):
                et.merge_reports([one, two], root, root/'out', True)
            report = et.read_json(root/'out/error_taxonomy.json')
            self.assertEqual(report['summary']['d']['固定路线']['categories']['粘连误检']['pixel_share'], .5)
            with self.assertRaises(ValueError):
                et.merge_reports([one, one], root, root/'out', True)


if __name__ == '__main__':
    unittest.main()
