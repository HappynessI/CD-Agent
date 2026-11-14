import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from agent_baseline.whu_gt import truth, paths, update_task
from agent_baseline.evaluation import load_reference, pixel_counts
from agent_baseline.recorder import Recorder
from agent_baseline.state import Session
from agent_baseline.analysis.second_metrics import counts
from agent_baseline.analysis.pyes_band import component_pixels


class WhuFinalTests(unittest.TestCase):
    def test_retired_xor_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Retired WHU XOR'):
            load_reference('/some/run/whu/gt/new/train_1.png')

    def test_official_constraint_and_ambiguous_directions(self):
        a=np.array([[0,1,1,0,0,1,255]])
        b=np.array([[1,0,1,0,1,0,1]])
        official=np.array([[1,1,1,1,0,0,1]])
        t=truth(a,b,official)
        np.testing.assert_array_equal(t['c0'],official>0)
        np.testing.assert_array_equal(t['new'],[[1,0,0,0,0,0,0]])
        np.testing.assert_array_equal(t['demolished'],[[0,1,0,0,0,0,0]])
        np.testing.assert_array_equal(t['ignore'],[[0,0,1,1,0,0,1]])
        np.testing.assert_array_equal(t['new']|t['demolished']|t['ignore'],t['c0'])

    def test_official_empty_overrides_semantic_xor(self):
        t=truth(np.zeros((2,3)),np.ones((2,3)),np.zeros((2,3)))
        self.assertFalse(any(x.any() for x in t.values()))

    def test_swapping_dates_swaps_only_directions(self):
        a=np.array([[0,1,1]]);b=np.array([[1,0,1]]);g=np.ones_like(a)
        x,y=truth(a,b,g),truth(b,a,g)
        np.testing.assert_array_equal(x['new'],y['demolished'])
        np.testing.assert_array_equal(x['ignore'],y['ignore'])

    def test_ignore_excludes_unknown_not_official_negative(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);p,ignore=paths(run,'tile.png','appeared')
            p.parent.mkdir(parents=True);ignore.parent.mkdir(parents=True)
            Image.fromarray(np.array([[255,0,0,0]],np.uint8)).save(p)
            Image.fromarray(np.array([[0,255,0,0]],np.uint8)).save(ignore)
            gt,valid=load_reference(p)
            pred=np.ones_like(gt)
            self.assertEqual(pixel_counts(pred,gt,valid),dict(tp=1,fp=2,fn=0))
            self.assertEqual(counts(pred,gt,valid=valid)[:3],[1,2,0])
            self.assertEqual(component_pixels(pred,gt,valid),[(1,1,2)])
            session=Session(gt.shape,pred)
            recorder=Recorder(run/'trace','synthetic','tile.png',gt_path=p,gt_ignore_path=ignore)
            before=session.get(session.current_mask).data.copy()
            self.assertEqual(recorder.metric(session,session.current_mask)['fp'],2)
            np.testing.assert_array_equal(before,session.get(session.current_mask).data)
            recorder.start(session,{})
            self.assertEqual(json.loads((run/'trace/input.json').read_text())['ignored_pixels'],1)

    def test_missing_canonical_ignore_is_not_silently_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=paths(d,'a.png','appeared');p.parent.mkdir(parents=True)
            Image.fromarray(np.zeros((2,2),np.uint8)).save(p)
            with self.assertRaises(FileNotFoundError):load_reference(p)

    def test_manifest_sets_ignore_for_both_directions_only(self):
        row=dict(id='a.png',change_direction='any',meta={})
        self.assertIsNone(update_task(row,'/tmp/test-whu')['gt_ignore_mask'])
        for direction in ['appeared','disappeared']:
            fixed=update_task(dict(row,change_direction=direction),'/tmp/test-whu')
            self.assertTrue(fixed['gt_ignore_mask'].endswith('/ignore/a.png'))

if __name__=='__main__':unittest.main()
