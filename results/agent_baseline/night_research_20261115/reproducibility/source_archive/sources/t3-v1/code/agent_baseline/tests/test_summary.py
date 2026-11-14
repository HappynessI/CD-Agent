import json
from pathlib import Path
import unittest
import numpy as np
from agent_baseline.summarize import summarize
from agent_baseline.recorder import Recorder
from common import read_jsonl
from .helpers import run, call, response, mask, tempdir


class SummaryTests(unittest.TestCase):
    def test_whu_ignore_applies_to_recomputed_summary(self):
        from agent_baseline.whu_gt import paths
        from common import save_mask_png
        with tempdir() as root:
            root=Path(root)
            gt,ignore=paths(root,'a.png','appeared')
            truth=mask();pred=truth.copy();pred[0,0]=True
            uncertain=np.zeros_like(truth);uncertain[0,0]=True
            save_mask_png(gt,truth);save_mask_png(ignore,uncertain)
            directory=root/'sessions'
            run(directory/'a',[response([call('submit',dict(mask='mask#0',summary='Keep.'))])],init=pred,gt=truth)
            info=directory/'a/input.json';meta=json.loads(info.read_text())
            meta.update(gt_mask_path=str(gt),gt_ignore_mask_path=str(ignore));info.write_text(json.dumps(meta))
            result=summarize(directory,'synthetic','T1.1xC0')
            self.assertEqual(result['fp'],0)

    def test_input_failure_is_an_empty_scored_session(self):
        with tempdir() as root:
            recorder=Recorder(Path(root)/'broken','synthetic','broken.png',gt=mask())
            result=recorder.input_failure('T1.1xC0',{},'synthetic image read failure')
            self.assertEqual(result['eval']['fn'],36)
            self.assertFalse(result['initial_images_present'])
            summary=summarize(root,'synthetic','T1.1xC0',gts={'broken.png':mask()})
            self.assertEqual(summary['failure_rate'],1.)
            self.assertEqual(summary['images'],1)

    def test_failures_counted_and_initial_delta(self):
        with tempdir() as root:
            root=Path(root)
            run(root/'a',[response([call('submit',dict(mask='mask#0',summary='Keep.'))])],init=mask(),gt=mask())
            run(root/'b',[response([call('remove_region',dict(mask='mask#0',ids=[1]))])],init=mask(),gt=mask(None),budget=1)
            result=summarize(root,'synthetic','T5.1',errors=True,gts={'a.png':mask(),'b.png':mask(None)})
            self.assertEqual(result['images'],2)
            self.assertEqual(result['failure_rate'],.5)
            self.assertEqual(result['unchanged_submission_rate'],.5)
            self.assertEqual((result['improved'],result['worsened'],result['unchanged']),(1,0,1))
            self.assertGreater(result['delta_f1_c'],0)
            self.assertEqual(result['modification_rate'],.5)
            self.assertEqual(result['routes']['other'],2)
            diagnostics=json.loads((root/'errors.json').read_text())
            self.assertEqual(diagnostics[1]['submitted_by'],'budget')
            self.assertEqual(diagnostics[1]['modification_calls'],1)
            self.assertEqual(diagnostics[1]['final_f1'],1.)

    def test_inst_assets_can_be_restored(self):
        with tempdir() as root:
            script=[response([call('segment_text',dict(time='t1',text='building'))]),
                    response([call('select',dict(set='inst#0',ids=[1]))]),
                    response([call('to_mask',dict(sets=['inst#1']))])]
            final,toolbox,_,_=run(root,script,budget=3)
            with np.load(Path(root)/'insts'/'inst_1.npz') as z:
                h,w=z['shape']
                restored=np.unpackbits(z['masks'],axis=1,count=int(h*w)).reshape(-1,h,w).astype(bool)
                np.testing.assert_array_equal(restored[0],mask())
                self.assertEqual(z['ids'].tolist(),[1])
                self.assertEqual(z['parents'].tolist(),['inst#0'])
                self.assertEqual(str(z['time']),'t1')


if __name__ == '__main__':
    unittest.main()
