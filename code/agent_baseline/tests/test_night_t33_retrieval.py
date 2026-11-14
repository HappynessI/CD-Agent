import unittest
import numpy as np
from agent_baseline.tools_t13 import Features
from agent_baseline.analysis.night_t33_retrieval import external_scores,geometry_indices


class CrossImageRetrievalTests(unittest.TestCase):
    def test_external_reference_coordinates_and_direction(self):
        masks=np.array([[[1,0],[0,0]],[[0,1],[0,0]]],bool)
        q1=Features(np.zeros((2,2,2),np.float32),(2,2),'t1')
        q2=Features(np.array([[[1,0],[0,1]],[[0,0],[0,0]]],np.float32),(2,2),'t2')
        r1=Features(np.zeros((1,1,2),np.float32),(3,3),'t1')
        r2=Features(np.array([[[0,1]]],np.float32),(3,3),'t2')
        support=np.ones((3,3),bool)
        scores=external_scores(masks,[q1,q2],[r1,r2],support)
        self.assertGreater(scores[1],scores[0])
        reversed_scores=external_scores(masks,[q1,q2],[r2,r1],support)
        self.assertLess(reversed_scores[1],reversed_scores[0])

    def test_geometry_compares_different_image_shapes(self):
        masks=np.ones((1,4,4),bool);reference=np.ones((2,2),bool)
        query=[np.full((4,4,3),100,np.uint8)]*2
        ref=[np.full((2,2,3),100,np.uint8)]*2
        self.assertEqual(geometry_indices(masks,query,reference,ref),[0])


if __name__=='__main__':unittest.main()


class CrossImagePrepareTests(unittest.TestCase):
    def test_prepare_has_no_api_gt_scoring_or_final_summary(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch,MagicMock
        from contextlib import ExitStack
        from agent_baseline.analysis import night_t33_retrieval as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'t33';(out/'B0').mkdir(parents=True)
            (out/'audit.json').write_text(json.dumps(dict(pilot_ids=['case'],split={'case':'tune'})))
            (out/'B0/summary.json').write_text(json.dumps(dict(records=[dict(id='case',pair=[1,2])])))
            row=dict(id='case',image_t1='first',image_t2='second',gt_mask='truth',examples=[dict(image_t1='external1',image_t2='external2',mask='support')])
            manifest=root/'t3x/tasks/dev/tasks.jsonl';manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps(row)+'\n')
            folder=out/'retrieval/B0/tasks/case';folder.mkdir(parents=True);(folder/'prepared.npz').touch()
            masks=np.ones((1,8,8),bool)
            def image(path,**kwargs):
                self.assertNotEqual(path,'truth')
                return np.ones((8,8,3),np.uint8) if kwargs.get('rgb') else masks[0]
            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'test'}))
                stack.enter_context(patch('sys.argv',['prepare','--run',directory,'--night',directory,'--parse-variant','B0','--prepare-only']))
                stack.enter_context(patch.object(module,'load_dev',return_value=[row]))
                stack.enter_context(patch.object(module,'read_image',side_effect=image))
                stack.enter_context(patch.object(module,'read_npz',return_value=dict(r1=masks,r2=masks,order1=np.array([0]),order3=np.array([0]))))
                for name in ('CachedVision','TextProposals','Features'):stack.enter_context(patch.object(module,name,MagicMock()))
                for name in ('request','score'):stack.enter_context(patch.object(module,name,side_effect=AssertionError('prepare must not verify/score')))
                module.main()
            self.assertFalse((out/'retrieval/B0/summary.json').exists())
            self.assertEqual(json.loads((out/'retrieval/B0/prepare_progress.json').read_text())['tasks'],1)
            self.assertNotIn('prepare_only',json.loads((out/'retrieval/B0/settings.json').read_text()))
