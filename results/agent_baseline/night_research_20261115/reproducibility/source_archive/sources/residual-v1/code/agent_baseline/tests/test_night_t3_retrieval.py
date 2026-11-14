import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.night_t3_retrieval import ranked,local_views,score,TextProposals
from agent_baseline.analysis.t3_routes_cache import write_npz


class RetrievalTests(unittest.TestCase):
    def test_ranking_and_exemplar_exclusion(self):
        masks=np.array([[[1,0],[0,0]],[[0,1],[0,0]]],bool);ex=masks[0];truth=masks.any(axis=0)
        with patch('agent_baseline.analysis.night_t3_retrieval.task_scores',return_value=np.array([.2,.8])):
            self.assertEqual(ranked(masks,[None,None],ex),[1,0])
        self.assertEqual(score(masks,[0,1],truth,ex,'test',2)['counts'],[1,0,0])

    def test_local_verifier_has_exactly_two_date_views(self):
        mask=np.zeros((80,80),bool);mask[30:50,30:50]=True
        pictures=local_views([np.full((80,80,3),200,np.uint8),np.full((80,80,3),100,np.uint8)],mask)
        self.assertEqual(len(pictures),2)
        self.assertIn('T1',pictures[0][0]);self.assertIn('T2',pictures[1][0])
        self.assertGreater(np.asarray(pictures[0][1])[192,192,0],np.asarray(pictures[1][1])[192,192,0])

    def test_text_cache_requires_predicted_class_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'example';folder.mkdir()
            (folder/'t1_t1.json').write_text(json.dumps(dict(request=dict(query='/image/t1.png',text='water'),pixel_thr=.85,score_thr=.5)))
            mask=np.ones((1,2,2),bool)
            write_npz(folder/'t1_t1.npz',masks=np.packbits(mask.reshape(1,-1),axis=1),shape=(2,2))
            backend=TextProposals.__new__(TextProposals);backend.old=root;backend.reused=[]
            class Fake:
                def for_row(self,row):
                    return lambda **kwargs:np.zeros((0,2,2),bool)
            backend.pool=Fake();row=dict(id='example',image_t1='/image/t1.png')
            self.assertEqual(len(backend.get(row,'water','t1')),1)
            self.assertEqual(len(backend.get(row,'building','t1')),0)
            self.assertEqual(len(backend.reused),1)


if __name__=='__main__':unittest.main()


class VerificationContractTests(unittest.TestCase):
    def test_unparsed_is_reported_separately(self):
        from agent_baseline.analysis.night_t3_retrieval import verification_stats
        result=verification_stats({0:dict(answer='yes'),1:dict(answer='no'),2:dict(answer='unparsed')})
        self.assertEqual((result['requests'],result['yes'],result['no'],result['unparsed']),(3,1,1,1))


class PrepareOnlyTests(unittest.TestCase):
    def test_cached_prepare_never_verifies_or_scores(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch,MagicMock
        from contextlib import ExitStack
        from agent_baseline.analysis import night_t3_retrieval as module
        with tempfile.TemporaryDirectory() as directory:
            night=Path(directory);(night/'B0').mkdir()
            (night/'B0/summary.json').write_text(json.dumps(dict(tasks=60,records=[dict(id='case',interpretation=dict(pair=[1,2]))])))
            (night/'audit.json').write_text(json.dumps(dict(split={'case':'tune'})))
            folder=night/'retrieval/B0/tasks/case';folder.mkdir(parents=True);(folder/'prepared.npz').touch()
            row=dict(id='case',exemplar_mask='support',image_t1='first',image_t2='second',gt_mask='truth')
            masks=np.ones((1,8,8),bool)
            def image(path,**kwargs):
                self.assertNotEqual(path,'truth')
                return np.ones((8,8,3),np.uint8) if kwargs.get('rgb') else masks[0]
            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'test'}))
                stack.enter_context(patch('sys.argv',['prepare','--run',directory,'--night',directory,'--parse-variant','B0','--split','tune','--prepare-only']))
                stack.enter_context(patch.object(module,'load_dev',return_value=[row]))
                stack.enter_context(patch.object(module,'read_image',side_effect=image))
                stack.enter_context(patch.object(module,'read_npz',return_value=dict(r1=masks,r2=masks,order1=np.array([0]),order3=np.array([0]))))
                for name in ('CachedVision','TextProposals','Features'):
                    stack.enter_context(patch.object(module,name,MagicMock()))
                api=stack.enter_context(patch.object(module,'request',side_effect=AssertionError('No API during preparation')))
                scoring=stack.enter_context(patch.object(module,'score',side_effect=AssertionError('No completed scores during preparation')))
                module.main();api.assert_not_called();scoring.assert_not_called()
            self.assertFalse((night/'retrieval/B0/summary_tune.json').exists())
            progress=json.loads((night/'retrieval/B0/prepare_progress_tune.json').read_text())
            self.assertEqual(progress['tasks'],1)
            self.assertFalse(progress['scoring_complete'])
            settings=json.loads((night/'retrieval/B0/settings.json').read_text())
            self.assertEqual(settings['verify_k'],8)
            self.assertNotIn('prepare_only',settings)
