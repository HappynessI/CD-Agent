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
