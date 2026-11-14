import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from agent_baseline.frozen_ranked_candidates import FrozenRankedCandidates
from agent_baseline.analysis.missed_online_runner import arm_arguments,common_arguments


class FrozenRankedTest(unittest.TestCase):
    def fixture(self,root):
        shape=(16,16);masks=np.zeros((12,*shape),bool)
        for i in range(12):masks[i,2,i+1]=True
        np.savez(root/'masks.npz',shape=shape,masks=np.packbits(masks.reshape(12,-1),axis=1))
        order=[5,1,8,0,9,2,4,3,6,7,10,11]
        item=dict(group='second_appeared',stem='train_1884',s5_gate='dino_020',masks='masks.npz',
            rankings=dict(evidence=order),gt_mask='must-not-reach-planner',
            candidates=[dict(source='S5',sources=['S5'],old_answer='/not/read.json',gt_tp=999) for _ in range(12)])
        (root/'manifest.json').write_text(json.dumps([item]))
        (root/'settings.json').write_text(json.dumps(dict(variant='V0',s5_selection={'second_appeared':dict(gate='dino_020')})))
        return masks,order

    def test_frozen_order_current_subtraction_no_refill_no_labels(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);masks,order=self.fixture(root);provider=FrozenRankedCandidates(root/'manifest.json','second_appeared','train_1884',masks.shape[1:])
            current=np.zeros(masks.shape[1:],bool);before=current.copy()
            rows=provider.candidates(current,20)
            self.assertEqual([r['original'] for r in rows],order[:10])
            self.assertTrue(np.array_equal(current,before))
            self.assertTrue(all('gt_tp' not in r and 'old_answer' not in r for r in rows))
            current[:]=masks[:10].any(axis=0)
            self.assertEqual(provider.candidates(current,20),[]) # ranks 11/12 not refilled
            current[:]=False;current|=masks[order[0]]
            self.assertEqual(provider.candidates(current,10)[0]['source_rank'],2)
            # A fresh session gets the same static pool, no mutable answer cache.
            fresh=FrozenRankedCandidates(root/'manifest.json','second_appeared','train_1884',masks.shape[1:])
            self.assertEqual(len(fresh.candidates(np.zeros_like(current),10)),10)
            missing=FrozenRankedCandidates(root/'manifest.json','second_appeared','train_368',masks.shape[1:])
            with self.assertRaises(ValueError):missing.load()

    def test_runner_default_and_rank_opt_in_are_separate(self):
        self.assertEqual(arm_arguments('baseline','four','cache','rank.json'),[])
        self.assertEqual(arm_arguments('find_missed','four','cache'),['--missed-candidates-root','cache','--missed-verifier-variant','four'])
        self.assertEqual(arm_arguments('find_missed','four','cache','rank.json'),['--missed-ranked-manifest','rank.json','--missed-verifier-variant','four'])
        with self.assertRaises(ValueError):arm_arguments('find_missed','structured','cache','rank.json')
        self.assertEqual(common_arguments('tasks','config',1016).count('qwen3.7-flash'),2)


if __name__=='__main__':unittest.main()
