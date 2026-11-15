import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import subprocess
import sys
import numpy as np
from agent_baseline.frozen_ranked_candidates import FrozenRankedCandidates
from agent_baseline.analysis.missed_online_runner import arm_arguments,common_arguments
from agent_baseline.state import Session, Instances
from agent_baseline.tools import Toolbox


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
        self.assertEqual(arm_arguments('baseline','four','cache','rank.json'),['--missed-verifier-variant','four'])
        self.assertEqual(arm_arguments('baseline','four','cache'),[])
        self.assertEqual(arm_arguments('find_missed','four','cache'),['--missed-candidates-root','cache','--missed-verifier-variant','four'])
        self.assertEqual(arm_arguments('find_missed','four','cache','rank.json'),['--missed-ranked-manifest','rank.json','--missed-verifier-variant','four'])
        with self.assertRaises(ValueError):arm_arguments('find_missed','structured','cache','rank.json')
        self.assertEqual(common_arguments('tasks','config',1016).count('qwen3.7-flash'),2)

    def test_actual_toolbox_dispatch_keeps_answer_and_rank_metadata(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);masks,order=self.fixture(root)
            session=Session(masks.shape[1:],masks[order[0]].copy(),target='building',
                change_direction='appeared',scope='building_rebuild_excluded')
            toolbox=Toolbox([np.zeros((*session.shape,3),np.uint8)]*2,tmp,session=session)
            toolbox.missed_provider=FrozenRankedCandidates(root/'manifest.json','second_appeared','train_1884',session.shape)
            before=session.current_mask
            result=toolbox.find_missed(limit=20)
            self.assertEqual(session.current_mask,before)
            self.assertIsNone(session.submitted)
            self.assertEqual(result['ranking'],'frozen_evidence_top10_no_refill')
            self.assertFalse(result['refill_outside_top10'])
            self.assertEqual(result['candidates'][0]['rank'],2)
            self.assertEqual(len(session.get(result['props'],Instances).items),9)
            self.assertNotIn('must-not-reach-planner',json.dumps(result))

    def test_real_entrypoint_accepts_ranked_flag(self):
        result=subprocess.run([sys.executable,'-B','-m','agent_baseline.loop_json','--help'],
            capture_output=True,text=True,check=True)
        self.assertIn('--missed-ranked-manifest',result.stdout)


if __name__=='__main__':unittest.main()
