import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from agent_baseline.find_missed import CachedMissedCandidates, SOURCES, offer
from agent_baseline.state import Session, Instances
from agent_baseline.tools import Toolbox

class FindMissedTest(unittest.TestCase):
    def test_offer_preserves_answer_and_proposal_provenance(self):
        with TemporaryDirectory() as directory:
            shape=(8,8);a=np.zeros(shape,bool);a[:3,:3]=True
            base=np.zeros(shape,bool);base[0,0]=True
            for source in SOURCES:
                path=Path(directory)/'candidates'/'second_appeared'/source/'sample.npz'
                path.parent.mkdir(parents=True)
                masks=np.asarray([np.packbits(a)]) if source in ('S1_02','S2') else np.empty((0,8),np.uint8)
                np.savez(path,masks=masks,shape=shape)
            session=Session(shape,base,target='building',change_direction='appeared',scope='building_rebuild_excluded')
            toolbox=Toolbox([np.zeros((*shape,3),np.uint8)]*2,directory,session=session)
            before=session.current_mask
            provider=CachedMissedCandidates(directory,'second_appeared','sample',shape)
            result=offer(toolbox,provider,limit=5)
            self.assertEqual(session.current_mask,before)
            self.assertIsNone(session.submitted)
            self.assertEqual(result['candidates'][0]['sources'],['S1_02','S2'])
            proposals=session.get(result['props'],Instances)
            self.assertEqual(len(proposals.items),1)
            self.assertEqual(proposals.items[0]['area'],8)
            self.assertNotIn(0,proposals.items[0]['idx'])

if __name__=='__main__':unittest.main()
