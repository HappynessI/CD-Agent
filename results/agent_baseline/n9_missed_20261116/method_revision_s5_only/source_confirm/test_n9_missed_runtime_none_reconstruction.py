"""CPU/Slurm contract for the frozen directed S5-only runtime configuration.

No API, no Agent run; actual reconstruction and feature pooling are exercised.
"""
import unittest
import numpy as np
from agent_baseline.n9_missed_online_tool import CurrentMaskProvider
from agent_baseline.analysis.n8_missed_assets import SOURCES
from agent_baseline.tools_t13 import Features


class DirectedRuntimeNoneReconstruction(unittest.TestCase):
    def test_low_dino_direct_sources_survive_but_s5_stays_gated(self):
        shape=(48,48)
        masks=[]
        for y,x in ((4,4),(4,20),(20,4),(20,20)):
            mask=np.zeros(shape,bool);mask[y:y+6,x:x+6]=True;masks.append(mask)
        # Spatially constant matching features give DINO difference 0. S5's
        # fourth square has orthogonal features, hence difference 1.
        f1=np.zeros((*shape,2),np.float32);f1[...,0]=1
        f2=f1.copy();f2[masks[3]]=np.array([0,1],np.float32)
        features=[Features(f1,shape,'t1'),Features(f2,shape,'t2')]
        for profile,direction in (('build','appeared'),('demolish','disappeared')):
            with self.subTest(profile=profile):
                config=dict(profile=profile,sources='all' if profile=='build' else 'S1_S2_S5',semantic=0.,dino=None,alpha=0.,verifier='V0',k=10)
                provider=CurrentMaskProvider({},config,direction)
                provider.sources={s:np.zeros(shape,bool) for s in SOURCES}
                provider.sources['S1_02']=masks[0].copy()
                provider.sources['S2']=masks[1].copy()
                provider.sources['S5']=masks[2]|masks[3]
                provider.features=features
                provider.probabilities=[np.ones(shape) if direction=='disappeared' else np.zeros(shape),np.zeros(shape) if direction=='disappeared' else np.ones(shape)]
                candidates,audit=provider.candidates(np.zeros(shape,bool))
                self.assertEqual(audit['s5_before'],2)
                self.assertEqual(audit['s5_after_dino020'],1)
                self.assertEqual(len(candidates),3)
                for index,source in ((0,'S1_02'),(1,'S2'),(3,'S5')):
                    found=[c for c in candidates if np.array_equal(c['mask'],masks[index])]
                    self.assertEqual(len(found),1)
                    self.assertIn(source,found[0]['sources'])
                    if index<2:self.assertLess(found[0]['dino'],.2)
                self.assertFalse(any(np.array_equal(c['mask'],masks[2]) for c in candidates))
                # Same cache, changed current answer: real dilation subtraction
                # removes S1; S2 and the strong S5 remain, with no stale replay.
                updated,_=provider.candidates(masks[0])
                self.assertEqual(len(updated),2)
                self.assertFalse(any((c['mask']&masks[0]).any() for c in updated))

if __name__=='__main__':unittest.main()
