"""Regression for the observed truncated JSON score failure on actual mask assets."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from agent_baseline.analysis.n8_missed_prepare import save_masks
from agent_baseline.analysis.n9_missed_pipeline import config_key
from agent_baseline.analysis.n9_missed_score import score

class ScoreAbstainIntegration(unittest.TestCase):
    def test_six_separate_lines_freeze_without_turning_unknown_into_no(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);refs=[];items=[];states=[]
            configurations={p:dict(profile=p,sources='all',semantic=0.,dino=.4 if p=='any' else None,alpha=0.,verifier='V0',k=1) for p in ('any','build','demolish')}
            for group,profile in [('levir_any','any'),('whu_any','any'),('second_appeared','build'),('whu_appeared','build'),('second_disappeared','demolish'),('whu_disappeared','demolish')]:
                base=np.zeros((8,8),bool);base[2,2]=True
                candidate=np.zeros_like(base);candidate[3,3]=True;gt=base|candidate
                folder=out/group;folder.mkdir();Image.fromarray(gt.astype('uint8')*255).save(folder/'gt.png')
                np.savez_compressed(folder/'base.npz',start=base)
                save_masks(folder/'masks.npz',[candidate],base.shape)
                refs.append(dict(group=group,stem='case',task=dict(gt_mask=str(folder/'gt.png'))))
                items.append(dict(group=group,stem='case',profile=profile,split='tune',base=str(folder/'base.npz'),masks=str(folder/'masks.npz'),image_t1='t1',image_t2='t2'))
                states.append(dict(group=group,stem='case',rankings={config_key(configurations[profile]):[0]}))
                answer=out/'answers/V0'/f'{group}__case__n9_0000.json';answer.parent.mkdir(parents=True,exist_ok=True)
                answer.write_text(json.dumps(dict(answer='unknown' if group=='whu_any' else 'yes',finish_reason='length' if group=='whu_any' else 'stop',cost_cny=.001)))
            for name,value in [('evaluation_manifest',refs),('fm1_manifest',items),('states_tune',states)]:
                (out/(name+'.json')).write_text(json.dumps(value))
            with patch('agent_baseline.analysis.n9_missed_score.configs_for',side_effect=lambda o,p,s:[configurations[p]]):
                score(SimpleNamespace(out=out,split='tune',allow_abstain=True))
            data=json.loads((out/'scores_tune.json').read_text());profiles=json.loads((out/'frozen_profiles.json').read_text())
            lines={g:r for value in data['summaries'].values() for g,r in value['lines'].items()}
            self.assertEqual(len(lines),6);self.assertEqual(set(profiles['profiles']),{'any','build','demolish'})
            self.assertEqual(lines['whu_any']['delta_f1'],0)
            self.assertEqual(lines['whu_any']['invalid_calls'],1)
            self.assertEqual(lines['whu_any']['added_tp'],0)
            self.assertEqual(lines['second_appeared']['added_tp'],1)
            self.assertEqual(lines['second_appeared']['after']['f1'],100.)
            self.assertEqual(json.loads((out/'answers/V0/whu_any__case__n9_0000.json').read_text())['answer'],'unknown')

if __name__=='__main__':unittest.main()
