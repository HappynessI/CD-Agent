import unittest
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from agent_baseline.analysis.night_missed_rank import ordering,prepare,verify,candidate_images,SOURCES

class NightRankTest(unittest.TestCase):
    def test_new_v0_and_v3_candidates_preserve_original_background(self):
        mask=np.zeros((64,64),bool);mask[24:40,24:40]=True
        rgb=[np.full((64,64,3),200,np.uint8)]*2;base=np.zeros_like(mask)
        for variant in ('V0','V3'):
            images=candidate_images(rgb,mask,base,[24,24,40,40],variant)
            self.assertTrue(np.all(np.asarray(images[0])[0,0]==200))
        images=candidate_images(rgb,mask,base,[24,24,40,40],'V1')
        self.assertTrue(np.all(np.asarray(images[0])[0,0]==60))

    def test_rankings_are_explicit_stable_and_label_free(self):
        rows=[dict(area=10,dino=.2,evidence_score=.04),dict(area=20,dino=.1,evidence_score=.02),dict(area=10,dino=.2,evidence_score=.04)]
        self.assertEqual(ordering(rows,'balanced'),[0,1,2])
        self.assertEqual(ordering(rows,'area'),[1,0,2])
        self.assertEqual(ordering(rows,'evidence'),[0,2,1])
        rows[0]['dino']=None
        self.assertIsNone(ordering(rows,'evidence'))

    def test_exact_candidate_reuses_selected_verifier_without_request(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);gates=root/'gates';pilot=root/'pilot';views=root/'views';work=root/'work';out=root/'rank'
            for p in (gates,pilot,views,work):p.mkdir()
            group='second_appeared';stem='sample';uid=group+':'+stem+':1';shape=(8,8)
            mask=np.zeros(shape,bool);mask[2:4,2:4]=True;base=np.zeros(shape,bool)
            basepath=work/'base'/group/(stem+'.npz');basepath.parent.mkdir(parents=True);np.savez(basepath,start=base)
            image=root/'rgb.png';Image.new('RGB',(8,8),(120,120,120)).save(image)
            prior=pilot/'masks.npz';np.savez(prior,masks=np.asarray([np.packbits(mask)]),shape=shape)
            candidate=dict(uid=uid,area=4,bbox=[2,2,4,4],source='S1_02',sources=['S1_02'])
            reference=dict(group=group,stem=stem,masks=str(prior),candidates=[candidate],image_t1=str(image),image_t2=str(image),direction='appeared',scope='building_rebuild_excluded')
            (pilot/'manifest.json').write_text(json.dumps([reference]))
            arrays=gates/'arrays.npz'
            np.savez(arrays,shape=shape,baseline=np.packbits(mask),**{'source_'+s:np.packbits(mask if s in ('S1_02','S5') else base) for s in SOURCES})
            (gates/'manifest.json').write_text(json.dumps([dict(group=group,stem=stem,split='confirm',arrays=str(arrays))]))
            (gates/'score_summary.json').write_text(json.dumps(dict(selection={group:dict(gate='baseline')})))
            answer=views/'answers'/'V1'/(uid.replace(':','__')+'.json');answer.parent.mkdir(parents=True)
            answer.write_text(json.dumps(dict(uid=uid,answer='yes',cost_cny=.2)))
            a=argparse.Namespace(gates=gates,pilot=pilot,views=views,work=work,out=out,variant='V1',split='all',clear_stop=False)
            prepare(a);verify(a)
            saved=json.loads((out/'answers'/answer.name).read_text())
            self.assertEqual(saved['answer'],'yes');self.assertEqual(saved['cost_cny'],0)
            self.assertEqual(saved['original_cost_cny'],.2)
            self.assertFalse((out/'usage').exists())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertIsNone(manifest[0]['rankings']['evidence'])

if __name__=='__main__':unittest.main()
