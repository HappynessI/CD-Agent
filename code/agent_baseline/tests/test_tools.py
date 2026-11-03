import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from agent_baseline import instances
from agent_baseline.state import Mask, Instances
from agent_baseline.tools import Toolbox, SCHEMAS
from agent_baseline.fixed_routes import imc
from .helpers import mask, rows, rgb, segment, prompt, ask, tempdir, SHAPE


class InstanceTests(unittest.TestCase):
    def test_containment_direction(self):
        a = rows([mask((0,0,10,10), (40,40))])
        b = rows([mask((0,0,25,40), (40,40))])
        appeared, gone, present = instances.match(a,b,.25)
        self.assertEqual([len(appeared),len(gone),len(present)], [1,0,0])
        self.assertEqual(instances.match(b,a,.25)[1], b)

    def test_one_side_empty(self):
        a = rows([mask()])
        self.assertEqual([len(x) for x in instances.match(a,[])], [0,1,0])
        self.assertEqual([len(x) for x in instances.match([],a)], [1,0,0])
        self.assertEqual(instances.match([],[]), ([],[],[]))

    def test_fragment_maximum_single_coverage(self):
        a = rows([mask((0,0,10,10))])
        b = rows([mask((x,0,x+2,10)) for x in (0,2,4,6)])
        appeared, gone, present = instances.match(a,b,.25)
        self.assertEqual([len(appeared),len(gone),len(present)], [0,1,4])

    def test_disjoint_even_zero_threshold(self):
        self.assertEqual([len(x) for x in instances.match(rows([mask()]), rows([mask((20,20,25,25))]),0)], [1,1,0])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempdir()
        self.addCleanup(self.temp.cleanup)
        self.box = Toolbox(rgb(), self.temp.name, segment, prompt, ask, init_mask=mask() | mask((30,20,35,25)))

    def execute(self, name, **args):
        return self.box.execute(name,args).public

    def good(self, name, **args):
        result = self.execute(name,**args)
        self.assertTrue(result['ok'], result)
        return result

    def test_handle_immutability_and_ids(self):
        source = mask()
        key = self.box.session.add_mask(source)
        source[:] = False
        self.assertEqual(int(self.box.session.get(key).data.sum()),36)
        with self.assertRaises(ValueError):
            self.box.session.get(key).data[:] = False
        inst = self.good('segment_text', time='t1',text='building')['inst']
        subset = self.good('select', set=inst,ids=[1])['inst']
        a,b = self.box.session.get(inst), self.box.session.get(subset)
        self.assertEqual(a.items[0]['id'],b.items[0]['id'])
        self.assertFalse(np.shares_memory(a.items[0]['idx'],b.items[0]['idx']))
        numbers = [int(k.split('#')[1]) for k in self.box.session.handles]
        self.assertEqual(numbers, list(range(len(numbers))))

    def test_current_defaults_candidates_and_old_submit(self):
        p = self.good('segment_prompt',time='t1',box=[0,0,12,12])
        self.assertEqual(self.box.session.current_mask,'mask#0')
        mid = self.good('combine',op='union',a='mask#0',b=p['mask'],set_current=False)
        self.assertEqual(self.box.session.current_mask,'mask#0')
        answer = self.good('remove_region',mask='mask#0',ids=[1])
        self.assertEqual(self.box.session.current_mask,answer['mask'])
        self.good('submit',mask='mask#0',summary='Keep the initial answer.')
        self.assertEqual(self.box.session.submitted,'mask#0')

    def test_remove_region_keeps_ids_across_batches(self):
        a = self.good('segment_prompt',time='t1',box=[0,0,4,4])['mask']
        b = self.good('segment_prompt',time='t1',box=[10,0,14,4])['mask']
        c = self.good('segment_prompt',time='t1',box=[20,0,24,4])['mask']
        ab = self.good('combine',op='union',a=a,b=b)['mask']
        abc = self.box.session.get(self.good('combine',op='union',a=ab,b=c)['mask'])
        key = [k for k,v in self.box.session.handles.items() if v is abc][0]
        last = abc.components[-1]
        first = self.good('remove_region',mask=key,ids=[abc.components[0]['id']])['mask']
        kept = self.box.session.get(first)
        self.assertEqual([x['id'] for x in kept.components],[x['id'] for x in abc.components[1:]])
        second = self.good('remove_region',mask=first,ids=[abc.components[1]['id']])['mask']
        final = self.box.session.get(second)
        self.assertEqual([x['id'] for x in final.components],[last['id']])
        self.assertTrue((final.data==(abc.labels==last['id'])).all())
        self.assertTrue((final.labels[final.data]==last['id']).all())
        self.good('view',overlay=second,label_components=True)
        self.assertEqual(self.good('inspect',mask=second)['total_components'],1)

    def test_same_reply_handles(self):
        allowed = frozenset(self.box.session.handles)
        new = self.box.execute('segment_prompt',dict(time='t1',box=[0,0,10,10]),allowed).public['mask']
        result = self.box.execute('inspect',dict(mask=new),allowed).public
        self.assertEqual(result['error_code'],'bad_handle')
        self.assertIn('next reply',result['message'])
        self.good('inspect',mask=new)

    def test_ids_and_pagination(self):
        result = self.good('inspect',mask='mask#0',limit=1)
        self.assertEqual((result['total_components'],result['returned'],result['has_more']), (2,1,True))
        self.assertEqual(set(result['components'][0]), {'id','area','bbox'})
        self.assertEqual(self.good('inspect',mask='mask#0',offset=1)['components'][0]['id'],2)
        for name,args in [('remove_region',dict(mask='mask#0',ids=[9])),
                          ('add_region',dict(mask='mask#0',other='mask#0',ids=[9])),
                          ('refine_region',dict(mask='mask#0',id=9,time='t1'))]:
            self.assertEqual(self.execute(name,**args)['error_code'],'bad_id')
        self.assertEqual(self.box.session.current_mask,'mask#0')

    def test_bad_args_coordinates(self):
        for args in [dict(time='t3',box=[0,0,8,8]), dict(time='t1',box=[0,0,100,100]),
                     dict(time='t1',points=[[48,0]],point_labels=[1]), dict(time='t1',points=[[0,0]]),
                     dict(time='t1',box=[0,0,8,8],points=[[0,0]],point_labels=[1])]:
            self.assertEqual(self.execute('segment_prompt',**args)['error_code'],'bad_args')
        self.assertEqual(self.execute('inspect',mask='mask#0',limit=51)['error_code'],'bad_args')

    def test_select_expressions_and_missing_score(self):
        key = self.box.session.add_instances(rows([mask(),mask((20,20,25,25))],[.3,.9]),'t1')
        result = self.good('select',set=key,where='(score > 0.5 and area >= 20) or score < -1',area_max=30)
        self.assertEqual(result['ids'],[2])
        for expr in ('score.real > 0','__import__("os")','area + 1 > 2'):
            self.assertEqual(self.execute('select',set=key,where=expr)['error_code'],'bad_args')
        raw = dict(rows([mask()])[0]); raw.pop('score')
        no_score = self.box.session.add_instances([raw],'t1')
        self.assertEqual(self.execute('select',set=no_score,where='score > 0')['error_code'],'bad_args')
        self.assertEqual(self.execute('select',set=key,ids=[99])['error_code'],'bad_id')

    def test_refine_only_target_and_roi(self):
        data = mask((2,2,14,14))
        data[5:11,5:11] = False
        data[7:9,7:9] = True  # second component lies inside first component's box
        self.box = Toolbox(rgb(),self.temp.name,prompt=lambda *a,**kw:(np.ones(SHAPE,bool),.9,{}),init_mask=data)
        result = self.good('refine_region',mask='mask#0',id=1,time='t2',box=[0,0,4,4])
        out = self.box.session.get(result['mask']).data
        self.assertTrue(out[7:9,7:9].all())
        self.assertEqual(int(out.sum()),20)
        self.assertEqual(result['roi'],[0,0,4,4])
        self.assertEqual(result['sam_quality'],.9)

    def test_morph_and_mask_algebra(self):
        data = mask((2,2,8,8)); data[4,4]=False
        key = self.box.session.add_mask(data)
        filled = self.good('morph',mask=key,op='fill_holes')
        self.assertEqual(filled['added_px'],1)
        self.assertEqual(self.execute('morph',mask=key,op='open')['error_code'],'bad_args')
        self.assertEqual(self.execute('morph',mask=key,op='close',radius=0)['error_code'],'bad_args')
        diff = self.good('combine',op='subtract',a=filled['mask'],b=key)
        self.assertEqual(diff['area'],1)

    def test_empty_set_success_and_multiple_sets(self):
        self.box.segment_backend = lambda *a:([],{})
        key = self.good('segment_text',time='t1',text=' building ')['inst']
        self.assertEqual(self.good('to_mask',sets=[key])['area'],0)
        a = self.box.session.add_instances(rows([mask()]),'t1')
        b = self.box.session.add_instances(rows([mask((20,20,25,25))]),'t2')
        self.assertEqual(self.good('to_mask',sets=[a,b])['area'],61)

    def test_ask_limit_and_unparsed(self):
        self.box.ask_backend=lambda pairs,q:([dict(answer=None,p_yes=.4,type=None,reason=None) for p in pairs],{})
        result = self.good('ask',regions=[dict(mask='mask#0',ids=[1])])['results'][0]
        self.assertEqual((result['answer'],result['status']),(None,'unparsed'))
        self.assertEqual(self.execute('ask',regions=[dict(mask='mask#0',ids=[1,2])]*33)['error_code'],'bad_args')

    def test_backend_error_is_private(self):
        def broken(*args):
            raise RuntimeError('private source path')
        self.box.segment_backend = broken
        result = self.box.execute('segment_text',dict(time='t1',text='building'))
        self.assertEqual(result.public['error_code'],'backend_error')
        self.assertNotIn('private source',result.text)
        self.assertIn('private source',result.internal['traceback'])
        self.box.segment_backend=None
        self.assertEqual(self.execute('segment_text',time='t1',text='building')['error_code'],'backend_unavailable')

    def test_view_ids_geometry_jpeg(self):
        result = self.box.execute('view',dict(region=[28,18,40,30],overlay='mask#0',label_components=True,ids=[2]))
        self.assertTrue(result.public['ok'])
        self.assertEqual(result.public['labelled_ids'],[2])
        self.assertEqual(result.public['display_size'],[768,512])
        with Image.open(result.images[0]) as im:
            self.assertEqual(im.format,'JPEG')
            self.assertEqual(im.size,(796,540))
        self.assertEqual(self.execute('view',overlay='mask#0',ids=[99])['error_code'],'bad_id')

    def test_fixed_route_consistency(self):
        fixed = imc(self.box)
        other = Toolbox(rgb(),self.temp.name,segment=segment)
        a = other.execute('segment_text',dict(time='t1',text='building')).public['inst']
        b = other.execute('segment_text',dict(time='t2',text='building')).public['inst']
        matched = other.execute('match_instances',dict(inst_t1=a,inst_t2=b)).public
        result = other.execute('to_mask',dict(sets=[matched['appeared'],matched['disappeared']])).public
        np.testing.assert_array_equal(fixed,other.session.get(result['mask']).data)

    def test_union_basis_drops_scores(self):
        a=self.box.session.add_instances(rows([mask()]),'t1')
        b=self.box.session.add_instances(rows([mask()]),'t2')
        with patch('agent_baseline.tools.MATCH_BASIS','union'):
            result=self.good('match_instances',inst_t1=a,inst_t2=b)
        self.assertNotIn('score',self.box.session.get(result['present_t2']).items[0])


if __name__ == '__main__':
    unittest.main()
