"""Synthetic selection/rendering checks. No model or network use."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from agent_baseline.pick_demo import select_typical
from agent_baseline.make_demo import render, read_json, CATEGORIES
from agent_baseline.make_g3_pages import audit_html


class TypicalTest(unittest.TestCase):
    def record(self, name, f1, submitted='model', categories=1, run='s1'):
        return dict(image=name, f1=f1, submitted_by=submitted, category_count=categories, run=run)

    def test_odd_and_even_median(self):
        result = select_typical([self.record('a',.2),self.record('b',.7),self.record('c',.9)])
        self.assertEqual(result['image'],'b'); self.assertEqual(result['median'],.7)
        result = select_typical([self.record('z',.3),self.record('a',.8)])
        self.assertAlmostEqual(result['median'],.55)
        self.assertEqual(result['image'],'a')

    def test_fallback_exclusion_and_all_fallback(self):
        data = [self.record('a',0),self.record('b',.5,'budget'),self.record('c',1)]
        result = select_typical(data)
        self.assertEqual(result['median'],.5); self.assertEqual(result['image'],'a')
        self.assertEqual(result['eligible'],2)
        for r in data: r['submitted_by']='error'
        self.assertEqual(select_typical(data)['image'],'b')
        self.assertTrue(select_typical(data)['all_fallback'])

    def test_ties_categories_image_then_run(self):
        data=[self.record('a',.5,categories=1),self.record('z',.5,categories=3),self.record('b',.5,categories=3,run='s2'),self.record('b',.5,categories=3,run='s1')]
        result=select_typical(data)
        self.assertEqual((result['image'],result['run']),('b','s1'))

    def test_missing_f1_is_not_zero(self):
        self.assertEqual(select_typical([self.record('a',None),self.record('b',.9)])['median'],.9)
        with self.assertRaises(ValueError): select_typical([])

    def test_shared_storage_retry(self):
        with patch('agent_baseline.make_demo.Path.read_text',side_effect=[OSError('EIO'),OSError('EIO'),'{}']) as read, patch('agent_baseline.make_demo.time.sleep'):
            self.assertEqual(read_json('input.json'),{})
            self.assertEqual(read.call_count,3)


class RenderTest(unittest.TestCase):
    def fixture(self, root, task='T3.1'):
        from agent_baseline.make_tools_page import registered_schemas
        (root/'masks').mkdir(); (root/'insts').mkdir()
        image = np.full((32,48,3),90,np.uint8)
        mask = np.zeros((32,48),np.uint8);mask[4:20,5:22]=255
        for f in ('t1.png','t2.png'): Image.fromarray(image).save(root/f)
        for f in ('gt.png','masks/mask_0.png','masks/mask_1.png','masks/mask_2.png','masks/region_0.png'): Image.fromarray(mask).save(root/f)
        packed=np.packbits(mask.reshape(1,-1)>0,axis=1)
        np.savez(root/'insts/inst_3.npz',masks=packed,shape=[32,48])
        info=dict(task_id=task,source_images=['t1.png','t2.png'],gt_mask_path='gt.png',
                  schemas=registered_schemas(),backends=dict(gate3=True,gate4=True),
                  instruction='Find changes. /server/private/example/image.png',initial_view=dict(overlay='mask#0'))
        final=dict(task_id=task,final_handle='mask#2',eval=dict(f1=.75),submitted_by='model',
                   pre_region_handle='mask#1',pre_region_mask_path='masks/mask_1.png',region_clipped_px=0)
        calls=[]
        for tool,handle in [('segment_everything','props#3'),('extract_features','feat#4'),('compare_features','props#3'),('classify','props#3'),('segment_semantic','mask#1'),('match_exemplar','props#3')]:
            calls.append(dict(tool=tool,args=dict(time='t2'),ok=True,result_text=json.dumps(dict(handle=handle)),handles_created=[handle]))
        rows=[dict(step=0,calls=[],current_mask=None),dict(step=1,calls=calls,current_mask='mask#1',eval=dict(f1=.75)),dict(step=2,note='report',report='Your current answer is mask#1. Your most recent mask is mask#2.',calls=[])]
        (root/'input.json').write_text(json.dumps(info));(root/'final.json').write_text(json.dumps(final))
        (root/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        return info

    def test_region_example_new_tools_and_g4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            document=render(root,'val',selection=dict(median=.7,run='synthetic_s1'))
            audit_html(document)
            for word in ['region#0 框叠在 T2','示例掩码 mask#0','G4 当前答案','G4 提交时区域求交','按中位数选出的典型会话','返回候选集合并集 props#3']:
                self.assertIn(word,document)
            self.assertIn('class="inputs panel sticky"',document)
            self.assertIn('class="results sticky panel"',document)
            self.assertIn('.inputs.sticky,.results.sticky{position:sticky',document)
            self.assertIn('grid-template-rows:auto auto',document)
            self.assertNotIn('overflow:hidden',document)
            self.assertNotIn('/server/private',document)
            for tool in ('segment_everything','extract_features','compare_features','classify','segment_semantic','match_exemplar'):
                self.assertIn(tool,CATEGORIES)

    def test_point_and_language_input(self):
        for task,feedback in [('T5.2',[dict(type='false_positive',point=[12,10])]),('T5.3',[dict(type='missed',where='upper left')])]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); info=self.fixture(root,task)
                info.update(initial_handle='mask#0',task_meta=dict(feedback=feedback))
                (root/'input.json').write_text(json.dumps(info))
                document=render(root,'val',selection=dict(median=.7,run='s1'))
                audit_html(document)
                self.assertIn('给定掩码 mask#0',document)
                self.assertIn('The user clicked' if task=='T5.2' else 'upper left',document)

    def test_audit_rejects_external_resources(self):
        with self.assertRaises(ValueError): audit_html('<html><img src="https://example.org/a.jpg"></html>')


class BaselineTest(unittest.TestCase):
    def data(self, root, key='levir_c0'):
        from agent_baseline.make_baselines_page import SPECS, load_baselines
        spec=SPECS[key]
        fixed=np.zeros((32,48),bool)
        fixed[4:12,5:14]=True;fixed[18:26,30:40]=True
        path=root/spec['fixed'];path.mkdir(parents=True)
        Image.fromarray(fixed.astype('uint8')*255).save(path/'sample.png')
        if spec.get('answers'):
            path=root/spec['answers'];path.parent.mkdir(parents=True)
            path.write_text('\n'.join(json.dumps(dict(uid=f'sample:{i}',answer=a,reason=r))
                for i,a,r in [(1,'yes','changed'),(2,'no','unchanged')]))
        record=dict(page=key,image='sample')
        return load_baselines(root,record),fixed

    def test_rule_deletion_outline_and_missing_artifacts(self):
        from agent_baseline.make_baselines_page import load_baselines, render_comparison, scored_image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,fixed=self.data(root)
            rule=fixed.copy();rule[18:26,30:40]=False
            np.testing.assert_array_equal(data['rule'],rule)
            self.assertEqual([r['answer'] for r in data['answers']],['yes','no'])
            t2=Image.new('RGB',(48,32),'gray')
            plain=np.array(scored_image(rule,fixed,t2))
            marked=np.array(scored_image(rule,fixed,t2,fixed&~rule))
            self.assertTrue(np.any(plain!=marked))
            self.assertTrue(np.any(np.all(marked==[255,218,65],axis=2)))
            document=render_comparison(data,fixed,fixed,t2)
            audit_html(document)
            for word in ('黄色虚线轮廓','组件答案表','unchanged','F1 100.00%'):
                self.assertIn(word,document)
            missing=load_baselines(root,dict(page='second_c1',image='missing'))
            document=render_comparison(missing,fixed,fixed,t2)
            audit_html(document);self.assertIn('产物未找到',document)

    def test_no_rule_response_is_not_missing_answer_ledger(self):
        from agent_baseline.make_baselines_page import render_comparison
        with tempfile.TemporaryDirectory() as tmp:
            data,fixed=self.data(Path(tmp),'levir_t52')
            document=render_comparison(data,fixed,fixed,Image.new('RGB',(48,32)))
            audit_html(document)
            self.assertIn('固定响应 C',document)
            self.assertIn('本任务无规则对照',document)
            self.assertNotIn('组件答案：产物未找到',document)

    def test_comparison_precedes_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session';session.mkdir()
            RenderTest().fixture(session,'T1.1xC0')
            data,_=self.data(root)
            document=render(session,'val',selection=dict(median=.7,run='s1'),baselines=data)
            audit_html(document)
            self.assertLess(document.index('class="panel system"'),document.index('class="panel baseline-panel"'))
            self.assertLess(document.index('class="panel baseline-panel"'),document.index('<main class="layout">'))

    def test_baselines_page_from_cached_artifacts(self):
        from agent_baseline.make_baselines_page import render_baselines, cached_levir_steps, select_example
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,_=self.data(root)
            record=dict(page='levir_c0',image='sample',task='T1.1xC0',run='synthetic_s1',
                        f1=.75,median=.7,file='levir_c0.html')
            session=root/record['run']/record['task']/'val'/record['image'];session.mkdir(parents=True)
            RenderTest().fixture(session,'T1.1xC0')
            final=json.loads((session/'final.json').read_text());final['mask_path']='masks/mask_2.png'
            (session/'final.json').write_text(json.dumps(final))
            example=select_example(root,[record])
            self.assertEqual(example['image'],'sample')
            document=render_baselines(root,[record],example)
            audit_html(document)
            for word in ('<svg','固定路线','规则','Agent','levir_c0.html','未缓存，只显示最终掩码','T5.4','T3.4'):
                self.assertIn(word,document)
            # These rendering functions have no backend dependency: absence of
            # all intermediate caches must remain a visible gap.
            self.assertEqual(cached_levir_steps(root,'missing').count('未缓存，只显示最终掩码'),4)


if __name__=='__main__': unittest.main()
