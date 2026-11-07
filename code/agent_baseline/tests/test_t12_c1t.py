"""Tiny fake-backend contracts; no model, API, Slurm or real dataset processing."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import save_mask_png, write_json
from agent_baseline.state import Session
from agent_baseline.tasks import read_tasks, instruction_for, needs_init
from agent_baseline.scope_t13 import ASK_CANDIDATES, session_card, session_prompts
from agent_baseline.analysis.make_t12_tasks import generate, task_row, t12_truth, SOURCE_CLASSES
from agent_baseline.analysis.make_c1t_tasks import conversion_truth
from agent_baseline.analysis.t12_fixed import route, main as fixed_main
from agent_baseline.analysis.c1t_fixed import intersect, source_mask
from agent_baseline.analysis import t13_ask_select as ask
from agent_baseline.analysis.second_labels import ENCODING
from agent_baseline.analysis.second_metrics import paired_bootstrap, direction_violation


def snapshot_requests(directory):
    """Runs under either checkout, then returns actual encoded planner/verifier payloads."""
    import copy
    import json
    from pathlib import Path
    import numpy as np
    from agent_baseline.state import Session
    from agent_baseline.backends import ApiAsk
    from agent_baseline.planner import Planner, clean_messages
    from agent_baseline.tools import Toolbox
    from agent_baseline.recorder import Recorder
    from agent_baseline.loop_json import run_session
    captured = []
    for gate3 in (False, True):
        mask = np.zeros((8,8), bool)
        mask[:2,:2] = True
        rgb = [np.zeros((8,8,3), np.uint8)]*2
        session = Session(mask.shape, mask)
        def verifier(payload):
            captured.append(copy.deepcopy(payload))
            return dict(choices=[dict(message=dict(role='assistant', content='{"change":"yes","type":"new_building","reason":"New."}'))], usage={})
        box = Toolbox(rgb, Path(directory), session=session)
        box.ask_backend = ApiAsk(session=session, client=verifier, correspondence=True)
        def planner(payload):
            captured.append(copy.deepcopy(payload))
            call = dict(id='s', type='function', function=dict(name='submit',arguments=json.dumps(dict(mask='mask#0',summary='Keep.'))))
            return dict(choices=[dict(message=dict(role='assistant', tool_calls=[call]))], usage={})
        recorder = Recorder(Path(directory), 'fake', 'x.png', gt=mask)
        final = run_session(box, Planner(client=planner, seed=7), recorder, 'T5.1',
            budget=3, prompt_version='v4', verify=True, gate2=True, gate3=gate3)
        if final['submitted_by'] != 'model':
            raise AssertionError(final)
    for payload in captured:
        payload['messages'] = clean_messages(payload['messages'], encode=True)
    return captured


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        self.dataset.mkdir()
        write_json(self.dataset/'cls_gray.json', ENCODING)
        self.names = dict(dev=['a.png', 'empty.png'], val=['b.png'])
        a = np.array([[0,1,2,5,5,3,1,4]], np.uint8)
        b = np.array([[0,5,5,1,5,5,2,6]], np.uint8)
        for name in ('a.png','b.png','empty.png'):
            for part, array in [('slabel1',a),('slabel2',a if name == 'empty.png' else b),
                                ('img1',np.zeros((1,8,3),np.uint8)),('img2',np.zeros((1,8,3),np.uint8))]:
                path = self.dataset/part/name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(array).save(path)

    def test_truth_encoding_unlabelled_replacement_and_reverse(self):
        a = np.array([[0,1,2,5,5,3,1,4]])
        b = np.array([[0,5,5,1,5,5,2,6]])
        np.testing.assert_array_equal(t12_truth(a,b), [[0,1,1,1,0,1,1,0]])
        np.testing.assert_array_equal(conversion_truth(a,b)['new'], [[0,1,1,0,0,0,0,0]])
        np.testing.assert_array_equal(conversion_truth(a,b)['demolished'], [[0,0,0,1,0,0,0,0]])
        np.testing.assert_array_equal(conversion_truth(b,a)['new'], conversion_truth(a,b)['demolished'])
        for fn in (t12_truth, conversion_truth):
            with self.assertRaises(ValueError):
                fn(np.array([[7]]),np.array([[5]]))
            with self.assertRaises(ValueError):
                fn(np.array([[0]]),np.array([[5]]))

    def test_manifests_deterministic_empty_images_and_attributes(self):
        with patch('agent_baseline.analysis.make_t12_tasks.inventories', return_value=self.names):
            for kind, task in [('t12','T1.2xC0_any'),('c1t','T1.1xC1t_appeared')]:
                report = generate(self.dataset,self.root,kind)
                path = self.root/kind/'tasks'/f'{task}_dev.jsonl'
                first = path.read_bytes()
                generate(self.dataset,self.root,kind)
                self.assertEqual(first,path.read_bytes())
                rows = read_tasks(path)
                self.assertEqual([r['id'] for r in rows],self.names['dev'])
                self.assertEqual(report['splits']['dev']['nonempty_truth'],1)
                for row in rows:
                    session = Session((1,8), target=row['target'], change_direction=row['change_direction'],
                        scope=row.get('scope'),source_classes=row.get('source_classes'))
                    text = instruction_for(row['task_type'],session)
                    self.assertEqual(text,row['instruction'])
                    self.assertNotIn('SECOND',text)
                    self.assertFalse(needs_init(row['task_type']))
                    if kind == 't12':
                        self.assertEqual(row['target'],('building','low vegetation'))
                        self.assertIn('buildings or low vegetation',text)
                    else:
                        self.assertEqual(session.source_classes,tuple(SOURCE_CLASSES))
                        self.assertIn('T1',json.dumps(session_card(session)))
                        self.assertIn('non-vegetated ground surface',session_prompts(session)['SYSTEM'])

    def test_source_requirement_and_multiclass_api_prompt(self):
        from agent_baseline.ask_eval import api_answer, VARIANTS
        from agent_baseline.planner import Planner
        with self.assertRaises(ValueError):
            instruction_for('T1.1xC1t',Session((1,1)))
        requests=[]
        def fake(this,messages,tools,number):
            requests.append(messages)
            return dict(message=dict(content='{"change":"yes"}'), usage={},raw=dict(choices=[dict(logprobs=None)]))
        session=Session((1,1),targets=['building','low vegetation'])
        with patch.object(Planner,'complete',fake):
            api_answer([],VARIANTS['f37_corr'],None,1,session)
        self.assertIn('building, low vegetation',requests[0][0]['content'])
        self.assertNotIn('SECOND',json.dumps(requests))

    def test_default_g2_g3_requests_match_code_q_bytes(self):
        reference = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_q')
        if not (reference/'agent_baseline/loop_json.py').is_file():
            self.skipTest('Read-only code_q reference unavailable')
        source = inspect.getsource(snapshot_requests)+'\nimport json\nprint(json.dumps(snapshot_requests('+repr(str(self.root/'wire'))+')))\n'
        env = dict(os.environ, PYTHONPATH=str(reference), OVCD_CODE=str(reference/'ovcd_v0'))
        result = subprocess.run([sys.executable,'-B','-c',source],env=env,text=True,capture_output=True,check=True)
        baseline = json.loads(result.stdout)
        current = snapshot_requests(self.root/'wire')
        self.assertEqual(json.dumps(current).encode(),json.dumps(baseline).encode())

    def vision(self, operation, time, classes):
        self.assertEqual(operation,'classify')
        masks = {('building','t1'):[1,1,0,0], ('building','t2'):[1,0,1,0],
            ('low vegetation','t1'):[0,0,0,1], ('low vegetation','t2'):[0,1,0,0],
            ('non-vegetated ground','t1'):[1,0,0,0]}
        m = np.array([masks[(classes[0],time)]], bool)
        return np.stack([~m,m]).astype(float)

    def test_t12_xor_is_symmetric_and_sam3_building_option(self):
        np.testing.assert_array_equal(route(self.vision),[[0,1,1,1]])
        def reverse(operation,time,classes):
            return self.vision(operation,'t2' if time == 't1' else 't1',classes)
        np.testing.assert_array_equal(route(self.vision),route(reverse))
        np.testing.assert_array_equal(route(self.vision,sam3_building=np.array([[1,0,0,0]],bool)),[[1,1,0,1]])
        self.assertFalse(route(self.vision,min_area=4).any())

    def test_c1_intersection_pixel_component_and_source_union(self):
        np.testing.assert_array_equal(source_mask(self.vision),[[1,0,0,1]])
        directed=np.array([[1,1,0,1]],bool)
        source=np.array([[1,0,0,0]],bool)
        np.testing.assert_array_equal(intersect(directed,source),[[1,0,0,0]])
        np.testing.assert_array_equal(intersect(directed,source,'component',.5),[[1,1,0,0]])
        np.testing.assert_array_equal(intersect(directed,source,'none'),directed)
        self.assertFalse(intersect(directed,source,'component',.6).any())
        self.assertAlmostEqual(direction_violation(directed,source),1/3)

    def test_fixed_entrypoints_freeze_resume_and_keep_reference(self):
        from agent_baseline.analysis import t12_fixed
        config=self.root/'config.json'
        config.write_text('{}')
        for name in ('a.png','empty.png'):
            save_mask_png(self.root/'sam3'/name,np.ones((1,8),bool))
        class Pool:
            def __init__(self,*args): pass
            def for_pair(this,*args):
                return lambda operation,**kwargs: np.stack([np.zeros((1,8)),np.ones((1,8))])
            def close(this): pass
        for kind in ('t12','c1t'):
            argv=['fixed','--vision-config',str(config),'--cache',str(self.root/'cache'),
                '--output-root',str(self.root),'--run-root',str(self.root),'--dataset-root',str(self.dataset),
                '--min-areas','0','--sam3-mask-dir',str(self.root/'sam3')]
            with patch('sys.argv',argv),patch.object(t12_fixed,'selected',return_value=self.names['dev']), \
                 patch.object(t12_fixed,'inventories',return_value=self.names),patch.object(t12_fixed,'WorkerPool',Pool):
                fixed_main(kind)
                with patch.object(Pool,'for_pair',side_effect=AssertionError('Resume must not infer')):
                    fixed_main(kind)
            report=json.loads((self.root/kind/'fixed/dev/report_0of1.json').read_text())
            self.assertTrue(report['complete'])
            self.assertIn('selected',report)
            if kind == 'c1t':
                self.assertIn('unintersected',report['scores'])
                self.assertIn('before_tp',report['diagnostics']['pixel_min0'])

    def test_component_labels_auc_ties_missing_probability_and_full_image_f1(self):
        pred=np.array([[1,1,0,1,0,1]],bool)
        truth=np.array([[1,0,0,0,0,1]],bool)
        answers={'a:1':dict(answer='yes',p_yes=.8),'a:2':dict(answer='no',p_yes=.2),'a:3':dict(answer='yes',p_yes=.8)}
        images=[('a.png',pred,truth),('empty.png',np.zeros_like(pred),np.array([[0,0,1,0,0,0]],bool))]
        report=ask.evaluate(images,answers,[.5])
        self.assertEqual(report['auc'],1.)
        self.assertEqual(report['component_positive_scored'],2)
        self.assertEqual(report['delete_no']['f1_c'],200*2/6)
        self.assertEqual(report['thresholds']['0.5'],report['delete_no'])
        self.assertEqual(ask.auc([.5],[.5]),.5)
        answers['a:2']['p_yes']=None
        self.assertIsNone(ask.evaluate(images,answers)['auc'])
        with self.assertRaises(KeyError):
            ask.evaluate(images,{})
        base=np.array(report['per_image_delete_no'])
        self.assertEqual(paired_bootstrap(np.array([base]*3),base,samples=5),[0.,0.])

    def test_fixed_shards_merge_before_val_and_reject_partial_selection(self):
        from agent_baseline.analysis import t12_fixed
        config=self.root/'vision.json'
        config.write_text('{}')
        class Pool:
            def __init__(this,*args): pass
            def for_pair(this,*args):
                return lambda operation,**kwargs: np.stack([np.ones((1,8)),np.zeros((1,8))])
            def close(this): pass
        base=['fixed','--run-root',str(self.root),'--dataset-root',str(self.dataset),
              '--vision-config',str(config),'--cache',str(self.root/'cache'),
              '--output-root',str(self.root),'--min-areas','0']
        def selected(args):
            return self.names[args.phase][args.shard::args.shards]
        with patch.object(t12_fixed,'selected',side_effect=selected), \
             patch.object(t12_fixed,'inventories',return_value=self.names),patch.object(t12_fixed,'WorkerPool',Pool):
            for shard in (0,1):
                with patch('sys.argv',base+['--shard',str(shard),'--shards','2']):
                    fixed_main()
            shards=[self.root/'t12/fixed/dev'/f'report_{n}of2.json' for n in (0,1)]
            self.assertNotIn('selected',json.loads(shards[0].read_text()))
            with patch('sys.argv',base+['--phase','val','--settings',str(shards[0])]),self.assertRaises(SystemExit):
                fixed_main()
            with patch('sys.argv',base+['--merge',*(str(s) for s in shards)]), \
                 patch.object(Pool,'for_pair',side_effect=AssertionError('Merge must not infer')):
                fixed_main()
            merged=self.root/'t12/fixed/dev/report_0of1_merged.json'
            self.assertTrue(json.loads(merged.read_text())['complete'])
            with patch('sys.argv',base+['--phase','val','--settings',str(merged)]):
                fixed_main()
            val=json.loads((self.root/'t12/fixed/val/report_0of1.json').read_text())
            self.assertEqual(len(val['scores']),1)

    def test_question_selection_uses_delete_no_f1(self):
        reports={q:dict(delete_no=dict(f1_c=i),thresholds={'0.2':dict(f1_c=2),'0.5':dict(f1_c=2)})
                 for i,q in enumerate(ASK_CANDIDATES)}
        self.assertEqual(ask.select(reports)['question'],'describe')
        self.assertEqual(ask.select(reports)['p_threshold'],.2)

    def test_t13_resume_fake_verifier_and_frozen_val_single_question(self):
        candidates=self.root/'candidates'
        provenance=dict(phase='dev',names=self.names['dev'],threshold=.22,backend_config={},dataset=str(self.dataset.resolve()))
        write_json(candidates/'complete.json',provenance)
        for name in self.names['dev']:
            save_mask_png(candidates/'masks'/name,np.array([[0,1,1,0,0,1,0,0]],bool))
        calls=[]
        def answer(images,spec,guard,seed,session):
            calls.append(session.ask_variant)
            self.assertEqual(spec,ask.VARIANTS['f37_corr'])
            self.assertNotIn('SECOND',json.dumps(session_prompts(session)))
            guard.charge(.001)
            return dict(answer='yes',p_yes=.8,cost_cny=.001)
        with patch.object(ask,'inventories',return_value=self.names):
            result=ask.answer_questions(self.dataset,self.root,'dev',candidates,self.root/'answers',1,answer_fn=answer)
            before=len(calls)
            ask.answer_questions(self.dataset,self.root,'dev',candidates,self.root/'answers',1,answer_fn=answer)
            self.assertEqual(len(calls),before)
            provenance.update(phase='val',names=self.names['val'])
            val=self.root/'val_candidates'
            write_json(val/'complete.json',provenance)
            save_mask_png(val/'masks/b.png',np.array([[0,1,0,0,0,0,0,0]],bool))
            report=ask.answer_questions(self.dataset,self.root,'val',val,self.root/'val_answers',1,
                settings=self.root/'answers/selection.json',answer_fn=answer)
            self.assertEqual(list(report['reports']),[result['selected']['question']])
            self.assertEqual(len(next(iter(report['reports'].values()))['thresholds']),1)
            altered=copy.deepcopy(result)
            altered['selected']['p_threshold']=.99
            write_json(self.root/'altered_selection.json',altered)
            with self.assertRaises(ValueError):
                ask.answer_questions(self.dataset,self.root,'val',val,self.root/'val_answers',1,
                    settings=self.root/'altered_selection.json',answer_fn=answer)

    def test_cache_reader_and_prepare_without_models(self):
        cache=self.root/'cache'
        (cache/'dynamic').mkdir(parents=True)
        paths=[self.dataset/part/'a.png' for part in ('img1','img2')]
        records=[]
        for time in ('t1','t2'):
            for operation in ('segment_everything','extract_features'):
                records.append(dict(version=1,sources=[dict(path=str(p.resolve()),size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in paths],
                    config={},operation=operation,time=time,classes=None))
                target=cache/'dynamic'/f'item_{len(records)-1:06d}.npz'
                if operation == 'segment_everything':
                    np.savez(target,data=np.packbits(np.ones((1,8),bool),axis=1),shape=[1,8])
                else:
                    np.savez(target,data=np.array([[[1.,0.]]] if time == 't1' else [[[0.,1.]]]))
        write_json(cache/'dynamic/inventory.json',records)
        backend=ask.cached_pair(cache,{'dynamic':{}},paths)
        self.assertEqual(backend('segment_everything',time='t1').shape,(1,1,8))
        with self.assertRaises(FileNotFoundError):
            backend('classify',time='t1',classes=['building'])
        probe=self.root/'probe.json'
        write_json(probe,dict(phase='dev',complete=True,names=['a.png'],backend_config={'dynamic':{}},thresholds=[.22],selected_threshold=.22))
        with patch.object(ask,'inventories',return_value=dict(dev=['a.png'])):
            ask.prepare(self.dataset,self.root,'dev',cache,{'dynamic':{}},probe,self.root/'prepared')
        self.assertTrue((self.root/'prepared/complete.json').is_file())
        self.assertTrue(np.asarray(Image.open(self.root/'prepared/masks/a.png')).all())

    def test_api_budget_stops_and_preserves_completed_answers(self):
        from agent_baseline.planner import RunStopped
        provenance=dict(phase='dev',names=['a.png'],threshold=.22,backend_config={},dataset=str(self.dataset.resolve()))
        candidates=self.root/'budget_candidates'
        write_json(candidates/'complete.json',provenance)
        save_mask_png(candidates/'masks/a.png',np.array([[0,1,0,1,0,0,0,0]],bool))
        calls=[]
        def answer(images,spec,guard,seed,session):
            calls.append(session.ask_variant)
            guard.charge(.01)
            return dict(answer='no',p_yes=.1,cost_cny=.01)
        with patch.object(ask,'inventories',return_value=dict(dev=['a.png'])):
            with self.assertRaises(RunStopped):
                ask.answer_questions(self.dataset,self.root,'dev',candidates,self.root/'budget_answers',.015,answer_fn=answer)
            self.assertEqual(calls,['category','category'])
            ledger=self.root/'budget_answers/category/answers.jsonl'
            self.assertEqual(len(ledger.read_text().splitlines()),1)
            self.assertFalse((self.root/'budget_answers/selection.json').exists())
            with self.assertRaises(RunStopped):
                ask.answer_questions(self.dataset,self.root,'dev',candidates,self.root/'budget_answers',.015,answer_fn=answer)
            self.assertEqual(len(calls),2)


if __name__ == '__main__':
    unittest.main()
