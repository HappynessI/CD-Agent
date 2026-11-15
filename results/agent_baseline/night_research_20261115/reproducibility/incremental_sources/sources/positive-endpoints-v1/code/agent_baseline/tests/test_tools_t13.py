"""Tiny fake-backend contracts. No models, network, real datasets or GPU jobs."""
import copy
import itertools
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from agent_baseline import instances
from agent_baseline.tools import Toolbox, SCHEMAS, session_schemas
from agent_baseline.tools_t13 import Features, NAMES, cosine, pool_features
from agent_baseline.scope_t13 import scope_card, verifier_prompts
from agent_baseline.state import Session, Mask, Instances
from agent_baseline.backends import ApiAsk
from agent_baseline.loop_json import run_session, system_prompt, referenced_handles
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.tasks import instruction_for, read_tasks
from agent_baseline.analysis.replay_g1 import compare, ReplayMismatch
from agent_baseline.analysis.make_t3_tasks import make_tasks, generate
from agent_baseline.analysis.probe_t13_common import (coverage_upper, selection_upper, annotated_counts,
    binary_truth, metrics, collect, inventories)
from agent_baseline.analysis.probe_t3 import retrieval, task_scores

SHAPE = (4,4)


class FakeVision:
    def __init__(self):
        self.calls = []
        self.masks = np.zeros((2,*SHAPE),bool)
        self.masks[0,:2,:2] = True
        self.masks[1,2:,2:] = True
        self.a = np.ones((*SHAPE,2),np.float32)
        self.a[...,1] = 0
        self.b = self.a.copy()
        self.b[:2,:2] = [0,1]
        self.b[2:,2:] = [2,-1]

    def __call__(self,operation,**args):
        self.calls.append((operation,args))
        if operation=='segment_everything':
            return self.masks.copy()
        if operation=='extract_features':
            return (self.a if args['time']=='t1' else self.b).copy()
        if operation=='classify':
            p = np.zeros((len(args['classes'])+1,*SHAPE),np.float32)
            p[0] = .9
            p[1] = .1
            p[0,:2,:2],p[1,:2,:2] = .1,.9
            return p
        raise AssertionError(operation)


class T13Tests(unittest.TestCase):
    def setUp(self):
        scratch = Path.cwd()/'.runtime/tests_t13'
        scratch.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.vision = FakeVision()
        self.rgb = [np.zeros((*SHAPE,3),np.uint8)]*2
        self.box = Toolbox(self.rgb,self.root,tools_t13=True,vision=self.vision)

    def run_tool(self,name,**kwargs):
        result = self.box.execute(name,kwargs)
        self.assertTrue(result.public['ok'],(result.public,result.internal))
        return result.public

    def test_off_exact_frozen_schema_and_prompts(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/g1_prompts.json').read_text())
        session = Session(SHAPE)
        for task in ('T1.1xC0','T5.1'):
            expected = fixture[task]
            self.assertEqual(json.dumps(session_schemas(session)).encode(),json.dumps(expected['schemas']).encode())
            self.assertEqual(system_prompt('v4',30,session)[2],expected['system_prompt'])
            self.assertEqual(instruction_for(task,session),expected['instruction'])
        self.assertEqual(json.dumps(verifier_prompts()).encode(),json.dumps(fixture['verifier']).encode())
        box = Toolbox(self.rgb,self.root,vision=self.vision)
        for name in NAMES:
            result = box.execute(name,{})
            self.assertFalse(result.public['ok'])
            self.assertEqual(result.public['error_code'],'bad_args')
        self.assertFalse(box.session.handles)
        self.assertFalse(self.vision.calls)

    def test_individual_tool_list(self):
        box = Toolbox(self.rgb,self.root,tools_t13=['segment_everything'],vision=self.vision)
        names = [x['function']['name'] for x in session_schemas(box.session,tools_t13=box.tools_t13)]
        self.assertEqual(names[-1:],[NAMES[0]])
        self.assertTrue(box.execute('segment_everything',dict(time='t1')).public['ok'])
        self.assertFalse(box.execute('extract_features',dict(time='t1')).public['ok'])

    def test_proposal_handles_selection_and_mask(self):
        props = self.run_tool('segment_everything',time='t1')['props']
        self.assertTrue(props.startswith('props#'))
        self.assertIsInstance(self.box.session.get(props),Instances)
        result = self.run_tool('select',set=props,ids=[2],where='score >= 0.5')
        self.assertTrue(result['props'].startswith('props#'))
        mask = self.run_tool('to_mask',sets=[result['props']])['mask']
        np.testing.assert_array_equal(self.box.session.get(mask,Mask).data,self.vision.masks[1])
        self.assertEqual(self.box.session.current_mask,mask)
        inspected = self.run_tool('inspect',mask=props)
        self.assertEqual(inspected['total_components'],2)

    def test_compare_one_minus_cos_and_feature_order(self):
        p = self.run_tool('segment_everything',time='t1')['props']
        a,b = [self.run_tool('extract_features',time=t)['feat'] for t in ('t1','t2')]
        out = self.run_tool('compare_features',props=p,feat_t1=a,feat_t2=b)
        rows = self.box.session.get(out['props']).items
        self.assertAlmostEqual(rows[0]['score'],1.)
        self.assertAlmostEqual(rows[1]['score'],1-2/np.sqrt(5),places=6)
        self.assertFalse(self.box.execute('compare_features',dict(props=p,feat_t1=b,feat_t2=a)).public['ok'])
        np.testing.assert_allclose(cosine([[1,0],[0,0]],[[0,1],[1,0]]),[0,0])

    def test_interpolated_grid_pool(self):
        grid = np.array([[[0.],[2.]],[[4.],[6.]]])
        f = Features(grid,(3,3),'t1')
        self.assertAlmostEqual(float(pool_features(f,[4])[0]),3.)
        self.assertAlmostEqual(float(pool_features(f,np.arange(9))[0]),3.)

    def test_classification_and_dense_mask(self):
        p = self.run_tool('segment_everything',time='t1')['props']
        out = self.run_tool('classify',props=p,time='t1',classes=['tree','water'])
        self.assertAlmostEqual(out['probabilities'][0]['probabilities']['tree'],.9,places=6)
        self.assertEqual(out['classes'],['tree','water'])
        result = self.run_tool('segment_semantic',time='t1',text='tree')
        np.testing.assert_array_equal(self.box.session.get(result['mask']).data,self.vision.masks[0])
        self.assertIsNone(self.box.session.current_mask)

    def test_change_exemplar_uses_difference_not_appearance(self):
        p = self.run_tool('segment_everything',time='t1')['props']
        exemplar = self.box.session.add_mask(self.vision.masks[0])
        out = self.run_tool('match_exemplar',props=p,exemplar=dict(mask=exemplar))
        rows = self.box.session.get(out['props']).items
        self.assertAlmostEqual(rows[0]['score'],1.,places=6)
        self.assertAlmostEqual(rows[1]['score'],-1.,places=6)
        point = self.run_tool('match_exemplar',props=p,exemplar=dict(point=[0,0]))
        self.assertEqual(point['similarities'],out['similarities'])
        self.assertFalse(self.box.execute('match_exemplar',dict(props=p,exemplar=dict(mask=exemplar,point=[0,0]))).public['ok'])
        self.assertFalse(self.box.execute('match_exemplar',dict(props=p,exemplar=dict(point=[4,0]))).public['ok'])
        empty = self.box.session.add_mask(np.zeros(SHAPE,bool))
        self.assertFalse(self.box.execute('match_exemplar',dict(props=p,exemplar=dict(mask=empty))).public['ok'])
        tasks = dict(instances=[np.flatnonzero(m).tolist() for m in self.vision.masks],exemplar_index=0)
        expected = task_scores(self.vision.masks,Features(self.vision.a,SHAPE,'t1'),Features(self.vision.b,SHAPE,'t2'),tasks)
        np.testing.assert_allclose(expected,[r['score'] for r in rows])

    def test_same_reply_handle_is_not_available(self):
        p = self.run_tool('segment_everything',time='t1')['props']
        result = self.box.execute('match_exemplar',dict(props=p,exemplar=dict(point=[0,0])),allowed=set())
        self.assertEqual(result.public['error_code'],'bad_handle')
        self.assertEqual(referenced_handles(dict(x=[p,'feat#3','mask#4'])),[p,'feat#3','mask#4'])

    def test_unrestricted_and_multiclass_verifier(self):
        raw = dict(choices=[dict(message=dict(content='{"change":"yes","type":"land_cover_change"}'))],usage={})
        for session in (Session(SHAPE,target='any change'),Session(SHAPE,targets=['tree','water'])):
            settings = verifier_prompts(session.target)
            self.assertNotIn('building',json.dumps(settings))
            self.assertIn('land-cover',system_prompt('v4',30,session)[2])
            self.assertEqual(scope_card(session.target)['sam3_cache'],{})
            captured = []
            backend = ApiAsk(session=session,client=lambda p:captured.append(p) or raw,correspondence=True)
            backend.one([Image.fromarray(self.rgb[0])]*4,settings['VERIFY'],1)
            self.assertEqual(captured[0]['messages'][0]['content'],settings['SYSTEM'])
            self.assertEqual(captured[0]['messages'][1]['content'][-1]['text'],settings['CORRESPONDENCE'])
            self.assertEqual(session_schemas(session)[5]['function']['parameters'],SCHEMAS[5]['function']['parameters'])
        text = verifier_prompts(('tree','water'))['VERIFY']
        self.assertIn('tree',text)
        self.assertIn('water',text)
        self.assertIn('without a category restriction',verifier_prompts('any change')['VERIFY'])

    def test_g2_off_requests_and_replay_comparison(self):
        # Two complete fake G2 sessions; use same output path to keep JPEG references identical.
        def run(explicit):
            session = Session(SHAPE,init_mask=self.vision.masks[0])
            box = Toolbox(self.rgb,self.root/'g2',session=session,**explicit)
            raw = dict(choices=[dict(message=dict(role='assistant',content='{"change":"yes","type":"new_building","reason":"Appeared."}'))],usage={})
            verifier_requests = []
            box.ask_backend = ApiAsk(session=session,client=lambda p:verifier_requests.append(copy.deepcopy(p)) or raw)
            submit = dict(id='submit',type='function',function=dict(name='submit',arguments=json.dumps(dict(mask='mask#0',summary='Keep.'))))
            requests = []
            def client(payload):
                requests.append(copy.deepcopy(payload))
                return dict(choices=[dict(message=dict(role='assistant',content='',tool_calls=[submit]))],usage={})
            recorder = Recorder(self.root/'g2','fake','x.png',gt=self.vision.masks[0])
            result = run_session(box,Planner(client=client),recorder,'T5.1',budget=2,prompt_version='v4',verify=True,gate2=True)
            self.assertEqual(result['submitted_by'],'model')
            return requests,verifier_requests
        original = run({})
        explicit_off = run(dict(tools_t13=False,vision=self.vision))
        compare(original,explicit_off,'G2 new tools off')
        self.assertEqual(json.dumps(original).encode(),json.dumps(explicit_off).encode())
        self.assertFalse(self.vision.calls)

    def test_feature_recorder(self):
        key = self.run_tool('extract_features',time='t1')['feat']
        recorder = Recorder(self.root/'trace','fake','x.png',gt=np.zeros(SHAPE,bool))
        recorder.assets(self.box.session)
        with np.load(self.root/'trace/features'/('feat_'+key.split('#')[1]+'.npz')) as z:
            np.testing.assert_array_equal(z['data'],self.vision.a)

    def test_annotated_and_binary_metrics(self):
        label = np.array([[1,0],[2,0]])
        self.assertEqual(annotated_counts(np.ones((2,2),bool),label,1),[1,1,0])
        b = np.array([[2,0],[2,0]])
        np.testing.assert_array_equal(binary_truth(label,b),[[True,False],[False,False]])
        with self.assertRaises(ValueError):
            binary_truth(label,np.ones((2,2),int))

    def test_coverage_and_exact_whole_mask_oracle(self):
        truth = np.array([[1,1],[0,0]],bool)
        masks = np.array([[[1,0],[1,0]],[[0,1],[0,0]],[[0,0],[1,1]]],bool)
        bounds = coverage_upper(masks,truth)
        self.assertEqual(bounds['coverage'],1.)
        self.assertEqual(bounds['pixel_oracle_f1_upper'],1.)
        brute = max(metrics([int((np.any(masks[list(ids)],axis=0)&truth).sum()),int((np.any(masks[list(ids)],axis=0)&~truth).sum()),int((~np.any(masks[list(ids)],axis=0)&truth).sum())])['f1'] or 0
            for size in range(1,4) for ids in itertools.combinations(range(3),size))
        oracle = selection_upper(masks,truth,time_limit=2)
        self.assertTrue(oracle['exact'])
        self.assertAlmostEqual(oracle['lower'],brute)
        self.assertAlmostEqual(oracle['upper'],.8)
        none = np.zeros((0,2,2),bool)
        self.assertEqual(coverage_upper(none,truth)['coverage'],0.)
        self.assertEqual(selection_upper(none,truth)['upper'],0.)

    def test_t3_reproducible_and_transition_truth(self):
        a = np.zeros((10,10),np.uint8)
        b = a.copy()
        a[:4,:4],b[:4,:4] = 1,5
        a[6:,6:],b[6:,6:] = 1,5
        first = make_tasks(a,b,'image.png')
        self.assertEqual(first,make_tasks(a,b,'image.png'))
        self.assertEqual(len(first),1)
        self.assertEqual(len(first[0]['instances']),2)
        self.assertEqual((first[0]['source'],first[0]['target']),(1,5))
        self.assertEqual(first,make_tasks(a,b,'image.png',min_area=16))
        self.assertEqual(make_tasks(a,b,'image.png',min_area=17),[])

    def test_retrieval_one_to_one_and_exemplar_removal(self):
        masks = self.vision.masks
        truths = [np.flatnonzero(m) for m in masks]
        duplicated = masks[[0,0,1]]
        result = retrieval(duplicated,[1,.9,.8],truths)
        self.assertAlmostEqual(result['ap'],(1+2/3)/2)
        self.assertEqual(result['recall_all'],1.)
        excluded = retrieval(duplicated,[1,.9,.8],truths,exclude=0)
        self.assertAlmostEqual(excluded['ap'],1/3)
        self.assertIsNone(retrieval(masks,[1,0],truths[:1],exclude=0)['ap'])

    def test_tiny_transition_is_retained_in_truth(self):
        a = np.zeros((8,8),np.uint8)
        b = a.copy()
        a[:4,:4],b[:4,:4] = 1,5
        a[7,7],b[7,7] = 1,5
        task = make_tasks(a,b,'x.png')[0]
        self.assertEqual([len(x) for x in task['instances']],[16,1])
        self.assertEqual(task['exemplar_index'],0)

    def test_task_manifest_and_exemplar_stay_separate_from_truth(self):
        from argparse import Namespace
        from agent_baseline.analysis.second_labels import ENCODING
        data = self.root/'data'
        data.mkdir()
        (data/'cls_gray.json').write_text(json.dumps(ENCODING))
        for part in ('img1','img2','slabel1','slabel2'):
            (data/part).mkdir()
        a = np.ones(SHAPE,np.uint8)
        b = np.full(SHAPE,5,np.uint8)
        for part,im in [('img1',self.rgb[0]),('img2',self.rgb[1]),('slabel1',a),('slabel2',b)]:
            Image.fromarray(im).save(data/part/'x.png')
        args = Namespace(dataset_root=data,run_root=self.root,phase='dev',shard=0,shards=1,
            output_root=self.root/'tasks',seed=1016,min_area=16)
        with patch('agent_baseline.analysis.make_t3_tasks.inventories',return_value=dict(dev=['x.png'])):
            generate(args)
        manifest = read_tasks(self.root/'tasks/t3_tasks_dev_0of1/tasks.jsonl')[0]
        self.assertEqual(manifest['task_type'],'T3.1')
        self.assertTrue(Path(manifest['exemplar_mask']).is_file())
        self.assertNotIn('low vegetation',manifest['instruction'])
        session = Session(SHAPE,target=manifest['target'],instruction=manifest['instruction'])
        session.exemplar_handle = session.add_mask(np.asarray(Image.open(manifest['exemplar_mask']))>0)
        box = Toolbox(self.rgb,self.root/'example_run',session=session)
        requests = []
        def client(payload):
            requests.append(copy.deepcopy(payload))
            call = dict(id='s',type='function',function=dict(name='submit',arguments=json.dumps(dict(mask='empty',summary='Empty.'))))
            return dict(choices=[dict(message=dict(role='assistant',tool_calls=[call]))],usage={})
        recorder = Recorder(self.root/'example_run','fake','x.png',gt_path=manifest['gt_mask'])
        final = run_session(box,Planner(client=client),recorder,'T3.1',budget=1,prompt_version='v4')
        self.assertEqual(final['submitted_by'],'model')
        payload = json.dumps(requests)
        self.assertIn('Example region: mask#0',payload)
        for word in ('low vegetation','SECOND','gt_mask',manifest['gt_mask'],manifest['exemplar_mask']):
            self.assertNotIn(word,payload)

    def test_replay_g1_synthetic_roundtrip(self):
        from agent_baseline.analysis.replay_g1 import replay, RUN
        reference = RUN/'code_g'
        if not (reference/'agent_baseline/backends.py').is_file():
            self.skipTest('Frozen reference unavailable; archived G1 replay remains unverified')
        archive = self.root/'archive/T5.1/val/x'
        paths = []
        for i,im in enumerate(self.rgb):
            path = self.root/f'source_{i}.png'
            Image.fromarray(im).save(path)
            paths.append(str(path))
        box = Toolbox(self.rgb,archive,init_mask=self.vision.masks[0])
        raw = dict(choices=[dict(message=dict(role='assistant',content='{"change":"yes","type":"new_building","reason":"Appeared."}'))],usage={})
        box.ask_backend = ApiAsk(client=lambda _:raw,correspondence=True)
        def client(payload):
            call = dict(id='s',type='function',function=dict(name='submit',arguments=json.dumps(dict(mask='mask#0',summary='Keep.'))))
            return dict(choices=[dict(message=dict(role='assistant',tool_calls=[call]))],usage={})
        recorder = Recorder(archive,'fake','x.png',gt=np.zeros(SHAPE,bool))
        run_session(box,Planner(client=client,seed=7),recorder,'T5.1',budget=3,prompt_version='v4',verify=True,
            metadata=dict(source_images=paths,backends=dict(ask_model='qwen3.7-plus',ask_correspondence=True,pixel_thr=.85,verify_stage=True)))
        with patch('urllib.request.urlopen',side_effect=AssertionError('No network')):
            result = replay(archive,self.root/'replayed',reference)
        self.assertEqual(result['planner_requests'],2)
        self.assertEqual(result['verifier_requests'],1)

    def test_probe_entrypoints_emit_json_and_markdown(self):
        from agent_baseline.analysis import probe_t13, probe_t12_segearth, probe_t3
        config = self.root/'vision.json'
        config.write_text('{}')
        a,b = np.ones(SHAPE,np.uint8),np.full(SHAPE,5,np.uint8)
        class Pool:
            def __init__(this,*args):
                pass
            def for_pair(this,*args):
                return self.vision
            def close(this):
                pass
        for module,stem in [(probe_t13,'probe_t13'),(probe_t12_segearth,'probe_t12_segearth'),(probe_t3,'probe_t3')]:
            argv = ['probe','--phase','dev','--vision-config',str(config),'--cache',str(self.root/'cache'),
                    '--output-root',str(self.root/'reports')]
            with patch('sys.argv',argv),patch.object(module,'selected',return_value=['x.png']), \
                 patch.object(module,'WorkerPool',Pool),patch.object(module,'load_labels',return_value=(a,b)):
                module.main()
            path = self.root/'reports'/f'{stem}_dev_0of1.json'
            report = json.loads(path.read_text())
            self.assertTrue(path.with_suffix('.md').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(set(report['per_image']),{'x.png'})
        self.assertEqual(json.loads((self.root/'reports/probe_t3_dev_0of1.json').read_text())['summary']['tasks'],1)

    def test_shards_reject_missing_and_duplicate_inventory(self):
        p = self.root/'shard.json'
        p.write_text(json.dumps(dict(phase='dev',per_image={'a':{}})))
        self.assertEqual(collect([p],'dev',['a'])[1],{'a':{}})
        with self.assertRaises(ValueError):
            collect([p,p],'dev',['a'])
        with self.assertRaises(ValueError):
            collect([p],'dev',['a','b'])


if __name__=='__main__':
    unittest.main()
