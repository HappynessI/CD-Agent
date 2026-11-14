"""Focused semantic bridge regression, executed in the smoke Slurm allocation."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.t3_semantic_bridge import (
    canonical, bridge_scores, rank, single_date_views, PARSE_QUESTION, union, cached_dino_ranking)


class BridgeTests(unittest.TestCase):
    def test_real_loop_advertises_tools_without_task_metadata(self):
        from agent_baseline.loop_json import run_session
        from agent_baseline.tools import Toolbox
        from agent_baseline.state import Session
        from agent_baseline.recorder import Recorder
        from agent_baseline.planner import Planner
        from .helpers import ScriptClient, response, call, rgb, mask
        with tempfile.TemporaryDirectory() as folder:
            session=Session(mask().shape,target='any change',instruction='Find the same transition as the visual example.')
            session.exemplar_handle=session.add_mask(mask())
            toolbox=Toolbox(rgb(),folder,session=session,tools_t13=True)
            client=ScriptClient([response([call('submit',dict(mask='empty',summary='No other region.'))])])
            planner=Planner(client=client,parameters=dict(model='qwen3.7-flash'))
            recorder=Recorder(folder,'synthetic','example.png',gt=mask())
            run_session(toolbox,planner,recorder,task='T3.1',t3_semantic_bridge=True,
                metadata=dict(session_id='private_task_6to3',task_meta=dict(transition='private_label')))
            names={s['function']['name'] for s in client.requests[0]['tools']}
            self.assertTrue({'interpret_exemplar','rank_semantic_transition'} <= names)
            payload=json.dumps(client.requests[0])
            self.assertNotIn('private_task_6to3',payload)
            self.assertNotIn('private_label',payload)
            self.assertNotIn('gt_mask',payload)

    def test_class6_background_contract_and_ranked_selection(self):
        from agent_baseline.tools import Toolbox
        from agent_baseline import instances
        from agent_baseline.state import Mask
        from agent_baseline.exemplar_semantics import CLASSES
        rgb=np.zeros((2,2,3),np.uint8)
        a,b=np.zeros((7,2,2)),np.zeros((7,2,2))
        a[0]=1; b[0]=1  # Background must not be used as a requested class.
        a[6]=[[.2,.9],[.1,.1]]; b[3]=[[.8,.9],[.1,.1]]
        masks=np.array([[[1,0],[0,0]],[[0,1],[0,0]]],bool)
        calls=[]
        def backend(operation,**kwargs):
            calls.append((operation,kwargs))
            self.assertEqual(kwargs['classes'],list(CLASSES.values()))
            return a if kwargs['time']=='t1' else b
        with tempfile.TemporaryDirectory() as folder:
            toolbox=Toolbox([rgb,rgb],folder,vision=backend); toolbox.t3_semantic_bridge=True
            current=toolbox.session.add_mask(np.eye(2,dtype=bool)); toolbox.session.current_mask=current
            props=toolbox._props(instances.from_dense(masks,np.ones(2)),'t1')['props']
            result=toolbox.execute('rank_semantic_transition',dict(props=props,before='playground',after='tree'))
            self.assertTrue(result.public['ok'],result.public)
            self.assertEqual(result.public['ranked_ids'],[2,1])
            self.assertAlmostEqual(result.public['scores'][1]['score'],.81)
            selected=toolbox.execute('select',dict(set=result.public['props'],ids=result.public['ranked_ids'][:1]))
            made=toolbox.execute('to_mask',dict(sets=[selected.public['props']],set_current=False))
            self.assertTrue(np.array_equal(toolbox.session.get(made.public['mask'],Mask).data,masks[1]))
            self.assertEqual(toolbox.session.current_mask,current)
            self.assertEqual(len(calls),2)

    def test_partial_cost_deduplicates_and_archive_preserves_success(self):
        from agent_baseline.analysis.t3_bridge_online_metrics import session_cost, archive_incomplete, completion
        from agent_baseline.planner import request_cost, usage_tokens
        usage = dict(prompt_tokens=1000,completion_tokens=100)
        event = dict(request_id='response-1',usage=usage,request=dict(model='qwen3.7-flash'))
        semantic = dict(requested_model='qwen3.7-flash',usage=usage,attempt=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); session=root/'bridge_s1016/T3.1/val/partial'
            session.mkdir(parents=True)
            (session/'messages.jsonl').write_text('\n'.join(json.dumps(e) for e in (event,event,semantic)))
            attempts=session/'semantic_bridge/mask_0/t1'; attempts.mkdir(parents=True)
            (attempts/'attempt_001.json').write_text(json.dumps(semantic))
            bill=session_cost(session)
            self.assertEqual(bill['known_requests'],2)
            self.assertAlmostEqual(bill['cost_cny'],2*request_cost('qwen3.7-flash',usage_tokens(usage)))
            success=root/'bridge_s1016/T3.1/val/success'; success.mkdir(parents=True)
            (success/'final.json').write_text('{}')
            archive_incomplete(root,'bridge_s1016',[dict(id='partial'),dict(id='success')])
            self.assertTrue((success/'final.json').exists())
            archived=root/'interrupted_attempts/bridge_s1016/partial/attempt_000'
            self.assertTrue((archived/'messages.jsonl').exists())
            self.assertAlmostEqual(session_cost(archived)['cost_cny'],bill['cost_cny'])
        self.assertEqual(completion(None),'missing_final')
        self.assertEqual(completion(dict(submitted_by='model',termination_reason='submit')),'successful_submit')
        self.assertEqual(completion(dict(submitted_by='error',termination_reason='error')),'error')
        self.assertEqual(completion(dict(submitted_by='budget',termination_reason='budget')),'fallback')

    def test_historical_dino_identity_and_order(self):
        masks = np.array([[[1,0],[0,0]], [[0,1],[0,0]], [[0,0],[1,0]]],bool)
        indices = np.array([2,0,1])
        saved = dict(indices=indices, shape=np.array([2,2]), scores=np.array([.9,.7,.1]),
            masks=np.packbits(masks[indices].reshape(3,-1),axis=1))
        with patch('agent_baseline.analysis.t3_semantic_bridge.read_npz',return_value=saved):
            self.assertEqual(cached_dino_ranking('unused',masks),[2,0,1])
            changed = masks.copy(); changed[0,1,1] = True
            with self.assertRaisesRegex(ValueError,'masks differ'):
                cached_dino_ranking('unused',changed)

    def test_online_opt_in_and_current_unchanged(self):
        from agent_baseline.tools import Toolbox, session_schemas
        from agent_baseline import instances
        rgb = np.zeros((2,2,3), np.uint8)
        a, b = np.zeros((7,2,2)), np.zeros((7,2,2))
        a[1] = .7; b[5] = .8
        def backend(operation, **kwargs):
            return a if kwargs['time']=='t1' else b
        with tempfile.TemporaryDirectory() as folder:
            toolbox = Toolbox([rgb,rgb], folder, vision=backend)
            current = toolbox.session.add_mask(np.eye(2,dtype=bool))
            toolbox.session.current_mask = current
            props = toolbox._props(instances.from_dense(np.ones((1,2,2),bool),np.ones(1)), 't1')['props']
            args = dict(props=props,before='low vegetation',after='building')
            self.assertFalse(toolbox.execute('rank_semantic_transition',args).public['ok'])
            off = [s['function']['name'] for s in session_schemas(toolbox.session)]
            self.assertNotIn('interpret_exemplar', off)
            toolbox.t3_semantic_bridge = True
            result = toolbox.execute('rank_semantic_transition',args)
            self.assertTrue(result.public['ok'], result.public)
            self.assertAlmostEqual(result.public['scores'][0]['score'], .56)
            self.assertEqual(toolbox.session.current_mask,current)
            self.assertIn('interpret_exemplar', [s['function']['name'] for s in session_schemas(toolbox.session,t3_semantic_bridge=True)])

    def test_vocab_unknown(self):
        self.assertEqual(canonical(' BUILDING '), 'building')
        self.assertEqual(canonical('roof'), 'unknown')
        self.assertEqual(canonical(None), 'unknown')

    def test_single_date_isolation(self):
        image = np.zeros((4,4,3), np.uint8)
        mask = np.ones((4,4), bool)
        with patch('agent_baseline.exemplar_semantics.ask_images', return_value=['a','b','c','d']) as render:
            self.assertEqual(single_date_views(image, mask), [('close','a'),('wide','c')])
        dates = render.call_args.args[0]
        self.assertIs(dates[0], image)
        self.assertIs(dates[1], image)
        self.assertNotIn('ground truth', PARSE_QUESTION)

    def test_direction_and_unknown(self):
        masks = np.array([[[1,0]], [[0,1]]], bool)
        a, b = np.zeros((7,1,2)), np.zeros((7,1,2))
        a[1] = [1,.2]; b[5] = [.8,.9]
        a[5] = [.1,.9]; b[1] = [.1,.9]
        self.assertEqual(rank(bridge_scores(masks,[a,b],[1,5])), [0,1])
        self.assertEqual(rank(bridge_scores(masks,[a,b],[5,1])), [1,0])
        self.assertEqual(rank(bridge_scores(masks,[a,b],[None,5])), [])
        self.assertEqual(rank(bridge_scores(masks,[a,b],[5,5])), [])
        self.assertEqual(rank([.5,.5]), [0,1])

    def test_exemplar_excluded(self):
        masks = np.ones((2,2,2), bool)
        exemplar = np.eye(2, dtype=bool)
        self.assertFalse(union(masks,[0],exemplar)[exemplar].any())
        self.assertFalse(union(masks,[],exemplar).any())


if __name__ == '__main__':
    unittest.main()
