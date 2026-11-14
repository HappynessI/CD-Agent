"""Small synthetic tests; no model initialization, scheduler or network."""
from io import BytesIO
import json
from pathlib import Path
import unittest
import tempfile
import urllib.error
import numpy as np
from agent_baseline.analysis.io_retry import write_json, write_text, save_image
from agent_baseline.analysis.t3_routes import (semantic_route, evaluate_a, group, breakdown,
    candidate_estimate, ledger_estimate, GRID)
from agent_baseline.analysis.t3_routes_match import (ranked_indices, build_messages, filtered_union,
    match_ranked)
from agent_baseline.analysis.t3_routes_runtime import (verify_jobs, Stopped, REQUEST_RESERVE,
    Budget, prepare_b, score_b)
from agent_baseline.analysis.t3_routes_cache import write_npz, CachedVision
from agent_baseline.analysis.t3_routes_agent import sample_ten, inspect_trace, audit
from agent_baseline.backends import ApiAsk
from agent_baseline.view import ask_images


def tempdir():
    return tempfile.TemporaryDirectory(prefix='t3_routes_', dir=Path(__file__).parent)


def region(box=(1, 1, 3, 3), shape=(10, 10)):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def response(answer='yes', tokens=1000):
    return dict(choices=[dict(message=dict(role='assistant', content=json.dumps(dict(change=answer,
        example_t1='water', example_t2='tree', candidate_t1='water', candidate_t2='tree'))))],
        usage=dict(prompt_tokens=tokens, completion_tokens=20))


def diag_task(task_id, counts, proposals=9):
    return dict(id=task_id, methods=dict(sam_dino=dict(scores={'0.6':dict(counts=counts)}, proposals=dict(
        gt_instances=2, aligned_instances=1, truth_pixels=100, covered_pixels=90, proposals=proposals))))


class RouteTests(unittest.TestCase):
    def test_group_and_pooled_diagnostic(self):
        rows = [dict(id='x_1to5'), dict(id='y_6to2'), dict(id='z_1to4')]
        self.assertEqual([group(r) for r in rows], ['目标类', '目标类', '地表类'])
        diag = dict(tasks=[diag_task(r['id'], c) for r, c in zip(rows, ([1,0,0], [0,9,9], [10,0,0]))])
        result = breakdown(rows, diag)
        self.assertAlmostEqual(result['目标类']['fixed']['f1_c'], 10.)
        self.assertEqual(result['地表类']['fixed']['f1_c'], 100.)
        self.assertEqual(result['目标类']['proposals']['instance_recall_upper'], .5)
        self.assertEqual(result['目标类']['proposals']['pixel_coverage_upper'], .9)
        with self.assertRaises(ValueError):
            breakdown(rows[:-1], diag)

    def test_semantic_pair_mean_probability_and_direction(self):
        ex = region((0,0,2,2), (4,4))
        a, b = np.zeros((7,4,4)), np.zeros((7,4,4))
        a[1] = .6; a[5] = .4
        # Majority of exemplar pixels favour 1 slightly, but mean favours 5.
        a[1,0,0] = 0; a[5,0,0] = 1
        a[1,ex & ~region((0,0,1,1),(4,4))] = .51
        a[5,ex & ~region((0,0,1,1),(4,4))] = .49
        a[5,3,3] = .9
        b[4] = 1
        pair, prediction = semantic_route([a,b], ex)
        self.assertEqual(pair, [5,4])
        self.assertTrue(prediction[3,3])
        self.assertFalse(prediction[ex].any())
        self.assertFalse(semantic_route([a,a], ex)[1].any())
        # Background is not a named SECOND class for the exemplar decision.
        a[0] = 10
        self.assertEqual(semantic_route([a,b], ex)[0], [5,4])
        self.assertFalse(semantic_route([a,b], ex)[1].any())

    def test_route_a_fake_backend_only_sees_images_and_classes(self):
        with tempdir() as d:
            d = Path(d); ex = region(); other = region((6,6,8,8)); truth = ex | other
            save_image(d/'ex.png', ex.astype('uint8')*255)
            save_image(d/'gt.png', truth.astype('uint8')*255)
            row = dict(id='x_1to5', exemplar_mask=d/'ex.png', gt_mask=d/'gt.png')
            seen = []
            def backend(op, **kw):
                seen.append((op, kw))
                p = np.zeros((7,10,10)); p[0] = 1
                p[1 if kw['time']=='t1' else 5, truth] = 2
                return p
            result = evaluate_a(row, backend)
            self.assertTrue(result['pair_correct'])
            self.assertEqual(result['counts'], [4,0,0])
            self.assertEqual(len(seen), 2)
            self.assertNotIn('gt_mask', str(seen))

    def test_counts_and_ledger_pricing(self):
        with tempdir() as d:
            path = Path(d)/'ledger.jsonl'
            write_text(path, json.dumps(dict(answer='yes', tokens=dict(input=1000, output=50, cached=0)))+'\n')
            unit = ledger_estimate(path)
            self.assertEqual(unit['estimated_input'], 2512)
            self.assertAlmostEqual(unit['per_request_cny'], (2512*.2+128*.8)/1e6)
            rows = [dict(id='x_1to5'),dict(id='y_1to4')]
            diag = dict(tasks=[diag_task(rows[0]['id'],[1,0,0],3),diag_task(rows[1]['id'],[1,0,0],10)])
            estimates = candidate_estimate(rows,diag,unit)
            self.assertEqual(estimates['grid']['4']['candidates'], 7)
            self.assertEqual(estimates['grid']['32']['requests'], 13)
            self.assertEqual(estimates['unique_requests_all_k'], 13)
            self.assertIsNone(estimates['grid']['8']['score'])

    def test_request_has_both_regions_and_legacy_default_unchanged(self):
        ex, candidate = region(), region((6,6,8,8))
        rgb = [np.zeros((10,10,3),np.uint8), np.full((10,10,3), 180,np.uint8)]
        messages = build_messages(rgb, ex, candidate, 't3_transition')
        content = messages[1]['content']
        self.assertEqual(sum(b['type']=='image_url' for b in content), 8)
        self.assertEqual(content[0]['text'], 'Example T1 close-up')
        self.assertEqual(content[8]['text'], 'Candidate T1 close-up')
        self.assertIn('example_t1', content[-1]['text'])
        requests = []
        def client(payload):
            requests.append(payload)
            return response()
        ApiAsk(client=client, model='qwen3.7-flash', correspondence=True).one(
            ask_images(rgb,candidate,[6,6,8,8],True), '', 1)
        self.assertEqual(build_messages(rgb,ex,candidate), requests[0]['messages'])

    def test_reusable_match_ranks_without_threshold_and_deletes_no(self):
        ex = region(); good = region((4,4,6,6)); bad = region((7,7,9,9))
        masks = np.asarray([good,bad,ex])
        self.assertEqual(ranked_indices([-.4,-.4,-.8],2), [0,1])
        replies = iter([dict(answer='yes'),dict(answer='no')])
        result = match_ranked(masks,[-.4,-.4,-.8],ex,
            [np.zeros((10,10,3),np.uint8)]*2,2,lambda _: next(replies),'t3_transition')
        np.testing.assert_array_equal(result['mask'],good)
        with self.assertRaises(ValueError):
            filtered_union(masks,[0],{0:dict(answer=None)},ex)
        np.testing.assert_array_equal(filtered_union(masks,[2],{2:dict(answer='yes')},ex),np.zeros_like(ex))
        # Unparsed is not no: the candidate stays; only explicit no is deleted.
        np.testing.assert_array_equal(filtered_union(masks,[0,1],{0:dict(answer='unparsed'),1:dict(answer='no')},ex),good)

    def test_verify_resume_reuses_answers_and_restores_cost(self):
        with tempdir() as d:
            d = Path(d); calls = []
            def client(payload):
                calls.append(payload)
                return response()
            jobs = [('one',{}),('two',{})]
            kwargs = dict(folder=d, concurrency=2, max_cost=1, policy={'version':1},
                          request=lambda _: [{'role':'user','content':'fake'}], client=client)
            spent = verify_jobs(jobs,resume=False,**kwargs)
            self.assertEqual(len(calls),2)
            self.assertEqual(verify_jobs(jobs,resume=True,**kwargs),spent)
            self.assertEqual(len(calls),2)
            self.assertEqual(Budget(d,1).spent,spent)
            with self.assertRaises(ValueError):
                verify_jobs(jobs,resume=False,**kwargs)

    def test_provider_stops_without_retry_and_keeps_status(self):
        for code in (429,403,402,400):
            with self.subTest(code=code), tempdir() as d:
                d = Path(d); calls = []
                def client(payload):
                    calls.append(payload)
                    raise urllib.error.HTTPError('https://example.invalid',code,'test',{},BytesIO(b'insufficient balance'))
                with self.assertRaises(Stopped):
                    verify_jobs([('one',{}),('two',{})],d,1,1,False,{},lambda _: [],client)
                self.assertEqual(len(calls),1)
                self.assertIn(str(code),json.loads((d/'STOP.json').read_text())['reason'])
                self.assertFalse((d/'answers/one.json').exists())

    def test_budget_prevents_dispatch_and_counts_malformed_bills(self):
        with tempdir() as d:
            d = Path(d); calls = []
            with self.assertRaises(Stopped):
                verify_jobs([('one',{})],d,1,REQUEST_RESERVE/2,False,{},lambda _: [],lambda p:calls.append(p))
            self.assertEqual(calls,[])
        with tempdir() as d:
            d = Path(d)
            # A billed reply that is not yes/no (e.g. cut off at max_tokens) is kept as unparsed; the run goes on.
            verify_jobs([('one',{}),('two',{})],d,1,1,False,{},lambda _: [],
                        lambda p: response('maybe') if not (d/'answers/one.json').exists() else response('no'))
            self.assertGreater(Budget(d,1).spent,0)
            self.assertFalse((d/'STOP.json').exists())
            one = json.loads((d/'answers/one.json').read_text())
            self.assertEqual(one['answer'],'unparsed'); self.assertIn('maybe',one['raw'])
            self.assertEqual(json.loads((d/'answers/two.json').read_text())['answer'],'no')

    def test_agent_sample_and_exemplar_classification_evidence(self):
        rows = [dict(id=f'x_{i:02}_1to5') for i in range(60)]
        chosen = sample_ten(rows)
        self.assertEqual(chosen,sample_ten(list(reversed(rows))))
        self.assertEqual(len({r['id'] for r in chosen}),10)
        with tempdir() as d:
            d = Path(d); ex = region()
            write_npz(d/'insts/inst_3.npz', masks=np.packbits(ex.reshape(1,-1),axis=1),shape=ex.shape)
            calls = [dict(tool='classify',ok=True,args=dict(props='props#3',time='t1')),
                     dict(tool='segment_semantic',ok=True,args=dict(time='t2',text='building'),output_handles=['mask#2'])]
            write_text(d/'steps.jsonl',json.dumps(dict(step=1,calls=calls))+'\n')
            result = inspect_trace(d,ex)
            self.assertTrue(result['exemplar_category_judgement'])
            self.assertFalse(result['both_dates_judged'])
            self.assertEqual(len(result['exemplar_category_evidence']),1)
            # A full-image semantic call alone does not count as exemplar judgement.
            write_text(d/'steps.jsonl',json.dumps(dict(step=1,calls=calls[1:]))+'\n')
            self.assertFalse(inspect_trace(d,ex)['exemplar_category_judgement'])
            follow=dict(step=2,planner=dict(thought='Identify the land-cover class of the example at T2.'),
                calls=[dict(tool='combine',ok=True,args=dict(op='intersect',a='mask#0',b='mask#2'))])
            write_text(d/'steps.jsonl','\n'.join(json.dumps(s) for s in [dict(step=1,calls=calls),follow])+'\n')
            self.assertTrue(inspect_trace(d,ex)['both_dates_judged'])

    def test_prepare_reuses_small_cached_arrays_and_scores_all_k(self):
        with tempdir() as d:
            d = Path(d); run=d/'run'; out=d/'out'; ex=region(); other=region((6,6,8,8))
            save_image(d/'ex.png',ex.astype('uint8')*255)
            save_image(d/'gt.png',(ex|other).astype('uint8')*255)
            row=dict(id='x_1to5',image_t1=str(d/'img1.png'),image_t2=str(d/'img2.png'),
                     exemplar_mask=d/'ex.png',gt_mask=d/'gt.png')
            for k in ('image_t1','image_t2'):
                save_image(row[k],np.zeros((10,10,3),np.uint8))
            sources=[]
            for k in ('image_t1','image_t2'):
                p=Path(row[k]); st=p.stat()
                sources.append(dict(path=str(p.resolve()),size=st.st_size,mtime=st.st_mtime_ns))
            records=[]; root=run/'t3_diag/cache/vision/dynamic'
            for t in ('t1','t2'):
                for op in ('segment_everything','extract_features'):
                    records.append(dict(sources=sources,operation=op,time=t,classes=None,config={}))
                    path=root/f'item_{len(records)-1:06d}.npz'
                    if op=='segment_everything':
                        write_npz(path,data=np.packbits(np.asarray([ex,other]).reshape(2,-1),axis=1),shape=ex.shape)
                    else:
                        write_npz(path,data=np.full((2,2,3),1 if t=='t2' else 0,np.float32))
            write_json(root/'inventory.json',records)
            write_json(run/'t3_diag/report.json',dict(settings=dict(vision_config=dict(dynamic={}))))
            prepare_b([row],run,out,False)
            first=json.loads((out/'route_b/prepared.json').read_text())
            prepare_b([row],run,out,True)
            self.assertEqual(first,json.loads((out/'route_b/prepared.json').read_text()))
            for i in first['tasks'][0]['indices']:
                write_json(out/'route_b/answers'/f'x_1to5__{i}.json',dict(answer='yes'))
            score_b([row],out)
            result=json.loads((out/'route_b/scores.json').read_text())
            self.assertEqual(result['selected_k'],4)
            self.assertTrue(all(result['scores'][str(k)]['全部']['score']['f1_c']==100 for k in GRID))

    def test_agent_audit_excludes_example_and_resumes(self):
        with tempdir() as d:
            d=Path(d); ex=region(); gt=ex|region((6,6,8,8))
            save_image(d/'ex.png',ex.astype('uint8')*255)
            save_image(d/'gt.png',gt.astype('uint8')*255)
            save_image(d/'sessions/x/pred.png',gt.astype('uint8')*255)
            write_json(d/'sessions/x/final.json',dict(mask_path='pred.png',submitted_by='model'))
            write_text(d/'sessions/x/steps.jsonl',json.dumps(dict(step=0,calls=[]))+'\n')
            rows=[dict(id='x',exemplar_mask=d/'ex.png',gt_mask=d/'gt.png')]
            result=audit(rows,d/'sessions',d/'out')
            self.assertEqual(result['score']['tp'],4)
            self.assertEqual(audit(rows,d/'sessions',d/'out',True)['score'],result['score'])


if __name__ == '__main__':
    unittest.main()
