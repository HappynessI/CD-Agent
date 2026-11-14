"""Small CPU/fake-client tests; never access API credentials or real images."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np
from PIL import Image

from agent_baseline.analysis import xbd_damage as x
from agent_baseline.analysis.io_retry import read_text, write_json
from agent_baseline.analysis.missed_sources_ask import AskStopped


def row(name, positive, event, kind):
    counts = dict.fromkeys((*x.GRADES, x.UNKNOWN), 0)
    counts['major-damage' if positive else 'no-damage'] = 1
    return dict(name=name, counts=counts, buildings=1, event=event, disaster_type=kind)


def record(i, truth='no-damage', event='event'):
    return dict(uid=f'fake:{i}', truth=truth, event=event)


def response(answer='{"damage":"major-damage"}'):
    return dict(id='fake-request', choices=[dict(message=dict(role='assistant', content=answer))],
                usage=dict(prompt_tokens=1000, completion_tokens=10))


class Sampling(unittest.TestCase):
    def test_deterministic_balanced_event_type_coverage(self):
        rows = [row(f'{event}:{pos}:{i}', pos, event, kind)
                for kind in ('wind','fire','water','earthquake','volcano')
                for event in (kind+'1', kind+'2') for pos in (False, True) for i in range(10)]
        first = x.sample(rows, 60)
        self.assertEqual(first, x.sample(list(reversed(rows)), 60))
        self.assertEqual(sum(map(x.severe, first)), 30)
        self.assertEqual(len({r['name'] for r in first}), 60)
        self.assertEqual(len({r['event'] for r in first}), 10)
        self.assertEqual(len({r['disaster_type'] for r in first}), 5)
        probe = x.sample(first, 10)
        self.assertEqual(sum(map(x.severe, probe)), 5)
        self.assertLessEqual({r['name'] for r in probe}, {r['name'] for r in first})

    def test_empty_unknown_and_insufficient(self):
        rows = [row('positive',True,'e','fire'), row('negative',False,'e','fire')]
        bad = row('unknown',False,'e','fire'); bad['counts'][x.UNKNOWN] = 1
        empty = row('empty',False,'e','fire'); empty['buildings'] = 0
        self.assertEqual(x.sample(rows+[bad,empty],2), x.sample(rows,2))
        for size in (3,4,0):
            with self.assertRaises(ValueError): x.sample(rows,size)
        with self.assertRaises(ValueError): x.sample(rows+rows,2)

    def test_label_count_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'hold/labels/fire_0001_post_disaster.json'
            write_json(path, dict(metadata=dict(disaster_type='fire'), features=dict(xy=[
                dict(properties=dict(feature_type='building',subtype='destroyed'),wkt='unused'),
                dict(properties=dict(feature_type='building',subtype=x.UNKNOWN),wkt='unused')])))
            value = x.label_row(path, Path(tmp))
            self.assertEqual(value['post'], 'hold/images/fire_0001_post_disaster.png')
            self.assertEqual(value['event'],'fire')
            self.assertEqual(value['counts'][x.UNKNOWN],1)
            self.assertEqual(value['buildings'],2)


class CropsAndParsing(unittest.TestCase):
    def test_edge_crop_contour_and_same_geometry(self):
        rgb = np.zeros((100,100,3),dtype=np.uint8)
        images, box = x.building_crops([rgb,rgb.copy()], 'POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
        self.assertEqual(box,[0,0,64,64])
        self.assertEqual([im.size for im in images],[(384,384)]*2)
        self.assertTrue(np.any(np.all(np.asarray(images[0]) == (255,0,0),axis=-1)))
        np.testing.assert_array_equal(images[0],images[1])

    def test_holes_and_multipolygon(self):
        mask = x.polygon_mask('POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))',(32,32))
        self.assertTrue(mask[2,2]); self.assertFalse(mask[10,10])
        multi = x.polygon_mask('MULTIPOLYGON (((0 0, 3 0, 3 3, 0 3, 0 0)), ((10 10, 14 10, 14 14, 10 14, 10 10)))',(32,32))
        self.assertTrue(multi[1,1]); self.assertTrue(multi[12,12]); self.assertFalse(multi[7,7])
        with self.assertRaises(ValueError): x.polygon_rings('POINT (1 2)')
        rgb = np.zeros((32,32,3),dtype=np.uint8)
        with self.assertRaises(ValueError):
            x.building_crops([rgb,rgb],'POLYGON ((100 100, 110 100, 110 110, 100 100))')

    def test_strict_answers(self):
        for grade in x.GRADES:
            self.assertEqual(x.parse_answer(json.dumps(dict(damage=grade))),grade)
        for text in ('major-damage', '```json\n{"damage":"destroyed"}\n```',
                     '{"damage":"destroyed","reason":"x"}', '{"damage":"unknown"}',
                     '{"damage":["destroyed"]}', '{}', 'null', '[]', ''):
            self.assertIsNone(x.parse_answer(text))


class Reporting(unittest.TestCase):
    def test_matrix_binary_abstentions_pending_event(self):
        records = [record(0,'no-damage'),record(1,'minor-damage'),record(2,'major-damage'),
                   record(3,'destroyed'),record(4,'major-damage'),record(5,'no-damage'),record(6,'destroyed')]
        preds = ['no-damage','destroyed','destroyed','minor-damage',None,None]
        answers = {r['uid']:dict(prediction=p,cost_cny=.001) for r,p in zip(records,preds)}
        value = x.metrics(records,answers)
        self.assertEqual(value['confusion_matrix'],[[1,0,0,0],[0,0,0,1],[0,0,0,1],[0,1,0,0]])
        self.assertEqual(value['unparsed'],[1,0,1,0])
        self.assertEqual(value['pending'],[0,0,0,1])
        self.assertAlmostEqual(value['binary']['precision'],.5)
        self.assertAlmostEqual(value['binary']['recall'],1/3)
        self.assertAlmostEqual(value['binary']['f1'],.4)
        self.assertEqual(value['per_class']['destroyed']['recall'],0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for uid, result in answers.items():
                write_json(root/'requests/answer_parts'/(uid.replace(':','__')+'.json'),dict(uid=uid,**result))
            report = x.make_report(records,2,root/'requests',root)
            self.assertEqual(report['status'],'pending_or_incomplete')
            self.assertEqual(report['by_event']['event']['f1'],.4)
            self.assertEqual(report['skipped_unclassified'],2)
            self.assertAlmostEqual(report['cost_cny'],.006)
            self.assertTrue((root/'xbd_probe10_report.md').is_file())

    def test_empty_template_null_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = x.make_report([record(0)],3,Path(tmp)/'requests',tmp)
            self.assertEqual(report['requests'],0)
            self.assertEqual(report['pending'],[1,0,0,0])
            self.assertIsNone(report['binary']['f1'])


class Requests(unittest.TestCase):
    def test_request_matches_existing_style_and_pricing(self):
        payloads = []
        class Guard:
            def check(self): pass
            def charge(self,cost): self.cost=cost
            def trip(self,reason): raise AskStopped(reason)
        guard = Guard(); snapshots=[]; attempts=[]
        images = [Image.new('RGB',(20,20))]*2
        result = x.damage_answer(images,guard,1016,x.QUESTION_VARIANT,
            client=lambda payload: payloads.append(payload) or response(), snapshot=snapshots.append,
            attempt_record=attempts.append)
        payload = payloads[0]
        self.assertEqual(payload['model'],'qwen3.7-flash')
        self.assertEqual(payload['response_format'],{'type':'json_object'})
        self.assertFalse(payload['enable_thinking'])
        blocks = payload['messages'][1]['content']
        self.assertEqual(sum(b['type']=='image_url' for b in blocks),2)
        self.assertIn('Joint Damage Scale', blocks[-1]['text'])
        self.assertEqual(result['prediction'],'major-damage')
        self.assertAlmostEqual(guard.cost,.000208)
        self.assertEqual(snapshots[0]['cost_cny'],result['cost_cny'])
        self.assertEqual(attempts[0]['status'],'response')
        with self.assertRaises(ValueError): x.damage_answer(images,guard,1,'default',client=lambda p:response())

    def test_http_stop_no_retry_and_resume(self):
        for code, body in ((429,'busy'),(403,'denied'),(402,'payment'),(400,'InsufficientBalance'),(400,'欠费')):
            with self.subTest(code=code,body=body), tempfile.TemporaryDirectory() as tmp:
                records = [record(i) for i in range(3)]
                calls=[]; client_calls=[]
                def backend(r,guard):
                    calls.append(r['uid'])
                    if r['uid']=='fake:0':
                        guard.charge(.001)
                        return dict(prediction='no-damage',cost_cny=.001)
                    def client(payload):
                        client_calls.append(payload)
                        raise urllib.error.HTTPError('unused',code,'failure',{},BytesIO(body.encode()))
                    return x.damage_answer([Image.new('RGB',(8,8))]*2,guard,1,x.QUESTION_VARIANT,client=client)
                with self.assertRaises(AskStopped):
                    x.run_probe(records,backend,tmp,1)
                self.assertEqual(calls,['fake:0','fake:1'])
                self.assertEqual(len(client_calls),1)
                self.assertTrue((Path(tmp)/'STOP.json').is_file())
                with self.assertRaises(AskStopped): x.run_probe(records,backend,tmp,1,resume=True)
                resumed=[]
                def good(r,guard):
                    resumed.append(r['uid']); guard.charge(.002)
                    return dict(prediction='minor-damage',cost_cny=.002)
                result = x.run_probe(records,good,tmp,1,resume=True,clear_stop=True)
                self.assertEqual(resumed,['fake:1','fake:2'])
                self.assertAlmostEqual(result['cost_cny'],.005)
                self.assertEqual(result['answers'],3)

    def test_budget_reservation_resume_and_input_freeze(self):
        records=[record(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def good(r,guard):
                calls.append(r['uid']);guard.charge(.01)
                return dict(prediction='no-damage',cost_cny=.01)
            with self.assertRaises(AskStopped): x.run_probe(records,good,tmp,.025,concurrency=2,allowance=.02)
            self.assertEqual(calls,['fake:0'])
            self.assertAlmostEqual(json.loads(read_text(Path(tmp)/'budget.json'))['spent_cny'],.01)
            with self.assertRaises(ValueError): x.run_probe(records,good,tmp,1)
            changed = [dict(r,truth='destroyed') for r in records]
            with self.assertRaises(ValueError): x.run_probe(changed,good,tmp,1,resume=True,clear_stop=True)
            result=x.run_probe(records,good,tmp,1,resume=True,clear_stop=True)
            self.assertEqual(calls,['fake:0','fake:1','fake:2'])
            self.assertAlmostEqual(result['cost_cny'],.03)

    def test_concurrent_inflight_success_retained(self):
        records=[record(i) for i in range(4)]
        barrier=threading.Barrier(2)
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def backend(r,guard):
                calls.append(r['uid']);barrier.wait(timeout=5)
                if r['uid']=='fake:0': guard.trip('HTTP 429')
                guard.charge(.001)
                return dict(prediction='no-damage',cost_cny=.001)
            with self.assertRaises(AskStopped): x.run_probe(records,backend,tmp,1,concurrency=2)
            self.assertEqual(set(calls),{'fake:0','fake:1'})
            self.assertTrue((Path(tmp)/'answer_parts/fake__1.json').is_file())

    def test_crash_snapshot_cost_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[record(0)]
            write_json(Path(tmp)/'answer_parts/fake__0.json',dict(uid='fake:0',prediction='destroyed',cost_cny=.123))
            with self.assertRaises(ValueError): x.run_probe(records,lambda *_:None,tmp,1)
            result = x.run_probe(records,lambda *_:self.fail('must not call again'),tmp,1,resume=True)
            self.assertEqual(result['answers'],1)
            self.assertAlmostEqual(result['cost_cny'],.123)

    def test_oversized_charge_retained_and_stops_next_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def costly(r,guard):
                calls.append(r['uid']); guard.charge(.03)
                return dict(prediction='no-damage',cost_cny=.03)
            with self.assertRaises(AskStopped):
                x.run_probe([record(0),record(1)],costly,tmp,1,allowance=.02)
            self.assertEqual(calls,['fake:0'])
            self.assertTrue((Path(tmp)/'answer_parts/fake__0.json').is_file())
            self.assertAlmostEqual(json.loads(read_text(Path(tmp)/'budget.json'))['spent_cny'],.03)

    def test_invalid_budget_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cap in (0,-1,float('nan'),float('inf')):
                with self.assertRaises(ValueError):
                    x.run_probe([record(0)],lambda *_:self.fail('must not call'),tmp,cap)

    def test_transient_eio_local_write_never_retries_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path.write_text; failed=[]; calls=[]
            def flaky(path,*args,**kwargs):
                if 'fake__0.json' in path.name and not failed:
                    failed.append(True);raise OSError(5,'I/O error')
                return original(path,*args,**kwargs)
            def backend(r,guard):
                calls.append(r['uid']);guard.charge(.001)
                return dict(prediction='no-damage',cost_cny=.001)
            with patch.object(Path,'write_text',flaky):
                x.run_probe([record(0)],backend,tmp,1)
            self.assertEqual(calls,['fake:0'])
            self.assertEqual(len(failed),1)


if __name__ == '__main__':
    unittest.main()
