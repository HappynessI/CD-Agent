import copy
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
from PIL import Image
from agent_baseline.backends import SegmentBackend, SharedPrompter, ApiAsk, yes_probability
from agent_baseline.planner import Planner, PlannerError
from agent_baseline.recorder import Recorder
from agent_baseline.state import ToolError
from agent_baseline.workers.ask_worker import MicroBatch
from agent_baseline.workers.sam3_worker import Segmenter
from prompts import VERIFY
from .helpers import mask, tempdir, rows, rgb, run, call, response


def raw():
    return dict(id='fake',choices=[dict(message=dict(role='assistant',content='{"change":"yes","type":"new_building","reason":"A new roof."}'),
        logprobs=dict(content=[dict(token='yes',top_logprobs=[dict(token='yes',logprob=-.2),dict(token='no',logprob=-2.)])]))],
        usage=dict(prompt_tokens=100,completion_tokens=20,prompt_tokens_details=dict(cached_tokens=10)))


class BackendTests(unittest.TestCase):
    def test_sam3_worker_packed_cache_with_fake_processor(self):
        class Tensor:
            def __init__(self,data):self.data=np.asarray(data)
            def __len__(self):return len(self.data)
            def __getitem__(self,key):return Tensor(self.data[key])
            def cpu(self):return self
            def float(self):return self
            def numpy(self):return self.data
        class Processor:
            def set_image(self,image):
                self.shape=(image.height,image.width)
                return {}
            def set_text_prompt(self,prompt,state):
                return dict(masks=Tensor(np.ones((1,1,*self.shape),bool)),scores=Tensor([.75]))
        class Engine:
            autocast=staticmethod(nullcontext)
        class Torch:
            inference_mode=staticmethod(nullcontext)
        with tempdir() as root, patch.dict('sys.modules',torch=Torch()), patch(
                'agent_baseline.workers.sam3_worker.load_rgb',return_value=rgb()) as loader:
            processor=Processor()
            worker=Segmenter(processor,Engine(),root)
            a=worker.infer('synthetic','image.png','t2','tree',0,.1)
            b=worker.infer('synthetic','image.png','t2','tree',0,.1)
            self.assertFalse(a['cache_hit'])
            self.assertTrue(b['cache_hit'])
            self.assertEqual(loader.call_count,1)
            with np.load(a['path']) as z:
                self.assertEqual(str(z['text']),'tree')
                self.assertEqual(float(z['conf']),.1)
                self.assertEqual(int(z['tile']),0)
                self.assertEqual(np.unpackbits(z['masks'],axis=1,count=mask().size).sum(),mask().size)
            processor.set_text_prompt=lambda **kw:dict(scores=Tensor([]))
            empty=worker.infer('synthetic','image.png','t2','empty class',0,.1)
            with np.load(empty['path']) as z:
                self.assertEqual(len(z['scores']),0)
                self.assertEqual(z['masks'].shape,(0,(mask().size+7)//8))

    def test_retries_status_and_attempt_limits(self):
        events, waits, payloads = [],[],[]
        def client(payload):
            payloads.append(payload)
            if len(payloads)==1:
                raise urllib.error.HTTPError('unused',429,'busy',{'Retry-After':'3'},None)
            if len(payloads)==2:
                raise TimeoutError()
            return raw()
        planner=Planner(client=client,record=events.append,sleep=waits.append,jitter=lambda:0)
        result=planner.complete([dict(role='user',content='Inspect.')],[],1)
        self.assertEqual((planner.attempts,result['attempts'],len(events)),(3,3,3))
        self.assertEqual(waits,[3,2])
        self.assertEqual(planner.tokens['input'],100)
        self.assertNotIn('tools',payloads[-1])
        self.assertNotIn('Authorization',json.dumps(events))
        def fail(payload):
            raise TimeoutError()
        events=[]
        planner=Planner(client=fail,record=events.append,sleep=lambda n:None,max_attempts=5)
        with self.assertRaises(PlannerError):
            planner.complete([],[],1)
        self.assertEqual(len(events),5)
        planner=Planner(client=fail,record=events.append,sleep=lambda n:None)
        with self.assertRaises(PlannerError):
            planner.complete([],[],1)
        self.assertEqual(planner.attempts,6)

    def test_nonretryable_and_record_failure(self):
        def denied(payload):
            raise urllib.error.HTTPError('unused',400,'bad request',{},None)
        planner=Planner(client=denied,sleep=lambda n: self.fail('unexpected retry'))
        with self.assertRaises(PlannerError):
            planner.complete([],[],1)
        self.assertEqual(planner.attempts,1)
        def broken_record(event):
            raise OSError('disk error')
        planner=Planner(client=lambda payload:raw(),record=broken_record)
        with self.assertRaises(OSError):
            planner.complete([],[],1)
        self.assertEqual(planner.attempts,1)

    def test_cache_dispatch_exact_thresholds(self):
        import sam3_t7
        fake=[(np.array([mask()]),np.array([.8])),(np.array([mask()]),np.array([.9]))]
        backend=SegmentBackend('synthetic','image.png')
        with patch.object(sam3_t7,'read',return_value=fake) as read:
            result,meta=backend('t1',' B uilding ',512,.3)
            read.assert_called_once_with('synthetic','image.png',512,conf=.1,min_score=.3)
            self.assertTrue(meta['cache_hit'])
        with patch.object(sam3_t7,'read',return_value=fake) as read:
            backend('t2','building',0,.6)
            read.assert_called_once_with('synthetic','image.png',0,conf=.5,min_score=.6)
        for tile,thr,text in [(0,.4,'building'),(512,.05,'building'),(512,.5,'tree')]:
            with self.assertRaises(ToolError) as caught:
                backend('t1',text,tile,thr)
            self.assertEqual(caught.exception.code,'backend_unavailable')

    def test_worker_dispatch_and_empty_cache(self):
        with tempdir() as root:
            path=root+'/worker.npz'
            np.savez(path,shape=mask().shape,masks=np.zeros((0,(mask().size+7)//8),np.uint8),scores=np.array([]))
            with patch('agent_baseline.backends.post',return_value=dict(ok=True,path=path,cache_hit=False)) as post:
                result,meta=SegmentBackend('synthetic','image.png','fake')('t1','tree',512,.05)
            self.assertEqual(result,[])
            self.assertEqual(post.call_args.args[1]['conf'],.05)

    def test_shared_prompter_serializes_full_call(self):
        calls=[]
        class Fake:
            def box(self,split,name,t,box):
                calls.append(('start',name))
                time.sleep(.005)
                calls.append(('end',name))
                return mask(box),.8
        shared=SharedPrompter(factory=Fake)
        threads=[threading.Thread(target=shared.for_image('synthetic',n),args=('t1',),kwargs=dict(box=[0,0,2,2])) for n in ['a','b']]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual([x[0] for x in calls],['start','end','start','end'])
        self.assertEqual(calls[0][1],calls[1][1])

    def test_soft_cache_keeps_the_grid_and_applies_thresholds(self):
        from agent_baseline.backends import read_soft
        with tempdir() as root:
            folder=Path(root)/'sam3_soft'/'t512s256'/'synthetic'
            folder.mkdir(parents=True)
            # Instance 0 on the 512 grid, instance 1 on an overlap tile, instance 2 below the score threshold.
            np.savez(folder/'a.npz',shape=np.array([1024,1024]),
                scores0=np.array([.6,.9,.2],np.float32),origins0=np.array([[512,0],[256,0],[0,0]]),offs0=np.array([0,3,4,5]),
                idx0=np.array([1024*2+600,1024*1+600,1024*3+700,7,8],np.int32),val0=np.array([250,200,240,255,255],np.uint8))
            rows_=read_soft('synthetic','a.png',0,.85,.3,root)
            self.assertEqual(len(rows_),1)
            self.assertEqual(rows_[0]['idx'].tolist(),[1024*2+600,1024*3+700])
            self.assertEqual(rows_[0]['bbox'],[600,2,701,4])
            self.assertEqual(rows_[0]['area'],2)
            self.assertEqual(len(read_soft('synthetic','a.png',0,.5,.3,root)[0]['idx']),3)

    def test_api_ask_readout_and_cost(self):
        payloads=[]
        backend=ApiAsk(client=lambda p:(payloads.append(p) or raw()))
        pair=[Image.fromarray(im).resize((384,384)) for im in rgb()]
        answers,meta=backend([pair],VERIFY)
        self.assertEqual(answers[0]['answer'],'yes')
        self.assertGreater(answers[0]['p_yes'],.8)
        self.assertEqual(payloads[0]['model'],'qwen3.7-plus')
        self.assertTrue(payloads[0]['logprobs'])
        self.assertEqual(payloads[0]['top_logprobs'],5)
        self.assertGreater(meta['cost_cny'],0)
        self.assertIsNone(yes_probability({'logprobs':None}))
        self.assertNotIn('data:image',json.dumps(meta))
        with tempdir() as root:
            recorder=Recorder(root,'synthetic','image.png',gt=mask())
            recorder.archive_images(meta)

    def test_api_ask_correspondence_question(self):
        from agent_baseline.backends import CORRESPONDENCE
        payloads=[]
        four=[Image.fromarray(im).resize((384,384)) for im in rgb()]*2
        ApiAsk(client=lambda p:(payloads.append(p) or raw()),model='qwen3.7-flash',correspondence=True)([four],VERIFY)
        self.assertEqual(payloads[0]['model'],'qwen3.7-flash')
        self.assertEqual(payloads[0]['messages'][1]['content'][-1]['text'],CORRESPONDENCE)

    def test_api_ask_context_order_and_failure_accounting(self):
        payloads=[]
        backend=ApiAsk(client=lambda p:(payloads.append(p) or raw()))
        four=[Image.fromarray(im).resize((384,384)) for im in rgb()]*2
        answers,meta=backend([four,four,four],VERIFY)
        self.assertEqual(len(answers),3)
        text=[b['text'] for b in payloads[0]['messages'][1]['content'] if b['type']=='text']
        self.assertEqual(text[:4],['T1 close-up','T2 close-up','T1 wider view','T2 wider view'])
        self.assertEqual(meta['tokens']['input'],3*raw()['usage']['prompt_tokens'])
        def flaky(p,calls=[]):
            calls.append(1)
            if len(calls)==2:
                raise ValueError('bad reply')
            return raw()
        with self.assertRaises(Exception) as caught:
            ApiAsk(client=flaky)([four,four,four],VERIFY)
        self.assertGreater(caught.exception.backend_meta['cost_cny'],0)

    def test_microbatch_order_limit_and_wait(self):
        sizes=[]
        class Fake:
            def verify(self,pairs):
                sizes.append(len(pairs))
                return [dict(answer='yes',p_yes=.9,reason=str(p[0])) for p in pairs]
        batcher=MicroBatch(Fake())
        try:
            futures=[batcher.submit((i,i)) for i in range(10)]
            results=[f.result(timeout=2) for f in futures]
        finally:
            batcher.close()
        self.assertEqual([r['reason'] for r in results],[str(i) for i in range(10)])
        self.assertEqual(sizes,[8,2])
        self.assertTrue(all(r['queue_seconds']>=0 for r in results))


if __name__ == '__main__':
    unittest.main()
