import unittest
from agent_baseline.analysis.n8_t3_online_loop import inference_rows
from agent_baseline.tasks import instruction_for
from agent_baseline.state import Session
from agent_baseline.loop_json import system_prompt

class InputTests(unittest.TestCase):
    def test_same_neutral_prompt_and_recorder_only_truth(self):
        original=dict(id='example',task_type='T3.1',target='any change',change_direction='any',
            instruction='Find all land-cover changes with the same transition as the marked example. Output a binary mask.',
            image_t1='/rgb/a',image_t2='/rgb/b',exemplar_mask='/example',gt_mask='/private_truth',
            meta={'transition':'PRIVATE_ENDPOINT_A -> PRIVATE_ENDPOINT_B'},source_classes=['PRIVATE_ENDPOINT_A'],
            constraint=None,init_mask=None)
        row=inference_rows(lambda *a,**k:[original])('tasks') [0]
        self.assertEqual(row['gt_mask'],'/private_truth');self.assertEqual(row['meta'],{})
        self.assertNotIn('source_classes',row)
        session=Session((2,2),target=row['target'],change_direction=row['change_direction'],instruction=row['instruction'])
        text=system_prompt('v4',30,session)[2]+'\n'+instruction_for('T3.1',session)
        self.assertNotIn('PRIVATE_ENDPOINT',text);self.assertNotIn('/private_truth',text)
        self.assertEqual(original['instruction'],instruction_for('T3.1',session))


class MetricsTests(unittest.TestCase):
    def test_fallback_is_distinct_and_usage_failure_is_counted(self):
        import tempfile,json
        from pathlib import Path
        from agent_baseline.analysis.n8_t3_online_metrics import completion,session_cost,support_summary
        self.assertEqual(completion(dict(termination_reason='budget',submitted_by='fallback')),'fallback')
        self.assertEqual(completion(dict(error='API error')),'error')
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            event=dict(request_id='paid',request={'model':'qwen3.7-flash'},usage={'prompt_tokens':100,'completion_tokens':10},error_type='MalformedResponse')
            (folder/'messages.jsonl').write_text('\n'.join(map(json.dumps,[event,event,{'error_type':'ConnectionError'}])))
            bill=session_cost(folder)
            self.assertEqual(bill['known_requests'],1);self.assertGreater(bill['known_failed_cost_cny'],0)
            self.assertEqual(bill['usage_missing_failures'],1)
        result=support_summary([dict(counts=[0,0,16]),dict(counts=[0,3,0])])
        self.assertEqual(result['positive']['empty_predictions'],1)
        self.assertEqual(result['empty']['fp'],3)

if __name__=='__main__':unittest.main()
