import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.gate6_pair_report import retries,report


class PairReportTests(unittest.TestCase):
    def test_filtered_unchanged_retry_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'input.json').write_text(json.dumps(dict(image_shape=[512,512])))
            args=dict(region=[0,0,768,768])
            call=dict(tool='view',args=args,ok=False,error_code='bad_args',result_text=json.dumps(dict(message='region must be an ordered box inside the full image.')))
            rows=[dict(step=1,calls=[call]),dict(step=2,calls=[],planner=dict(raw=dict(choices=[dict(message=dict(tool_calls=[dict(function=dict(name='view',arguments=json.dumps(args)))]))])))]
            (root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=retries(root)
            self.assertEqual(len(result),1);self.assertTrue(result[0]['filtered_by_harness']);self.assertTrue(result[0]['prior_failure']['coordinate'])

    def test_ask_after_current_answer_repair_is_not_invalid_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'input.json').write_text(json.dumps(dict(image_shape=[512,512])))
            rows=[dict(step=14,current_mask=None,calls=[dict(tool='ask',args={},ok=False,error_code='bad_args',current_before=None)]),
                  dict(step=15,current_mask='mask#9',calls=[dict(tool='select',args={},ok=True,current_before=None,current_after='mask#9')]),
                  dict(step=16,current_mask='mask#9',calls=[dict(tool='ask',args={},ok=True,current_before='mask#9')])]
            (root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(retries(root),[])

    def test_failed_arm_still_preserves_budget_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ledger=root/'budget/gate6_actual';ledger.mkdir(parents=True)
            (ledger/'worker.txt').write_text('0.37')
            manifest=root/'tasks.jsonl';manifest.write_text(json.dumps(dict(id='sample.png'))+'\n')
            with self.assertRaises(SystemExit):report(root,manifest,root/'result')
            metric=json.loads((root/'result/metric.json').read_text())
            self.assertEqual(metric['cost_cny'],.37);self.assertEqual(metric['paired_tasks'],0)


if __name__=='__main__':unittest.main()
