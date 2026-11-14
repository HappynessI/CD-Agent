import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.missed_online_runner import prepare,common_arguments,arm_arguments,successful,SEEDS,GROUPS,SOURCES
from agent_baseline.analysis.missed_online_report import report,ledger_cost,complete_final,lines

class MissedOnlineTest(unittest.TestCase):
    def test_real_process_ledger_and_legacy_file_not_double_counted(self):
        with TemporaryDirectory() as tmp:
            run=Path(tmp);directory=run/'budget'/'missed_online';directory.mkdir(parents=True)
            (directory/'gpu42-1.txt').write_text('0.7')
            (directory/'gpu42-2.txt').write_text('0.3')
            self.assertAlmostEqual(ledger_cost(run)[0],1.)
            (run/'budget'/'missed_online.txt').write_text('1.0')
            self.assertAlmostEqual(ledger_cost(run)[0],1.)

    def test_error_final_is_not_completion_even_with_eval(self):
        self.assertFalse(complete_final(dict(submitted_by='error',eval=dict(tp=1,fp=1,fn=1))))
        self.assertTrue(complete_final(dict(submitted_by='system',fallback_kind='current',eval=dict(tp=1,fp=1,fn=1))))

    def test_partial_jsonl_tail_is_evidence_not_report_failure(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'messages.jsonl';path.write_text('{"raw":"text"}\n{"raw":')
            errors=[];rows=lines(path,errors)
            self.assertEqual(rows,[dict(raw='text')]);self.assertEqual(errors[0]['line'],2)
            self.assertEqual(errors[0]['kind'],'parse_error')

    def test_default_tool_difference_is_explicit(self):
        common=common_arguments('tasks','config',1016)
        self.assertEqual(common[common.index('--concurrency')+1],'1')
        self.assertEqual(common[common.index('--planner-model')+1],'qwen3.7-flash')
        self.assertEqual(arm_arguments('baseline','structured','cache'),[])
        self.assertEqual(arm_arguments('find_missed','structured','cache'),
            ['--missed-candidates-root','cache','--missed-verifier-variant','structured'])

    def test_prepare_uses_all_sources_without_reading_gt(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);selection=root/'selection';selection.mkdir();cache=root/'cache';out=root/'out'
            selected={}
            for group in GROUPS:
                rows=[];selected[group]=['one','two']
                for stem in selected[group]:
                    rows.append(dict(id=stem,task_type='T5.1',init_mask='/not/read/initial.png',
                        image_t1='/not/read/A.png',image_t2='/not/read/B.png',gt_mask='/must/not/read/gt.png'))
                    for source in SOURCES:
                        path=cache/'candidates'/group/source/(stem+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        mask=np.zeros((8,8),bool);mask[0,0]=True
                        np.savez(path,masks=np.asarray([np.packbits(mask)]),shape=[8,8])
                (selection/(group+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (selection/'design.json').write_text(json.dumps(dict(selection_seed=1016,seed_offsets=list(SEEDS),selected=selected)))
            args=argparse.Namespace(selection=selection,cache=cache,out=out,variant='structured',code=root/'code'/'agent_baseline',python='python',ovcd=root/'ovcd')
            with patch.dict(os.environ,SLURM_JOB_ID='unit-test'):prepare(args)
            self.assertEqual(len(list((out/'candidate_manifest').glob('*/*.json'))),6)
            self.assertEqual(len(list((out/'jobs').glob('*.sbatch'))),9)
            archived=json.loads((out/'candidate_manifest'/GROUPS[0]/'one.json').read_text())
            self.assertEqual(archived['candidates'][0]['sources'],list(SOURCES))
            self.assertNotIn('gt_mask',archived)

    def test_report_retains_failed_cost_and_deduplicates_backend_copy(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'online';out.mkdir();task=dict(group='levir_any',stem='one',task_type='T5.1')
            (out/'settings.json').write_text(json.dumps(dict(tasks=[task],seeds=[1016],arms=['baseline','find_missed'])))
            directory=out/'run'/'levir_any'/'seed1016'/'find_missed'/'one'/'attempt_000'/'T5.1'/'val'/'one'
            directory.mkdir(parents=True)
            event=dict(request_id='request-one',usage=dict(prompt_tokens=1000,completion_tokens=100))
            (directory/'messages.jsonl').write_text(json.dumps(event)+'\n'+json.dumps(dict(raw='plain assistant text'))+'\n{"partial":')
            backend=directory/'missed_verifier'/'000001';backend.mkdir(parents=True)
            (backend/'response.json').write_text(json.dumps(event))
            result=report(out)
            self.assertEqual(result['completed'],0)
            self.assertAlmostEqual(result['usage_derived_cost_cny'],0.00028)
            self.assertEqual(len(result['sessions'][1]['attempts']),1)
            self.assertFalse(successful(directory.parent.parent.parent,'T5.1'))
            self.assertEqual(len(result['parse_errors']),1)
            # Root publication is explicit; report-only calls do not touch offline metric.
            rootmetric=out.parent/'metric.json'
            self.assertFalse(rootmetric.exists())
            offline=out.parent/'smoke'/'answers'/'four'/'usage';offline.mkdir(parents=True,exist_ok=True)
            (offline/'one.json').write_text(json.dumps(dict(cost_cny=.01,raw=dict(id='offline-one'))))
            report(out,publish_root=True)
            combined=json.loads(rootmetric.read_text())
            self.assertAlmostEqual(combined['cost_cny'],.01028)

if __name__=='__main__':unittest.main()
