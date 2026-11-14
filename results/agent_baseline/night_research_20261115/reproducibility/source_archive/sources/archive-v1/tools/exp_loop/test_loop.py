import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('engine',Path(__file__).with_name('loop.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.exp=Path(self.temp.name);(self.exp/'jobs').mkdir()
        for i in range(4): (self.exp/'jobs'/f'{i}.sbatch').write_text('#!/bin/sh\ntrue\n')
        self.cfg=dict(metric=['python3','metric.py'],partition='gre',notify=False,
                      stop=dict(no_gain_rounds=2,min_gain=1.,budget_cny=10))
        self.config();self.calls=[];self.next_id=100;self.states={};self.metrics=[]
        self.fail_hook=False;self.fail_qq=False;self.fail_parent=False
        self.mock=patch.object(m,'run',self.fake);self.mock.start();self.addCleanup(self.mock.stop)
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__();self.addCleanup(self.output.__exit__,None,None,None)
    def config(self): (self.exp/'exp.json').write_text(json.dumps(self.cfg))
    def fake(self,args,**kwargs):
        self.calls.append(args)
        if args[0]=='scontrol': return 0,'Slurmctld(primary) is UP',''
        if args[0]=='sbatch':
            if self.fail_hook and '/hooks/' in args[-1]: return 1,'','hook rejected'
            if self.fail_parent and '/jobs/' in args[-1]: return 1,'','invalid resource request'
            self.next_id+=1;return 0,str(self.next_id),''
        if args[0]=='sacct': return 0,self.states.get(args[2],'COMPLETED')+'|0:0',''
        if args==self.cfg['metric']:
            item=self.metrics.pop(0)
            return 0,json.dumps(item),''
        if args[0]==m.QQ: return (1,'','mail failed') if self.fail_qq else (0,'sent','')
        raise AssertionError(args)
    def state(self): return m.load(self.exp)
    def end(self,primary,cost=0):
        self.metrics.append(dict(primary=primary,cost_cny=cost));s=self.state();m.advance(self.exp,s['job_id'])
    def next_round(self):
        m.add(self.exp,decision='continue',reason='Next planned comparison is needed to test the hypothesis')
        m.advance(self.exp,'start')
    def conclude(self):
        m.add(self.exp,decision='finish',reason='Evidence sufficient for this test; record conclusion')
    def test_small_gains_and_declines_do_not_stop_research(self):
        m.init(self.exp)
        for primary in (10,10.4,9):
            self.end(primary)
            self.assertEqual(self.state()['status'],'awaiting_agent')
            self.assertFalse(self.state()['finalize_requested'])
            self.next_round()
        self.end(20)
        s=self.state();self.assertEqual(s['rounds'],4)
        self.assertEqual(s['status'],'awaiting_agent');self.assertFalse(s['queue'])
        self.assertNotIn('no_gain',s)
        self.assertEqual(len(s['decisions']),3)
        self.conclude();self.assertEqual(self.state()['status'],'done')
    def test_submit_failure_is_reviewable_and_retryable(self):
        self.fail_parent=True;m.init(self.exp)
        s=self.state();self.assertEqual(s['review']['kind'],'submission_failed')
        self.assertEqual(len(s['queue']),4);self.assertFalse(s['finalize_requested'])
        self.fail_parent=False
        m.add(self.exp,decision='retry',reason='Corrected invalid resource request')
        m.advance(self.exp,'start')
        self.assertEqual(self.state()['rounds_submitted'],1)
    def test_missing_binding_is_not_reported_as_delivered(self):
        m.init(self.exp);self.end(1)
        self.assertEqual(self.state()['review']['delivery'],'pending')
        self.assertIn('NOT reached',self.state()['review']['delivery_error'])
    def test_duplicate_hook_and_reinit_do_not_duplicate_jobs(self):
        m.init(self.exp);old=self.state()['job_id'];self.end(10)
        before=len(self.calls);m.advance(self.exp,old);self.assertEqual(len(self.calls),before)
        with self.assertRaises(ValueError):m.init(self.exp)
    def test_tick_repairs_missing_or_failed_hook_without_experiment(self):
        self.fail_hook=True;m.init(self.exp);s=self.state();self.assertIsNone(s['hook_id'])
        self.fail_hook=False;start=len(self.calls);m.tick(self.exp)
        self.assertTrue(all('/hooks/' in c[-1] for c in self.calls[start:] if c[0]=='sbatch'))
        self.assertEqual(self.state()['rounds'],0)
        failed_hook=self.state()['hook_id'];self.states[failed_hook]='FAILED';m.tick(self.exp)
        self.assertEqual(self.state()['rounds'],0)
        self.assertEqual(self.state()['rounds_submitted'],1)
        queried=[c[2] for c in self.calls if c[0]=='sacct']
        self.assertTrue(all(x==failed_hook for x in queried),queried)
    def test_running_hook_untouched(self):
        m.init(self.exp);self.states[self.state()['hook_id']]='RUNNING';start=len(self.calls)
        m.tick(self.exp);self.assertFalse(any(c[0]=='sbatch' for c in self.calls[start:]))
    def test_estimate_does_not_block_submission(self):
        self.cfg['round_cost_cny']={'0.sbatch':11};self.config();m.init(self.exp)
        self.assertEqual(self.state()['rounds_submitted'],1)
        self.assertTrue(any(c[0]=='sbatch' for c in self.calls))
    def test_cumulative_cost_is_not_charged_again_and_legacy_state_repairs(self):
        m.init(self.exp)
        self.metrics.append(dict(cost_cny=.277627,cumulative=True))
        m.advance(self.exp,self.state()['job_id']);self.next_round()
        self.metrics.append(dict(cost_cny=.513725,cumulative=True))
        m.advance(self.exp,self.state()['job_id'])
        self.assertEqual(self.state()['cost_cny'],.513725)
        self.assertEqual(self.state()['history'][-1]['cost_cny'],.236098)
        s=self.state();s['cost_cny']=.791352;m.save(self.exp,s)
        self.next_round()
        self.assertEqual(self.state()['cost_cny'],.513725)
        self.metrics.append(dict(cost_cny=.8,cumulative=True))
        m.advance(self.exp,self.state()['job_id'])
        self.assertEqual(self.state()['cost_cny'],.8)

    def test_actual_budget_stops(self):
        m.init(self.exp);self.end(10,10);self.assertEqual(self.state()['rounds_submitted'],1)
    def test_diagnostic_does_not_require_scalar_score(self):
        m.init(self.exp);self.metrics.append({'checks_passed':17});m.advance(self.exp,self.state()['job_id'])
        s=self.state();self.assertEqual(s['status'],'awaiting_agent')
        self.assertIsNone(s['history'][0]['metric_error'])
    def test_invalid_metric_requests_review(self):
        m.init(self.exp);self.metrics.append({'primary':'not a number'});m.advance(self.exp,self.state()['job_id'])
        self.assertEqual(self.state()['status'],'awaiting_agent')
        self.assertIn('finite number',self.state()['review']['error'])
        self.assertFalse(self.state()['finalize_requested'])
    def test_failed_parent_preserves_cost_and_allows_repair_retry(self):
        m.init(self.exp);self.states[self.state()['job_id']]='TIMEOUT';self.end(2,.3)
        self.assertEqual(self.state()['status'],'awaiting_agent');self.assertEqual(self.state()['cost_cny'],.3)
        m.add(self.exp,decision='retry',reason='Resume saved checkpoint with corrected time limit')
        m.advance(self.exp,'start');self.end(2,.2)
        self.assertEqual(self.state()['cost_cny'],.5)
        self.assertEqual(self.state()['rounds'],2)
    def test_add_after_queue_drained_uses_start_hook(self):
        self.cfg['jobs_glob']='jobs/0.sbatch';self.config();m.init(self.exp);self.end(10);self.conclude()
        m.finalize(self.exp);self.assertTrue(self.state()['finalized'])
        m.add(self.exp,self.exp/'jobs/1.sbatch');s=self.state()
        self.assertFalse(s['finalized']);self.assertIsNone(s['job_id']);self.assertTrue(s['hook_id'])
        m.advance(self.exp,'start');self.assertEqual(self.state()['rounds_submitted'],2)
    def test_add_does_not_bypass_budget(self):
        m.init(self.exp);self.end(10,10)
        (self.exp/'extra.sbatch').write_text('true')
        with self.assertRaisesRegex(ValueError,'budget'):m.add(self.exp,self.exp/'extra.sbatch',reason='more work')
    def test_recorded_event_recovery_does_not_double_bill(self):
        m.init(self.exp);s=self.state();self.metrics.append(dict(primary=5,cost_cny=1))
        m.record(self.exp,self.cfg,s,'COMPLETED','0:0')
        m.advance(self.exp,s['job_id']);self.assertEqual(self.state()['cost_cny'],1)
        self.assertEqual(self.state()['rounds'],1);self.assertEqual(self.state()['rounds_submitted'],1)
    def test_qq_failure_remains_pending_then_retries(self):
        self.cfg['notify']=True;self.cfg['finalize_retry_seconds']=0;self.config()
        m.init(self.exp);self.end(10,10);self.conclude();self.fail_qq=True
        self.assertEqual(m.finalize(self.exp),1);self.assertFalse(self.state()['finalized'])
        self.fail_qq=False;m.finalize(self.exp);self.assertTrue(self.state()['finalized'])
        before=len(self.calls);m.finalize(self.exp);self.assertEqual(len(self.calls),before)
    def test_push_failure_is_retryable_without_recommit(self):
        self.cfg['finalize_retry_seconds']=0;self.config();m.init(self.exp);self.end(10,10);self.conclude()
        def git_fail(exp,cfg,s):
            s['git_committed']=True;m.save(exp,s);raise ValueError('push failed')
        with patch.object(m,'git_finalize',git_fail):m.finalize(self.exp)
        self.assertFalse(self.state()['finalized']);self.assertTrue(self.state()['git_committed'])
        with patch.object(m,'git_finalize') as g:m.finalize(self.exp)
        self.assertTrue(self.state()['finalized']);self.assertEqual(g.call_count,1)
    def test_retry_limit_reports_unfinished(self):
        self.cfg.update(notify=True,finalize_retry_seconds=0,max_finalize_attempts=2);self.config()
        m.init(self.exp);self.end(1,10);self.conclude();self.fail_qq=True
        m.finalize(self.exp);m.finalize(self.exp);before=len(self.calls);m.finalize(self.exp)
        self.assertTrue(self.state()['finalize_blocked']);self.assertFalse(self.state()['finalized'])
        self.assertEqual(len(self.calls),before)
    def test_status_never_contacts_slurm(self):
        m.init(self.exp);before=len(self.calls);m.status(self.exp);self.assertEqual(len(self.calls),before)

class GitFinalizationTests(unittest.TestCase):
    def test_push_retry_preserves_unrelated_staged_file(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repo=root/'repo';repo.mkdir();exp=root/'exp';exp.mkdir()
            def git(*args):
                return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-b','research');git('config','user.name','Loop Test');git('config','user.email','loop@example.invalid')
            (repo/'initial').write_text('initial');git('add','initial');git('commit','-m','initial')
            (repo/'user.txt').write_text('user change');git('add','user.txt')
            (repo/'result.txt').write_text('experiment result')
            cfg=dict(git=dict(repo=str(repo),branch='research',paths=['result.txt']))
            state=dict(name='test',rounds=3);attempts=[];original=m.run
            def commands(args,**kw):
                if args[0]==m.WGIT:
                    attempts.append(args)
                    return (1,'','network failure') if len(attempts)==1 else (0,'','')
                return original(args,**kw)
            with patch.object(m,'run',commands):
                with self.assertRaises(ValueError):m.git_finalize(exp,cfg,state)
                self.assertTrue(state['git_committed']);self.assertNotIn('git_pushed',state)
                m.git_finalize(exp,cfg,state)
            self.assertEqual(git('rev-list','--count','HEAD'),'2')
            self.assertEqual(git('diff','--cached','--name-only'),'user.txt')
            self.assertTrue(state['git_pushed'])

if __name__=='__main__':unittest.main()
