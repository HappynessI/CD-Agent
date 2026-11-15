import json,os
from pathlib import Path
root=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115')
job=Path(os.environ['EXP_LOOP_JOB']).name
if job.startswith(('07-','08-','09-','10-','91-')):
 p=root/'metric.json'
 value=json.loads(p.read_text()) if p.exists() else dict(stage='online_not_reached',cost_cny=json.loads((root/'api_cost_status.json').read_text())['known_cost_cny'],cumulative=True)
 print(json.dumps(value));raise SystemExit(0)
stage='step3/verification/confirm' if job.startswith(('05-','06-','90-')) else 'step1' if job.startswith('01-') else 'step2' if job.startswith('02-') else 'step3/verification/'+('confirm' if 'confirm' in job else 'tune')
p=root/stage/'summary.json'
s=json.loads(p.read_text()) if p.exists() else {}
requests=root/'step3/verification'
answers=list(requests.glob('*/tasks/*/requests/*/answer.json'))
failures=list(requests.glob('*/tasks/*/requests/*/failure_*.json'))
cost=sum(float(json.loads(q.read_text()).get('cost_cny',0)) for q in answers+failures)
ledger_path=root/'api_cost_status.json'
ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
cost=ledger.get('known_cost_cny',cost)
online_check=root/'online/provider_check.json'
check=json.loads(online_check.read_text()) if online_check.exists() else {}
print(json.dumps(dict(online_provider_tasks=check.get('tasks',0),diagnostics_exists=(root/'diagnostics10/index.html').exists(),archive_exists=(root/'reproducibility/inventory.json').exists(),usage_missing_attempts=ledger.get('usage_missing_attempts',0),stage=stage,summary_exists=p.exists(),tasks=s.get('tasks'),new_calls=len(answers),failed_requests=len(failures),cost_cny=cost,cumulative=True)))
