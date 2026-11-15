import json,os
from pathlib import Path
root=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115')
job=Path(os.environ['EXP_LOOP_JOB']).name
stage='step3/verification/confirm' if job.startswith(('05-','90-')) else 'step1' if job.startswith('01-') else 'step2' if job.startswith('02-') else 'step3/verification/'+('confirm' if 'confirm' in job else 'tune')
p=root/stage/'summary.json'
s=json.loads(p.read_text()) if p.exists() else {}
requests=root/'step3/verification'
answers=list(requests.glob('*/tasks/*/requests/*/answer.json'))
failures=list(requests.glob('*/tasks/*/requests/*/failure_*.json'))
cost=sum(float(json.loads(q.read_text()).get('cost_cny',0)) for q in answers+failures)
ledger_path=root/'api_cost_status.json'
ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
cost=ledger.get('known_cost_cny',cost)
print(json.dumps(dict(diagnostics_exists=(root/'diagnostics10/index.html').exists(),archive_exists=(root/'reproducibility/inventory.json').exists(),usage_missing_attempts=ledger.get('usage_missing_attempts',0),stage=stage,summary_exists=p.exists(),tasks=s.get('tasks'),new_calls=len(answers),failed_requests=len(failures),cost_cny=cost,cumulative=True)))
