import json, os
from pathlib import Path
out=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115')
job=Path(os.environ.get('EXP_LOOP_JOB','')).name
if 'verify-tune' in job:
 p=out/'verification/tune/summary.json'
elif 'verify-confirm' in job:
 p=out/'verification/confirm/summary.json'
elif 'tradeoff' in job:
 p=out/'tradeoff_diagnostic/summary.json'
elif 'archive' in job:
 p=out/'reproducibility/inventory.json'
else:
 p=out/'all_summary.json'
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(cost_cny=d.get('cost_cny',0),artifact_exists=p.exists(),tasks=d.get('tasks'),new_calls=d.get('new_calls',0),reused_calls=d.get('reused_calls',0))))
