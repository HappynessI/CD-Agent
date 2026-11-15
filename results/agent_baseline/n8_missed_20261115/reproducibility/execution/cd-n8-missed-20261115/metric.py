import json,os
from pathlib import Path
root=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_missed_20261115')
v=json.loads((root/'metric.json').read_text())
job=Path(os.environ['EXP_LOOP_JOB']).name
if job.startswith('07-'):
 p=root/'diagnostic_link_audit.json';audit=json.loads(p.read_text()) if p.exists() else {}
 v.update(stage='mechanism_audit',link_audit_complete=audit.get('complete',False),pair_audit_exists=(root/'paired_input_audit.json').exists())
if job.startswith('90-'):
 p=root/'reproducibility/inventory.json';audit=json.loads(p.read_text()) if p.exists() else {}
 v.update(stage='archive',archive_complete=audit.get('complete',False))
print(json.dumps(v))
