"""Count actual prepared candidate request slots without labels or inference."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import write_json


def inventory(root):
    manifest=root/'manifest.json'
    if not manifest.exists():return dict(status='pending_prepare',path=str(manifest))
    rows=json.loads(manifest.read_text());groups={};details=[]
    for row in rows:
        values=dict(images=1,candidates=len(row['candidates']),already_answered=0,recoverable_usage=0,
                    reusable_old_answer=0,new_request_slots=0)
        for candidate in row['candidates']:
            name=candidate['uid'].replace(':','__')+'.json'
            if (root/'answers'/name).exists():values['already_answered']+=1
            elif (root/'usage'/name).exists() and json.loads((root/'usage'/name).read_text()).get('raw'):
                values['recoverable_usage']+=1
            elif candidate.get('old_answer') and Path(candidate['old_answer']).exists():values['reusable_old_answer']+=1
            else:values['new_request_slots']+=1
        key=row['group']+'/'+row['split'];group=groups.setdefault(key,{k:0 for k in values})
        for k,v in values.items():group[k]+=v
        details.append(dict(group=row['group'],stem=row['stem'],split=row['split'],gate=row['s5_gate'],**values))
    total={k:sum(v[k] for v in groups.values()) for k in next(iter(groups.values()),{})}
    return dict(status='prepared',groups=groups,total=total,images=details,
        model='qwen3.7-flash',api_parallelism=1,
        bound='new_request_slots counts candidates; bounded transient retries may add attempts, maximum 3 attempts per slot',
        max_attempts_bound=3*total.get('new_request_slots',0),
        stop_file_exists=(root/'STOP.json').exists(),gt_role='no GT read; inherited tune-selected gate only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run inventory through Slurm')
    write_json(a.out/'request_inventory.json',inventory(a.out))


if __name__=='__main__':main()
