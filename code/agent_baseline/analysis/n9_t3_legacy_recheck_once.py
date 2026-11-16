"""One bounded legacy-file recovery check; no clean-data or model changes."""
import argparse,json,os,signal
from pathlib import Path
import numpy as np
from PIL import Image
from .n9_t3_data import audit_old
from .io_retry import write_json

TASK='train_2020_T33_1to5'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    folder=a.data/'legacy_recheck_once'/TASK;result_path=folder/'result.json'
    if result_path.exists():raise SystemExit('Single attempt already recorded; do not retry automatically')
    status_path=a.data/'legacy_io_status.json';status=json.loads(status_path.read_text())
    deferred=next(r for r in status['deferred'] if r['id']==TASK);row=deferred['original_task']
    write_json(folder/'previous_legacy_status.json',status)
    result=dict(id=TASK,status='unresolved',attempt_state='started',reason='No completed comparison yet; interruption/time limit leaves unresolved',original_mask_unchanged=True,clean_manifests_unchanged=True)
    write_json(result_path,result)
    def timed_out(signum,frame):raise TimeoutError('Single legacy check exceeded 180 seconds')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(180)
    try:
        with Image.open(row['examples'][0]['mask']) as image:original=np.asarray(image)>0
        with Image.open(a.data/'legacy_recovery'/TASK/'recovered_candidate.png') as image:recovered=np.asarray(image)>0
        if original.shape!=recovered.shape or not np.array_equal(original,recovered):
            result.update(attempt_state='finished',reason='Original and independent recovered candidate differ',comparison='different');write_json(result_path,result);return
        partition=json.loads((a.data/'partition.json').read_text());parts={name:part for part,names in partition['query'].items() for name in names}
        records,_=audit_old([row],parts);record=records[0]
        # Stage the single completed audit first; existing deferred evidence remains in previous_legacy_status.json.
        write_json(folder/'verified_audit.json',record)
        checkpoint=a.data/'checkpoints/legacy'/row['task_type']/(TASK+'.json');write_json(checkpoint,record)
        path=a.data/'old_task_audit.json';all_records=json.loads(path.read_text());all_records=[r for r in all_records if r['id']!=TASK];all_records.append(record)
        write_json(path,all_records)
        write_json(a.data/'t3_empty_abstention_diag.json',dict(status='sealed_not_evaluated',tasks=[r for r in all_records if r['query_support']['pixels']==0]))
        remaining=[r for r in status['deferred'] if r['id']!=TASK]
        status.update(status='complete' if not remaining and len(all_records)==status['expected'] else 'incomplete',completed=len(all_records),deferred=remaining)
        status.setdefault('resolved',[]).append(dict(id=TASK,comparison='pixel_equal',evidence=str(folder/'verified_audit.json'),previous_deferred_evidence=str(folder/'previous_legacy_status.json')))
        write_json(status_path,status)
        summary_path=a.data/'summary.json';summary=json.loads(summary_path.read_text());summary.update(legacy_audit_status=status['status'],legacy_audit_completed=len(all_records));write_json(summary_path,summary)
        result.update(status='resolved',attempt_state='finished',reason='Original equals independently recovered candidate; single audit appended',comparison='pixel_equal',original_pixels=int(original.sum()),completed_legacy_tasks=len(all_records),audit_path=str(folder/'verified_audit.json'))
        write_json(result_path,result)
    except Exception as exc:
        result.update(attempt_state='finished',reason=str(exc),error_type=type(exc).__name__);write_json(result_path,result)
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
