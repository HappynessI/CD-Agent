"""Incremental completion/usage ledger, retaining interruptions and normal fallback."""
import argparse,json
from pathlib import Path
from .n8_t3_online_metrics import session_cost,completion
from .io_retry import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--expect',action='store_true');a=p.parse_args()
    rows=[json.loads(s) for s in a.tasks.read_text().splitlines() if s];records=[]
    for row in rows:
        folder=a.root/a.config/row['task_type']/'val'/row['id'];final=folder/'final.json'
        if a.prepare and folder.exists() and not final.exists():
            dst=a.root/'interrupted_attempts'/a.config/row['task_type']/row['id'];dst.mkdir(parents=True,exist_ok=True)
            folder.rename(dst/f'attempt_{len(list(dst.glob("attempt_*"))):03d}')
        value=json.loads(final.read_text()) if final.exists() else None
        records.append(dict(id=row['id'],task_type=row['task_type'],state=completion(value),usage=session_cost(folder),final=str(final)))
    interrupted=[dict(path=str(f.parent),usage=session_cost(f.parent)) for f in (a.root/'interrupted_attempts'/a.config).glob('**/messages.jsonl')]
    recorded=sum(r['usage']['cost_cny'] for r in records)+sum(r['usage']['cost_cny'] for r in interrupted)
    ledger_files=list((a.root/'budget'/a.config).glob('*.txt'))
    ledger=sum(float(f.read_text()) for f in ledger_files)
    write_json(a.root/(a.config+'_status.json'),dict(records=records,interrupted=interrupted,completed=sum(r['state'] not in ('missing_final','error') for r in records),
        recorded_usage_cost_cny=recorded,run_guard_ledger_cny=ledger,ledger_files=[str(f) for f in ledger_files],
        known_cost_cny=max(recorded,ledger),ledger_unattributed_difference_cny=max(0.,ledger-recorded),
        cost_policy='One config/seed only, including its interrupted attempts. max(recorded usage, unique per-process RunGuard ledger); difference is known spending without attributed usage, not fabricated tokens.'))
    if a.expect and any(r['state'] in ('missing_final','error') for r in records):raise SystemExit('Engineering incomplete/error; retain records, do not replace normal no_submit outcomes')

if __name__=='__main__':main()
