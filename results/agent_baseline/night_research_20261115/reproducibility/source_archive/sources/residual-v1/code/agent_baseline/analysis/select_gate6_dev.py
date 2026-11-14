"""Select paired tasks from development histories only, independently of full audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .coordinate_audit import audit,select_dev,write_json,load_journal

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--existing-journal',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    names={Path(json.loads(line)['image_t1']).stem for line in a.manifest.read_text().splitlines() if line.strip()}
    cache={}
    if a.existing_journal and a.existing_journal.exists():
        # Read-only snapshot: never repair/truncate another active job's journal.
        with a.existing_journal.open() as f:
            for line in f:
                try:r=json.loads(line)
                except ValueError:break
                if r['task']=='T1.3' and r['name'] in names and not r['other_bad_args']:cache[r['session']]=r
    journal=a.output/'sessions.jsonl';rows=load_journal(journal);done={r['session'] for r in rows};items=[]
    for config in sorted(a.root.iterdir()):
        if config.name.startswith(('.', 'code')) or 'test' in config.name.lower():continue
        directory=config/'T1.3'/'val'
        if not directory.is_dir():continue
        for name in sorted(names):
            session=directory/name
            if str(session) not in done and (session/'input.json').is_file():items.append((config.name,'T1.3','val',session))
    def work(item):return cache.get(str(item[3])) or audit(item)
    with journal.open('a') as sink,ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(work,items):
            sink.write(json.dumps(r,ensure_ascii=False)+'\n');sink.flush();rows.append(r)
    select_dev(rows,a.manifest,a.output)
    write_json(a.output/'metric.json',dict(historical_dev_sessions=len(rows),selected_tasks=10,cost_cny=0))
    print(f'development histories={len(rows)}; selected=10; independent of full audit')

if __name__=='__main__':main()
