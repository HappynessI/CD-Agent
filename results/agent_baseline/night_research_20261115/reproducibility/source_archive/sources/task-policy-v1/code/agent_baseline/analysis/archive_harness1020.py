"""Archive completed paired-run evidence; preserve full original request streams."""
import argparse
import json
from pathlib import Path
import shutil


def archive(output):
    records=[]
    for arm in sorted((output/'arms').glob('*.json')):
        data=json.loads(arm.read_text())
        for row in data['rows']:
            source=Path(row['session'])
            dest=output/'traces'/row['config']/row['task']/row['name']
            dest.mkdir(parents=True,exist_ok=True)
            copied=[]
            for name in ['input.json','final.json','steps.jsonl','verification.json','gate.json']:
                if (source/name).is_file():
                    shutil.copyfile(source/name,dest/name);copied.append(name)
            for directory in ['masks','insts']:
                if (source/directory).is_dir():
                    shutil.copytree(source/directory,dest/directory,dirs_exist_ok=True);copied.append(directory+'/')
            records.append(dict(config=row['config'],task=row['task'],name=row['name'],archived=str(dest),
                full_original=str(source),requests=str(source/'messages.jsonl'),views=str(source/'images'),copied=copied))
    (output/'archive_manifest.json').write_text(json.dumps(dict(sessions=records,
        policy='Original request streams and rendered views remain at their original shared paths; nothing deleted. Input, complete calls, final, verification, masks and instances copied for review.'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(archived_sessions=len(records),api_calls=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);archive(p.parse_args().output)
