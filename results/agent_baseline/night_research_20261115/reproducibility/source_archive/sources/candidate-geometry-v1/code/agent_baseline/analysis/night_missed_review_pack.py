"""Fixed no-GT visual review contacts; run through Slurm, original inputs untouched."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw
from .io_retry import write_json


def load(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--night',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Render on a Slurm compute node')
    output=a.night/'review_pack';output.mkdir(parents=True,exist_ok=True)
    rows=load(a.night/'manifest.json');first={}
    for row in rows:
        if row['split']=='tune':first.setdefault(row['group'],row)
    manifest=[]
    for group,row in first.items():
        candidates=row['candidates']
        chosen=candidates[:1]
        secondary=next((c for c in candidates[1:] if 'S5' in c['sources']),None)
        if secondary is None:secondary=next(iter(candidates[1:]),None)
        if secondary:chosen.append(secondary)
        for candidate in chosen:
            uid=candidate['uid'];name=uid.replace(':','__');canvas=Image.new('RGB',(4*384,5*416),'white');draw=ImageDraw.Draw(canvas)
            entries=[]
            for i,variant in enumerate(('V0','V1','V2','V3','V4')):
                answer_path=(a.pilot/'answers'/'four' if variant=='V0' else a.night/'answers'/variant)/(name+'.json')
                if not answer_path.exists():
                    entries.append(dict(variant=variant,status='not_available'));continue
                answer=load(answer_path)
                directory=Path(answer.get('input_path') or (a.night/'inputs'/variant/name))
                if variant=='V0' and not answer.get('input_path'):directory=a.pilot/'answers'/'four'/'inputs'/name
                draw.text((8,i*416+4),uid+' | '+variant+' | T1 close / T2 close / T1 wide / T2 wide',fill='black')
                for j in range(4):
                    with Image.open(directory/f'{j}.jpg') as im:canvas.paste(im.convert('RGB'),(j*384,i*416+32))
                entries.append(dict(variant=variant,request=str(directory/'request.json'),answer=str(answer_path),images=[str(directory/f'{j}.jpg') for j in range(4)]))
            path=output/(name+'.png');canvas.save(path)
            manifest.append(dict(uid=uid,source=candidate['source'],area=candidate['area'],contact=str(path),variants=entries))
    write_json(output/'manifest.json',dict(selection='first tune image per group; first rank and first subsequent S5 candidate (else rank2); no GT/answer-based selection',
        blind_display='contact sheets omit answers; separate paths retain evidence',cases=manifest))

if __name__=='__main__':main()
