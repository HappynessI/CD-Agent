"""Slurm delivery checks using recorded QC fields, without rescoring pixels."""
import argparse
import json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links += [v for k,v in attrs if k in ('href','src') and v]

def check(data, review):
    read=lambda p:json.loads(p.read_text())
    partition=read(data/'partition.json')
    groups=[set(v) for section in partition.values() for v in section.values()]
    for i,a in enumerate(groups):
        for b in groups[i+1:]: assert not a&b, 'Image partition overlap'
    counts={}
    for kind in ('t3_1','t3_3'):
        for split in ('tune','confirm','report'):
            name=f'{kind}_clean_{split}'
            rows=read(data/(name+'.json')); counts[name]=len(rows)
            assert len({r['id'] for r in rows})==len(rows)
            for r in rows:
                m=r['meta'];g=m['geometry'];s=m['query_support']
                assert r['review_status']==m['review_status']=='pending_review'
                assert m['split']==split and Path(r['image_t1']).name in partition['query'][split]
                assert 64<=g['area']<=4096 and g['hole_ratio']<=.1 and g['bbox_fill']>=.25 and g['components']==1
                assert s['pixels']>=64 and s['largest_component']>=32
                assert all(Path(r[k]).is_file() for k in ('image_t1','image_t2','gt_mask'))
                if kind=='t3_1': assert Path(r['exemplar_mask']).is_file()
                else:
                    assert m['example_provenance'][0]['name'] in partition['external_sources'][split]
    cases=read(review/'review_manifest.json');assert len(cases)==6
    parser=Links();parser.feed((review/'index.html').read_text())
    checked=0
    for link in parser.links:
        part=urlsplit(link)
        if part.scheme or part.netloc or not part.path: continue
        assert (review/unquote(part.path)).is_file(),link
        checked+=1
    result=dict(status='passed',counts=counts,review_cases=6,local_links_checked=checked,
                status_remains='pending_review',scope='Recorded QC geometry and partition checks; no new pixel computation or approval')
    (data/'delivery_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();check(a.data,a.review)
