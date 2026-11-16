"""N9 approved-manifest launcher. GT paths remain recorder-only."""
import os
import sys
import json
from pathlib import Path
from .n9_t3_online_adapter import install

FIELDS=('id','task_type','image_t1','image_t2','exemplar_mask','gt_mask','gt_ignore_mask','instruction')
def inference_rows(reader):
    def read(*args,**kwargs):
        result=[]
        for row in reader(*args,**kwargs):
            if row['task_type'] not in ('T3.1','T3.3'):raise ValueError('N9 T3 only')
            clean={k:row[k] for k in FIELDS if k in row}
            clean.update(target='any change',change_direction='any',constraint=None,init_mask=None,meta={})
            if row.get('examples'):
                clean['examples']=[{k:e[k] for k in ('role','image_t1','image_t2','mask') if k in e} for e in row['examples']]
            result.append(clean)
        return result
    return read

def main():
    from agent_baseline import loop_json
    from .n9_t3_online_manifest import validate_frozen
    root=Path(os.environ['N9_FROZEN_MANIFEST']);value=validate_frozen(root)
    if Path(sys.argv[sys.argv.index('--tasks')+1]).resolve()!=(root/'tasks.jsonl').resolve():raise ValueError('Use frozen manifest')
    if json.loads(Path(os.environ['N9_INFERENCE_CONFIG']).read_text())!=value['configuration']:raise ValueError('Configuration differs from freeze')
    install();loop_json.read_tasks=inference_rows(loop_json.read_tasks);loop_json.main()

if __name__=='__main__':main()
