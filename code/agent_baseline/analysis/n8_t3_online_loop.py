"""Opt-in provider and common task input whitelist; scoring paths stay recorder-only."""
import os
from .n8_t3_online_provider import install


def inference_rows(reader):
    def read(*args,**kwargs):
        rows=reader(*args,**kwargs)
        clean=[]
        fields=('id','task_type','image_t1','image_t2','exemplar_mask','gt_mask','gt_ignore_mask','instruction')
        for row in rows:
            if row['target']!='any change' or row['change_direction']!='any':
                raise ValueError('N8 fixed example task must be category-neutral')
            clean.append(dict({k:row[k] for k in fields if k in row},target='any change',change_direction='any',
                constraint=None,init_mask=None,meta={}))
        return clean
    return read


def main():
    if os.environ.get('N8_ONLINE_ARM') not in ('baseline','n8'):raise RuntimeError('Set explicit N8_ONLINE_ARM')
    install()
    from agent_baseline import loop_json
    loop_json.read_tasks=inference_rows(loop_json.read_tasks)
    loop_json.main()


if __name__=='__main__':main()
