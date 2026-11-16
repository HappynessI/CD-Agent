"""Opt-in process-local integration; no changes to default Agent entry or tools."""
from contextlib import contextmanager
import os
from .n8_missed_assets import read


@contextmanager
def installed(runtime):
    from agent_baseline import loop_json, tools
    from agent_baseline.n9_missed_online_tool import N9Toolbox, schema
    previous=(loop_json.Toolbox,loop_json.run_session,tools.find_missed_schema,
              tools.SCHEMA_BY_NAME['find_missed'],N9Toolbox.runtime)
    N9Toolbox.runtime=runtime
    loop_json.Toolbox=N9Toolbox
    if runtime['arm']=='find_missed':
        tools.find_missed_schema=lambda: schema(runtime['config']['k'])
        tools.SCHEMA_BY_NAME['find_missed']=schema(runtime['config']['k'])['function']['parameters']
    original=loop_json.run_session
    def run_session(*args,**kwargs):
        meta=kwargs.setdefault('metadata',{})
        meta.setdefault('backends',{}).update(n9_profile=runtime['config']['profile'], n9_arm=runtime['arm'],
            n9_find_missed=runtime['arm']=='find_missed', n9_config=runtime['config'],
            n9_current_mask_reconstruction=runtime['arm']=='find_missed')
        return original(*args,**kwargs)
    loop_json.run_session=run_session
    try:
        yield loop_json
    finally:
        loop_json.Toolbox,loop_json.run_session,tools.find_missed_schema,tools.SCHEMA_BY_NAME['find_missed'],N9Toolbox.runtime=previous


def main():
    runtime=read(os.environ['N9_MISSED_RUNTIME'])
    with installed(runtime) as loop_json: loop_json.main()

if __name__=='__main__': main()
