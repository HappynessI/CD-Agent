import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from agent_baseline import instances
from agent_baseline.tools import Toolbox
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.loop_json import run_session

SHAPE = (32, 48)


def mask(box=(3, 4, 9, 10), shape=SHAPE):
    result = np.zeros(shape, bool)
    if box:
        x0,y0,x1,y1 = box
        result[y0:y1,x0:x1] = True
    return result


def rows(masks, scores=None):
    return instances.from_dense(np.asarray(masks), scores if scores is not None else [.8]*len(masks))


def rgb():
    yy, xx = np.indices(SHAPE)
    a = np.stack((xx*4, yy*6, np.full(SHAPE, 50)), axis=2).astype(np.uint8)
    return [a, a.copy()]


def segment(time, text, tile, threshold):
    return rows([mask((3,4,9,10)) if time == 't1' else mask((20,18,28,26))]), dict(cache_hit=True)


def prompt(time, box=None, points=None, point_labels=None):
    return mask(box or (2,2,12,12)), .9, dict(cache_hit=True)


def ask(pairs, question):
    return [dict(answer='yes', type='new_building', reason='A building appeared.', p_yes=.8) for pair in pairs], {}


def call(name, args, id='call1'):
    return dict(id=id, type='function', function=dict(name=name, arguments=json.dumps(args)))


def response(calls=None, content='Inspect the evidence.'):
    return dict(role='assistant', content=content, **({'tool_calls': calls} if calls is not None else {}))


class ScriptClient:
    def __init__(self, script):
        self.script = script
        self.requests = []

    def __call__(self, payload):
        self.requests.append(copy.deepcopy(payload))
        message = self.script[len(self.requests)-1]
        return dict(id=f'request{len(self.requests)}', choices=[dict(message=message)],
            usage=dict(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=dict(cached_tokens=20)))


def tempdir():
    return tempfile.TemporaryDirectory(prefix='synthetic_', dir=Path(__file__).parent)


def run(directory, script, gt=None, init=None, budget=30, task=None, review=False, status=False, verify=False, plan=False, ask_backend=ask, skills=(), gate2=False, prompt_version='v1'):
    toolbox = Toolbox(rgb(), directory, segment, prompt, ask_backend, init_mask=init)
    client = ScriptClient(script)
    planner = Planner(client=client, sleep=lambda n: None, seed=7)
    recorder = Recorder(directory, 'synthetic', 'image.png', gt=mask() if gt is None else gt)
    final = run_session(toolbox, planner, recorder, task or ('T5.1' if init is not None else 'T1.1xC0'), budget, prompt_version=prompt_version, gate2=gate2, review=review, status=status, verify=verify, plan=plan, skills=skills)
    return final, toolbox, client, recorder
