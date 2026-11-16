"""Explicit N9-only Agent adapter; default Toolbox and tool registry stay unchanged."""
from pathlib import Path
import time
import numpy as np
from .state import Mask, ToolError
from .tools import Toolbox
from .tools_t13 import Features
from .view import ask_images
from .planner import Planner, usage_tokens, request_cost
from .analysis.n8_missed_assets import read, write, SOURCES
from .analysis.n8_missed_prepare import load_union, reconstruct, save_masks
from .analysis.n9_missed_protocol import profile_for, candidate_indices
from .analysis.n9_missed_pipeline import make_request
from .analysis.night_missed_views import render
from .analysis.missed_verifier_study import parse_response


def schema(k):
    return dict(type='function', function=dict(name='find_missed', description=(
        'Find and verify possible missed building changes outside the CURRENT answer. '
        'Uses the task-direction profile, cached image evidence and freshly cropped candidates. '
        'Returns proposals with advisory yes/no evidence; does not change the answer. '
        'You decide whether to call this tool and which proposals to select, convert with '
        'to_mask(set_current=false), and combine with your answer. No call is required.'),
        parameters=dict(type='object', properties=dict(limit=dict(type='integer', minimum=1, maximum=k,
            description=f'Optional proposal cap, at most the frozen profile limit {k}.')), additionalProperties=False)))


class CurrentMaskProvider:
    def __init__(self, asset, config, direction):
        if profile_for(direction) != config['profile']:
            raise ValueError('Profile must match task direction')
        self.asset, self.config, self.direction = asset, config, direction
        self.sources = self.probabilities = self.features = None

    def load(self, shape):
        if self.sources is not None: return
        self.sources = {s: load_union(self.asset['sources'][s]) for s in SOURCES}
        self.probabilities = []; self.features = []
        for t in ('t1', 't2'):
            with np.load(self.asset['semantic'][t], allow_pickle=False) as z:
                probability = z['data']
                assert probability.shape == (7,*shape)
                self.probabilities.append(probability[5].astype(np.float32))
            with np.load(self.asset['features'][t], allow_pickle=False) as z:
                self.features.append(Features(z['data'], shape, t))

    def candidates(self, current, limit=None):
        self.load(current.shape)
        candidates, _, audit = reconstruct(self.sources, current, self.direction, self.probabilities, self.features)
        config = dict(self.config)
        if limit is not None: config['k'] = min(limit, config['k'])
        indexes = candidate_indices(candidates, config)
        return [candidates[i] for i in indexes], audit


class N9Toolbox(Toolbox):
    runtime = None  # Set only by the opt-in entry in its isolated process.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n9_call_number = 0
        if self.runtime['arm'] == 'find_missed':
            self.missed_provider = CurrentMaskProvider(self.runtime['asset'], self.runtime['config'], self.session.change_direction)

    def find_missed(self, limit=None):
        if self.missed_provider is None: raise ToolError('bad_args','Unknown tool name.')
        current_handle = self.session.current_mask
        if current_handle is None: raise ToolError('bad_args','Choose a current answer first.')
        current = self.get(current_handle, Mask).data
        self.n9_call_number += 1
        folder = self.directory/'n9_missed'/f'call_{self.n9_call_number:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        selected, audit = self.missed_provider.candidates(current, limit)
        config = self.runtime['config']; masks = [c['mask'] for c in selected]
        save_masks(folder/'candidates.npz', masks, current.shape, current=np.packbits(current))
        write(folder/'generation.json', dict(config=config, reference_answer=current_handle,
            candidates=[{k:v for k,v in c.items() if k!='mask'} for c in selected], audit=audit,
            seconds=time.monotonic()-started, labels_accessed=False, answer_changed=False))
        events = []; tokens = dict(input=0, cached=0, output=0); charged = 0.; proposals = []; outcomes = []
        self.meta.update(backend='n9_find_missed', attempts=events, tokens=tokens, cost_cny=0.)
        guard = getattr(self.ask_backend,'guard',None)
        for rank, candidate in enumerate(selected, 1):
            directory = folder/f'candidate_{rank:02d}'; directory.mkdir(exist_ok=True)
            mask = candidate['mask']; bbox = candidate['bbox']; variant = config['verifier']
            images = (render(self.rgb, mask, current, bbox, 'V2')[0] if variant in ('V2','V4')
                      else ask_images(self.rgb, mask, bbox, True))
            paths = [directory/f'{i}.jpg' for i in range(4)]
            for path, image in zip(paths, images): image.save(path, quality=92)
            row = dict(shape=current.shape, direction=self.session.change_direction, scope=self.session.scope,
                       mask_index=candidate['index'])
            request = make_request(row, paths, variant)
            # Pairing uses the session seed; no offline answers or cross-session responses.
            request['parameters']['seed'] = self.runtime['session_seed'] + self.n9_call_number*1000 + rank
            write(directory/'request.json', dict(request, verifier=variant, candidate=candidate['index'],
                current_answer=current_handle, reuse='none; actual current residual rendered anew'))
            parameters = dict(request['parameters']); seed = parameters.pop('seed')
            def record(event):
                nonlocal charged
                import os
                safe = dict(event); body = safe.get('error_body'); secret = os.environ.get('DASHSCOPE_API_KEY')
                if body and secret: safe['error_body'] = body.replace(secret, '[credential omitted]')
                token = usage_tokens(event.get('usage')); cost = request_cost('qwen3.7-flash', token)
                safe.update(model='qwen3.7-flash', cost_cny=cost, n9_find_missed=True)
                write(directory/f"attempt_{event['attempt_id']:02d}.json", safe)
                events.append(safe)
                for key in tokens: tokens[key] += token[key]
                charged += cost; self.meta['cost_cny'] = charged
            planner = Planner(seed=seed, parameters=parameters, guard=guard, max_attempts=3, record=record)
            answer = planner.complete(request['messages'], [], rank)
            parsed = parse_response(answer['raw'])
            write(directory/'answer.json', dict(parsed, usage=answer['usage'], cost_cny=planner.cost))
            idx = np.flatnonzero(mask).astype(np.int32)
            proposals.append(dict(id=rank, idx=idx, area=len(idx), score=1./rank, bbox=bbox,
                                  sources=candidate['sources']))
            outcomes.append(dict(id=rank, area=len(idx), sources=candidate['sources'],
                answer=parsed['answer'], reason=parsed.get('parsed',{}).get('reason')))
            # Populate only the existing exact-outline in-session answer cache.
            # Generic ask remains unchanged; a changed or merged outline gets a new request.
            outline = np.flatnonzero(mask)
            entry = self.asked.setdefault(outline.tobytes(), dict(outline=outline, area=len(idx), bbox=bbox, records=[]))
            entry['records'].append(dict(answer=parsed['answer'] if parsed['answer'] in ('yes','no') else None,
                type=parsed.get('parsed',{}).get('type'), reason=parsed.get('parsed',{}).get('reason'),
                p_yes=None, level='L1', source='find_missed', mask=None, id=rank))
        result = self._props(proposals, 't1' if self.session.change_direction=='disappeared' else 't2', parents=(current_handle,))
        result.update(reference_answer=current_handle, answer_changed=False, profile=config['profile'],
            candidates=outcomes, accepted_ids=[r['id'] for r in outcomes if r['answer']=='yes'],
            note='Verifier decisions are advisory. Select proposals and combine only if you choose.',
            current_mask_reconstructed=True)
        assert self.session.current_mask == current_handle
        self.meta['backend_seconds'] = time.monotonic()-started
        return result
