"""JSON tool loop. Run as a module, or directly from the installed directory."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import traceback
import numpy as np
from common import names, load_rgb, load_mask_png, write_json, WORKSPACE
from .scope_t13 import scope_card, session_card
from .tools import Toolbox, PLAN_SCHEMA, SKILL_SCHEMA, failure, skill_index
from . import tools as tool_module
from .state import Mask, Session
from .planner import Planner, RunGuard, RunStopped, usage_tokens, stop_path
from .recorder import Recorder
from .backends import SegmentBackend, SharedPrompter, LocalAsk, ApiAsk

from .tasks import INSTRUCTIONS, instruction_for, seed_for, read_tasks, image_name, load_pair, needs_init
from .feedback import public_feedback
from .gate4 import remind, clip_submission

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))


def system_prompt(version, budget, session=None):
    path = Path(__file__).parent / 'prompts' / f'system_{version}.txt'
    template = path.read_text()
    target = session.target if session else 'building'
    direction = session.change_direction if session else 'any'
    card = session_card(session) if session else scope_card(target, direction)
    # Earlier templates retain their text on the default route too.
    rendered = template.replace('buildings', card['plural']).replace('building', card['target'])
    text = rendered.format(counts='\n'.join('- ' + s for s in card['counts']),
                           not_counts='\n'.join('- ' + s for s in card['not_counts']),
                           target=card['target'], plural=card['plural'], budget=budget)
    if target == 'any change' or isinstance(target,(list,tuple)):
        text = text.replace('one object class','the requested land-cover categories')
    if direction != 'any':
        text += f'\nCount only changes in the {direction} direction.'
    if session and getattr(session, 'old_map', None):
        text = text.replace('two co-registered remote sensing images',
                            'an old building map and a co-registered new optical image')
    return path, template, text


def image_blocks(result):
    info = {k: result.public[k] for k in ('view_id', 'region', 'overlay')}
    blocks = []
    for t, path in enumerate(result.images, 1):
        blocks += [dict(type='text', text=f'{info["view_id"]} T{t}', _view=info),
                   dict(type='image_url', image_url=dict(url=path), _view=info)]
    return blocks


# Editing an earlier message breaks the provider's prefix cache from that point on, so images leave in one
# batch: nothing changes until more than MAX_VIEW_PAIRS view pairs are live, then only the latest KEEP_VIEW_PAIRS stay.
MAX_VIEW_PAIRS, KEEP_VIEW_PAIRS = 7, 3


def trim_images(messages):
    views = []
    for message in messages:
        for block in message.get('content', []) if isinstance(message.get('content'), list) else []:
            info = block.get('_view')
            if info and info['view_id'] not in views:
                views.append(info['view_id'])
    later = [v for v in views if v != 'view#0']
    if len(later) <= MAX_VIEW_PAIRS:
        return
    keep = {'view#0', *later[-KEEP_VIEW_PAIRS:]}
    for message in messages:
        if not isinstance(message.get('content'), list):
            continue
        dropped = set()
        blocks = []
        for block in message['content']:
            info = block.get('_view')
            if info and info['view_id'] not in keep:
                if info['view_id'] not in dropped:
                    blocks.append(dict(type='text', text=f'[image dropped: {info["view_id"]}, region={info["region"]}, overlay={info["overlay"]}]'))
                    dropped.add(info['view_id'])
            else:
                blocks.append(block)
        message['content'] = blocks


def referenced_handles(args):
    if isinstance(args, str):
        return [args] if args.startswith(('mask#', 'inst#', 'region#', 'props#', 'feat#', 'example#', 'exmask#', 'map#')) else []
    if isinstance(args, dict):
        return [key for v in args.values() for key in referenced_handles(v)]
    if isinstance(args, list):
        return [key for v in args for key in referenced_handles(v)]
    return []


REVIEW_REPLIES = 3
EXTRA_REPLIES = 2  # G2: one grant per session to act on a report that would otherwise come too late.


def latest_mask(session):
    masks = [key for key, value in session.handles.items() if isinstance(value, Mask)
             and key not in (session.region_mask, getattr(session, 'old_map', None))]
    return masks[-1] if masks else None


def run_session(toolbox, planner, recorder, task='T1.1xC0', budget=30, prompt_version='v1', metadata=None, review=False, status=False, verify=False, plan=False, skills=(), gate2=False, gate3=False, gate4=False, tool_hints='off', gate5=False, gate6=False, answer_state=False, segment_pair=False, progressive_tools=False, t3_semantic_bridge=False):
    session = toolbox.session
    if t3_semantic_bridge:
        toolbox.t3_semantic_bridge = True
        toolbox.bridge_seed = planner.parameters.get('seed', 0)
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['t3_semantic_bridge'] = True
    if progressive_tools:
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['progressive_tools'] = True
    if segment_pair or toolbox.segment_pair:
        toolbox.segment_pair = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['segment_pair'] = True
    if answer_state or toolbox.answer_state:
        toolbox.answer_state = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['answer_state'] = True
    if gate6 or toolbox.gate6:
        toolbox.gate6 = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault("backends", {})["gate6"] = True
    if gate5 or toolbox.gate5:
        gate5 = gate4 = True
        toolbox.gate5 = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['gate5'] = True
    if gate4:
        gate3 = True
        toolbox.gate4 = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {})['gate4'] = True
    if (task == 'T5.4') != bool(getattr(session, 'old_map', None)):
        raise ValueError('T5.4 requires the opt-in map input; use --map-input with a T5.4 manifest')
    if gate3:
        if not verify:
            raise ValueError('G3 requires --verify-stage')
        toolbox.gate3 = True
        gate2 = True
        metadata = copy.deepcopy(metadata or {})
        metadata.setdefault('backends', {}).update(gate3=True, gate2=True, big_box_share=toolbox.big_box_share)
    gate2 = gate2 and verify
    limit, grant = budget, None
    rounds, no_tools = 0, 0
    previous_calls = None
    reason, error = 'budget', None
    reviewed = not (review or verify)
    schemas = [*tool_module.session_schemas(session, toolbox.python_enabled, toolbox.tools_t13, tool_hints, gate5=gate5, answer_state=toolbox.answer_state, segment_pair=toolbox.segment_pair, t3_semantic_bridge=toolbox.t3_semantic_bridge, missed_candidates=toolbox.missed_provider is not None), *([PLAN_SCHEMA] if plan else []), *([SKILL_SCHEMA] if skills else [])]
    generated_calls = []
    toolbox.skills = tuple(skills)
    messages = []
    planner.record = recorder.request
    try:
        path, template, system = system_prompt(prompt_version, budget, session)
        if progressive_tools:
            system += '\nMask-edit tools become visible automatically once an existing mask is available. Core and task-specific tools remain available; no extra skill-loading step is needed.'
        if toolbox.tools_t13:
            system += '\nAdditional handles: props#k proposal sets and feat#k patch features. Proposal sets work with select and to_mask. On compared proposals score is 1-cos temporal difference; on matched proposals it is exemplar cosine similarity. Classification probabilities are reported by ID.'
        if skills:
            # Index only; a skill's text enters the conversation when the planner loads it.
            system += '\n\nSkills\nThese notes hold working methods. Before your first other tool call, call load_skill for the skill that fits the start of your work. Later, load another one whenever its description fits what you are about to do.\n' + skill_index(skills)
        instruction = instruction_for(task, session)
        exemplar = getattr(session, 'exemplar_handle', None)
        initial_overlay = session.initial_mask or exemplar
        initial = toolbox.execute('view', dict(overlay=initial_overlay) if initial_overlay else {})
        if not initial.public['ok']:
            raise RuntimeError('Initial view failed')
        text = instruction
        if getattr(session, 'old_map', None):
            from .map_input import describe
            text += ' ' + describe(session)
        if exemplar:
            text += f' Example region: {exemplar}, marked by the red contour. Find other changes of the same transition as this example, including the example itself.'
        if session.initial_mask:
            item = session.get(session.initial_mask, Mask)
            text += f' mask#0: area={int(item.data.sum())} pixels; component_count={len(item.components)}.'
        if session.region_mask:
            item = session.get(session.region_mask, Mask)
            text += f' region#0: area={int(item.data.sum())} pixels; bbox={session.region_bbox}. Coordinates are integer pixels; x1 and y1 are exclusive right and bottom bounds. region#0 is a region mask usable wherever a mask handle is accepted. It is not the current answer.'
        messages = [dict(role='system', content=system),
                    dict(role='user', content=[dict(type='text', text=text)] + image_blocks(initial))]
        if getattr(session, 'external_examples', ()):
            from .examples_t3x import example_blocks
            messages[1]['content'].extend(example_blocks(session, toolbox.directory))
        recorder.start(session, dict(metadata or {}, task_id=task, instruction=instruction,
            target=session.target, change_direction=session.change_direction, python=toolbox.python_enabled,
            image_shape=list(session.shape), system_prompt_version=prompt_version, system_prompt_path=str(path),
            system_prompt_template=template, system_prompt=system, tool_schema_version='v2',
            schemas=tool_module.disclosed_schemas(schemas, session) if progressive_tools else schemas,
            model_parameters=planner.parameters, seed=planner.parameters['seed'], seed_acceptance='not_reported',
            budget=budget, max_calls_per_reply=8, total_attempt_limit=planner.max_attempts,
            initial_view=initial.public, initial_handle=session.initial_mask))
        if needs_init(task) and session.initial_mask is None:
            raise ValueError('T5.1 requires the fixed initial mask')
        number = 0
        while number < limit:
            number += 1
            # Default auto keeps the model's one-line reasons (required returns empty content); after a
            # reply without a function call, the next request forces one and changes the seed.
            if toolbox.answer_state:
                # Replace a single compact summary; do not grow another trajectory.
                messages[:] = [m for m in messages if not (m.get('role') == 'user' and isinstance(m.get('content'), str) and m['content'].startswith('Session state index: '))]
                messages.append(dict(role='user', content='Session state index: ' + json.dumps(toolbox.state_summary())))
            visible_schemas = tool_module.disclosed_schemas(schemas, session) if progressive_tools else schemas
            response = planner.complete(messages, visible_schemas, number, 'required' if no_tools else None, no_tools)
            rounds += 1
            original = response['message']
            assistant = {k: copy.deepcopy(original[k]) for k in ('role', 'content', 'tool_calls') if k in original}
            assistant.setdefault('role', 'assistant')
            messages.append(assistant)
            calls = original.get('tool_calls') or []
            # The provider rejects a history with the same call in consecutive rounds, and rerunning it teaches nothing.
            signature = [(c.get('function') or {}).get('name') for c in calls], [(c.get('function') or {}).get('arguments') for c in calls]
            if toolbox.answer_state:
                # Only implicit ask depends on mutable current/verification state.
                # A new unrelated handle or another view does not justify a repeat.
                ask_state = []
                for i, candidate in enumerate(calls):
                    function = candidate.get('function') or {}
                    if function.get('name') != 'ask':
                        continue
                    try:
                        arguments = json.loads(function.get('arguments', ''))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(arguments, dict) and arguments.get('regions') is None:
                        item = session.handles.get(session.current_mask)
                        checked = tuple((k, len(v)) for k, v in toolbox.records(item).items()) if isinstance(item, Mask) else ()
                        ask_state.append((i, session.current_mask, checked))
                signature = (signature, tuple(ask_state))
            repeated = bool(calls) and signature == previous_calls
            if repeated:
                calls = []
            elif calls:
                previous_calls = signature
            call_records, images = [], []
            allowed = frozenset(session.handles)
            more_to_check = False
            for index, call in enumerate(calls):
                fun = call.get('function') or {}
                name, raw_args = fun.get('name', ''), fun.get('arguments', '')
                args = None
                empty_handle = None
                before = session.current_mask
                if session.submitted is not None:
                    result = failure('bad_args', 'skipped_due_to_submit')
                    result.internal['status'] = 'skipped_due_to_submit'
                elif index >= 8:
                    result = failure('bad_args', 'At most 8 tool calls execute per reply.')
                else:
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        result = failure('bad_args', 'function.arguments must be a valid JSON object; check its fields and quoting.')
                    else:
                        if name == 'submit' and isinstance(args, dict) and args.get('mask') == 'empty':
                            args = dict(args, mask=session.add_mask(np.zeros(session.shape, bool)))
                            empty_handle = args['mask']
                            allowed = allowed | {args['mask']}
                        handle = args.get('mask') if isinstance(args, dict) else None
                        report_calls = (generated_calls + [dict(tool=name, handles_created=[empty_handle])]
                                        if empty_handle is not None else generated_calls)
                        # The first valid submit opens one report, and with the verifier stage so does any
                        # submit of a mask with unchecked components. The last reply can only submit.
                        known = name == 'submit' and handle in allowed and isinstance(session.handles[handle], Mask)
                        owed = known and (toolbox.owed if gate2 else toolbox.pending)(session.handles[handle])
                        if known and number < limit and (not reviewed or (verify and owed)):
                            reviewed = True
                            previous_calls = None
                            result = toolbox.pre_submit(handle, limit - number, verify)
                            if gate4 and limit - number <= REVIEW_REPLIES:
                                remind(result, session, report_calls)
                        elif known and gate2 and grant is None and number == limit and owed:
                            # The last reply: its report still reaches the planner, with replies to act on it.
                            grant = dict(kind='last_reply', at_reply=number)
                            limit += EXTRA_REPLIES
                            reviewed = True
                            previous_calls = None
                            result = toolbox.pre_submit(handle, EXTRA_REPLIES, verify)
                            if gate4:
                                remind(result, session, report_calls)
                        else:
                            result = toolbox.execute(name, args, allowed)
                            # ask checks a batch per call, so the same empty call is the way to get the next batch.
                            more_to_check = more_to_check or (gate2 and name == 'ask' and bool(result.public.get('remaining_unverified')))
                messages.append(dict(role='tool', tool_call_id=call['id'], content=result.text))
                images.extend(image_blocks(result) if result.images else [])
                internal = result.internal
                if gate4 and empty_handle is not None:
                    internal['handles_created'] = [empty_handle, *internal.get('handles_created', [])]
                call_records.append(dict(tool=name, tool_call_id=call['id'], raw_arguments=raw_args, args=args,
                    ok=result.public['ok'], error_code=result.public.get('error_code'), result_text=result.text,
                    result_images=result.images, input_handles=referenced_handles(args),
                    output_handles=internal.get('handles_created', []), handles_created=internal.get('handles_created', []),
                    current_before=before, current_after=session.current_mask, cache_hit=internal.get('cache_hit'),
                    backend_seconds=internal.get('backend_seconds', 0.), seconds=internal.get('seconds', 0.),
                    status=internal.get('status', 'executed'), provenance=internal,
                    set_current=args.get('set_current', True) if isinstance(args, dict) and name in
                        ('to_mask','combine','morph','remove_region','add_region','refine_region','python') else None))
                if gate4:
                    generated_calls.append(call_records[-1])
            if more_to_check:
                previous_calls = None
            if images:
                messages.append(dict(role='user', content=images))
            trim_images(messages)
            recorder.step(session, number, dict(request_id=response['request_id'],
                **{k + '_tokens': v for k, v in usage_tokens(response['usage']).items()},
                seconds=response['seconds'], thought=original.get('content'),
                raw=response['raw'], code=None, attempts=response['attempts']), call_records)
            if session.submitted is not None:
                reason = 'submit'
                break
            late_key = None
            if not calls:
                no_tools += 1
                if no_tools == 3:
                    key = session.current_mask if session.current_mask is not None else latest_mask(session)
                    if not (gate2 and grant is None and key is not None and toolbox.owed(session.get(key, Mask))):
                        reason = 'no_submit'
                        break
                    grant = dict(kind='no_tools', at_reply=number)
                    limit = max(limit, number + EXTRA_REPLIES)
                    no_tools, late_key = 0, key
                # Pseudo-calls written as text prime the next reply to do the same, even under
                # tool_choice=required, so the reply stays in the trace but leaves the context.
                messages.pop()
                if late_key is not None:
                    pass
                elif repeated:
                    messages.append(dict(role='user', content='You repeated your previous tool call, so it did not run again. Its result is above. Choose a different action, or call submit with an existing mask and a summary.'))
                elif no_tools == 1:
                    messages.append(dict(role='user', content='Your last reply contained no function call, so nothing ran. Writing a tool call as text does not execute it. Make a function call now, or call submit with an existing mask and a summary.'))
            else:
                no_tools = 0
            left = limit - number
            note = ''
            if gate2 and left == 0 and grant is None and session.current_mask is not None and toolbox.owed(session.get(session.current_mask, Mask)):
                # The budget ran out with unchecked components in the current answer.
                grant = dict(kind='budget', at_reply=number)
                limit += EXTRA_REPLIES
                left, late_key = EXTRA_REPLIES, session.current_mask
            if gate2 and late_key is None and 0 < left <= REVIEW_REPLIES and not reviewed and session.current_mask is None and latest_mask(session) is not None:
                late_key = latest_mask(session)
                note = 'You have no current answer yet. This is the mask you generated most recently.\n'
            if late_key is not None:
                reviewed = True
                previous_calls = None
                late = toolbox.pre_submit(late_key, left, verify)
                # G4 keeps G2's behaviour here: the reported mask becomes the current answer.
                if gate4 and (left <= REVIEW_REPLIES or (grant and grant['kind'] in ('budget', 'last_reply'))):
                    remind(late, session, generated_calls)
                recorder.ask_spending(late.internal, f'report-{number}')
                recorder.note(number, late)
                messages.append(dict(role='user', content=note + late.public['report']))
            elif 0 < left <= REVIEW_REPLIES and not reviewed and session.current_mask is not None:
                reviewed = True
                late = toolbox.pre_submit(session.current_mask, left, verify)
                if gate4:
                    remind(late, session, generated_calls)
                recorder.ask_spending(late.internal, f'report-{number}')
                recorder.note(number, late)
                messages.append(dict(role='user', content=late.public['report']))
            elif 0 < left <= REVIEW_REPLIES:
                messages.append(dict(role='user', content=f'You have {left} replies left. Submit your best mask before they run out.'))
            if status and left > 0:
                messages.append(dict(role='user', content=toolbox.status(left)))
    except RunStopped:
        raise
    except Exception:
        reason, error = 'error', traceback.format_exc()
    if session.submitted is not None:
        submitted_by, fallback = ('error', 'none') if reason == 'error' else ('model', 'none')
    else:
        fallback = 'current' if session.current_mask is not None else 'empty'
        submitted_by = 'error' if reason == 'error' else ('empty' if fallback == 'empty' else reason)
        if session.current_mask is None:
            session.current_mask = session.add_mask(np.zeros(session.shape, bool))
        session.submitted = session.current_mask
        session.summary = f'系统兜底：{reason}；使用当前掩码。' if fallback == 'current' else f'系统兜底：{reason}；使用空掩码。'
        if gate4:
            session.submitted, report = clip_submission(toolbox, session.submitted)
            session.current_mask = session.submitted
            if report:
                session.summary += '\n' + report
    if verify and reason != 'error':
        # Whatever ended the run, the trace holds an answer for every component of the final mask.
        _, _, meta = toolbox.stage(session.submitted)
        recorder.ask_spending(meta, 'final-stage')
    write_json(toolbox.directory / 'verification.json', toolbox.ledger(session.submitted))
    if gate2:
        write_json(toolbox.directory / 'gate.json', dict(grant=grant, rounds=rounds, budget=budget,
            extra_used=rounds - grant['at_reply'] if grant else 0))
    result = recorder.final(session, task, submitted_by, reason, fallback, rounds, planner, error)
    if gate3:
        result['backends'] = metadata['backends']
        write_json(toolbox.directory / 'final.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=list(INSTRUCTIONS))
    parser.add_argument('--tasks', help='Task manifest JSONL; optionally filter by --task.')
    parser.add_argument('--map-ask-variant', choices=['t2_state'], help='Opt-in T2-only building-state verifier; requires --map-input and API ask.')
    parser.add_argument('--map-input', action='store_true', help='Enable T5.4 old building map as T1; requires a T5.4 manifest.')
    parser.add_argument('--missed-candidates-root', type=Path, help='Opt-in frozen SECOND/LEVIR development candidate cache; returns proposals only.')
    parser.add_argument('--missed-verifier-variant', choices=['four','som','structured'], help='Opt-in serial flash verifier with controlled position marks/question.')
    parser.add_argument('--t3-semantic-bridge', action='store_true', help='Offer independent visual exemplar interpretation and semantic proposal scoring. Default off.')
    parser.add_argument('--tools-t13', nargs='?', const='all', default='', help='Enable all new vision tools, or comma-separated tool names. Default off.')
    parser.add_argument('--tool-hints', choices=['off', 'a', 'b'], default='off', help='Append purpose hints to enabled T1.3 tool descriptions only.')
    parser.add_argument('--external-examples', action='store_true', help='Load external example crops from --tasks. Requires match_exemplar in --tools-t13. Default off.')
    parser.add_argument('--vision-config', type=Path, help='Local worker environment, source and weight paths JSON.')
    parser.add_argument('--python', action='store_true', help='Offer the isolated Python array tool.')
    parser.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--names-file')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--ask-backend', choices=['local', 'api', 'none'], default='local')
    parser.add_argument('--sam3-worker')
    parser.add_argument('--ask-worker')
    parser.add_argument('--planner-model', default='qwen3.7-plus')
    parser.add_argument('--review', action='store_true', help='One pre-submission review from the session records.')
    parser.add_argument('--status', action='store_true', help='A state line in every reply.')
    parser.add_argument('--plan', action='store_true', help='Offer the plan tool.')
    parser.add_argument('--skills', default='', help='Comma-separated skill names, or all.')
    parser.add_argument('--verify-stage', action='store_true', help='Each submit first verifies every unchecked component and reports all answers.')
    parser.add_argument('--gate2', action='store_true', help='Close the gaps where verifier answers never reach the planner (G2). Needs --verify-stage.')
    parser.add_argument('--gate3', action='store_true', help='G3 box guard, constraint reports and feedback verification scope. Implies --gate2; requires --verify-stage.')
    parser.add_argument('--gate4', action='store_true', help='G4 current-answer reminders and region clipping at submission. Implies --gate3; requires --verify-stage.')
    parser.add_argument('--progressive-tools', action='store_true', help='Reveal mask-edit schemas after a mask exists; core and task-specific tools stay visible.')
    parser.add_argument('--segment-pair', action='store_true', help='Offer paired T1/T2 text segmentation; two sequential cached backend requests.')
    parser.add_argument('--answer-state', action='store_true', help='Offer explicit set_answer for selecting an existing answer mask; default off.')
    parser.add_argument('--gate6', action='store_true', help='Expose original image dimensions and feature score quantiles; independent of other gates.')
    parser.add_argument('--gate5', action='store_true', help='G4 plus optional region clipping for segment_text and segment_everything. Implies --gate4; requires --verify-stage.')
    parser.add_argument('--big-box-share', type=float, default=tool_module.BIG_BOX_SHARE)
    parser.add_argument('--ask-correspondence', action='store_true', help='Ask about the object across the dates.')
    parser.add_argument('--ask-model', default='qwen3.7-plus', help='Model of the api ask backend.')
    parser.add_argument('--pixel-thr', type=float, default=.5, help='SAM3 soft mask threshold of segment_text.')
    parser.add_argument('--sam-prompt-worker', help='URL of workers/sam_prompt_worker.py; one SAM decoder for the whole job. Default off.')
    parser.add_argument('--sam-prompt-device', choices=['cpu', 'cuda'], default='cpu', help='SAM decoder device; cached LEVIR embeddings are reused on either device.')
    parser.add_argument('--init-name', default='sam3_imc', help='Directory under init/ with the T5.1 given masks.')
    parser.add_argument('--prompt-version', default='v1', choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--seed-offset', type=int, default=0, help='Repeat runs of one configuration.')
    parser.add_argument('--batch', help='Budget ledger name; parallel jobs of one batch share it.')
    parser.add_argument('--max-cost-cny', type=float, help='Stop every job of the batch above this estimated cost.')
    args = parser.parse_args()
    if args.gate5:
        if not args.verify_stage:
            parser.error('--gate5 requires --verify-stage')
        args.gate4 = True
    if args.gate4:
        if not args.verify_stage:
            parser.error('--gate4 requires --verify-stage')
        args.gate3 = True
    if args.external_examples and ('match_exemplar' not in tool_module.enabled_names(True if args.tools_t13 == 'all' else args.tools_t13) or not args.tasks):
        parser.error('--external-examples requires --tasks and match_exemplar in --tools-t13')
    if args.gate3 and not args.verify_stage:
        parser.error('--gate3 requires --verify-stage')
    if not 0 < args.big_box_share <= 1:
        parser.error('--big-box-share must be in (0, 1]')
    vision = None
    if args.tools_t13 or args.t3_semantic_bridge:
        from .backends_t13 import WorkerPool
        if not args.vision_config:
            parser.error('--tools-t13 requires --vision-config')
        vision = WorkerPool(json.loads(args.vision_config.read_text()), RUN / 'vision_t13')
    if not args.tasks and not args.task:
        parser.error('--task is required without --tasks')
    if args.task in ('T3.2', 'T3.3', 'T3.4') and not args.tasks:
        parser.error('External example tasks require an example manifest (--tasks)')
    if args.task in ('T5.2', 'T5.3') and not args.tasks:
        parser.error(f'{args.task} requires a feedback task manifest (--tasks)')
    if (args.map_input or args.task == 'T5.4') and not args.tasks:
        parser.error('T5.4 map input requires a task manifest (--tasks)')
    skills = sorted(p.name for p in tool_module.SKILLS.iterdir() if (p / 'SKILL.md').exists()) if args.skills == 'all' else [x for x in args.skills.split(',') if x]
    guard = RunGuard(stop_path(RUN / 'STOP.json'), RUN / 'budget' / f'{args.batch}.txt' if args.batch else None, args.max_cost_cny)
    if guard.stopped():
        raise SystemExit(f'{guard.active_stop()} exists; check quota or budget, then delete it before running: {guard.active_stop().read_text()}')
    todo = read_tasks(args.tasks, args.task) if args.tasks else (Path(args.names_file).read_text().split() if args.names_file else names(args.split))
    if args.tasks and any(r.get('examples') for r in todo) and not args.external_examples:
        parser.error('External example tasks require --external-examples')
    if args.map_ask_variant and (not args.map_input or args.ask_backend != 'api'):
        parser.error('--map-ask-variant requires --map-input and --ask-backend api')
    if args.map_input and (not args.tasks or not todo or any(r['task_type'] != 'T5.4' for r in todo)):
        parser.error('--map-input requires a nonempty T5.4 manifest')
    if (args.task == 'T5.4' or (args.tasks and any(r['task_type'] == 'T5.4' for r in todo))) and not args.map_input:
        parser.error('T5.4 requires --map-input')
    if args.missed_candidates_root and (not args.tasks or any((r.get('meta') or {}).get('group') not in ('second_appeared','second_disappeared','levir_any') for r in todo)):
        parser.error('--missed-candidates-root requires the frozen development group manifests')
    if args.missed_verifier_variant and (args.ask_backend != 'api' or args.ask_model != 'qwen3.7-flash'):
        parser.error('--missed-verifier-variant requires API qwen3.7-flash')
    if args.limit is not None:
        todo = todo[:args.limit]
    if args.tasks:
        keys = [(r['task_type'], Path(r['id']).stem) for r in todo]
        if len(set(keys)) != len(keys):
            parser.error('Run each direction in a separate config/manifest to avoid output collisions.')
    def gpu_prompter():
        from sam_tools import Prompter
        return Prompter(device='cuda')
    if args.sam_prompt_worker:
        from .backends import PromptWorker
        shared = PromptWorker(args.sam_prompt_worker)
    else:
        shared = SharedPrompter(factory=gpu_prompter) if args.sam_prompt_device == 'cuda' else SharedPrompter()
    def identity(item):
        if isinstance(item, dict):
            return item['task_type'], image_name(item), Path(item['id']).stem
        return args.task, item, Path(item).stem

    def output_directory(item):
        task, name, stem = identity(item)
        return RUN / args.config / task / args.split / stem

    def process(item):
        task, name, stem = identity(item)
        row = item if isinstance(item, dict) else None
        directory = output_directory(item)
        if (directory / 'final.json').exists():
            return name, 'skipped'
        if guard.stopped():
            return name, 'stopped'
        # The label path is consumed only by the one-way scoring recorder.
        recorder = Recorder(directory, args.split, name, gt_path=row['gt_mask'] if row else None, gt_ignore_path=row.get('gt_ignore_mask') if row else None)
        try:
            rgb = load_pair(row) if row else load_rgb(args.split, name)
        except Exception:
            recorder.input_failure(task, dict(session_id=stem), traceback.format_exc())
            return name, 'error'
        init_path = Path(row['init_mask']) if row and row['init_mask'] else RUN / 'init' / args.init_name / args.split / (Path(name).stem + '.png')
        planner = Planner(guard=guard, parameters=dict(model=args.planner_model), seed=seed_for(name, args.seed_offset))
        init, init_error = None, None
        if needs_init(task) or (row and row['init_mask']):
            try:
                init = load_mask_png(init_path)
            except Exception:
                init_error = traceback.format_exc()
        session = Session(rgb[0].shape[:2], init, target=row['target'] if row else 'building',
            change_direction=row['change_direction'] if row else 'any',
            instruction=row['instruction'] if row else None, constraint=row['constraint'] if row else None,
            scope=row.get('scope') if row else None,
            source_classes=row.get('source_classes') if row else None,
            ask_variant=args.map_ask_variant or (row.get('ask_variant') if row else None),
            region_mask=load_mask_png(row['region_mask']) if row and row.get('region_mask') else None,
            area_min_m2=row.get('area_min_m2') if row else None, gsd_m=row.get('gsd_m') if row else None,
            **(public_feedback(row) if row else {}))
        if row and row.get('exemplar_mask'):
            session.exemplar_handle = session.add_mask(load_mask_png(row['exemplar_mask']))
        if args.external_examples and row and row.get('examples'):
            from .examples_t3x import attach_examples, load_examples
            attach_examples(session, load_examples(row['examples'], vision))
        ask = LocalAsk(args.ask_worker) if args.ask_backend == 'local' else ApiAsk(guard=guard, model=args.ask_model, correspondence=args.ask_correspondence, session=session) if args.ask_backend == 'api' else None
        if args.missed_verifier_variant:
            from .missed_verifier import MissedVerifier
            ask = MissedVerifier(directory/'missed_verifier', args.missed_verifier_variant, session=session, guard=guard, seed=seed_for(name,args.seed_offset))
        source_images = [str(WORKSPACE / 'datasets' / 'LEVIR-CD' / f'{args.split}_original' / part / name) for part in ('A','B')]
        custom_images = row is not None and [row['image_t1'], row['image_t2']] != source_images
        if task in ('T5.2', 'T5.3'):
            from .tasks import levir_cache_name
            levir_cache_name(row, args.split)
            custom_images = False
        image_paths = [row['image_t1'], row['image_t2']] if custom_images else None
        toolbox = Toolbox(rgb, directory, SegmentBackend(args.split, name, args.sam3_worker, args.pixel_thr,
            target=session.target, image_paths=image_paths),
            shared.for_image(args.split, name, rgb=rgb if custom_images else None), ask, session=session, python=args.python,
            tools_t13=True if args.tools_t13 == 'all' else args.tools_t13,
            vision=vision.for_pair(image_paths or source_images) if vision else None,
            map_input=args.map_input,
            big_box_share=args.big_box_share)
        if args.missed_candidates_root:
            from .find_missed import CachedMissedCandidates
            toolbox.missed_provider = CachedMissedCandidates(args.missed_candidates_root, row['meta']['group'], Path(row['id']).stem, session.shape)
        try:
            result = run_session(toolbox, planner, recorder, task, prompt_version=args.prompt_version, review=args.review, status=args.status, verify=args.verify_stage, plan=args.plan, skills=skills, gate2=args.gate2, gate3=args.gate3, gate4=args.gate4, tool_hints=args.tool_hints, gate5=args.gate5, gate6=args.gate6, answer_state=args.answer_state, segment_pair=args.segment_pair, progressive_tools=args.progressive_tools, t3_semantic_bridge=args.t3_semantic_bridge,
                metadata=dict(**({'scope': session.scope} if session.scope else {}), session_id=stem, source_images=image_paths or source_images,
                    init_mask_path=str(init_path) if init is not None or needs_init(task) else None,
                    init_config=dict(route=args.init_name, text=session.target, tile=512, score_thr=.5,
                        overlap_thr=.25, match_basis=tool_module.MATCH_BASIS) if needs_init(task) else None,
                    backends=dict(**({'gate2': True} if args.gate2 else {}), ask=args.ask_backend, review=args.review, status=args.status, verify_stage=args.verify_stage,
                        plan=args.plan, skills=skills, ask_correspondence=args.ask_correspondence,
                        missed_candidates=bool(args.missed_candidates_root), missed_verifier_variant=args.missed_verifier_variant,
                        ask_model=args.ask_model if args.ask_backend == 'api' else 'Qwen3.5-4B' if args.ask_backend == 'local' else None,
                        sam3_worker=bool(args.sam3_worker), pixel_thr=args.pixel_thr, sam_prompt='worker' if args.sam_prompt_worker else 'shared_' + args.sam_prompt_device, sam_model='SAM ViT-H'),
                    task_meta=row['meta'] if row else {}, init_error=init_error))
        except RunStopped:
            return name, 'stopped'
        return name, result['submitted_by']
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, item): item for item in todo}
        for future in as_completed(futures):
            try:
                print(*future.result(), flush=True)
            except Exception as exc:
                item = futures[future]
                task, name, stem = identity(item)
                directory = output_directory(item)
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / 'final.json', dict(task_id=task, submitted_by='error',
                    termination_reason='input_error', fallback_kind='empty', error_type=type(exc).__name__,
                    error=traceback.format_exc(), summary='系统兜底：输入数据无法读取，无法生成掩码。', elapsed_seconds=0))
                print(name, 'input_error', flush=True)
    if vision:
        vision.close()
    if guard.stopped():
        raise SystemExit(3)  # a failed job state, so a stop cannot pass as a finished run


if __name__ == '__main__':
    main()
