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
from prompts import SCOPE_CARD
from .tools import Toolbox, SCHEMAS, PLAN_SCHEMA, SKILL_SCHEMA, Result, failure, skill_index
from . import tools as tool_module
from .state import Mask
from .planner import Planner, RunGuard, RunStopped, usage_tokens
from .recorder import Recorder
from .backends import SegmentBackend, SharedPrompter, LocalAsk, ApiAsk

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
INSTRUCTIONS = {
    'T1.1xC0': 'Find all buildings that changed between T1 and T2 (newly built or demolished). Output a binary change mask.',
    'T5.1': 'Here is a building change detection mask (mask#0). Check it against T1 (earlier) and T2 (later), and correct it where needed. Submit the final binary change mask and briefly summarize your decision.'}


def system_prompt(version, budget):
    path = Path(__file__).parent / 'prompts' / f'system_{version}.txt'
    template = path.read_text()
    text = template.format(counts='\n'.join('- ' + s for s in SCOPE_CARD['counts']),
                           not_counts='\n'.join('- ' + s for s in SCOPE_CARD['not_counts']), budget=budget)
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
        return [args] if args.startswith(('mask#', 'inst#')) else []
    if isinstance(args, dict):
        return [key for v in args.values() for key in referenced_handles(v)]
    if isinstance(args, list):
        return [key for v in args for key in referenced_handles(v)]
    return []


REVIEW_REPLIES = 3


def run_session(toolbox, planner, recorder, task='T1.1xC0', budget=30, prompt_version='v1', metadata=None, review=False, status=False, verify=False, plan=False, skills=()):
    session = toolbox.session
    rounds, no_tools = 0, 0
    previous_calls = None
    reason, error = 'budget', None
    reviewed = not (review or verify)
    schemas = [*SCHEMAS, *([PLAN_SCHEMA] if plan else []), *([SKILL_SCHEMA] if skills else [])]
    toolbox.skills = tuple(skills)
    messages = []
    planner.record = recorder.request
    try:
        path, template, system = system_prompt(prompt_version, budget)
        if skills:
            # Index only; a skill's text enters the conversation when the planner loads it.
            system += '\n\nSkills\nThese notes hold working methods. Call load_skill with a name when its description fits what you are doing.\n' + skill_index(skills)
        instruction = INSTRUCTIONS[task]
        initial = toolbox.execute('view', dict(overlay=session.initial_mask) if session.initial_mask else {})
        if not initial.public['ok']:
            raise RuntimeError('Initial view failed')
        text = instruction
        if session.initial_mask:
            item = session.get(session.initial_mask, Mask)
            text += f' mask#0: area={int(item.data.sum())} pixels; component_count={len(item.components)}.'
        messages = [dict(role='system', content=system),
                    dict(role='user', content=[dict(type='text', text=text)] + image_blocks(initial))]
        recorder.start(session, dict(metadata or {}, task_id=task, instruction=instruction,
            image_shape=list(session.shape), system_prompt_version=prompt_version, system_prompt_path=str(path),
            system_prompt_template=template, system_prompt=system, tool_schema_version='v2', schemas=schemas,
            model_parameters=planner.parameters, seed=planner.parameters['seed'], seed_acceptance='not_reported',
            budget=budget, max_calls_per_reply=8, total_attempt_limit=planner.max_attempts,
            initial_view=initial.public, initial_handle=session.initial_mask))
        if task == 'T5.1' and session.initial_mask is None:
            raise ValueError('T5.1 requires the fixed initial mask')
        for number in range(1, budget + 1):
            # Default auto keeps the model's one-line reasons (required returns empty content); after a
            # reply without a function call, the next request forces one and changes the seed.
            response = planner.complete(messages, schemas, number, 'required' if no_tools else None, no_tools)
            rounds += 1
            original = response['message']
            assistant = {k: copy.deepcopy(original[k]) for k in ('role', 'content', 'tool_calls') if k in original}
            assistant.setdefault('role', 'assistant')
            messages.append(assistant)
            calls = original.get('tool_calls') or []
            # The provider rejects a history with the same call in consecutive rounds, and rerunning it teaches nothing.
            signature = [(c.get('function') or {}).get('name') for c in calls], [(c.get('function') or {}).get('arguments') for c in calls]
            repeated = bool(calls) and signature == previous_calls
            if repeated:
                calls = []
            elif calls:
                previous_calls = signature
            call_records, images = [], []
            allowed = frozenset(session.handles)
            for index, call in enumerate(calls):
                fun = call.get('function') or {}
                name, raw_args = fun.get('name', ''), fun.get('arguments', '')
                args = None
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
                            allowed = allowed | {args['mask']}
                        handle = args.get('mask') if isinstance(args, dict) else None
                        # The first valid submit opens one report, and with the verifier stage so does any
                        # submit of a mask with unchecked components. The last reply can only submit.
                        if (name == 'submit' and number < budget and handle in allowed and isinstance(session.handles[handle], Mask)
                                and (not reviewed or (verify and toolbox.pending(session.handles[handle])))):
                            reviewed = True
                            previous_calls = None
                            result = toolbox.pre_submit(handle, budget - number, verify)
                        else:
                            result = toolbox.execute(name, args, allowed)
                messages.append(dict(role='tool', tool_call_id=call['id'], content=result.text))
                images.extend(image_blocks(result) if result.images else [])
                internal = result.internal
                call_records.append(dict(tool=name, tool_call_id=call['id'], raw_arguments=raw_args, args=args,
                    ok=result.public['ok'], error_code=result.public.get('error_code'), result_text=result.text,
                    result_images=result.images, input_handles=referenced_handles(args),
                    output_handles=internal.get('handles_created', []), handles_created=internal.get('handles_created', []),
                    current_before=before, current_after=session.current_mask, cache_hit=internal.get('cache_hit'),
                    backend_seconds=internal.get('backend_seconds', 0.), seconds=internal.get('seconds', 0.),
                    status=internal.get('status', 'executed'), provenance=internal,
                    set_current=args.get('set_current', True) if isinstance(args, dict) and name in
                        ('to_mask','combine','morph','remove_region','add_region','refine_region') else None))
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
            if not calls:
                no_tools += 1
                if no_tools == 3:
                    reason = 'no_submit'
                    break
                # Pseudo-calls written as text prime the next reply to do the same, even under
                # tool_choice=required, so the reply stays in the trace but leaves the context.
                messages.pop()
                if repeated:
                    messages.append(dict(role='user', content='You repeated your previous tool call, so it did not run again. Its result is above. Choose a different action, or call submit with an existing mask and a summary.'))
                elif no_tools == 1:
                    messages.append(dict(role='user', content='Your last reply contained no function call, so nothing ran. Writing a tool call as text does not execute it. Make a function call now, or call submit with an existing mask and a summary.'))
            else:
                no_tools = 0
            left = budget - number
            if 0 < left <= REVIEW_REPLIES and not reviewed and session.current_mask is not None:
                reviewed = True
                late = toolbox.pre_submit(session.current_mask, left, verify)
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
    if verify and reason != 'error':
        # Whatever ended the run, the trace holds an answer for every component of the final mask.
        _, _, meta = toolbox.stage(session.submitted)
        recorder.ask_spending(meta, 'final-stage')
    write_json(toolbox.directory / 'verification.json', toolbox.ledger(session.submitted))
    return recorder.final(session, task, submitted_by, reason, fallback, rounds, planner, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=list(INSTRUCTIONS), required=True)
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
    parser.add_argument('--ask-correspondence', action='store_true', help='Ask about the object across the dates.')
    parser.add_argument('--ask-model', default='qwen3.7-plus', help='Model of the api ask backend.')
    parser.add_argument('--pixel-thr', type=float, default=.5, help='SAM3 soft mask threshold of segment_text.')
    parser.add_argument('--init-name', default='sam3_imc', help='Directory under init/ with the T5.1 given masks.')
    parser.add_argument('--prompt-version', default='v1', choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--seed-offset', type=int, default=0, help='Repeat runs of one configuration.')
    parser.add_argument('--batch', help='Budget ledger name; parallel jobs of one batch share it.')
    parser.add_argument('--max-cost-cny', type=float, help='Stop every job of the batch above this estimated cost.')
    args = parser.parse_args()
    skills = sorted(p.name for p in tool_module.SKILLS.iterdir() if (p / 'SKILL.md').exists()) if args.skills == 'all' else [x for x in args.skills.split(',') if x]
    guard = RunGuard(RUN / 'STOP.json', RUN / 'budget' / f'{args.batch}.txt' if args.batch else None, args.max_cost_cny)
    if guard.stopped():
        raise SystemExit(f'{guard.active_stop()} exists; check quota or budget, then delete it before running: {guard.active_stop().read_text()}')
    todo = Path(args.names_file).read_text().split() if args.names_file else names(args.split)
    if args.limit is not None:
        todo = todo[:args.limit]
    shared = SharedPrompter()
    def process(index_name):
        index, name = index_name
        directory = RUN / args.config / args.task / args.split / Path(name).stem
        if (directory / 'final.json').exists():
            return name, 'skipped'
        if guard.stopped():
            return name, 'stopped'
        recorder = Recorder(directory, args.split, name)
        try:
            rgb = load_rgb(args.split, name)
        except Exception:
            recorder.input_failure(args.task, dict(session_id=Path(name).stem), traceback.format_exc())
            return name, 'error'
        init_path = RUN / 'init' / args.init_name / args.split / (Path(name).stem + '.png')
        # Recorder is created before optional init loading so an absent initial asset gets an error record.
        planner = Planner(guard=guard, parameters=dict(model=args.planner_model), seed=(sum((i+1)*ord(c) for i,c in enumerate(name)) + 1000 * args.seed_offset) % (2**31))
        init = None
        init_error = None
        if args.task == 'T5.1':
            try:
                init = load_mask_png(init_path)
            except Exception:
                init_error = traceback.format_exc()
        ask = LocalAsk(args.ask_worker) if args.ask_backend == 'local' else ApiAsk(guard=guard, model=args.ask_model, correspondence=args.ask_correspondence) if args.ask_backend == 'api' else None
        toolbox = Toolbox(rgb, directory, SegmentBackend(args.split, name, args.sam3_worker, args.pixel_thr),
            shared.for_image(args.split, name), ask, init_mask=init)
        try:
          result = run_session(toolbox, planner, recorder, args.task, prompt_version=args.prompt_version, review=args.review, status=args.status, verify=args.verify_stage, plan=args.plan, skills=skills,
            metadata=dict(session_id=Path(name).stem, source_images=[str(WORKSPACE / 'datasets' / 'LEVIR-CD' / f'{args.split}_original' / part / name) for part in ('A','B')],
                init_mask_path=str(init_path) if args.task == 'T5.1' else None,
                init_config=dict(route=args.init_name, text='building', tile=512, score_thr=.5,
                                 overlap_thr=.25, match_basis=tool_module.MATCH_BASIS) if args.task == 'T5.1' else None,
                backends=dict(ask=args.ask_backend, review=args.review, status=args.status, verify_stage=args.verify_stage, plan=args.plan, skills=skills, ask_correspondence=args.ask_correspondence, ask_model=args.ask_model if args.ask_backend == 'api'
                    else 'Qwen3.5-4B' if args.ask_backend == 'local' else None,
                    sam3_worker=bool(args.sam3_worker), pixel_thr=args.pixel_thr, sam_prompt='shared_cpu', sam_model='SAM ViT-H'),
                init_error=init_error))
        except RunStopped:
            return name, 'stopped'  # no final.json, so a later run redoes this image
        return name, result['submitted_by']
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, item): item[1] for item in enumerate(todo)}
        for future in as_completed(futures):
            try:
                print(*future.result(), flush=True)
            except Exception as exc:
                # Setup failures (including unavailable reference labels) remain visible per image.
                directory = RUN / args.config / args.task / args.split / Path(futures[future]).stem
                directory.mkdir(parents=True, exist_ok=True)
                from common import write_json
                write_json(directory / 'final.json', dict(task_id=args.task, submitted_by='error',
                    termination_reason='input_error', fallback_kind='empty', error_type=type(exc).__name__,
                    error=traceback.format_exc(),
                    summary='系统兜底：输入数据无法读取，无法生成掩码。', elapsed_seconds=0))
                print(futures[future], 'input_error', flush=True)
    if guard.stopped():
        raise SystemExit(3)  # a failed job state, so a stop cannot pass as a finished run


if __name__ == '__main__':
    main()
