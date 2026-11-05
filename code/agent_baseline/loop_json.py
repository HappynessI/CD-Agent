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
from prompts import scope_card
from .tools import Toolbox, PLAN_SCHEMA, SKILL_SCHEMA, failure, skill_index
from . import tools as tool_module
from .state import Mask, Session
from .planner import Planner, RunGuard, RunStopped, usage_tokens
from .recorder import Recorder
from .backends import SegmentBackend, SharedPrompter, LocalAsk, ApiAsk

from .tasks import INSTRUCTIONS, instruction_for, seed_for, read_tasks, image_name, load_pair, needs_init

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))


def system_prompt(version, budget, session=None):
    path = Path(__file__).parent / 'prompts' / f'system_{version}.txt'
    template = path.read_text()
    target = session.target if session else 'building'
    direction = session.change_direction if session else 'any'
    card = scope_card(target, direction, session.scope if session else None)
    # Earlier templates retain their text on the default route too.
    rendered = template.replace('buildings', card['plural']).replace('building', target)
    text = rendered.format(counts='\n'.join('- ' + s for s in card['counts']),
                           not_counts='\n'.join('- ' + s for s in card['not_counts']),
                           target=target, plural=card['plural'], budget=budget)
    if direction != 'any':
        text += f'\nCount only changes in the {direction} direction.'
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
EXTRA_REPLIES = 2  # G2: one grant per session to act on a report that would otherwise come too late.


def latest_mask(session):
    masks = [key for key, value in session.handles.items() if isinstance(value, Mask)]
    return masks[-1] if masks else None


def run_session(toolbox, planner, recorder, task='T1.1xC0', budget=30, prompt_version='v1', metadata=None, review=False, status=False, verify=False, plan=False, skills=(), gate2=False):
    session = toolbox.session
    gate2 = gate2 and verify
    limit, grant = budget, None
    rounds, no_tools = 0, 0
    previous_calls = None
    reason, error = 'budget', None
    reviewed = not (review or verify)
    schemas = [*tool_module.session_schemas(session, toolbox.python_enabled), *([PLAN_SCHEMA] if plan else []), *([SKILL_SCHEMA] if skills else [])]
    toolbox.skills = tuple(skills)
    messages = []
    planner.record = recorder.request
    try:
        path, template, system = system_prompt(prompt_version, budget, session)
        if skills:
            # Index only; a skill's text enters the conversation when the planner loads it.
            system += '\n\nSkills\nThese notes hold working methods. Before your first other tool call, call load_skill for the skill that fits the start of your work. Later, load another one whenever its description fits what you are about to do.\n' + skill_index(skills)
        instruction = instruction_for(task, session)
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
            target=session.target, change_direction=session.change_direction, python=toolbox.python_enabled,
            image_shape=list(session.shape), system_prompt_version=prompt_version, system_prompt_path=str(path),
            system_prompt_template=template, system_prompt=system, tool_schema_version='v2', schemas=schemas,
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
            more_to_check = False
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
                        known = name == 'submit' and handle in allowed and isinstance(session.handles[handle], Mask)
                        owed = known and (toolbox.owed if gate2 else toolbox.pending)(session.handles[handle])
                        if known and number < limit and (not reviewed or (verify and owed)):
                            reviewed = True
                            previous_calls = None
                            result = toolbox.pre_submit(handle, limit - number, verify)
                        elif known and gate2 and grant is None and number == limit and owed:
                            # The last reply: its report still reaches the planner, with replies to act on it.
                            grant = dict(kind='last_reply', at_reply=number)
                            limit += EXTRA_REPLIES
                            reviewed = True
                            previous_calls = None
                            result = toolbox.pre_submit(handle, EXTRA_REPLIES, verify)
                        else:
                            result = toolbox.execute(name, args, allowed)
                            # ask checks a batch per call, so the same empty call is the way to get the next batch.
                            more_to_check = more_to_check or (gate2 and name == 'ask' and bool(result.public.get('remaining_unverified')))
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
                        ('to_mask','combine','morph','remove_region','add_region','refine_region','python') else None))
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
                recorder.ask_spending(late.internal, f'report-{number}')
                recorder.note(number, late)
                messages.append(dict(role='user', content=note + late.public['report']))
            elif 0 < left <= REVIEW_REPLIES and not reviewed and session.current_mask is not None:
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
    if gate2:
        write_json(toolbox.directory / 'gate.json', dict(grant=grant, rounds=rounds, budget=budget,
            extra_used=rounds - grant['at_reply'] if grant else 0))
    return recorder.final(session, task, submitted_by, reason, fallback, rounds, planner, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=list(INSTRUCTIONS))
    parser.add_argument('--tasks', help='Task manifest JSONL; optionally filter by --task.')
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
    parser.add_argument('--ask-correspondence', action='store_true', help='Ask about the object across the dates.')
    parser.add_argument('--ask-model', default='qwen3.7-plus', help='Model of the api ask backend.')
    parser.add_argument('--pixel-thr', type=float, default=.5, help='SAM3 soft mask threshold of segment_text.')
    parser.add_argument('--init-name', default='sam3_imc', help='Directory under init/ with the T5.1 given masks.')
    parser.add_argument('--prompt-version', default='v1', choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--seed-offset', type=int, default=0, help='Repeat runs of one configuration.')
    parser.add_argument('--batch', help='Budget ledger name; parallel jobs of one batch share it.')
    parser.add_argument('--max-cost-cny', type=float, help='Stop every job of the batch above this estimated cost.')
    args = parser.parse_args()
    if not args.tasks and not args.task:
        parser.error('--task is required without --tasks')
    skills = sorted(p.name for p in tool_module.SKILLS.iterdir() if (p / 'SKILL.md').exists()) if args.skills == 'all' else [x for x in args.skills.split(',') if x]
    guard = RunGuard(RUN / 'STOP.json', RUN / 'budget' / f'{args.batch}.txt' if args.batch else None, args.max_cost_cny)
    if guard.stopped():
        raise SystemExit(f'{guard.active_stop()} exists; check quota or budget, then delete it before running: {guard.active_stop().read_text()}')
    todo = read_tasks(args.tasks, args.task) if args.tasks else (Path(args.names_file).read_text().split() if args.names_file else names(args.split))
    if args.limit is not None:
        todo = todo[:args.limit]
    if args.tasks:
        keys = [(r['task_type'], Path(r['id']).stem) for r in todo]
        if len(set(keys)) != len(keys):
            parser.error('Run each direction in a separate config/manifest to avoid output collisions.')
    shared = SharedPrompter()
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
        recorder = Recorder(directory, args.split, name, gt_path=row['gt_mask'] if row else None)
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
            scope=row.get('scope') if row else None)
        ask = LocalAsk(args.ask_worker) if args.ask_backend == 'local' else ApiAsk(guard=guard, model=args.ask_model, correspondence=args.ask_correspondence, session=session) if args.ask_backend == 'api' else None
        source_images = [str(WORKSPACE / 'datasets' / 'LEVIR-CD' / f'{args.split}_original' / part / name) for part in ('A','B')]
        custom_images = row is not None and [row['image_t1'], row['image_t2']] != source_images
        image_paths = [row['image_t1'], row['image_t2']] if custom_images else None
        toolbox = Toolbox(rgb, directory, SegmentBackend(args.split, name, args.sam3_worker, args.pixel_thr,
            target=session.target, image_paths=image_paths),
            shared.for_image(args.split, name, rgb=rgb if custom_images else None), ask, session=session, python=args.python)
        try:
            result = run_session(toolbox, planner, recorder, task, prompt_version=args.prompt_version, review=args.review, status=args.status, verify=args.verify_stage, plan=args.plan, skills=skills, gate2=args.gate2,
                metadata=dict(**({'scope': session.scope} if session.scope else {}), session_id=stem, source_images=image_paths or source_images,
                    init_mask_path=str(init_path) if init is not None or needs_init(task) else None,
                    init_config=dict(route=args.init_name, text=session.target, tile=512, score_thr=.5,
                        overlap_thr=.25, match_basis=tool_module.MATCH_BASIS) if needs_init(task) else None,
                    backends=dict(**({'gate2': True} if args.gate2 else {}), ask=args.ask_backend, review=args.review, status=args.status, verify_stage=args.verify_stage,
                        plan=args.plan, skills=skills, ask_correspondence=args.ask_correspondence,
                        ask_model=args.ask_model if args.ask_backend == 'api' else 'Qwen3.5-4B' if args.ask_backend == 'local' else None,
                        sam3_worker=bool(args.sam3_worker), pixel_thr=args.pixel_thr, sam_prompt='shared_cpu', sam_model='SAM ViT-H'),
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
    if guard.stopped():
        raise SystemExit(3)  # a failed job state, so a stop cannot pass as a finished run


if __name__ == '__main__':
    main()
