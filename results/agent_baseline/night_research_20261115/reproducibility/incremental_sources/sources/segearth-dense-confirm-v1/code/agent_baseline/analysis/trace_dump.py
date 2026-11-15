"""Print one Agent session step by step (planner thought, tool calls, short results, current mask F1).

Offline, no API. Usage: python -m agent_baseline.analysis.trace_dump <session_dir>
Used for the T1.3 G4 trajectory review (2026-11-12).
"""
import json
import sys
from pathlib import Path


def dump(folder, width=700):
    folder = Path(folder)
    final = json.loads((folder / 'final.json').read_text())
    ev = final['eval']
    yield (f"FINAL {final['final_handle']} {final['termination_reason']} f1={ev['f1']:.3f} "
           f"P={final['precision']:.3f} R={final['recall']:.3f} rounds={final['planning_rounds']} "
           f"calls={final['tool_calls']} cost={final['cost_cny']:.4f}")
    yield 'SUMMARY ' + final['summary'][:400]
    for line in (folder / 'steps.jsonl').read_text().splitlines():
        step = json.loads(line)
        if not step['calls']:
            continue
        thought = ((step.get('planner') or {}).get('thought') or '').strip().replace('\n', ' ')
        f1 = (step.get('eval') or {}).get('f1') if isinstance(step.get('eval'), dict) else None
        yield f"\n## step {step['step']} current={step.get('current_mask')} f1={f1}"
        if thought:
            yield '  THOUGHT: ' + thought[:500]
        for call in step['calls']:
            status = 'ok' if call['ok'] else 'ERR ' + str(call['error_code'])
            yield f"  CALL {call['tool']} {json.dumps(call['args'], ensure_ascii=False)[:300]} {status}"
            text = call.get('result_text') or ''
            try:
                result = json.loads(text)
                for key in ('view_id', 'display_size'):
                    result.pop(key, None)
                text = json.dumps(result, ensure_ascii=False)
            except ValueError:
                pass
            yield '    -> ' + text[:width]


if __name__ == '__main__':
    print('\n'.join(dump(sys.argv[1])))
