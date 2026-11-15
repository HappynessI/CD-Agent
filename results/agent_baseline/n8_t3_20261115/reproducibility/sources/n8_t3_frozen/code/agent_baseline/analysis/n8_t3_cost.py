"""Account once per returned attempt usage, including failed responses and resumes."""
import json
from .io_retry import read_text,exists
from agent_baseline.planner import request_cost,usage_tokens
from agent_baseline.exemplar_semantics import MODEL


def request_ledger(folder):
    attempts=sorted(folder.glob('attempt_*.json'))
    cost=failed_cost=0.;unknown=failed=0
    if attempts:
        for path in attempts:
            event=json.loads(read_text(path));usage=event.get('usage');bad=bool(event.get('error_type'))
            failed+=bad
            if not usage or not any(k in usage for k in ('prompt_tokens','completion_tokens')):unknown+=1;continue
            amount=request_cost(event.get('requested_model') or MODEL,usage_tokens(usage));cost+=amount
            if bad:failed_cost+=amount
        source='attempt usage only; answer/failure aggregate excluded to prevent double counting'
    else:
        # Only for artifacts without attempt records: separate failed invocations plus final successful invocation.
        for path in [folder/'answer.json',*sorted(folder.glob('failure_*.json'))]:
            if not exists(path):continue
            value=json.loads(read_text(path));amount=float(value.get('cost_cny',0));cost+=amount
            if path.name.startswith('failure_'):failed+=1;failed_cost+=amount
        source='fallback independent invocation summaries; no attempt records present'
    return dict(known_cost_cny=cost,known_failed_cost_cny=failed_cost,attempts=len(attempts),failed_attempts=failed,
                usage_missing_attempts=unknown,source=source)


def aggregate_ledgers(folders):
    totals=dict(known_cost_cny=0.,known_failed_cost_cny=0.,attempts=0,failed_attempts=0,usage_missing_attempts=0)
    for folder in folders:
        record=request_ledger(folder)
        for key in totals:totals[key]+=record[key]
    return dict(**totals,note='Returned-usage estimate, not account bill. Missing-usage transport failures are unknown and excluded from known cost, not asserted free.')


def line_ledger(root):return aggregate_ledgers(root.glob('*/tasks/*/requests/*'))
def stage_ledger(root):return aggregate_ledgers(root.glob('tasks/*/requests/*'))
