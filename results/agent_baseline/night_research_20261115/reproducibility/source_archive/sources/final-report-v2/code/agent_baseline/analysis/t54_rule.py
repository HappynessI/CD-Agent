"""T5.4 flash correspondence verifier on each XOR component; delete parsed no only.

The map-specific question enters both offline and online verifiers via Session.
Reuses SECOND rendering, flash parameters, per-component checkpoint and cost cap.
"""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
from agent_baseline.analysis.second_rule import run as second_rule
from agent_baseline.map_input import check_task
from agent_baseline.tasks import read_tasks


def run(rows, mask_dir, output, max_cost, concurrency=1, ask_variant=None):
    if not rows or any(row['task_type'] != 'T5.4' for row in rows):
        raise ValueError('Use a nonempty T5.4 manifest')
    for row in rows:
        check_task(row)
    if ask_variant not in (None, 't2_state'):
        raise ValueError('Unknown map ask variant')
    return second_rule(rows, mask_dir, output, max_cost, concurrency, ask_variant=ask_variant)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-ask-variant', choices=['t2_state'])
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-cost-cny', type=float, required=True)
    p.add_argument('--concurrency', type=int, default=1)
    a = p.parse_args()
    print(json.dumps(run(read_tasks(a.tasks), a.mask_dir, a.output_dir, a.max_cost_cny, a.concurrency, a.map_ask_variant)))


if __name__ == '__main__':
    main()
