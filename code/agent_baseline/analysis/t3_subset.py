"""Pick one T3.1 task per image for the Agent run (Ethan 2026-11-08: 703 tasks x 3 would exceed 100 CNY).

Images in sorted order; each takes the task whose transition has been picked least often so far,
ties broken by a seeded random draw (seed 1016). Ground truth is not used. Writes the subset manifest.
"""
import argparse
import collections
import json
import random
from pathlib import Path

SEED = 1016


def transition(row):
    return row['id'].rsplit('_', 1)[1]


def pick(rows, seed=SEED):
    rng = random.Random(seed)
    by_image = collections.defaultdict(list)
    for r in rows:
        by_image[r['id'].rsplit('_', 1)[0]].append(r)
    used, out = collections.Counter(), []
    for image in sorted(by_image):
        tasks = sorted(by_image[image], key=lambda r: r['id'])
        low = min(used[transition(r)] for r in tasks)
        choice = rng.choice([r for r in tasks if used[transition(r)] == low])
        used[transition(choice)] += 1
        out.append(choice)
    return out, used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    out, used = pick(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in out))
    print(json.dumps(dict(tasks=len(out), transitions=dict(sorted(used.items())))))


if __name__ == '__main__':
    main()
