"""Copy one round's trajectories (jsonl.gz), eval JSON (no per-image), val selection and the Codex
discussion into results/ovcd_v0/round_<n>/.

    python save_round.py --round 2 [--discussion path/to/DISCUSSION.md]
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

from common import REPO, RUN

p = argparse.ArgumentParser()
p.add_argument('--round', type=int, required=True)
p.add_argument('--discussion')
a = p.parse_args()
out = REPO / 'results' / 'ovcd_v0' / f'round_{a.round}'
out.mkdir(parents=True, exist_ok=True)
for k in (10, 20, 50):
    for s in ('val', 'test'):
        fs = sorted((RUN / 'agent' / f'r{a.round}_k{k}' / s / 'traj').glob('*.json'))
        with gzip.open(out / f'traj_r{a.round}_k{k}_{s}.jsonl.gz', 'wt') as g:
            for f in fs:
                g.write(json.dumps(json.loads(f.read_text()), ensure_ascii=False) + '\n')
        e = json.loads((RUN / 'eval' / f'r{a.round}_k{k}_{s}.json').read_text())
        e.pop('per_image', None)
        (out / f'eval_r{a.round}_k{k}_{s}.json').write_text(json.dumps(e, indent=1))
shutil.copy(RUN / 'select' / f'r{a.round}' / 'val.json', out / 'select_val.json')
if a.discussion:
    shutil.copy(a.discussion, out / 'codex_discussion.md')
print(sorted(x.name for x in out.iterdir()))
