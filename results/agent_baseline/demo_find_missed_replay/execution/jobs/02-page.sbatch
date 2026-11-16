#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-missed-replay-20261116/logs/02-page-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-missed-replay-20261116/logs/02-page-%j.err
set -euo pipefail
ROOT=/guisongxia01/pangchao/wangyihan/wyh
cd "$ROOT/CD-Agent"
export PYTHONPATH="$ROOT/CD-Agent/code:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.missed_replay_page
mkdir -p results/agent_baseline/demo_find_missed_replay/validation
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/browsers"
export PLAYWRIGHT_MODULE="$ROOT/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/browser-tools/node_modules/playwright"
export LD_LIBRARY_PATH="$ROOT/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/browser-deps/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
node tools/agent_baseline/check_missed_replay_browser.cjs results/agent_baseline/demo_find_missed_replay

/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B - <<'PACK'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path('results/agent_baseline/demo_find_missed_replay')
with ZipFile(root/'download.zip','w',compression=ZIP_DEFLATED) as z:
    for p in [root/'index.html',root/'page_summary.json',*sorted((root/'assets').glob('*.png'))]:z.write(p,p.relative_to(root))
with ZipFile(root/'download.zip') as z:assert z.testzip() is None
print('Portable HTML + assets bundle verified.')
PACK
