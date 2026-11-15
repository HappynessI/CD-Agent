#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/build-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/build-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/code
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m agent_baseline.make_site --catalog --source-repo /guisongxia01/pangchao/wangyihan/wyh/CD-Agent --run-root /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --cache /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/cache --out-dir results/agent_baseline/demo_g3 --selections /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/inventory.json
"$PY" -B -m unittest agent_baseline.tests.test_demo_g3 agent_baseline.tests.test_site_store agent_baseline.tests.test_site_svg
