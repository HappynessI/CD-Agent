#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/audit-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/audit-%j.err
set -euo pipefail

cd /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review
export PYTHONPATH=code
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source_repo=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent
python_bin=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$python_bin" -B -m agent_baseline.make_site --catalog --stage page \
    --source-repo "$source_repo" --cache .runtime/demo-g3-review/cache \
    --out-dir results/agent_baseline/demo_g3
