#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/confirm-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/confirm-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/code PYTHONDONTWRITEBYTECODE=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/tools/run_n7_t3_rejection.sh confirm
"$PY" -B -m agent_baseline.analysis.n7_t3_diagnostics --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
"$PY" -B -m unittest agent_baseline.tests.test_n7_t3_verify
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/tools/run_n7_t3_verify.sh tune reuse-only
