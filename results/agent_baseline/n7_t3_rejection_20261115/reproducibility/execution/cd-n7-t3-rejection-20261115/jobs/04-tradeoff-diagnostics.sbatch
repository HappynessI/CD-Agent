#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/tradeoff-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/tradeoff-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/code PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m agent_baseline.tests.test_n7_t3_verify
"$PY" -B -m agent_baseline.analysis.n7_t3_tradeoff --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
"$PY" -B -m agent_baseline.analysis.n7_t3_top1_audit --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
"$PY" -B -m agent_baseline.analysis.n7_t3_diagnostics --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
