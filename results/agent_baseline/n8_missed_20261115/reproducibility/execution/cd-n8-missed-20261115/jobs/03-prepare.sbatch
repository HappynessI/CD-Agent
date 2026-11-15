#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/prepare-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/prepare-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N8_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_n8_missed_prepare
"$PY" -B -m agent_baseline.analysis.n8_missed_prepare prepare
"$PY" -B -m agent_baseline.analysis.n8_missed_prepare semantic-select
