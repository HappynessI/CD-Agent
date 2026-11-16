#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Main agent submits offline job}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="$ROOT/cd-agent-worktrees/n7-missed/code:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_n9_missed_legacy_abstain
"$PY" -B -m agent_baseline.analysis.n9_missed_legacy_abstain
