#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${BAILIAN_CSV:?Existing workspace credential export path}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N8_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
SPLIT=${1:-tune}
"$PY" -B -m unittest agent_baseline.tests.test_n8_missed_prepare agent_baseline.tests.test_n8_missed_pipeline
"$PY" -B -m agent_baseline.analysis.n8_missed_pipeline prepare --split "$SPLIT"
"$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.n8_missed_pipeline verify --split "$SPLIT"
