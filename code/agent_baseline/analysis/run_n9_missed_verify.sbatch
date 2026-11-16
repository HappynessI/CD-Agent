#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${BAILIAN_CSV:?Main agent provides the existing credential file path}"
PART=${1:?tune or confirm}
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N9_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_n9_missed_protocol agent_baseline.tests.test_n9_missed_pipeline
"$PY" -B -m agent_baseline.analysis.n9_missed_pipeline prepare --split "$PART"
"$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.n9_api_preflight --out "$ROOT/CD-Agent/results/agent_baseline/n9_missed_20261116/preflight/${PART}-${SLURM_JOB_ID}.json" -- "$PY" -B -m agent_baseline.analysis.n9_missed_pipeline verify --split "$PART" --budget "${N9_FM_BUDGET:-20}"
"$PY" -B -m agent_baseline.analysis.n9_missed_score --split "$PART"
