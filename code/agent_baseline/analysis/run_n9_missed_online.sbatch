#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${BAILIAN_CSV:?Main agent provides the workspace credential CSV}"
LINE=${1:?Positive dataset task line}
SEED=${2:?Frozen seed 1016 1017 or 1018}
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N9_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0:/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.n9_missed_online run --group "$LINE" --seed "$SEED" --budget "${N9_FM_BUDGET:-20}"
