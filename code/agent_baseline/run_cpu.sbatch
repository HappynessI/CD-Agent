#!/usr/bin/env bash
#SBATCH --job-name=agent-cpu
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.err
# Pre-create AGENT_RUN/logs before sbatch; override --output/--error for a custom root.
set -euo pipefail
WORKSPACE=${WORKSPACE:-/guisongxia01/pangchao/wangyihan}
export AGENT_RUN=${AGENT_RUN:-$WORKSPACE/wyh/exp6_staging/agent_baseline_run}
CODE=${AGENT_CODE:-$WORKSPACE/wyh/CD-Agent/code/agent_baseline}
PY=$WORKSPACE/conda_envs/exp5-env/bin/python
mkdir -p "$AGENT_RUN/logs" "$AGENT_RUN/.tmp/${SLURM_JOB_ID:-manual}"
exec >>"$AGENT_RUN/logs/${SLURM_JOB_NAME:-agent-cpu}-${SLURM_JOB_ID:-manual}.runtime.log" 2>&1
started=$SECONDS
finish() {
    status=$?; trap - EXIT
    QQ_SUB="agent-cpu ${SLURM_JOB_ID:-manual}" \
    QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$AGENT_RUN; details in logs" \
        "$WORKSPACE/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
    exit "$status"
}
trap finish EXIT
if (($# == 0)); then
    echo 'Supply a Python entrypoint and its arguments.' >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=$AGENT_RUN/.tmp/${SLURM_JOB_ID:-manual}
cd "$CODE"
"$PY" -B "$@"
