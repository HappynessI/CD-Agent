#!/usr/bin/env bash
#SBATCH --job-name=ovcd-cpu
#SBATCH --partition=gre
#SBATCH --exclude=gpu04,gpu06,gpu16,gpu20,gpu37
#SBATCH --exclude=gpu04,gpu06,gpu16,gpu20,gpu37
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/ovcd_v0_run/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/ovcd_v0_run/logs/%x-%j.err
# CPU stage of OVCD-v0 (agent replay, SAM decoder on cached embeddings, analysis, figures).
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/ovcd_v0
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/ovcd_v0_run/.tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
started=$SECONDS
notify() {
    status=$?; trap - EXIT
    QQ_SUB="ovcd-v0 ${SLURM_JOB_NAME} ${SLURM_JOB_ID}" QQ_BODY="exit=$status elapsed=$((SECONDS-started))s args=$*" \
        /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true
    exit "$status"
}
trap 'notify "$@"' EXIT
"$PY" -B "$@"
