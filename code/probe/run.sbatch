#!/usr/bin/env bash
# Usage: sbatch [resources] --job-name=NAME --output=$OUT/logs/%x-%j.out probe/run.sbatch ENV SCRIPT [ARGS...]
#   ENV: omniovcd | vlm   SCRIPT: file name inside probe/
#SBATCH --partition=gre
#SBATCH --exclude=gpu06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
set -euo pipefail
PROBE=/guisongxia01/pangchao/wangyihan/Chang-Agent/probe
ENVS=/guisongxia01/pangchao/wangyihan/conda_envs
case "$1" in
  omniovcd) PY=$ENVS/omniovcd-env/bin/python ;;
  vlm) PY=$ENVS/vlm-vs-onejev-env/bin/python ;;
  *) echo "unknown env $1"; exit 2 ;;
esac
shift
STARTED=$(date +%s)
JOB_TMP=
CMD="$*"
finish() {
  code=$?
  trap - EXIT
  rm -rf "$JOB_TMP"
  status=$([ "$code" -eq 0 ] && echo success || echo "failed(exit $code)")
  QQ_SUB="CD_1016_probe ${SLURM_JOB_NAME} ${status}" \
  QQ_BODY="job ${SLURM_JOB_ID} ${SLURM_JOB_NAME}: ${status}; elapsed $(( $(date +%s) - STARTED ))s; cmd: ${CMD}; output: /guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe (log in logs/)" \
  /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || echo "QQ notification failed"
  exit "$code"
}
trap finish EXIT
# Node-local /tmp can be full (seen on gpu37); keep compiler/triton scratch per job in the workspace.
JOB_TMP=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/.tmp/${SLURM_JOB_ID}
mkdir -p "$JOB_TMP"
export TMPDIR=$JOB_TMP TRITON_CACHE_DIR=$JOB_TMP/triton
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$PROBE"
echo "node=$(hostname -s) job=${SLURM_JOB_ID} cmd=$*"
"$PY" "$@"
