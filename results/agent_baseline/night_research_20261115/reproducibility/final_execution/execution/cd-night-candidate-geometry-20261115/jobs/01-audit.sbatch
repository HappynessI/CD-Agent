#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-candidate-geometry-20261115/logs/audit-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-candidate-geometry-20261115/logs/audit-%j.err
set -euo pipefail
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/candidate-geometry-v1/code
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night candidate geometry $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/candidate_geometry_tune; logs=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-candidate-geometry-20261115/logs" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_candidate_geometry.sbatch"
