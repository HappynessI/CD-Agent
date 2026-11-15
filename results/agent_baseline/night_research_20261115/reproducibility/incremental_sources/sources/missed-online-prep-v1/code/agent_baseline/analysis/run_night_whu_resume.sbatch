#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
export NIGHT_MISSED_CODE="$CODE"
# Only after the previous WHU writer finishes/stops; no extraction/evaluation
# of SECOND/LEVIR and no model requests. Existing official records are reused.
bash "$CODE/agent_baseline/analysis/run_night_missed_candidates.sbatch" whu --resume-whu
