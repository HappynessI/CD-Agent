#!/usr/bin/env bash
#SBATCH --job-name=missed-pilot-verify
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
: "${MISSED_GROUP:?second_appeared|second_disappeared|levir_any}"
: "${MISSED_VARIANT:?four|som|structured}"
: "${BAILIAN_CSV:?Existing credentials export path}"
ENTRY=/guisongxia01/pangchao/wangyihan/wyh/cd-missed-verifier/code/agent_baseline/analysis/run_missed_verifier_study.sbatch
bash "$ENTRY" verify --group "$MISSED_GROUP" --variant "$MISSED_VARIANT"
bash "$ENTRY" score
