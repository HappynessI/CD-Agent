#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/candidate-quality-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/candidate-quality-%j.err
set -euo pipefail
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/candidate-quality-v1/code
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_candidate_quality.sbatch"
