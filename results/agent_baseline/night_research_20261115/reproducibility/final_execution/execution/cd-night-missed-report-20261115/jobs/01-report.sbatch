#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --dependency=afterok:186463:186475
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-report-20261115/logs/report-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-report-20261115/logs/report-%j.err
set -euo pipefail
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/final-report-v1/code
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_finalize.sbatch"
