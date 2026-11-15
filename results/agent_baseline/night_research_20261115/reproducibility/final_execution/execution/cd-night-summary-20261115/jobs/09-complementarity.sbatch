#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --dependency=afterok:186709
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/complementarity-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/complementarity-%j.err
set -euo pipefail
export T3_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/t3-complementarity-v1
bash "$T3_CODE/tools/run_night_t3_complementarity.sh"
