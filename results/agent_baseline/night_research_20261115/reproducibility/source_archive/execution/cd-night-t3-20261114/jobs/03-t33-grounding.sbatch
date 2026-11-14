#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/t33-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/t33-%j.err
set -euo pipefail
export T3_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/followup-v1
bash "$T3_CODE/tools/run_night_t33_grounding.sh"
