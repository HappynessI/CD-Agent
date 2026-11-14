#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --dependency=afterok:186439
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t33-proposals-20261115/logs/prepare-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t33-proposals-20261115/logs/prepare-%j.err
set -euo pipefail
export T3_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/t33-retrieval-v1
bash "$T3_CODE/tools/run_night_t33_retrieval.sh" B0 prepare
