#!/usr/bin/env bash
# Usage: sbatch probe/exp6_timing.sbatch --fold 0 --max_iters 50000 --report PATH
#SBATCH --job-name=exp6-timing
#SBATCH --partition=gre
#SBATCH --exclude=gpu06,gpu37
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/logs/%x-%j.err
set -euo pipefail
PROBE=/guisongxia01/pangchao/wangyihan/Chang-Agent/probe
JOB_TMP_ROOT=/tmp exec bash "$PROBE/run.sbatch" vlm exp6_timing.py "$@"
