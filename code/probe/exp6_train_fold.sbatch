#!/usr/bin/env bash
# Usage: sbatch probe/exp6_train_fold.sbatch --fold K --max_iters N --timing_report PATH [ARGS]
#SBATCH --job-name=exp6-train
#SBATCH --partition=gre
#SBATCH --exclude=gpu06,gpu37
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/logs/%x-%j.err
set -euo pipefail
PROBE=/guisongxia01/pangchao/wangyihan/Chang-Agent/probe
JOB_TMP_ROOT=/tmp RUNNER=torchrun2 exec bash "$PROBE/run.sbatch" vlm exp6_train_fold.py "$@"
