#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/02-road-anchors-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/02-road-anchors-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/CD-Agent
export PYTHONPATH="$PWD/code" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 T44_ROAD_STAGE=1
/guisongxia01/pangchao/wangyihan/conda_envs/ovcd-de-env/bin/python -B -m agent_baseline.analysis.task_reconstruct_anchor_fill
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.task_reconstruct_t44
