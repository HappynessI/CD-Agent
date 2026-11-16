#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/03-page-preview-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/03-page-preview-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/CD-Agent
export PYTHONPATH="$PWD/code" PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.task_reconstruct_page
