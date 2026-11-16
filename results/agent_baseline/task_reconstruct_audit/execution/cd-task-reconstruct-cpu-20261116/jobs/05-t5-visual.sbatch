#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-cpu-20261116/logs/05-t5-visual-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-cpu-20261116/logs/05-t5-visual-%j.err
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/tools/run_task_reconstruct_t5_visual.sh
