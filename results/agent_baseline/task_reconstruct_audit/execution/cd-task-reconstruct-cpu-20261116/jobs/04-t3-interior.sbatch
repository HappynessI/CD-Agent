#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-cpu-20261116/logs/04-t3-interior-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-cpu-20261116/logs/04-t3-interior-%j.err
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_task_reconstruct_t3_interior.sbatch
