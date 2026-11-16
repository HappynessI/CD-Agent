#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/04-t44-interior-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-task-reconstruct-20261116/logs/04-t44-interior-%j.err
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_task_reconstruct_t44_interior.sbatch
