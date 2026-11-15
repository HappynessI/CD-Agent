#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/verify-tune-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/verify-tune-%j.err
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/tools/run_n7_t3_verify.sh tune
