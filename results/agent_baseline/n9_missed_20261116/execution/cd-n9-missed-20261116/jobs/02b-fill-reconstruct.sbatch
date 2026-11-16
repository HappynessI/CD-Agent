#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/02b-fill-reconstruct-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/02b-fill-reconstruct-%j.err
set -euo pipefail
export N9_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code
bash "$N9_MISSED_CODE/agent_baseline/analysis/run_n9_missed_online_tests.sbatch"
bash "$N9_MISSED_CODE/agent_baseline/analysis/run_n9_missed_fill.sbatch"
bash "$N9_MISSED_CODE/agent_baseline/analysis/run_n9_missed_prepare.sbatch"
