#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/01-assets-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/01-assets-%j.err
set -euo pipefail
export N9_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_n9_missed_assets.sbatch
