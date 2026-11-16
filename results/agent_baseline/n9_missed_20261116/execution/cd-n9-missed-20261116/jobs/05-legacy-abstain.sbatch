#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/05-legacy-abstain-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/05-legacy-abstain-%j.err
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_n9_missed_legacy_abstain.sbatch
