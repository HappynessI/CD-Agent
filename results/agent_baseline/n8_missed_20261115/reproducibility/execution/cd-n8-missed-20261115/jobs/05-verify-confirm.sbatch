#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/verify-confirm-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/verify-confirm-%j.err
set -euo pipefail
export BAILIAN_CSV='/guisongxia01/pangchao/wangyihan/wyh/默认业务空间-apiKey-6214720.csv'
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_n8_missed_verify.sbatch confirm
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code/agent_baseline/analysis/run_n8_missed_score.sbatch confirm
