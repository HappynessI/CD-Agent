#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/03-tune-verify-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/03-tune-verify-%j.err
set -euo pipefail
export N9_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed/code
export BAILIAN_CSV=/guisongxia01/pangchao/wangyihan/wyh/默认业务空间-apiKey-6214720.csv
export N9_FM_BUDGET=20
bash "$N9_MISSED_CODE/agent_baseline/analysis/run_n9_missed_verify.sbatch" tune
