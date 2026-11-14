#!/usr/bin/env bash
#SBATCH --job-name=missed-sources-full
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
bash /guisongxia01/pangchao/wangyihan/wyh/cd-missed-verifier/code/agent_baseline/analysis/run_missed_verifier_study.sbatch tests
bash /guisongxia01/pangchao/wangyihan/wyh/cd-missed-verifier/code/agent_baseline/analysis/run_missed_verifier_study.sbatch online_selection
exec bash /guisongxia01/pangchao/wangyihan/wyh/cd-missed-verifier/code/agent_baseline/analysis/run_missed_verifier_study.sbatch sources
