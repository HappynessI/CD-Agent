#!/usr/bin/env bash
#SBATCH --job-name=missed-pilot-prepare
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
exec bash /guisongxia01/pangchao/wangyihan/wyh/cd-missed-verifier/code/agent_baseline/analysis/run_missed_verifier_study.sbatch prepare --images 10
