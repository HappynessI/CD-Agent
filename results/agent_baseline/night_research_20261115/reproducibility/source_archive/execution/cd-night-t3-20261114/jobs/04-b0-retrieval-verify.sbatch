#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --dependency=afterok:186439
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/retrieval-verify-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/retrieval-verify-%j.err
set -euo pipefail
export T3_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/retrieval-v2
/usr/bin/python3 -c 'import json;from pathlib import Path;d=json.loads(Path("/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/retrieval/B0/prepare_progress_all.json").read_text());assert d["tasks"]==60,d["tasks"]'
bash "$T3_CODE/tools/run_night_t3_retrieval.sh" B0 all
