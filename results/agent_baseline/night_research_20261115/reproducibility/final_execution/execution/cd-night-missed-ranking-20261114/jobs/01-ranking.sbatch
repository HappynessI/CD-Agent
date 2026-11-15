#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --dependency=afterok:186417:186425
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-ranking-20261114/logs/ranking-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-ranking-20261114/logs/ranking-%j.err
set -euo pipefail
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/rank-v1/code
export BAILIAN_CSV=/guisongxia01/pangchao/wangyihan/wyh/默认业务空间-apiKey-6214720.csv
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night missed rankings $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114; logs=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-ranking-20261114/logs" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_prepare_report.sbatch"
export NIGHT_VERIFIER=$(/usr/bin/python3 -c 'import json;print(json.load(open("/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/selected_verifier.json"))["selected_verifier"])')
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_rank.sbatch" verify --split all
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_rank.sbatch" score
export PYTHONPATH="$NIGHT_MISSED_CODE:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python -B -m agent_baseline.analysis.night_missed_report --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114
