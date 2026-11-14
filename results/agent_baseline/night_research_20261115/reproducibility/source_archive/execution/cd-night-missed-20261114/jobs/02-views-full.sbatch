#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-20261114/logs/views-full-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-20261114/logs/views-full-%j.err
set -euo pipefail
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/missed-views-v2/code
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/missed-views-v2/code:/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/missed-views-v2/code/ovcd_v0
export BAILIAN_CSV=/guisongxia01/pangchao/wangyihan/wyh/默认业务空间-apiKey-6214720.csv
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
ENTRY="$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_views.sbatch"
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night full verifier $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114; logs=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-20261114/logs" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
for variant in V1 V2 V3 V4; do bash "$ENTRY" verify --variant "$variant" --split tune; done
bash "$ENTRY" score --split tune
cp /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/score_summary.json /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/score_tune_frozen.json
"$PY" -B -m agent_baseline.analysis.night_verifier_selection --report /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/score_tune_frozen.json --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/selected_verifier.json
for variant in V1 V2 V3 V4; do bash "$ENTRY" verify --variant "$variant" --split confirm; done
bash "$ENTRY" score --split all
"$PY" -B -m agent_baseline.analysis.night_missed_review_pack --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114 --pilot /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot
