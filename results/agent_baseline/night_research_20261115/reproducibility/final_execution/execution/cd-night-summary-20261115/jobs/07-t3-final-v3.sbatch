#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/report-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/report-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/t3-report-final-v6/code:/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/t3-report-final-v6/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night T3 final report $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/report_final_v3; API0" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_night_t3_report agent_baseline.tests.test_night_t3_positive_endpoints agent_baseline.tests.test_night_t3_segearth_dense
"$PY" -B -m agent_baseline.analysis.night_t3_report --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114 --old /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/t3_bridge_20261114/dev60/summary.json --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/report_final_v3
