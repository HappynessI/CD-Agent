#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-online-prep-tests-20261115/logs/tests-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-online-prep-tests-20261115/logs/tests-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/missed-online-prep-v1/code:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Ranked provider preparation $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/online_preparation_tests; Agent sessions0; API0" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python -B /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/missed-online-prep-v1/code/test_online_preparation.py
