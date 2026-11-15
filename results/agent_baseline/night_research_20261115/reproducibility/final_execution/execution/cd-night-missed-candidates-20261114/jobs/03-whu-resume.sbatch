#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-candidates-20261114/logs/whu-resume-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-candidates-20261114/logs/whu-resume-%j.err
set -euo pipefail
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="WHU official source audit $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/candidate_gates_pilot/whu_official_summary.json; API0" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/whu-resume-v1/code
export PYTHONPATH="$NIGHT_MISSED_CODE:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python -B -m unittest agent_baseline.tests.test_night_missed_candidates
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_whu_resume.sbatch"
