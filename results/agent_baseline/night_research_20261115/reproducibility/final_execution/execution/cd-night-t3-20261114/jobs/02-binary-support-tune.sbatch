#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/binary-support-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/binary-support-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/followup-v1/code:/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/followup-v1/code/ovcd_v0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night T3 binary support $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/B3; logs=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m agent_baseline.analysis.night_t3_blind_review --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114 --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/grounding_blind_v1 --key /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/grounding_blind_v1_key.json
"$PY" /guisongxia01/pangchao/wangyihan/Chang-Agent/tools/run_with_bailian_csv.py /guisongxia01/pangchao/wangyihan/wyh/默认业务空间-apiKey-6214720.csv "$PY" -B -m agent_baseline.analysis.night_t3_mask_panel --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114 --split tune
