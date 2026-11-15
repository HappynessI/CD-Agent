#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-validation-20261114/logs/validate-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-validation-20261114/logs/validate-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/followup-v1/code:/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/followup-v1/code/ovcd_v0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest -v agent_baseline.tests.test_night_t3_retrieval agent_baseline.tests.test_night_t33_grounding agent_baseline.tests.test_night_t33_retrieval agent_baseline.tests.test_night_t3_blind_review agent_baseline.tests.test_night_t3_mask_panel agent_baseline.tests.test_night_missed_rank agent_baseline.tests.test_night_verifier_selection
"$PY" -B -m agent_baseline.analysis.night_t33_grounding --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114 --variant audit
"$PY" -c 'import json;from pathlib import Path;Path("/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_validation_20261114/metric.json").write_text(json.dumps(dict(stage="followup_validation_and_T33_audit",completed=1,cost_cny=0)))'
