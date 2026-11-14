#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-task-policy-20261115/logs/policy-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-task-policy-20261115/logs/policy-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/task-policy-v1/code PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_night_verifier_by_task
"$PY" -B -m agent_baseline.analysis.night_verifier_by_task select --report /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/score_tune_frozen.json --selection /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/selected_verifier_by_task.json
