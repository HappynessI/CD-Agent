#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-missed-replay-20261116/logs/01b-online-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-missed-replay-20261116/logs/01b-online-%j.err
set -euo pipefail
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="$ROOT/CD-Agent/code:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0:/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python

"$PY" -B -m agent_baseline.run_with_key "$ROOT/默认业务空间-apiKey-6214720.csv" "$PY" -B -m agent_baseline.analysis.missed_replay_single
