#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/r1-joint-confirm-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t3-20261114/logs/r1-joint-confirm-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/r1-joint-v2/code:/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/r1-joint-v2/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="T3 fixed R1 joint confirmation $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm; API0" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" - <<'CHECK'
from pathlib import Path
import json
root=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune')
records=[json.loads(p.read_text()) for p in sorted((root/'tasks').glob('*/ranking_provenance.json'))]
result=dict(tasks=len(records),all_delta_top8_equal=len(records)==30 and all(r['cached_delta_top8']==r['recomputed_delta_top8'] for r in records))
(root/'delta_order_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert result['all_delta_top8_equal']
CHECK
"$PY" -B -m agent_baseline.analysis.night_t3_r1_joint --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --night /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114 --split confirm
