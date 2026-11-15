#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --dependency=afterok:186463:186741
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-report-20261115/logs/refresh-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-missed-report-20261115/logs/refresh-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/final-report-v6/code:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night missed final report $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/report_revisions/job_$SLURM_JOB_ID; API0" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
export NIGHT_RANK_REFERENCE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/rank_V0
export NIGHT_MISSED_CODE=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/final-report-v6/code
export NIGHT_VERIFIER=V0
mkdir -p "$NIGHT_RANK_REFERENCE/aggregation_revision"
cp "$NIGHT_RANK_REFERENCE/score_summary.json" "$NIGHT_RANK_REFERENCE/aggregation_revision/original_score_summary.json"
"$PY" -B -m unittest agent_baseline.tests.test_night_missed_report agent_baseline.tests.test_night_verifier_by_task agent_baseline.tests.test_night_missed_rank agent_baseline.tests.test_night_missed_rank_inventory agent_baseline.tests.test_night_missed_candidates agent_baseline.tests.test_night_rank_aggregate
bash "$NIGHT_MISSED_CODE/agent_baseline/analysis/run_night_missed_rank.sbatch" aggregate
"$PY" - <<'CHECK'
import os,json
from pathlib import Path
p=Path(os.environ['NIGHT_RANK_REFERENCE']);old=json.loads((p/'aggregation_revision/original_score_summary.json').read_text());new=json.loads((p/'score_summary.json').read_text());n=0
for group,splits in old['groups'].items():
 for split,ranks in splits.items():
  for rank,ks in ranks.items():
   for k,a in ks.items():
    b=new['groups'][group][split][rank][k]
    for field in ('metrics','delta_vs_ungated_same_verifier','paired_image_bootstrap95'):assert a[field]==b[field]
    n+=1
(p/'aggregation_revision/regression.json').write_text(json.dumps(dict(complete=True,comparisons=n,images=len(new['images']),pixel_scoring_repeated=False,new_api_requests=0,job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
CHECK
"$PY" -B -m agent_baseline.analysis.night_missed_report --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114
mkdir -p /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/report_revisions/job_$SLURM_JOB_ID
cp /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/night_report.json /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/night_report.md /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/report_revisions/job_$SLURM_JOB_ID/
