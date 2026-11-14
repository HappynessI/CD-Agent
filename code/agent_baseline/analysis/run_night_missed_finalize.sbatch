#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
OUT="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
export NIGHT_MISSED_CODE="$CODE" PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night missed offline report $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT/night_report.md" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_night_missed_report agent_baseline.tests.test_night_verifier_by_task agent_baseline.tests.test_night_missed_rank agent_baseline.tests.test_night_missed_rank_inventory agent_baseline.tests.test_night_missed_candidates
# Reuse the full views scoring from 186425; never overwrite its original output.
# Submit this stage after views completes, not while score_summary is tune-only.
test -s "$OUT/score_summary.json"
# Offline only. Never rewrite either frozen selection or launch verify/prepare.
if [[ -f "$OUT/selected_verifier_by_task.json" && ! -f "$OUT/score_verifier_by_task.json" ]]; then
  "$PY" -B -m agent_baseline.analysis.night_verifier_by_task score --report "$OUT/score_summary.json" --selection "$OUT/selected_verifier_by_task.json" --output "$OUT/score_verifier_by_task.json"
fi
export NIGHT_VERIFIER=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_verifier"])' "$OUT/selected_verifier.json")
if [[ -f "$OUT/rank_$NIGHT_VERIFIER/manifest.json" ]]; then
  bash "$CODE/agent_baseline/analysis/run_night_missed_rank.sbatch" score
fi
"$PY" -B -m agent_baseline.analysis.night_missed_report --out "$OUT"
# Retain each offline report revision as a job-specific review artifact.
ARCHIVE="$OUT/report_revisions/job_$SLURM_JOB_ID"
mkdir -p "$ARCHIVE"
cp "$OUT/night_report.json" "$OUT/night_report.md" "$ARCHIVE/"
