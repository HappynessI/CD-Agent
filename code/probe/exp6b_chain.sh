#!/usr/bin/env bash
# Management-node orchestration only (sbatch/squeue): wait for a backbone's feature shards,
# then train V1, freeze thresholds on second_cv, evaluate on the given datasets.
# Usage: exp6b_chain.sh BACKBONE [--include_pred] -- DATASET...
set -euo pipefail
BB=$1; shift
EXTRA=()
if [ "${1:-}" = "--include_pred" ]; then EXTRA=(--include_pred); shift; fi
[ "${1:-}" = "--" ] && shift
DATASETS=("$@")
PROBE=/guisongxia01/pangchao/wangyihan/Chang-Agent/probe
L=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/logs
TAG=$BB${EXTRA:+-pred}
cd "$PROBE"
wait_name() { until [ -z "$(squeue -h -u guisongxia -o %j 2>/dev/null | grep -E "$1")" ]; do sleep 60; done; }
wait_job() { until [ -z "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 60; done
  state=$(sacct -j "$1" -X -n --format=State | tr -d ' '); echo "$2: $state"; [ "$state" = COMPLETED ]; }
wait_name "cd1016-6b-feat-$BB-(cv|second)-"
J=$(sbatch --parsable --exclude=gpu04,gpu06,gpu37 --job-name=cd1016-6b-train-$TAG --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=12:00:00 --output=$L/%x-%j.out --error=$L/%x-%j.err run.sbatch exp5 exp6b_train.py --backbone "$BB" "${EXTRA[@]}")
wait_job "$J" train
for d in second_cv "${DATASETS[@]}"; do
  J=$(sbatch --parsable --job-name=cd1016-6b-eval-$TAG-$d --cpus-per-task=8 --mem=64G --time=12:00:00 \
    --output=$L/%x-%j.out --error=$L/%x-%j.err run.sbatch exp5 exp6b_eval.py --backbone "$BB" "${EXTRA[@]}" --dataset "$d")
  wait_job "$J" "eval $d"
done
