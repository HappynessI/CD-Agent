#!/usr/bin/env bash
#SBATCH --job-name=ab-fruns
#SBATCH --partition=gre
#SBATCH --exclude=gpu04,gpu06,gpu14,gpu16,gpu20,gpu37
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.err
# All runs of a batch share one SAM3 worker on one GPU: the runs spend their time waiting for the API.
set -uo pipefail
W=/guisongxia01/pangchao/wangyihan
export AGENT_RUN=$W/wyh/exp6_staging/agent_baseline_run
export OVCD_CODE=$AGENT_RUN/code_f/ovcd_v0
CODE=$AGENT_RUN/code_f/agent_baseline
PY=$W/conda_envs/exp5-env/bin/python
: "${BAILIAN_CSV:?BAILIAN_CSV must be supplied in the submission environment}"
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=$AGENT_RUN/.tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR"
cd "$CODE"
PORT=$($PY -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
$W/conda_envs/omniovcd-env/bin/python -B workers/sam3_worker.py --port $PORT > $AGENT_RUN/logs/sam3-$SLURM_JOB_ID.log 2>&1 &
WORKER=$!
trap 'kill $WORKER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -s --noproxy '*' -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 5; done
echo "worker ready $(date -u +%H:%M:%S)"
one() { "$PY" "$W/Chang-Agent/tools/run_with_bailian_csv.py" "$BAILIAN_CSV" "$PY" -B loop_json.py --sam3-worker "http://127.0.0.1:$PORT" "$@"; }
pids=()
for f in f0 f1 f2; do
  extra=(); [ $f = f1 ] && extra=(--review); [ $f = f2 ] && extra=(--review --status)
  for n in 1 2 3; do
    A=(--split val --config json_qwen37flash_${f}_s$n --concurrency ${CONC:-6} --prompt-version v2 --seed-offset $n --batch frun --max-cost-cny 100 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --init-name sam3_imc_p85 "${extra[@]}")
    ( one --task T1.1xC0 "${A[@]}" > $AGENT_RUN/logs/fruns-$f-s$n-T1.log 2>&1 && one --task T5.1 "${A[@]}" > $AGENT_RUN/logs/fruns-$f-s$n-T5.log 2>&1 ) &
    pids+=($!)
  done
done
status=0; for p in "${pids[@]}"; do wait $p || status=$?; done
QQ_SUB="ab-fruns $SLURM_JOB_ID" QQ_BODY="status=$status finals=$(find $AGENT_RUN/json_qwen37flash_f[012]_s* -name final.json | wc -l)" $W/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true
exit $status
