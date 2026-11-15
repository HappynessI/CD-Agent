#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through exp-loop}"
seed=${1:?1016, 2027 or 3038}
[[ "$seed" == 1016 || "$seed" == 2027 || "$seed" == 3038 ]]
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N8_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
TASKS=$ROOT/wyh/exp6_staging/batch98_t3_routes/results/agent10_tasks.jsonl
PY=$ROOT/conda_envs/exp5-env/bin/python
export AGENT_RUN=$OUT/online AGENT_CODE=$REPO/code/agent_baseline
export OVCD_CODE=$REPO/code/ovcd_v0 PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0
export BAILIAN_CSV=$ROOT/wyh/默认业务空间-apiKey-6214720.csv
export AGENT_STOP_FILE=$OUT/online/STOP.json
export N8_INFERENCE_CONFIG=$OUT/online/inference_config.json
export USE_ASK_WORKER=0 USE_LOCAL_WORKERS=1 AGENT_ASK_SLOTS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$AGENT_RUN/logs"
started=$SECONDS
finish() {
    status=$?; trap - EXIT; set +e
    "$PY" -B -m agent_baseline.analysis.n8_t3_online_metrics --output "$OUT" --tasks "$TASKS"
    QQ_SUB="N8 T3 online seed $seed" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$AGENT_RUN" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
    exit "$status"
}
trap finish EXIT
while IFS= read -r var; do
    if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi
done < <(compgen -e)
# Probe and real sessions run in this same GPU allocation; never print credentials or response body.
"$PY" "$ROOT/Chang-Agent/tools/run_with_bailian_csv.py" "$BAILIAN_CSV" "$PY" - "$AGENT_RUN/network_s${seed}.json" <<'PY'
import json,os,socket,sys,urllib.request,urllib.error
from pathlib import Path
from urllib.parse import urlsplit
base=os.environ['DASHSCOPE_BASE_URL'].rstrip('/')
request=urllib.request.Request(base+'/models',headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']})
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=30) as response: status=response.status
except urllib.error.HTTPError as error: status=error.code
record=dict(node=socket.gethostname(),endpoint_host=urlsplit(base).hostname,status=status,proxy_present=any('proxy' in k.lower() for k in os.environ))
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if status!=200:raise SystemExit('Workspace endpoint probe failed; real sessions not started')
PY
"$PY" -B -m unittest agent_baseline.tests.test_n8_t3_online_provider agent_baseline.tests.test_n8_t3_online_inputs
for arm in baseline n8; do
    export N8_ONLINE_ARM=$arm
    config=${arm}_s${seed}
    "$PY" -B -m agent_baseline.analysis.n8_t3_online_metrics --output "$OUT" --tasks "$TASKS" --prepare-config "$config"
    args=(--tasks "$TASKS" --split val --concurrency 1 --prompt-version v4 --seed-offset "$seed"
      --batch n8_t3_20261115 --max-cost-cny 49.9824638 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash
      --ask-correspondence --pixel-thr 0.85 --status --verify-stage --gate3 --tools-t13 all
      --vision-config "$REPO/code/agent_baseline/vision_t13.example.json" --config "$config")
    # First task is a real session. Continue the same config only after engineering completion (fallback remains a measured result).
    bash "$REPO/tools/run_n8_t3_agent_worker.sh" "${args[@]}" --limit 1
    "$PY" -B -m agent_baseline.analysis.n8_t3_online_metrics --output "$OUT" --tasks "$TASKS" --expect-config "$config" --first-only
    bash "$REPO/tools/run_n8_t3_agent_worker.sh" "${args[@]}"
    "$PY" -B -m agent_baseline.analysis.n8_t3_online_metrics --output "$OUT" --tasks "$TASKS" --expect-config "$config"
done
