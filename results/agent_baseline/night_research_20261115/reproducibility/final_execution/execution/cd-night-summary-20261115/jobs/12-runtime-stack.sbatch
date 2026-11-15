#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/runtime-stack-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-summary-20261115/logs/runtime-stack-%j.err
set -euo pipefail
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night own-job runtime diagnosis $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack; no API" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-deps --target /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-tools/py-spy-0.4.3 py-spy==0.4.3 --disable-pip-version-check --no-input --retries 0 --timeout 20 > /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/installation.log 2>&1
scontrol listpids 186463 > /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/own_job_pids.txt
if awk '$1==1973415 && $2==186463 {found=1} END {exit !found}' /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/own_job_pids.txt; then
  set +e
  /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-tools/py-spy-0.4.3/bin/py-spy dump --pid 1973415 > /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/stack.txt 2>&1
  status=$?
  set -e
  printf '{"job_id":"186463","pid":1973415,"profiler_exit":%s,"new_api_calls":0}\n' "$status" > /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/result.json
else
  printf '{"job_id":"186463","status":"original_prepare_process_finished_before_inspection","new_api_calls":0}\n' > /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/runtime_stack/result.json
fi
