#!/usr/bin/env bash
#SBATCH --job-name=CA0801V121E2E
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT required}")";WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT required}")";PARENT="$(realpath -e -- "${1:?usage: worker OUTPUT}")";MODEL_PATH="$(realpath -e -- "${MODEL_PATH:?MODEL_PATH required}")"
PYTHON="$WORKSPACE/omniovcd-env/bin/python";INPUT_ROOT="$WORKSPACE/datasets/LEVIR-CD/test_256";SEGAGENT_PYTHON="$WORKSPACE/segagent-env/bin/python";OMNIOVCD_PYTHON="$WORKSPACE/omniovcd-env/bin/python"
SIMPLECLICK="$WORKSPACE/models/SimpleClick/cocolvis_vit_large.pth";SAM3="$WORKSPACE/models/sam3/sam3.pt";BPE="$WORKSPACE/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz";COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT required}";PROFILE="${EVALUATION_PROFILE:-e2e3}";JOB_ID="${SLURM_JOB_ID:-manual-$$}"
case "$PROFILE" in e2e3) EVAL_ID=DEV-CORE-3-V1;SAMPLES=(test_20_15 test_1_1 test_50_8);; e2e5) EVAL_ID=DEV-EXTENDED-5-V1;SAMPLES=(test_20_15 test_78_13 test_85_16 test_1_1 test_50_8);; *) echo "profile must be e2e3/e2e5" >&2;exit 2;;esac
START="$SECONDS";STATUS=failed;RESULT=not_started
finish(){ local rc="$?";trap - EXIT;set +e;[[ "$rc" -eq 0 ]]&&STATUS=success;local elapsed="$((SECONDS-START))";printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\nprofile=%s\nresult=%s\n' "$STATUS" "$rc" "$elapsed" "$PROFILE" "$RESULT" >"$PARENT/logs/status-${JOB_ID}.txt";QQ_SUB="Qwen3-VL v12.1 post-SFT ${PROFILE}" QQ_BODY="status=${STATUS}; result=${RESULT}; job=${JOB_ID}; elapsed=${elapsed}s; output=${PARENT}" "$WORKSPACE/.local/bin/wangyihan-send-qq" >"/tmp/v121_eval_${JOB_ID}.qq.log" 2>&1||true;rm -f "/tmp/v121_eval_${JOB_ID}.qq.log";exit "$rc";};trap finish EXIT
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$COMMIT" && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]||{ echo "code drift" >&2;exit 2; };[[ -f "$MODEL_PATH/model.safetensors" && -f "$MODEL_PATH/config.json" ]]||exit 2
export PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONHASHSEED=42
export PYTHONPATH="$ROOT:$WORKSPACE/Chang-Agent/third_party/OmniOVCD:$WORKSPACE/Chang-Agent/third_party/SegAgent:$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick" CHANGE_AGENT_STAGED_PROTOCOL_VERSION=v12.1
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A800' in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported()
print('gpu_contract=ok device='+torch.cuda.get_device_name(0))
PY
RUN="$PARENT/run"
CMD=("$PYTHON" "$ROOT/tools/run_levir_change_agent.py" --input-root "$INPUT_ROOT" --samples "${SAMPLES[@]}" --output "$RUN" --query building --max-steps 3 --selection-policy conservative_best --proposal-mode proposal --staged-verifier-max-total-regions 32 --verifier-max-selected-regions 3 --verifier-region-padding-ratio 0.25 --empty-mask-scan-grid-size 3 --matching-mode overlap_presence --overlap-threshold 0.25 --t12-min-instance-area 0 --cd-min-instance-area 0 --authorized-negative-point-input-scope full_session --max-point-proposal-change-ratio 0.25 --max-point-proposal-change-pixels 256 --max-point-image-change-ratio 0.004 --model-path "$MODEL_PATH" --agent-backend local --verifier qwen_staged --staged-verifier-backend local --verifier-max-new-tokens 1536 --verifier-accept-threshold 0.82 --verifier-retries 2 --verifier-repetition-penalty 1.05 --device-map auto --tool-device cuda --seed 42 --action-retries 3 --segagent-python "$SEGAGENT_PYTHON" --omniovcd-python "$OMNIOVCD_PYTHON" --simpleclick-checkpoint "$SIMPLECLICK" --sam3-checkpoint "$SAM3" --sam3-bpe "$BPE" --sam3-resolution 1008)
printf 'launch=%q ' "${CMD[@]}" >"$PARENT/launch_command.txt";printf '\n' >>"$PARENT/launch_command.txt";"${CMD[@]}"
[[ -f "$RUN/rollout_complete.json" && -f "$RUN/per_sample_metrics.json" && ! -e "$RUN/.run_incomplete" ]]||{ echo "rollout incomplete" >&2;exit 1; }
"$PYTHON" "$ROOT/tools/summarize_qwen3vl_e2e_run.py" --run-dir "$RUN" --output "$PARENT/post_sft_summary.json" --evaluation-profile "$PROFILE" --evaluation-id "$EVAL_ID" --model-state post_sft --job-id "$JOB_ID" --experiment-arm v12_1-compact-post-sft --protocol-version v12.1 --runtime-commit "$COMMIT" --harness-commit "$COMMIT" --expected-samples "${SAMPLES[@]}"
"$PYTHON" "$ROOT/tools/analyze_staged_verifier_run.py" --run-dir "$RUN" --output "$PARENT/protocol_summary.json" --evaluation-profile "$PROFILE" --evaluation-id "$EVAL_ID" --experiment-arm v12_1-compact-post-sft --protocol-version v12.1 --job-id "$JOB_ID" --runtime-commit "$COMMIT" --harness-commit "$COMMIT" --expected-samples "${SAMPLES[@]}"
"$PYTHON" - "$PARENT" "$MODEL_PATH" <<'PY'
import hashlib,json,pathlib,sys
p,m=map(pathlib.Path,sys.argv[1:]);h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();s=json.load(open(p/'post_sft_summary.json'));x={'model_path':str(m),'model_sha256':h(m/'model.safetensors'),'summary':s};(p/'evaluation_receipt.json').write_text(json.dumps(x,indent=2)+'\n')
PY
RESULT="post_sft_${PROFILE}_complete"
