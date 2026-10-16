#!/usr/bin/env bash
#SBATCH --job-name=CA0801V121BUILD
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT required}")"
RUN_DIR="$(realpath -e -- "${1:?usage: worker RUN_DIR}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
CONFIG="$ROOT/configs/qwen3vl_v121_curation_0801.json"
CURATION_ROOT="$(realpath -e -- "${CURATION_ROOT:-$RUN_DIR/curation}")"
DATA_ROUND="${DATA_ROUND:-v1}"
COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT required}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"; START="$SECONDS"; STATUS=failed; RESULT=not_started
finish(){ local rc="$?";trap - EXIT;set +e;[[ "$rc" -eq 0 ]]&&STATUS=success;local elapsed="$((SECONDS-START))";printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\nresult=%s\n' "$STATUS" "$rc" "$elapsed" "$RESULT" >"$RUN_DIR/logs/data-build-status-${JOB_ID}.txt";QQ_SUB="Qwen3-VL v12.1 data build" QQ_BODY="status=${STATUS}; result=${RESULT}; job=${JOB_ID}; elapsed=${elapsed}s; output=${RUN_DIR}" "$WORKSPACE/.local/bin/wangyihan-send-qq" >"/tmp/v121_build_${JOB_ID}.qq.log" 2>&1||true;rm -f "/tmp/v121_build_${JOB_ID}.qq.log";exit "$rc";};trap finish EXIT
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CPU build received GPU" >&2; exit 2; }
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$COMMIT" ]] || { echo "commit drift" >&2;exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { echo "dirty worktree" >&2;exit 2; }
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -p 'test_qwen3vl_v121_baseline_data.py'
if [[ ! -f "$RUN_DIR/data/.data_complete" ]]; then
  "$PYTHON" tools/build_qwen3vl_v121_baseline_data.py --curation-root "$CURATION_ROOT" --curation-config "$CONFIG" --output "$RUN_DIR/data" --runtime-commit "$COMMIT" --seed 801 --round "$DATA_ROUND"
fi
[[ -f "$RUN_DIR/data/.data_complete" ]] || { echo "data marker missing" >&2;exit 1; }
if [[ "$DATA_ROUND" == "v5-contrast-repair" ]]; then
  QWEN3VL_V121_V5_MANIFEST="$RUN_DIR/data/manifest.json" \
    "$PYTHON" -m unittest \
      tests.test_qwen3vl_v121_baseline_data.Qwen3VLV121BaselineDataTest.test_v5_manifest_contract_if_available
fi
if [[ "$DATA_ROUND" == "v6-matched-contrast" ]]; then
  QWEN3VL_V121_V6_MANIFEST="$RUN_DIR/data/manifest.json" \
    "$PYTHON" -m unittest \
      tests.test_qwen3vl_v121_baseline_data.Qwen3VLV121BaselineDataTest.test_v6_manifest_contract_if_available
fi
if [[ "$DATA_ROUND" == "v7-weighted-contrast" ]]; then
  QWEN3VL_V121_V7_MANIFEST="$RUN_DIR/data/manifest.json" \
    "$PYTHON" -m unittest \
      tests.test_qwen3vl_v121_baseline_data.Qwen3VLV121BaselineDataTest.test_v7_manifest_contract_if_available
fi
RESULT="$("$PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["valid"] is True;print("train={};val={};valid={}".format(d["train_rows"],d["val_rows"],d["valid"]))' "$RUN_DIR/data/validation.json")"
