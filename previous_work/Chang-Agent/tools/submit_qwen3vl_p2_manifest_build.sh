#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
WORKSPACE="/guisongxia01/pangchao/wangyihan"
PYTHON="$(realpath -e -- "$WORKSPACE/conda_envs/omniovcd-env/bin/python3.12")"
QQ_NOTIFIER="$WORKSPACE/.local/bin/wangyihan-send-qq"
TARGET_REPO_INPUT="${TARGET_REPO_ROOT:-}"
EXPECTED_COMMIT_INPUT="${EXPECTED_COMMIT:-}"
WORKER="$ROOT/tools/run_qwen3vl_p2_manifest_build.slurm"
BUILDER="$ROOT/tools/build_qwen3vl_p2_preregistration.py"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_preregistration.py"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
NODE="gpu48"
JOB_USER="guisongxia"
JOB_NAME="CA0803P2Manifest"
PARTITION="gre"

fail() { printf 'qwen3vl_p2_manifest_submit: %s\n' "$*" >&2; exit 2; }
[[ "$#" -eq 0 ]] || fail "this submitter accepts configuration only through EXPECTED_COMMIT and TARGET_REPO_ROOT"
[[ "$EXPECTED_COMMIT_INPUT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT must be exact implementation A"
[[ -n "$TARGET_REPO_INPUT" ]] || fail "TARGET_REPO_ROOT is required"
TARGET_REPO="$(realpath -e -- "$TARGET_REPO_INPUT")"
EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse --verify "$EXPECTED_COMMIT_INPUT^{commit}")"
[[ "$EXPECTED_COMMIT" == "$EXPECTED_COMMIT_INPUT" && "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] \
  || fail "source checkout is not exact implementation A"
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then fail "source A checkout must be detached"; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || fail "source A checkout must be clean"
[[ "$TARGET_REPO" != "$ROOT" ]] || fail "target development worktree must differ from detached A"
[[ "$(git -C "$TARGET_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "target worktree is not exact A"
[[ "$(git -C "$TARGET_REPO" branch --show-current)" == "feat/local-qwen3vl-p2-oracle-0803" ]] \
  || fail "target worktree is not the P2 development branch"
[[ -z "$(git -C "$TARGET_REPO" status --porcelain --untracked-files=all)" ]] || fail "target worktree must be clean"

git_common_dir() {
  local repo="$1" raw
  raw="$(git -C "$repo" rev-parse --git-common-dir)"
  [[ "$raw" == /* ]] || raw="$repo/$raw"
  realpath -e -- "$raw"
}
SOURCE_COMMON="$(git_common_dir "$ROOT")"
TARGET_COMMON="$(git_common_dir "$TARGET_REPO")"
[[ "$SOURCE_COMMON" == "$TARGET_COMMON" ]] || fail "source and target are not worktrees of the same repository"
MANIFEST_ROOT="$(realpath -e -- "$TARGET_REPO/doc/training_pipeline/0803_fourth_try")"
PREREG="$MANIFEST_ROOT/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="$MANIFEST_ROOT/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
[[ ! -e "$PREREG" && ! -e "$EXECUTION" ]] || fail "formal manifests already exist in target A"
for file in "$PYTHON" "$QQ_NOTIFIER" "$WORKER" "$BUILDER" "$VALIDATOR" "$RELEASE_HELPER"; do
  [[ -e "$file" ]] || fail "required file is missing: $file"
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" ]] || fail "python or notifier is not executable"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE is forbidden for release submission"

if ! PING="$(scontrol ping 2>&1)" || ! grep -q 'UP' <<<"$PING"; then printf '%s\n' "$PING" >&2; fail "Slurm controller is not UP"; fi
NODE_INFO="$(scontrol show node "$NODE" -o 2>&1)" || { printf '%s\n' "$NODE_INFO" >&2; fail "cannot inspect fixed node"; }
[[ "$NODE_INFO" =~ State=([^[:space:]]+) ]] || fail "node state is absent"
case "${BASH_REMATCH[1]}" in *DOWN*|*DRAIN*|*FAIL*|*MAINT*|*UNKNOWN*) fail "fixed node is not usable" ;; esac
ACTIVE="$(squeue -h -u "$JOB_USER" --name="$JOB_NAME" -o '%i %T')" || fail "cannot inspect duplicate materialization job"
[[ -z "$ACTIVE" ]] || fail "active duplicate materialization job exists: $ACTIVE"

DIAG_BASE="$ROOT/diagnostics/legacy/qwen3vl_p2_manifest_materialization_0803/implementation-$EXPECTED_COMMIT"
mkdir -p -- "$DIAG_BASE/locks"
ATTEMPT_ROOT=""
for number in $(seq 1 9999); do
  candidate="$DIAG_BASE/$(printf 'attempt-%04d' "$number")"
  if mkdir -- "$candidate" 2>/dev/null; then ATTEMPT_ROOT="$candidate"; break; fi
done
[[ -n "$ATTEMPT_ROOT" ]] || fail "cannot reserve append-only materialization attempt"
mkdir -- "$ATTEMPT_ROOT/logs" "$ATTEMPT_ROOT/staging"
STAGING_ROOT="$ATTEMPT_ROOT/staging"
LOCK_FILE="$DIAG_BASE/locks/materialization.lock"
INTENT="$ATTEMPT_ROOT/submission_intent.json"
SUBMISSION="$ATTEMPT_ROOT/submission_receipt.json"
WORKER_SHA="$(sha256sum -- "$WORKER" | awk '{print $1}')"
BUILDER_SHA="$(sha256sum -- "$BUILDER" | awk '{print $1}')"
VALIDATOR_SHA="$(sha256sum -- "$VALIDATOR" | awk '{print $1}')"
HELPER_SHA="$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')"

"$PYTHON" - "$INTENT" "$EXPECTED_COMMIT" "$ROOT" "$TARGET_REPO" "$MANIFEST_ROOT" \
  "$STAGING_ROOT" "$LOCK_FILE" "$WORKER_SHA" "$BUILDER_SHA" "$VALIDATOR_SHA" "$HELPER_SHA" <<'PY'
import json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent
payload={"schema_id":"qwen3vl_p2_manifest_submission_intent_v1","status":"prepared",
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"implementation_commit_a":sys.argv[2],
 "source_repo":str(pathlib.Path(sys.argv[3]).resolve()),"target_repo":str(pathlib.Path(sys.argv[4]).resolve()),
 "manifest_root":str(pathlib.Path(sys.argv[5]).resolve()),"staging_root":str(pathlib.Path(sys.argv[6]).resolve()),
 "lock_file":str(pathlib.Path(sys.argv[7]).resolve()),
 "workers":{"materializer":sys.argv[8],"builder":sys.argv[9],"validator":sys.argv[10],"release_helper":sys.argv[11]},
 "job_contract":{"name":"CA0803P2Manifest","user":"guisongxia","node":"gpu48","partition":"gre","gpus":0,"cpus":2,"memory":"16G"}}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
INTENT_SHA="$(sha256sum -- "$INTENT" | awk '{print $1}')"

set +e
SUBMIT_OUTPUT="$(sbatch --parsable --export=NIL --job-name="$JOB_NAME" --partition="$PARTITION" \
  --nodelist="$NODE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --output="$ATTEMPT_ROOT/logs/slurm-%j.out" --error="$ATTEMPT_ROOT/logs/slurm-%j.err" \
  "$WORKER" "$ROOT" "$TARGET_REPO" "$WORKSPACE" "$ATTEMPT_ROOT" "$MANIFEST_ROOT" \
  "$STAGING_ROOT" "$LOCK_FILE" "$PYTHON" "$QQ_NOTIFIER" "$EXPECTED_COMMIT" "$WORKER_SHA" \
  "$BUILDER_SHA" "$VALIDATOR_SHA" "$HELPER_SHA" "$INTENT" "$INTENT_SHA" 2>&1)"
SUBMIT_RC="$?"; set -e
JOB_ID="not_submitted"
STATUS="materialization_submit_failed"
if [[ "$SUBMIT_RC" -eq 0 ]]; then
  JOB_ID="$(cut -d ';' -f 1 <<<"$SUBMIT_OUTPUT")"
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || fail "materialization sbatch returned invalid Job ID"
  STATUS="materialization_submitted"
fi
"$PYTHON" - "$SUBMISSION" "$STATUS" "$JOB_ID" "$INTENT" "$EXPECTED_COMMIT" "$WORKER_SHA" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent;intent=pathlib.Path(sys.argv[4])
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
payload={"schema_id":"qwen3vl_p2_manifest_submission_receipt_v1","submission_status":sys.argv[2],
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"implementation_commit_a":sys.argv[5],
 "job":{"job_id":sys.argv[3],"name":"CA0803P2Manifest","user":"guisongxia","node":"gpu48",
        "partition":"gre","gpus":0,"cpus":2,"memory":"16G"},"worker_sha256":sys.argv[6],"intent":bind(intent)}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
if [[ "$SUBMIT_RC" -ne 0 ]]; then printf '%s\n' "$SUBMIT_OUTPUT" >&2; exit "$SUBMIT_RC"; fi
printf 'status=materialization_submitted\njob_id=%s\nattempt=%s\ntarget_repo=%s\n' "$JOB_ID" "$ATTEMPT_ROOT" "$TARGET_REPO"
