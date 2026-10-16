#!/usr/bin/env bash
set -euo pipefail
umask 027

# Submit the five frozen P1 allocations as held jobs, publish each immutable
# submission receipt, add one auxiliary zero-GPU afterany guard, then release
# all held jobs in one operation.
# No output reservation or Slurm mutation happens until the exact clean
# worktree, amendment, config, node and model/input preflights pass.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_ROOT="$(realpath -e -- "${WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd -P)}")"
ARTIFACT_ROOT="${ARTIFACT_ROOT:?ARTIFACT_ROOT must point to the canonical development repository}"
ARTIFACT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
CONFIG="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
AMENDMENT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json"
INPUT_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
FIXTURE_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json"
CPU_WORKER="$ROOT/tools/run_qwen3vl_p1_cpu_fixtures.slurm"
GPU_WORKER="$ROOT/tools/run_qwen3vl_p1_evaluation.slurm"
DEPENDENCY_GUARD_WORKER="$ROOT/tools/run_qwen3vl_p1_dependency_guard.slurm"
PAIR_BUILDER="$ROOT/tools/build_qwen3vl_p1_pair_comparison.py"
RUNTIME_MANIFEST_BUILDER="$ROOT/tools/build_qwen3vl_runtime_input_manifest.py"
PYTHON="$(realpath -e -- "${PYTHON:-$WORKSPACE_ROOT/omniovcd-env/bin/python}")"
OMNIOVCD_PYTHON="$(realpath -e -- "${OMNIOVCD_PYTHON:-$WORKSPACE_ROOT/omniovcd-env/bin/python}")"
SEGAGENT_PYTHON="$(realpath -e -- "${SEGAGENT_PYTHON:-$WORKSPACE_ROOT/segagent-env/bin/python}")"
MODEL_PATH="$(realpath -e -- "${MODEL_PATH:-$WORKSPACE_ROOT/models/Qwen3-VL-2B-Instruct}")"
INPUT_ROOT="$(realpath -e -- "${INPUT_ROOT:-$WORKSPACE_ROOT/datasets/LEVIR-CD/test_256}")"
SIMPLECLICK_CHECKPOINT="$(realpath -e -- "${SIMPLECLICK_CHECKPOINT:-$WORKSPACE_ROOT/models/SimpleClick/cocolvis_vit_large.pth}")"
SAM3_CHECKPOINT="$(realpath -e -- "${SAM3_CHECKPOINT:-$WORKSPACE_ROOT/models/sam3/sam3.pt}")"
SAM3_BPE="$(realpath -e -- "${SAM3_BPE:-$WORKSPACE_ROOT/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz}")"
PARTITION="gre"
NODE="gpu48"
SLURM_USER="guisongxia"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
[[ "$#" -eq 0 ]] || { echo "usage: $0 [--check-only]" >&2; exit 2; }

fail() {
  echo "qwen3vl_p1_submit: $*" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || fail "required Python is missing: $PYTHON"
for required in "$CONFIG" "$AMENDMENT" "$INPUT_MANIFEST" "$FIXTURE_MANIFEST" \
  "$CPU_WORKER" "$GPU_WORKER" "$DEPENDENCY_GUARD_WORKER" "$PAIR_BUILDER" \
  "$RUNTIME_MANIFEST_BUILDER"; do
  [[ -f "$required" ]] || fail "required P1 file is missing: $required"
done
[[ "$(realpath -e -- "$ROOT")" == "/guisongxia01/pangchao/wangyihan/Chang-Agent-baseline-0802-run" ]] \
  || fail "submit P1 only from the exact clean run worktree Chang-Agent-baseline-0802-run"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "run worktree is dirty"
EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
EXPECTED_IMPLEMENTATION_COMMIT="$(git -C "$ROOT" rev-parse "${EXPECTED_COMMIT}^")"
CONFIG_SHA256="$(sha256sum -- "$CONFIG" | awk '{print $1}')"
CPU_WORKER_SHA256="$(sha256sum -- "$CPU_WORKER" | awk '{print $1}')"
GPU_WORKER_SHA256="$(sha256sum -- "$GPU_WORKER" | awk '{print $1}')"
DEPENDENCY_GUARD_WORKER_SHA256="$(sha256sum -- "$DEPENDENCY_GUARD_WORKER" | awk '{print $1}')"
SUBMITTER_SHA256="$(sha256sum -- "$ROOT/tools/submit_qwen3vl_p1.sh" | awk '{print $1}')"
INPUT_MANIFEST_CANONICAL_SHA256="$($PYTHON - "$INPUT_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
print(hashlib.sha256(raw).hexdigest())
PY
)"
AMENDMENT_CANONICAL_SHA256="$($PYTHON - "$AMENDMENT" <<'PY'
import hashlib, json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
print(hashlib.sha256(raw).hexdigest())
PY
)"
FIXTURE_MANIFEST_CANONICAL_SHA256="$($PYTHON - "$FIXTURE_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
print(hashlib.sha256(raw).hexdigest())
PY
)"

# Validate the amendment and resolved allocation contract in Python so shell
# quoting cannot silently alter ordered samples, paths or hashes.
mapfile -t RESOLVED < <("$PYTHON" - "$CONFIG" "$AMENDMENT" "$INPUT_MANIFEST" "$FIXTURE_MANIFEST" \
  "$ROOT" "$WORKSPACE_ROOT" "$ARTIFACT_ROOT" "$OUTPUT_ROOT" \
  "$PYTHON" "$OMNIOVCD_PYTHON" "$SEGAGENT_PYTHON" "$MODEL_PATH" "$INPUT_ROOT" \
  "$SIMPLECLICK_CHECKPOINT" "$SAM3_CHECKPOINT" "$SAM3_BPE" \
  "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$CONFIG_SHA256" "$CPU_WORKER_SHA256" "$GPU_WORKER_SHA256" \
  "$DEPENDENCY_GUARD_WORKER_SHA256" "$SUBMITTER_SHA256" \
  "$INPUT_MANIFEST_CANONICAL_SHA256" "$FIXTURE_MANIFEST_CANONICAL_SHA256" \
  "$AMENDMENT_CANONICAL_SHA256" <<'PY'
import hashlib, json, pathlib, sys

(
    config_text, amendment_text, input_text, fixture_text, root_text,
    workspace_text, artifact_text, output_root_text,
    python_text, omniovcd_python_text, segagent_python_text, model_text, input_root_text,
    simpleclick_text, sam3_text, sam3_bpe_text,
    freeze_commit, implementation_commit, config_hash, cpu_worker_hash, gpu_worker_hash, dependency_guard_worker_hash,
    submitter_hash,
    input_hash, fixture_hash, amendment_hash,
) = sys.argv[1:]

def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

config = load(config_text)
amendment = load(amendment_text)
input_manifest = load(input_text)
fixture = load(fixture_text)

def resolved(path_text, *, strict=True):
    return pathlib.Path(path_text).resolve(strict=strict)

def configured_path(paths, key):
    value = paths.get(key)
    if not isinstance(value, str) or not pathlib.Path(value).is_absolute():
        raise SystemExit(f"resolved config paths.{key} is not an absolute path")
    return resolved(value)

if config.get("schema_id") != "qwen3vl_p1_resolved_run_config_v1":
    raise SystemExit("resolved config schema drifted")
if config.get("configuration_status") != "frozen":
    raise SystemExit("resolved config configuration_status is not frozen")
if config.get("execution_readiness") != "execution_ready":
    raise SystemExit("resolved config execution_readiness is not execution_ready")
if config.get("p1_execution_authorized") is not True:
    raise SystemExit("resolved config does not authorize P1 execution")
paths = config.get("paths")
if not isinstance(paths, dict):
    raise SystemExit("resolved config paths are missing")
observed_paths = {
    "workspace_root": workspace_text,
    "clean_run_worktree": root_text,
    "artifact_repository": artifact_text,
    "output_root": output_root_text,
    "input_root": input_root_text,
    "base_model": model_text,
    "simpleclick_checkpoint": simpleclick_text,
    "sam3_checkpoint": sam3_text,
    "sam3_bpe": sam3_bpe_text,
    "omniovcd_python": omniovcd_python_text,
    "segagent_python": segagent_python_text,
}
for key, observed in observed_paths.items():
    if configured_path(paths, key) != resolved(observed):
        raise SystemExit(f"resolved config paths.{key} differs from the submitter")
configured_omniovcd = configured_path(paths, "omniovcd_python")
if resolved(python_text) != configured_omniovcd or resolved(sys.executable) != configured_omniovcd:
    raise SystemExit("submitter PYTHON is not the resolved config omniovcd_python")
for key, observed in (
    ("evaluation_input_manifest", input_text),
    ("cpu_fixture_manifest", fixture_text),
    ("p1_amendment", amendment_text),
):
    declared = paths.get(key)
    if not isinstance(declared, str) or (resolved(root_text) / declared).resolve() != resolved(observed):
        raise SystemExit(f"resolved config paths.{key} differs from the submitter")
if amendment.get("record_status") != "frozen":
    raise SystemExit("amendment.record_status is not frozen")
if amendment.get("execution_readiness") != "execution_ready":
    raise SystemExit("amendment.execution_readiness is not execution_ready")
if amendment.get("p1_execution_authorized") is not True:
    raise SystemExit("amendment does not authorize P1 execution")
if amendment.get("implementation_commit") != implementation_commit:
    raise SystemExit("amendment implementation_commit differs from the freeze commit parent")
if pathlib.Path(amendment.get("clean_run_worktree", "")).resolve() != pathlib.Path(root_text).resolve():
    raise SystemExit("amendment clean_run_worktree differs from submitter root")
if amendment.get("resolved_run_config_sha256") != config_hash:
    raise SystemExit("resolved_run_config_sha256 differs from checked-in config")
if amendment.get("input_manifest_canonical_sha256") != input_hash:
    raise SystemExit("input manifest canonical hash differs")
if amendment.get("cpu_fixture_manifest_sha256") != fixture_hash:
    raise SystemExit("CPU fixture manifest hash differs")
if amendment.get("qq_notification_titles_verified") is not True:
    raise SystemExit("QQ notification titles are not verified")
for key in (
    "model_hash", "tool_hashes", "summary_schema_definition_and_validator_hashes",
    "receipt_schema_definition_and_validator_hashes", "slurm_node_partition_cpu_memory_time",
    "submission_script_path_and_sha256",
):
    value = amendment.get(key)
    if value is None or value == {} or value == []:
        raise SystemExit(f"amendment.{key} is unresolved")
if amendment.get("unique_output_paths") != [item["output_path"] for item in config["allocations"]]:
    raise SystemExit("amendment unique output paths differ from resolved config")
if config["slurm"]["fixed_node"] != "gpu48" or config["slurm"]["partition"] != "gre":
    raise SystemExit("resolved config node/partition drifted")
if config["slurm"]["maximum_ca0802_concurrent_gpus"] != 3:
    raise SystemExit("resolved config exceeds the three-card cap")
bindings = config.get("freeze_commit_hash_bindings", {})
if bindings.get("status") != "frozen":
    raise SystemExit("resolved config hash bindings are not frozen")
for group in ("model_files_sha256", "tool_files_sha256", "worker_sha256"):
    values = bindings.get(group)
    if not isinstance(values, dict) or not values or any(value is None for value in values.values()):
        raise SystemExit(f"resolved config {group} contains an unresolved hash")
if bindings["worker_sha256"].get("cpu_fixtures") != cpu_worker_hash:
    raise SystemExit("resolved config CPU worker hash differs")
if bindings["worker_sha256"].get("evaluation") != gpu_worker_hash:
    raise SystemExit("resolved config evaluation worker hash differs")
if bindings["worker_sha256"].get("dependency_guard") != dependency_guard_worker_hash:
    raise SystemExit("resolved config dependency-guard worker hash differs")
if bindings["tool_files_sha256"].get("harness_worker") != gpu_worker_hash:
    raise SystemExit("resolved config harness-worker hash differs")
amendment_model = amendment.get("model_hash")
if isinstance(amendment_model, str):
    amendment_model = {"model.safetensors": amendment_model}
if amendment_model != bindings["model_files_sha256"]:
    raise SystemExit("amendment model hash differs from resolved config")
if amendment.get("tool_hashes") != bindings["tool_files_sha256"]:
    raise SystemExit("amendment tool hashes differ from resolved config")
if fixture.get("schema_id") != "qwen3vl_h0_h1_cpu_fixture_manifest_v1" or fixture.get("status") != "frozen":
    raise SystemExit("CPU fixture manifest is not frozen")
if fixture.get("execution_readiness") != "execution_ready":
    raise SystemExit("CPU fixture manifest execution_readiness is not execution_ready")
if fixture.get("p1_execution_authorized") is not True:
    raise SystemExit("CPU fixture manifest does not authorize P1 execution")
if fixture.get("targeted_unittest_patterns") != config["cpu_fixture_suite"]["targeted_unittest_patterns"]:
    raise SystemExit("CPU fixture patterns differ from resolved config")
if input_manifest.get("schema_id") != "qwen3vl_evaluation_input_manifest_v1":
    raise SystemExit("evaluation input manifest schema drifted")

worker_field = amendment["worker_sha256"]
if isinstance(worker_field, dict):
    def pick(*names):
        for name in names:
            if name in worker_field:
                return worker_field[name]
        return None
    if pick("cpu_fixtures", "run_qwen3vl_p1_cpu_fixtures.slurm") != cpu_worker_hash:
        raise SystemExit("CPU worker SHA-256 is not amendment-bound")
    if pick("evaluation", "run_qwen3vl_p1_evaluation.slurm") != gpu_worker_hash:
        raise SystemExit("GPU worker SHA-256 is not amendment-bound")
    if pick("dependency_guard", "run_qwen3vl_p1_dependency_guard.slurm") != dependency_guard_worker_hash:
        raise SystemExit("dependency-guard worker SHA-256 is not amendment-bound")
    if pick("cpu_fixtures", "run_qwen3vl_p1_cpu_fixtures.slurm") != bindings["worker_sha256"]["cpu_fixtures"]:
        raise SystemExit("amendment/config CPU worker hashes differ")
    if pick("evaluation", "run_qwen3vl_p1_evaluation.slurm") != bindings["worker_sha256"]["evaluation"]:
        raise SystemExit("amendment/config evaluation worker hashes differ")
    if pick("dependency_guard", "run_qwen3vl_p1_dependency_guard.slurm") != bindings["worker_sha256"]["dependency_guard"]:
        raise SystemExit("amendment/config dependency-guard worker hashes differ")
else:
    raise SystemExit("amendment.worker_sha256 must map CPU, evaluation and dependency-guard workers")

submission_field = amendment["submission_script_path_and_sha256"]
if isinstance(submission_field, dict):
    observed = submission_field.get("sha256")
    path = submission_field.get("path", "")
    if observed != submitter_hash or pathlib.Path(path).name != "submit_qwen3vl_p1.sh":
        raise SystemExit("submitter hash/path is not amendment-bound")
else:
    raise SystemExit("submission_script_path_and_sha256 must be an object")

for item in config["allocations"]:
    declared_output = item.get("output_path")
    if not isinstance(declared_output, str) or not pathlib.Path(declared_output).is_absolute():
        raise SystemExit("resolved allocation output_path is not absolute")
    if resolved(declared_output, strict=False).parent != configured_path(paths, "output_root"):
        raise SystemExit("resolved allocation output_path escapes the configured output_root")
    resources = item["resources"]
    if item["allocation_id"] == "CA_0802(2)" and resources.get("gpus") != 0:
        raise SystemExit("CPU allocation requests a GPU")
    if item["allocation_id"] != "CA_0802(2)" and resources.get("gpus") != 1:
        raise SystemExit("GPU allocation does not request exactly one GPU")
    if item.get("dependency") and not item["dependency"].startswith("afterok:"):
        raise SystemExit("P1 dependency is not afterok")
    arm = item.get("arm")
    if arm == "h0_h1":
        if resources.get("gpus") != 0:
            raise SystemExit("combined CPU fixture arm unexpectedly requests a GPU")
        experiment_arm_id = "not_applicable"
        harness_protocol_id = "not_applicable"
    else:
        identity = config.get("frozen_identity", {}).get(arm)
        if not isinstance(identity, dict):
            raise SystemExit(f"resolved allocation has no frozen identity: {arm!r}")
        experiment_arm_id = identity.get("experiment_arm_id")
        harness_protocol_id = identity.get("harness_protocol_id")
        if not isinstance(experiment_arm_id, str) or not experiment_arm_id:
            raise SystemExit(f"resolved experiment arm identity is invalid: {arm!r}")
        if not isinstance(harness_protocol_id, str) or not harness_protocol_id:
            raise SystemExit(f"resolved harness protocol identity is invalid: {arm!r}")
    item["_resolved_experiment_arm_id"] = experiment_arm_id
    item["_resolved_harness_protocol_id"] = harness_protocol_id
guard = config.get("dependency_guard")
if not isinstance(guard, dict):
    raise SystemExit("resolved config dependency_guard is missing")
if guard.get("included_in_formal_allocations") is not False:
    raise SystemExit("dependency guard must remain outside the five formal allocations")
if guard.get("job_name") != "CA0802P1DependencyGuard":
    raise SystemExit("dependency guard Job name drifted")
if guard.get("worker_path") != "tools/run_qwen3vl_p1_dependency_guard.slurm":
    raise SystemExit("dependency guard worker path drifted")
if guard.get("dependency") != "afterany:<all_five_formal_job_ids>":
    raise SystemExit("dependency guard must depend afterany on all five formal jobs")
if guard.get("resources") != {
    "gpus": 0,
    "cpus_per_task": 1,
    "memory": "1G",
    "time_limit": "00:10:00",
}:
    raise SystemExit("dependency guard resources drifted")
for item in config["allocations"]:
    print("|".join([
        item["allocation_id"], item["job_name"], item["phase"], item["arm"],
        item["evaluation_profile"], item["evaluation_id"], str(resolved(item["output_path"], strict=False)),
        item["qq_title"], str(item["resources"].get("gpus", 0)),
        str(item["resources"]["cpus_per_task"]), item["resources"]["memory"],
        item["resources"]["time_limit"], item.get("dependency") or "none",
        " ".join(item.get("ordered_samples", [])),
        item["_resolved_experiment_arm_id"],
        item["_resolved_harness_protocol_id"],
    ]))
PY
); [[ "${#RESOLVED[@]}" -eq 5 ]] || fail "resolved config did not produce five allocations"
IFS='|' read -r -a FIRST_RESOLVED_FIELDS <<<"${RESOLVED[0]}"
CPU_OUTPUT="${FIRST_RESOLVED_FIELDS[6]}"
[[ -n "$CPU_OUTPUT" ]] || fail "resolved config did not produce a CPU output"

if ! SLURM_PING="$(scontrol ping 2>&1)"; then
  printf '%s\n' "$SLURM_PING" >&2
  fail "Slurm controller preflight failed; no job submitted"
fi
grep -q 'UP' <<<"$SLURM_PING" || fail "Slurm controller is not UP; no job submitted"
if ! DUPLICATE_JOBS="$(squeue -h -u "$SLURM_USER" -o '%i %j %T' 2>&1)"; then
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  fail "Slurm duplicate-job preflight failed"
fi
if grep -Eq 'CA0802P1(CPUFixtures|SmokeH0|SmokeH1|CoreH0|CoreH1|DependencyGuard)( |$)' <<<"$DUPLICATE_JOBS"; then
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  fail "an active CA0802 P1 job already exists"
fi
if ! NODE_INFO="$(scontrol show node "$NODE" 2>&1)"; then
  printf '%s\n' "$NODE_INFO" >&2
  fail "cannot inspect fixed node $NODE"
fi
grep -Eq 'State=(IDLE|MIXED|ALLOCATED)' <<<"$NODE_INFO" \
  || fail "fixed node $NODE is down, drained or unavailable"
grep -Eq 'CfgTRES=.*gpu' <<<"$NODE_INFO" \
  || fail "fixed node $NODE does not advertise a GPU resource"
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || fail "base model is missing"
[[ -d "$INPUT_ROOT/A" && -d "$INPUT_ROOT/B" ]] || fail "online input directories are missing"
for sample in test_1_1 test_20_15 test_50_8; do
  [[ -f "$INPUT_ROOT/A/$sample.png" && -f "$INPUT_ROOT/B/$sample.png" ]] \
    || fail "online A/B input is missing for $sample"
done
for row in "${RESOLVED[@]}"; do
  IFS='|' read -r allocation_id job_name phase arm profile evaluation_id output qq_title gpus cpus memory time_limit dependency samples experiment_arm_id harness_protocol_id <<<"$row"
  [[ ! -e "$output" ]] || fail "reserved output already exists: $output"
done

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  printf 'status=preflight_ok\nfreeze_commit=%s\nimplementation_commit=%s\nconfig_sha256=%s\namendment_canonical_sha256=%s\nnode=%s\npartition=%s\nallocations=5\nmax_concurrent_gpus=1\n' \
    "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$CONFIG_SHA256" "$AMENDMENT_CANONICAL_SHA256" "$NODE" "$PARTITION"
  exit 0
fi

# Reserve each absent output and its log child before the first held job is
# submitted.  The exact paths are never reused after a partial chain failure.
for row in "${RESOLVED[@]}"; do
  IFS='|' read -r allocation_id job_name phase arm profile evaluation_id output qq_title gpus cpus memory time_limit dependency samples experiment_arm_id harness_protocol_id <<<"$row"
  mkdir "$output"
  mkdir "$output/logs"
done

declare -a JOB_IDS=()
declare -a JOB_OUTPUTS=()
declare -a JOB_NAMES=()
declare -a JOB_ALLOCATION_IDS=()
GUARD_JOB_ID=""
SUBMISSION_FAILED=0
SUBMISSION_ERROR=""

cancel_submitted() {
  local submitted_ids=("${JOB_IDS[@]}")
  if [[ -n "$GUARD_JOB_ID" ]]; then
    submitted_ids+=("$GUARD_JOB_ID")
  fi
  if [[ "${#submitted_ids[@]}" -gt 0 ]]; then
    scancel "${submitted_ids[@]}" >/dev/null 2>&1 || true
  fi
}

write_submission_receipt() {
  local output="$1" allocation_id="$2" job_name="$3" profile="$4" evaluation_id="$5" arm="$6" experiment_arm="$7" protocol="$8" gpus="$9" cpus="${10}" memory="${11}" time_limit="${12}" dependency="${13}" job_id="${14}" samples="${15}"
  local temporary="$output/submission.txt.tmp-$$"
  printf 'submission_status=submitted\njob_id=%s\njob_name=%s\nallocation_id=%s\nevaluation_profile=%s\nevaluation_id=%s\nsamples=%s\narm=%s\nexperiment_arm_id=%s\nharness_protocol_id=%s\nexpected_commit=%s\nimplementation_commit=%s\nworker_sha256=%s\nconfig_sha256=%s\namendment_canonical_sha256=%s\npartition=%s\nnode=%s\nresources=gpu:%s,cpu:%s,mem:%s,time:%s\ndependency=%s\noutput=%s\n' \
    "$job_id" "$job_name" "$allocation_id" "$profile" "$evaluation_id" "$samples" "$arm" \
    "$experiment_arm" "$protocol" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
    "$([[ "$gpus" == 1 ]] && printf '%s' "$GPU_WORKER_SHA256" || printf '%s' "$CPU_WORKER_SHA256")" \
    "$CONFIG_SHA256" "$AMENDMENT_CANONICAL_SHA256" \
    "$PARTITION" "$NODE" "$gpus" "$cpus" "$memory" "$time_limit" "$dependency" "$output" \
    >"$temporary"
  mv -f -- "$temporary" "$output/submission.txt"
}

for row in "${RESOLVED[@]}"; do
  IFS='|' read -r allocation_id job_name phase arm profile evaluation_id output qq_title gpus cpus memory time_limit dependency samples experiment_arm_id harness_protocol_id <<<"$row"
  worker="$CPU_WORKER"
  worker_hash="$CPU_WORKER_SHA256"
  if [[ "$gpus" == 1 ]]; then
    worker="$GPU_WORKER"
    worker_hash="$GPU_WORKER_SHA256"
  fi
  dependency_arg=()
  dependency_value="none"
  if [[ "$dependency" != "none" ]]; then
    dependency_value="afterok:${JOB_IDS[-1]}"
    dependency_arg=("--dependency=$dependency_value")
  fi
  export_list="REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE_ROOT,ARTIFACT_ROOT=$ARTIFACT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,EXPECTED_IMPLEMENTATION_COMMIT=$EXPECTED_IMPLEMENTATION_COMMIT,EXPECTED_WORKER_SHA256=$worker_hash,EXPECTED_CONFIG_SHA256=$CONFIG_SHA256,EXPECTED_AMENDMENT_CANONICAL_SHA256=$AMENDMENT_CANONICAL_SHA256,EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256=$INPUT_MANIFEST_CANONICAL_SHA256,EXPECTED_FIXTURE_MANIFEST_CANONICAL_SHA256=$FIXTURE_MANIFEST_CANONICAL_SHA256,PYTHON=$PYTHON,OMNIOVCD_PYTHON=$OMNIOVCD_PYTHON,SEGAGENT_PYTHON=$SEGAGENT_PYTHON,MODEL_PATH=$MODEL_PATH,INPUT_ROOT=$INPUT_ROOT,SIMPLECLICK_CHECKPOINT=$SIMPLECLICK_CHECKPOINT,SAM3_CHECKPOINT=$SAM3_CHECKPOINT,SAM3_BPE=$SAM3_BPE,QQ_NOTIFIER=$WORKSPACE_ROOT/.local/bin/wangyihan-send-qq"
  worker_args=("$output")
  if [[ "$gpus" == 1 ]]; then
    worker_args+=("$profile" "$arm")
    if [[ "$arm" == "h1" ]]; then
      [[ "${#JOB_OUTPUTS[@]}" -gt 0 ]] || fail "H1 has no preceding H0 output"
      worker_args+=("${JOB_OUTPUTS[-1]}/evaluation_receipt.json")
    fi
  fi
  set +e
  job_output="$(sbatch --parsable --hold \
    --job-name="$job_name" --partition="$PARTITION" --nodelist="$NODE" \
    $(if [[ "$gpus" == 1 ]]; then printf '%s' '--gres=gpu:1'; fi) \
    --cpus-per-task="$cpus" --mem="$memory" --time="$time_limit" \
    --signal=B:TERM@60 --kill-on-invalid-dep=yes \
    "${dependency_arg[@]}" \
    --output="$output/logs/slurm-%j.out" --error="$output/logs/slurm-%j.err" \
    --export="$export_list" "$worker" "${worker_args[@]}" 2>&1)"
  submit_rc="$?"
  set -e
  if [[ "$submit_rc" -ne 0 || ! "$job_output" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]]; then
    SUBMISSION_FAILED=1
    SUBMISSION_ERROR="sbatch failed for $allocation_id: $job_output"
    break
  fi
  job_id="${job_output%%;*}"
  JOB_IDS+=("$job_id")
  JOB_OUTPUTS+=("$output")
  JOB_NAMES+=("$job_name")
  JOB_ALLOCATION_IDS+=("$allocation_id")
  write_submission_receipt "$output" "$allocation_id" "$job_name" "$profile" "$evaluation_id" "$arm" \
    "$experiment_arm_id" "$harness_protocol_id" \
    "$gpus" "$cpus" "$memory" "$time_limit" "$dependency_value" "$job_id" "$samples"
done

if [[ "$SUBMISSION_FAILED" -ne 0 ]]; then
  cancel_submitted
  printf 'submission_status=failed\nerror=%s\nfreeze_commit=%s\nimplementation_commit=%s\nconfig_sha256=%s\namendment_canonical_sha256=%s\n' \
    "$SUBMISSION_ERROR" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$CONFIG_SHA256" "$AMENDMENT_CANONICAL_SHA256" \
    >"$CPU_OUTPUT/submission_chain_failure.txt" 2>/dev/null || true
  fail "$SUBMISSION_ERROR"
fi

# The auxiliary zero-GPU guard is not a sixth formal allocation.  Its afterany
# dependency observes all five terminal Slurm states without weakening any of
# the four formal afterok edges above.
GUARD_DEPENDENCY="$(IFS=:; printf 'afterany:%s' "${JOB_IDS[*]}")"
GUARD_EXPORTS="REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE_ROOT,ARTIFACT_ROOT=$ARTIFACT_ROOT,PYTHON=$PYTHON,EXPECTED_COMMIT=$EXPECTED_COMMIT,EXPECTED_IMPLEMENTATION_COMMIT=$EXPECTED_IMPLEMENTATION_COMMIT,EXPECTED_WORKER_SHA256=$DEPENDENCY_GUARD_WORKER_SHA256,EXPECTED_CPU_WORKER_SHA256=$CPU_WORKER_SHA256,EXPECTED_GPU_WORKER_SHA256=$GPU_WORKER_SHA256,EXPECTED_CONFIG_SHA256=$CONFIG_SHA256,QQ_NOTIFIER=$WORKSPACE_ROOT/.local/bin/wangyihan-send-qq"
set +e
guard_output="$(sbatch --parsable --hold \
  --job-name=CA0802P1DependencyGuard --partition="$PARTITION" \
  --cpus-per-task=1 --mem=1G --time=00:10:00 \
  --signal=B:TERM@30 --dependency="$GUARD_DEPENDENCY" \
  --output="$CPU_OUTPUT/logs/dependency-guard-%j.out" \
  --error="$CPU_OUTPUT/logs/dependency-guard-%j.err" \
  --export="$GUARD_EXPORTS" "$DEPENDENCY_GUARD_WORKER" "${JOB_IDS[@]}" 2>&1)"
guard_submit_rc="$?"
set -e
if [[ "$guard_submit_rc" -ne 0 || ! "$guard_output" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]]; then
  cancel_submitted
  fail "sbatch failed for auxiliary dependency guard: $guard_output"
fi
GUARD_JOB_ID="${guard_output%%;*}"
GUARD_SUBMISSION="$CPU_OUTPUT/dependency_guard_submission.txt"
GUARD_SUBMISSION_TEMPORARY="${GUARD_SUBMISSION}.tmp-$$"
printf 'submission_status=submitted\njob_id=%s\njob_name=CA0802P1DependencyGuard\nrole=auxiliary_dependency_guard\nincluded_in_formal_allocations=false\nexpected_commit=%s\nimplementation_commit=%s\nworker_sha256=%s\nconfig_sha256=%s\npartition=%s\nnode=unconstrained\nresources=gpu:0,cpu:1,mem:1G,time:00:10:00\ndependency=%s\nformal_job_ids=%s\noutput=%s\n' \
  "$GUARD_JOB_ID" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$DEPENDENCY_GUARD_WORKER_SHA256" \
  "$CONFIG_SHA256" "$PARTITION" "$GUARD_DEPENDENCY" \
  "$(IFS=,; printf '%s' "${JOB_IDS[*]}")" \
  "$CPU_OUTPUT" \
  >"$GUARD_SUBMISSION_TEMPORARY"
mv -f -- "$GUARD_SUBMISSION_TEMPORARY" "$GUARD_SUBMISSION"

# Release all held jobs only after every submission receipt has been atomically
# published.  A comma-separated job list prevents the CPU job or the guard
# from starting while a downstream receipt is still being written.
FORMAL_JOB_LIST="$(IFS=,; printf '%s' "${JOB_IDS[*]}")"
RELEASE_JOB_LIST="${FORMAL_JOB_LIST},${GUARD_JOB_ID}"
if ! scontrol release "$RELEASE_JOB_LIST" >/dev/null 2>&1; then
  cancel_submitted
  fail "could not release the complete held afterok chain"
fi

CHAIN_RECEIPT="$CPU_OUTPUT/submission_chain.txt"
printf 'submission_status=released\nfreeze_commit=%s\nimplementation_commit=%s\nconfig_sha256=%s\namendment_canonical_sha256=%s\ninput_manifest_canonical_sha256=%s\nnode=%s\npartition=%s\ndependency=afterok\njob_ids=%s\nallocations=CA_0802(2),CA_0802(3),CA_0802(4),CA_0802(5),CA_0802(6)\nmax_concurrent_gpus=1\noutput_root=%s\n' \
  "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$CONFIG_SHA256" "$AMENDMENT_CANONICAL_SHA256" \
  "$INPUT_MANIFEST_CANONICAL_SHA256" "$NODE" "$PARTITION" "$FORMAL_JOB_LIST" "$OUTPUT_ROOT" \
  >"$CHAIN_RECEIPT"
printf 'auxiliary_dependency_guard_job_id=%s\nauxiliary_dependency=%s\nall_released_job_ids=%s\n' \
  "$GUARD_JOB_ID" "$GUARD_DEPENDENCY" "$RELEASE_JOB_LIST" >>"$CHAIN_RECEIPT"
printf 'status=released;formal_job_ids=%s;dependency_guard_job_id=%s;node=%s;partition=%s;max_concurrent_gpus=1\n' \
  "$FORMAL_JOB_LIST" "$GUARD_JOB_ID" "$NODE" "$PARTITION"
