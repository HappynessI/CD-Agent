#!/usr/bin/env bash
set -euo pipefail

# Bootstrap and audit the isolated package overlay used by the Qwen3-VL-2B
# LLaMA-Factory smoke pipeline. The venv reuses the base environment's read-only
# Torch/CUDA stack but all requested installs target TRAIN_ENV. Environment
# creation and validation run in Slurm; training uses a separate GPU job.

ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
ROOT="$(realpath -e -- "$ROOT")"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
BASE_ENV="${BASE_ENV:-$WORKSPACE/conda_envs/omniovcd-env}"
TRAIN_ENV="${TRAIN_ENV:-$WORKSPACE/conda_envs/qwen3vl-sft-env}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-$WORKSPACE/Chang-Agent/third_party/LLaMA-Factory}"
LLAMA_FACTORY_COMMIT="9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d"
GIT_WRAPPER="${GIT_WRAPPER:-$WORKSPACE/.local/bin/wangyihan-git}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
MODE="${1:-install}"

die() {
  echo "setup_qwen3vl_sft_env: $*" >&2
  exit 2
}

case "$MODE" in
  install|check) ;;
  *) die "usage: $0 [install|check]" ;;
esac

# Dependency installation/import checks can consume substantial CPU and
# filesystem bandwidth. They must run inside a Slurm CPU allocation rather
# than on the management node (for example: `sbatch --cpus-per-task=4
# --mem=16G --time=00:30:00 tools/setup_qwen3vl_sft_env.sh install`).
[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this script inside a Slurm CPU allocation"

[[ -d "$BASE_ENV" ]] || die "base environment does not exist: $BASE_ENV"
[[ -d "$MODEL_PATH" ]] || die "local model does not exist: $MODEL_PATH"
[[ -x "$GIT_WRAPPER" ]] || die "git wrapper is missing: $GIT_WRAPPER"

if [[ ! -d "$LLAMA_FACTORY_ROOT/.git" ]]; then
  if [[ "$MODE" == "check" ]]; then
    die "pinned LLaMA-Factory checkout is missing: $LLAMA_FACTORY_ROOT"
  fi
  mkdir -p "$(dirname "$LLAMA_FACTORY_ROOT")"
  "$GIT_WRAPPER" clone https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_FACTORY_ROOT"
fi

HEAD="$($GIT_WRAPPER -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [[ "$HEAD" != "$LLAMA_FACTORY_COMMIT" ]]; then
  if [[ "$MODE" == "check" ]]; then
    die "LLaMA-Factory HEAD is $HEAD; expected $LLAMA_FACTORY_COMMIT"
  fi
  DIRTY="$($GIT_WRAPPER -C "$LLAMA_FACTORY_ROOT" status --porcelain 2>/dev/null || true)"
  [[ -z "$DIRTY" ]] || die "refusing to change a dirty LLaMA-Factory checkout: $LLAMA_FACTORY_ROOT"
  "$GIT_WRAPPER" -C "$LLAMA_FACTORY_ROOT" fetch --depth=1 origin "$LLAMA_FACTORY_COMMIT"
  "$GIT_WRAPPER" -C "$LLAMA_FACTORY_ROOT" checkout --detach "$LLAMA_FACTORY_COMMIT"
  HEAD="$($GIT_WRAPPER -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD)"
fi
[[ "$HEAD" == "$LLAMA_FACTORY_COMMIT" ]] || die "could not pin LLaMA-Factory"

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  [[ "$MODE" == "install" ]] || die "training environment is missing: $TRAIN_ENV"
  [[ -x "$BASE_ENV/bin/python" ]] || die "base Python is missing: $BASE_ENV/bin/python"
  mkdir -p "$(dirname "$TRAIN_ENV")"
  # `conda create --clone` is intentionally avoided: on compute nodes it may
  # copy many GB and then contact conda-forge for uncached Python/Tk packages.
  # A venv overlay leaves BASE_ENV untouched and needs no network to create.
  "$BASE_ENV/bin/python" -m venv --system-site-packages "$TRAIN_ENV"
fi

PYTHON="$TRAIN_ENV/bin/python"
[[ -x "$PYTHON" ]] || die "training interpreter is missing: $PYTHON"

check_versions() {
  "$PYTHON" - "$LLAMA_FACTORY_ROOT" "$MODEL_PATH" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model = pathlib.Path(sys.argv[2])
expected = {
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "transformers": "4.57.6",
    "datasets": "4.0.0",
    "accelerate": "1.11.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "torchdata": "0.11.0",
    "qwen-vl-utils": "0.0.14",
}
required_present = (
    "torchaudio",
    "matplotlib",
    "fire",
    "tyro",
    "omegaconf",
    "sentencepiece",
    "tiktoken",
    "jieba",
    "nltk",
    "rouge-chinese",
)
bad = []
for name, want in expected.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "MISSING"
    print(f"{name}={got}")
    if got != want:
        bad.append(f"{name}={got} (expected {want})")
for name in required_present:
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "MISSING"
    print(f"{name}={got}")
    if got == "MISSING":
        bad.append(f"{name}=MISSING")

if not (root / "src" / "llamafactory" / "cli.py").is_file():
    bad.append(f"missing LLaMA-Factory source under {root}")
if not (model / "config.json").is_file():
    bad.append(f"missing model config under {model}")
else:
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        bad.append(f"unexpected model_type={config.get('model_type')!r}")

if bad:
    raise SystemExit("version/model check failed: " + "; ".join(bad))
PY
}

if [[ "$MODE" == "install" ]]; then
  # Reuse the base Torch/CUDA stack through the venv overlay while pinning the
  # packages imported by the image-only Qwen3-VL path.
  run_pip() {
    # Compute nodes can inherit a localhost proxy that is not listening there.
    # Clear proxy variables without reading or printing their values; retain the
    # configured package index so direct/internal-mirror access still works.
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      PIP_DISABLE_PIP_VERSION_CHECK=1 \
      "$PYTHON" -m pip "$@"
  }
  run_pip install --upgrade \
    "numpy==1.26.4" \
    "scipy==1.17.1" \
    "transformers==4.57.6" \
    "datasets==4.0.0" \
    "accelerate==1.11.0" \
    "peft==0.18.1" \
    "trl==0.24.0" \
    "torchdata==0.11.0" \
    "qwen-vl-utils==0.0.14" \
    "torchaudio==2.10.0" \
    "matplotlib==3.10.7" \
    "fire==0.7.1" \
    "tyro==0.8.14" \
    "omegaconf==2.3.0" \
    "sentencepiece==0.2.1" \
    "tiktoken==0.11.0" \
    "jieba==0.42.1" \
    "nltk==3.9.2" \
    "rouge-chinese==1.0.3"
  # Install the pinned source with its remaining declared dependencies. The
  # exact core versions above already satisfy the commit's constraints, while
  # this second pass supplies transitive CLI/data imports such as protobuf,
  # pandas/scipy and the declared API extras instead of relying on accidental
  # packages from the base environment. Keep NumPy in the base workspace's
  # supported ABI range: the source project declares no upper bound, so pip
  # would otherwise select NumPy 2 and violate sam3/change-agent contracts.
  run_pip install --upgrade -e "$LLAMA_FACTORY_ROOT" \
    "numpy==1.26.4" "scipy==1.17.1"
fi

export PYTHONPATH="$LLAMA_FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
check_versions

# Import the actual multimodal training path, not just package metadata. This
# catches missing torchaudio/matplotlib and incompatible transitive imports
# before a GPU job spends time loading the 4GB checkpoint.
"$PYTHON" - <<'PY'
import llamafactory.data.mm_plugin  # noqa: F401
import llamafactory.train.sft.workflow  # noqa: F401
print("llamafactory_import=ok")
PY
"$PYTHON" -m pip check

if [[ "$MODE" == "install" ]]; then
  # These manifests contain package versions and the source commit only; no
  # tokens, proxy values, or other credentials are written.
  "$GIT_WRAPPER" -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD >"$TRAIN_ENV/qwen3vl_sft_llamafactory_commit.txt"
  "$PYTHON" -m pip freeze >"$TRAIN_ENV/qwen3vl_sft_pip_freeze.txt"
  printf '%s\n' \
    "environment_kind=venv_system_site_overlay" \
    "base_environment=$BASE_ENV" \
    "base_environment_mutated=false" \
    >"$TRAIN_ENV/qwen3vl_sft_environment_kind.txt"
  echo "Qwen3-VL SFT environment installed: $TRAIN_ENV"
else
  echo "Qwen3-VL SFT environment check passed: $TRAIN_ENV"
fi
