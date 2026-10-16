#!/usr/bin/env bash
set -euo pipefail

# Submit only the Proposal v7 executable-diagnosis fallback validation.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0722_proposal_executable_v7.sh 7

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-7}"

export MODES=proposal
export OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0722(${RUN_INDEX})-bailian-proposal-executable-v7}"
export SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0722_${RUN_INDEX}_proposal_executable_v7}"

exec bash "$ROOT/tools/submit_ca0722_context_fix_ablation.sh" "$RUN_INDEX"
