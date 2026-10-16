#!/usr/bin/env bash
set -euo pipefail

# Submit only the Proposal v9 delta-only candidate-evidence validation.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0722_proposal_delta_only_v9.sh 9

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-9}"

export MODES=proposal
export PROTOCOL_VERSION=v9
export PROTOCOL_EVIDENCE='action-scoped delta masks plus original-color delta-only T1/T2 RGB crops with cyan contour and previous/candidate masks'
export PROTOCOL_TRANSITION='v9 deterministic T1/T2 facts plus unique editable-target resolution and delta-only candidate RGB evidence'
export OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0722(${RUN_INDEX})-bailian-proposal-delta-only-v9}"
export SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0722_${RUN_INDEX}_proposal_delta_only_v9}"

exec bash "$ROOT/tools/submit_ca0722_context_fix_ablation.sh" "$RUN_INDEX"
