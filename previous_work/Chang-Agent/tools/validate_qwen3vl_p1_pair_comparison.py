#!/usr/bin/env python3
"""Independently validate a frozen H0/H1 P1 comparison receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_h0_h1_pair_comparison_v1"
PROFILES = {
    "p1_smoke1": ("DEV-SMOKE-1-H1-V1", ["test_1_1"]),
    "e2e3": ("DEV-CORE-3-V1", ["test_20_15", "test_1_1", "test_50_8"]),
}
ARM_IDS = {
    "h0": "v12_1-h0-legacy-single-round-base-0802",
    "h1": "v12_1-h1-gtfree-terminal-review-base-0802",
}


class PairValidationError(ValueError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--h0-receipt", type=Path, required=True)
    parser.add_argument("--h1-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--receipt-validator", type=Path, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairValidationError(f"cannot read {source}: {error}") from error
    if not isinstance(payload, dict):
        raise PairValidationError(f"{source} must be an object")
    return payload


def mapping(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairValidationError(f"{source} must be an object")
    return value


def validate_individual(path: Path, validator: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(validator.resolve(strict=True)), "--receipt", str(path.resolve(strict=True))],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PairValidationError(
            f"individual receipt validation failed: {(completed.stderr or completed.stdout).strip()}"
        )
    return read(path.resolve(strict=True), source="individual receipt")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_id, samples = PROFILES[args.evaluation_profile]
    h0_path = args.h0_receipt.resolve(strict=True)
    h1_path = args.h1_receipt.resolve(strict=True)
    h0 = validate_individual(h0_path, args.receipt_validator)
    h1 = validate_individual(h1_path, args.receipt_validator)
    pair = read(args.pair.resolve(strict=True), source="pair receipt")
    if pair.get("schema_id") != SCHEMA_ID or pair.get("status") != "complete":
        raise PairValidationError("pair schema/status is not complete")
    if pair.get("evaluation_profile") != args.evaluation_profile:
        raise PairValidationError("pair profile differs from validator arguments")
    if pair.get("evaluation_id") != evaluation_id or pair.get("sample_order") != samples:
        raise PairValidationError("pair evaluation identity differs")
    identities = {
        arm: mapping(receipt.get("identity"), source=f"{arm}.identity")
        for arm, receipt in (("h0", h0), ("h1", h1))
    }
    if identities["h0"].get("job_id") == identities["h1"].get("job_id"):
        raise PairValidationError("paired Job IDs must differ")
    for arm in ("h0", "h1"):
        expected = {
            "arm": arm,
            "experiment_arm_id": ARM_IDS[arm],
            "evaluation_profile": args.evaluation_profile,
            "evaluation_id": evaluation_id,
            "sample_order": samples,
        }
        for key, value in expected.items():
            if identities[arm].get(key) != value:
                raise PairValidationError(f"{arm} identity {key} differs")
    if identities["h0"].get("runtime_commit") != identities["h1"].get("runtime_commit"):
        raise PairValidationError("paired runtime commits differ")
    provenances = {
        arm: mapping(receipt.get("provenance"), source=f"{arm}.provenance")
        for arm, receipt in (("h0", h0), ("h1", h1))
    }
    implementation_commit = provenances["h0"].get("implementation_commit")
    if implementation_commit != provenances["h1"].get("implementation_commit"):
        raise PairValidationError("paired implementation commits differ")
    freeze_commit = identities["h0"].get("runtime_commit")
    if any(
        provenance.get("freeze_commit") != freeze_commit
        for provenance in provenances.values()
    ):
        raise PairValidationError("paired provenance freeze commits differ")
    if pair.get("implementation_commit") != implementation_commit:
        raise PairValidationError("pair implementation commit is not reconstructed")
    if pair.get("freeze_commit") != freeze_commit:
        raise PairValidationError("pair freeze commit is not reconstructed")

    arm_artifacts = mapping(pair.get("arms"), source="pair.arms")
    for arm, receipt_path in (("h0", h0_path), ("h1", h1_path)):
        record = mapping(arm_artifacts.get(arm), source=f"pair.arms.{arm}")
        bound = mapping(record.get("receipt"), source=f"pair.arms.{arm}.receipt")
        if bound.get("path") != str(receipt_path) or bound.get("sha256") != sha256(receipt_path):
            raise PairValidationError(f"pair {arm} receipt binding drifted")

    gates = mapping(pair.get("hard_gates"), source="pair.hard_gates")
    if any(value is not True for value in gates.values()):
        raise PairValidationError("pair hard gates contain a failure")
    comparison = mapping(pair.get("comparison"), source="pair.comparison")
    h0_comp = mapping(comparison.get("h0"), source="pair.comparison.h0")
    h1_comp = mapping(comparison.get("h1"), source="pair.comparison.h1")
    if h1_comp.get("final_region_protocol_valid_count", -1) < h0_comp.get(
        "final_region_protocol_valid_count", -1
    ):
        raise PairValidationError("H1 protocol validity is below H0")
    if comparison.get("final_region_protocol_validity_delta_h1_minus_h0") != (
        h1_comp.get("final_region_protocol_valid_count")
        - h0_comp.get("final_region_protocol_valid_count")
    ):
        raise PairValidationError("pair validity delta is not reconstructed")
    if args.evaluation_profile == "p1_smoke1":
        if h1_comp.get("round_two_invoked_count") != 1:
            raise PairValidationError("smoke H1 did not invoke exactly one round two")
        if h1_comp.get("final_region_protocol_valid_count") != 1:
            raise PairValidationError("smoke H1 final protocol is invalid")
    return pair


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pair = validate(args)
    except (OSError, PairValidationError) as error:
        print(f"qwen3vl_p1_pair_comparison: {error}", file=sys.stderr)
        return 3
    print(
        "status=valid;"
        f"profile={pair['evaluation_profile']};"
        f"samples={len(pair['sample_order'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
