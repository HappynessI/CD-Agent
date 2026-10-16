#!/usr/bin/env python3
"""Build the runtime P1 input manifest without crossing the GT boundary.

The pre-rollout pass hashes online A/B inputs, model files, tools and runtime
sources, but it neither opens nor stats ``label_cvt`` members.  The
post-rollout pass requires the GT-free completion marker, rechecks every
pre-rollout hash, then and only then hashes labels and atomically replaces the
same parent-level ``input_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_runtime_input_manifest_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeManifestError(ValueError):
    """Raised when runtime inputs drift or GT is requested too early."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pre_rollout", "post_rollout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--evaluation-input-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-input-manifest-canonical-sha256", required=True)
    parser.add_argument("--resolved-run-config", type=Path, required=True)
    parser.add_argument("--resolved-run-config-sha256", required=True)
    parser.add_argument("--evaluation-profile", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--arm", choices=("h0", "h1"), required=True)
    parser.add_argument("--experiment-arm-id", required=True)
    parser.add_argument("--harness-protocol-id", required=True)
    parser.add_argument("--model-io-protocol", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--runtime-device", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument(
        "--tool-file", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--runtime-source", action="append", default=[], metavar="NAME=PATH"
    )
    return parser.parse_args(argv)


def read_object(path: Path, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeManifestError(f"cannot read {source} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeManifestError(f"{source} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeManifestError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def parse_bindings(values: Sequence[str], *, source: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in result:
            raise RuntimeManifestError(f"{source} must contain unique NAME=PATH values")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise RuntimeManifestError(f"{source} binding is not a file: {path}")
        result[name] = path
    return result


def frozen_hashes(value: Any, *, source: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeManifestError(f"{source} must be a non-empty object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name or name in result:
            raise RuntimeManifestError(f"{source} contains a malformed name")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise RuntimeManifestError(f"{source}.{name} is not a frozen SHA-256")
        result[name] = digest
    return result


def frozen_config_bindings(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if config.get("schema_id") != "qwen3vl_p1_resolved_run_config_v1":
        raise RuntimeManifestError("resolved-run-config schema drifted")
    if (
        config.get("configuration_status") != "frozen"
        or config.get("execution_readiness") != "execution_ready"
        or config.get("p1_execution_authorized") is not True
    ):
        raise RuntimeManifestError(
            "resolved-run-config is not frozen, execution-ready and authorized"
        )
    bindings = config.get("freeze_commit_hash_bindings")
    if not isinstance(bindings, Mapping) or bindings.get("status") != "frozen":
        raise RuntimeManifestError("resolved-run-config hash bindings are not frozen")
    return (
        frozen_hashes(
            bindings.get("model_files_sha256"),
            source="resolved-run-config model_files_sha256",
        ),
        frozen_hashes(
            bindings.get("tool_files_sha256"),
            source="resolved-run-config tool_files_sha256",
        ),
        frozen_hashes(
            bindings.get("worker_sha256"),
            source="resolved-run-config worker_sha256",
        ),
    )


def file_records(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeManifestError(f"model path is not a directory: {resolved}")
    records: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeManifestError(f"model tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        records[relative] = sha256_file(path)
        sizes[relative] = path.stat().st_size
    if not records:
        raise RuntimeManifestError("model tree contains no files")
    return records, sizes


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_payload(args: argparse.Namespace, *, include_gt: bool) -> dict[str, Any]:
    evaluation_path = args.evaluation_input_manifest.resolve(strict=True)
    evaluation = read_object(evaluation_path, source="evaluation input manifest")
    observed_evaluation_hash = canonical_sha256(evaluation)
    expected_evaluation_hash = args.evaluation_input_manifest_canonical_sha256
    if not SHA256_RE.fullmatch(expected_evaluation_hash):
        raise RuntimeManifestError("evaluation-input canonical SHA-256 is malformed")
    if observed_evaluation_hash != expected_evaluation_hash:
        raise RuntimeManifestError("evaluation-input manifest canonical hash drifted")
    config_path = args.resolved_run_config.resolve(strict=True)
    if not SHA256_RE.fullmatch(args.resolved_run_config_sha256):
        raise RuntimeManifestError("resolved-config SHA-256 is malformed")
    if sha256_file(config_path) != args.resolved_run_config_sha256:
        raise RuntimeManifestError("resolved-run-config byte hash drifted")
    config = read_object(config_path, source="resolved run config")
    frozen_model_hashes, frozen_tool_hashes, frozen_worker_hashes = (
        frozen_config_bindings(config)
    )
    if not SHA256_RE.fullmatch(args.worker_sha256):
        raise RuntimeManifestError("worker SHA-256 is malformed")
    if frozen_worker_hashes.get("evaluation") != args.worker_sha256:
        raise RuntimeManifestError(
            "evaluation worker SHA-256 differs from resolved-run-config"
        )
    observed_commit = git_head(Path.cwd())
    if observed_commit != args.runtime_commit:
        raise RuntimeManifestError(
            f"runtime commit drifted: expected {args.runtime_commit}, found {observed_commit}"
        )

    profiles = evaluation.get("evaluation_profiles")
    if not isinstance(profiles, list):
        raise RuntimeManifestError("evaluation manifest lacks profiles")
    matched = [
        item
        for item in profiles
        if isinstance(item, Mapping)
        and item.get("evaluation_id") == args.evaluation_id
        and item.get("ordered_sample_ids") == list(args.samples)
    ]
    if len(matched) != 1:
        raise RuntimeManifestError("evaluation ID/sample order is not frozen")
    samples_manifest = evaluation.get("samples")
    if not isinstance(samples_manifest, Mapping):
        raise RuntimeManifestError("evaluation manifest lacks samples")

    input_root = args.input_root.resolve(strict=True)
    sample_hashes: dict[str, dict[str, str | None]] = {}
    for sample in args.samples:
        frozen = samples_manifest.get(sample)
        if not isinstance(frozen, Mapping):
            raise RuntimeManifestError(f"sample is absent from evaluation manifest: {sample}")
        record: dict[str, str | None] = {}
        for directory, path_key, hash_key in (
            ("A", "a_path_relative_to_input_root", "a_sha256"),
            ("B", "b_path_relative_to_input_root", "b_sha256"),
        ):
            relative = frozen.get(path_key)
            expected = frozen.get(hash_key)
            if relative != f"{directory}/{sample}.png" or not isinstance(expected, str):
                raise RuntimeManifestError(f"frozen {directory} binding is malformed: {sample}")
            observed = sha256_file(input_root / relative)
            if observed != expected:
                raise RuntimeManifestError(f"online input hash drifted: {relative}")
            record[directory] = observed
        if include_gt:
            relative = frozen.get("label_path_relative_to_input_root")
            expected = frozen.get("label_sha256")
            if relative != f"label_cvt/{sample}.png" or not isinstance(expected, str):
                raise RuntimeManifestError(f"frozen label binding is malformed: {sample}")
            observed = sha256_file(input_root / relative)
            if observed != expected:
                raise RuntimeManifestError(f"offline label hash drifted: {relative}")
            record["label_cvt"] = observed
        else:
            # Deliberately do not resolve, stat, or open label_cvt here.
            record["label_cvt"] = None
        sample_hashes[sample] = record

    tool_paths = parse_bindings(args.tool_file, source="tool-file")
    runtime_paths = parse_bindings(args.runtime_source, source="runtime-source")
    model_path = args.model_path.resolve(strict=True)
    config_paths = config.get("paths")
    if not isinstance(config_paths, Mapping) or Path(
        str(config_paths.get("base_model", ""))
    ).resolve() != model_path:
        raise RuntimeManifestError("model path differs from resolved-run-config")
    model_records, model_sizes = file_records(model_path)
    observed_model_subset = {
        name: model_records.get(name) for name in frozen_model_hashes
    }
    if observed_model_subset != frozen_model_hashes:
        raise RuntimeManifestError(
            "actual model-file subset differs from resolved-run-config"
        )
    if set(tool_paths) != set(frozen_tool_hashes):
        raise RuntimeManifestError(
            "tool-file names differ from resolved-run-config exact set"
        )
    observed_tool_hashes = {
        name: sha256_file(path) for name, path in sorted(tool_paths.items())
    }
    if observed_tool_hashes != dict(sorted(frozen_tool_hashes.items())):
        raise RuntimeManifestError(
            "actual tool files differ from resolved-run-config"
        )
    if (
        observed_tool_hashes.get("harness_worker") != args.worker_sha256
        or frozen_tool_hashes.get("harness_worker") != args.worker_sha256
    ):
        raise RuntimeManifestError(
            "harness-worker bytes differ from evaluation worker binding"
        )
    run_dir = args.run_dir.resolve()
    marker_path = run_dir / "rollout_complete.json"
    marker_binding: dict[str, Any] | None = None
    if include_gt:
        if (run_dir / ".run_incomplete").exists():
            raise RuntimeManifestError("run remains incomplete; GT hashing is forbidden")
        marker = read_object(marker_path, source="rollout marker")
        if marker.get("execution_status") != "completed" or marker.get("gt_loaded") is not False:
            raise RuntimeManifestError("rollout marker does not prove GT-free completion")
        marker_binding = {
            "path": str(marker_path.resolve(strict=True)),
            "sha256": sha256_file(marker_path),
        }
    payload = {
        "schema_id": SCHEMA_ID,
        "phase": "post_rollout_complete" if include_gt else "pre_rollout_gt_free",
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "sample_order": list(args.samples),
        "terminal_review_arm": args.arm,
        "experiment_arm": args.experiment_arm_id,
        "harness_protocol_id": args.harness_protocol_id,
        "protocol_version": args.model_io_protocol,
        "input_root": str(input_root),
        "files_sha256": sample_hashes,
        "gt_boundary": {
            "label_content_accessed": include_gt,
            "label_hashed_after_rollout_complete": include_gt,
            "rollout_marker": marker_binding,
        },
        "gt_hashed_after_rollout": include_gt,
        "model_path": str(model_path),
        "model_files_sha256": model_records,
        "model_file_sizes": model_sizes,
        "model_tree_canonical_sha256": canonical_sha256(model_records),
        "tool_files_sha256": observed_tool_hashes,
        "tool_file_paths": {
            name: str(path) for name, path in sorted(tool_paths.items())
        },
        "runtime_source_sha256": {
            name: sha256_file(path) for name, path in sorted(runtime_paths.items())
        },
        "runtime_commit": args.runtime_commit,
        "observed_runtime_commit": observed_commit,
        "runtime_commits": {
            "chang_agent": observed_commit,
            "expected_chang_agent": args.runtime_commit,
            "harness": observed_commit,
            "expected_harness": args.runtime_commit,
        },
        "executed_worker_sha256": args.worker_sha256,
        "evaluation_input_manifest": {
            "path": str(evaluation_path),
            "canonical_sha256": observed_evaluation_hash,
        },
        "resolved_run_config": {
            "path": str(config_path),
            "sha256": args.resolved_run_config_sha256,
        },
        "slurm": {
            "job_id": args.job_id,
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "slurm_job_id": args.job_id,
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "runtime_environment": {
            "device": args.runtime_device,
        },
    }
    return payload


def atomic_publish(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        raise RuntimeManifestError(f"runtime manifest already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    include_gt = args.phase == "post_rollout"
    try:
        if include_gt:
            previous = read_object(args.output.resolve(strict=True), source="pre-rollout manifest")
            if previous.get("schema_id") != SCHEMA_ID or previous.get("phase") != "pre_rollout_gt_free":
                raise RuntimeManifestError("post-rollout pass requires the pre-rollout manifest")
        else:
            previous = None
        payload = build_payload(args, include_gt=include_gt)
        if previous is not None:
            for key in (
                "evaluation_profile",
                "evaluation_id",
                "sample_order",
                "terminal_review_arm",
                "experiment_arm",
                "harness_protocol_id",
                "protocol_version",
                "input_root",
                "model_path",
                "model_files_sha256",
                "model_file_sizes",
                "model_tree_canonical_sha256",
                "tool_files_sha256",
                "tool_file_paths",
                "runtime_source_sha256",
                "runtime_commit",
                "runtime_commits",
                "executed_worker_sha256",
                "evaluation_input_manifest",
                "resolved_run_config",
                "runtime_environment",
            ):
                if previous.get(key) != payload.get(key):
                    raise RuntimeManifestError(f"runtime input drifted during rollout: {key}")
            for sample in args.samples:
                for directory in ("A", "B"):
                    if previous["files_sha256"][sample][directory] != payload["files_sha256"][sample][directory]:
                        raise RuntimeManifestError(
                            f"online input drifted during rollout: {directory}/{sample}.png"
                        )
        atomic_publish(args.output, payload, replace=include_gt)
    except (OSError, RuntimeManifestError) as error:
        print(f"qwen3vl_runtime_input_manifest: {error}", file=sys.stderr)
        return 3
    print(
        f"status=complete;phase={payload['phase']};"
        f"samples={len(payload['sample_order'])};gt_hashed={include_gt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
