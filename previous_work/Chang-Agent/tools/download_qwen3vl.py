#!/usr/bin/env python3
"""Download the pinned Qwen3-VL checkpoint outside the Git working tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--local-dir",
        default="/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    destination = Path(args.local_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=args.max_workers,
    )
    files = []
    total_bytes = 0
    for file_path in sorted(destination.rglob("*")):
        if file_path.is_file():
            size = file_path.stat().st_size
            files.append({"path": str(file_path.relative_to(destination)), "bytes": size})
            total_bytes += size
    manifest = {
        "repo": args.repo,
        "local_dir": str(destination),
        "snapshot_path": str(path),
        "total_bytes": total_bytes,
        "files": files,
    }
    (destination / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
