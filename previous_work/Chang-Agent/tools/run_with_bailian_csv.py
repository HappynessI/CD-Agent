#!/usr/bin/env python3
"""Exec a child command with DASHSCOPE_API_KEY loaded from a local CSV.

The key is kept in memory and is never printed or written to an artifact.  The
CSV is expected to contain a row whose first column is ``api_key`` and whose
second column is the key value.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit("usage: run_with_bailian_csv.py CSV_PATH COMMAND [ARGS ...]")
    csv_path = Path(argv[1])
    command = argv[2:]
    key = None
    base_url = None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            field = row[0].strip().lower().replace("_", "")
            value = row[1].strip()
            if field == "apikey":
                key = value
            elif field == "openaicompatible":
                base_url = value
    if not key or not key.startswith("sk-"):
        raise SystemExit("CSV does not contain a valid api_key row")
    environment = os.environ.copy()
    environment["DASHSCOPE_API_KEY"] = key
    if base_url:
        # Workspace-scoped keys must use the workspace's compatible endpoint.
        environment["DASHSCOPE_BASE_URL"] = base_url.rstrip("/")
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
