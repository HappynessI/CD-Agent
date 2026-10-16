#!/usr/bin/env python3
"""Seed an isolated tool process before delegating to segmentation_worker.py."""

from __future__ import annotations

import json
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np


def _seed_worker() -> dict[str, object]:
    seed = int(os.environ.get("CHANGE_AGENT_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    details: dict[str, object] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": False,
        "deterministic_algorithms": False,
    }
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        details["torch"] = True
        details["torch_version"] = torch.__version__
    except ImportError:
        details["torch_version"] = None
    return details


def _report_path() -> Path | None:
    if "--report" not in sys.argv:
        return None
    index = sys.argv.index("--report")
    return Path(sys.argv[index + 1]) if index + 1 < len(sys.argv) else None


def main() -> None:
    seed_details = _seed_worker()
    report_path = _report_path()
    worker = Path(__file__).with_name("segmentation_worker.py")
    runpy.run_path(str(worker), run_name="__main__")
    if report_path is not None and report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["seed_runtime"] = seed_details
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
