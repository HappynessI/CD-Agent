"""Real-model tool adapters executed in isolated Python environments."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


_SENSITIVE_ENVIRONMENT_NAMES = frozenset(
    {
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
_SENSITIVE_ENVIRONMENT_SUFFIXES = (
    "_API_KEY",
    "_ACCESS_KEY",
    "_TOKEN",
    "_ACCESS_TOKEN",
    "_AUTH_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PRIVATE_KEY",
)


def _isolated_tool_environment(
    pythonpath: tuple[str, ...],
    seed: int | None,
) -> dict[str, str]:
    """Build a least-privilege environment for local segmentation workers."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _SENSITIVE_ENVIRONMENT_NAMES
        and not key.upper().endswith(_SENSITIVE_ENVIRONMENT_SUFFIXES)
    }
    # Tool environments are intentionally isolated. User-site NumPy can
    # override the pinned SimpleClick stack and break legacy imgaug.
    environment["PYTHONNOUSERSITE"] = "1"
    if seed is not None:
        environment["PYTHONHASHSEED"] = str(seed)
        environment["CHANGE_AGENT_SEED"] = str(seed)
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if pythonpath:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            pythonpath + ((existing,) if existing else ())
        )
    return environment


def _persist_failed_worker_streams(
    call_dir: Path,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    """Retain bounded worker diagnostics only when a worker fails."""

    for name, value in (("stdout.log", stdout), ("stderr.log", stderr)):
        if value is None:
            continue
        text = (
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
        )
        if text:
            (call_dir / name).write_text(text[:1_000_000], encoding="utf-8")


def _run_worker(
    command: list[str],
    *,
    worker: str,
    environment: dict[str, str],
    timeout_seconds: int,
    call_dir: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(Path(worker).parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        _persist_failed_worker_streams(call_dir, error.stdout, error.stderr)
        raise RuntimeError(
            f"worker timed out after {timeout_seconds}s; diagnostics: {call_dir}"
        ) from error
    if result.returncode != 0:
        _persist_failed_worker_streams(call_dir, result.stdout, result.stderr)
    return result


class _SubprocessTool:
    def __init__(
        self,
        python: str | Path,
        worker: str | Path,
        artifact_dir: str | Path,
        *,
        pythonpath: Iterable[str | Path] = (),
        device: str = "cuda",
        timeout_seconds: int = 300,
        seed: int | None = None,
    ):
        self.python = str(Path(python).resolve())
        self.worker = str(Path(worker).resolve())
        self.artifact_dir = Path(artifact_dir)
        self.pythonpath = tuple(str(Path(item).resolve()) for item in pythonpath)
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.call_index = 0
        self.last_evidence: dict[str, Any] = {}

    def _call(
        self,
        mode: str,
        image: np.ndarray,
        initial_mask: np.ndarray | None,
        extra_args: list[str],
    ) -> np.ndarray:
        call_dir = self.artifact_dir / f"{mode}_{self.call_index:03d}"
        self.call_index += 1
        call_dir.mkdir(parents=True, exist_ok=False)
        image_path = call_dir / "image.png"
        mask_path = call_dir / "initial_mask.npy"
        output_path = call_dir / "output_mask.npy"
        report_path = call_dir / "report.json"
        Image.fromarray(_uint8_image(image)).save(image_path)
        if initial_mask is not None:
            np.save(mask_path, np.asarray(initial_mask, dtype=np.uint8))

        command = [
            self.python,
            self.worker,
            mode,
            "--image",
            str(image_path),
            "--output-mask",
            str(output_path),
            "--report",
            str(report_path),
            "--device",
            self.device,
        ]
        if initial_mask is not None:
            command.extend(["--initial-mask", str(mask_path)])
        command.extend(extra_args)
        env = _isolated_tool_environment(self.pythonpath, self.seed)
        result = _run_worker(
            command,
            worker=self.worker,
            environment=env,
            timeout_seconds=self.timeout_seconds,
            call_dir=call_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{mode} worker failed with exit={result.returncode}")
        if not output_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"{mode} worker did not produce its declared artifacts")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.last_evidence = {
            "worker_mode": mode,
            "worker_command": command,
            "worker_report": report,
            "worker_artifact_dir": str(call_dir),
            "worker_seed": self.seed,
        }
        return np.asarray(np.load(output_path), dtype=bool)


class SubprocessPointBackend(_SubprocessTool):
    def __init__(self, *args: Any, checkpoint: str | Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.checkpoint = str(Path(checkpoint).resolve())

    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
        click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
    ) -> np.ndarray:
        x, y = coordinate
        history_args = [
            value
            for history_coordinate, history_is_positive in click_history
            for value in (
                "--history-click",
                str(history_coordinate[0]),
                str(history_coordinate[1]),
                "1" if history_is_positive else "0",
            )
        ]
        return self._call(
            "point",
            image,
            initial_mask,
            [
                "--checkpoint",
                self.checkpoint,
                "--coordinate",
                str(x),
                str(y),
                "--is-positive",
                "1" if is_positive else "0",
                *history_args,
            ],
        )


class SubprocessBoxBackend(_SubprocessTool):
    def __init__(
        self,
        *args: Any,
        checkpoint: str | Path,
        bpe: str | Path,
        resolution: int = 1008,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.bpe = str(Path(bpe).resolve())
        self.resolution = resolution

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        return self._call(
            "box",
            image,
            None,
            [
                "--checkpoint",
                self.checkpoint,
                "--bpe",
                self.bpe,
                "--resolution",
                str(self.resolution),
                "--query",
                query,
                "--box",
                *(str(value) for value in box_cxcywh_normalized),
            ],
        )


class SubprocessSAM3Initializer(_SubprocessTool):
    """Run fresh dual-view SAM3 text initialization and preserve all evidence arrays."""

    def __init__(
        self,
        *args: Any,
        checkpoint: str | Path,
        bpe: str | Path,
        resolution: int = 1008,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.bpe = str(Path(bpe).resolve())
        self.resolution = resolution

    def initialize_masks(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        call_dir = self.artifact_dir / f"initialize_{self.call_index:03d}"
        self.call_index += 1
        call_dir.mkdir(parents=True, exist_ok=False)
        t1_image_path = call_dir / "t1_image.png"
        t2_image_path = call_dir / "t2_image.png"
        t1_mask_path = call_dir / "t1_mask.npy"
        t2_mask_path = call_dir / "t2_mask.npy"
        evidence_dir = call_dir / "evidence"
        report_path = call_dir / "report.json"
        Image.fromarray(_uint8_image(t1_image)).save(t1_image_path)
        Image.fromarray(_uint8_image(t2_image)).save(t2_image_path)
        command = [
            self.python,
            self.worker,
            "initialize",
            "--image",
            str(t1_image_path),
            "--image-t2",
            str(t2_image_path),
            "--output-mask",
            str(t1_mask_path),
            "--output-mask-t2",
            str(t2_mask_path),
            "--evidence-dir",
            str(evidence_dir),
            "--report",
            str(report_path),
            "--device",
            self.device,
            "--checkpoint",
            self.checkpoint,
            "--bpe",
            self.bpe,
            "--resolution",
            str(self.resolution),
            "--query",
            query,
        ]
        env = _isolated_tool_environment(self.pythonpath, self.seed)
        result = _run_worker(
            command,
            worker=self.worker,
            environment=env,
            timeout_seconds=self.timeout_seconds,
            call_dir=call_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SAM3 initialization failed with exit={result.returncode}")
        required = (t1_mask_path, t2_mask_path, report_path)
        if not all(path.is_file() for path in required):
            raise RuntimeError("SAM3 initialization did not produce all declared artifacts")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        t1_mask = np.asarray(np.load(t1_mask_path), dtype=bool)
        t2_mask = np.asarray(np.load(t2_mask_path), dtype=bool)
        if t1_mask.shape != t1_image.shape[:2] or t2_mask.shape != t2_image.shape[:2]:
            raise ValueError("fresh SAM3 mask shape does not match its temporal image")
        artifacts = report["intermediate_artifacts"]
        t1_confidence = np.asarray(
            np.load(artifacts["t1"]["confidence_map"]["file"]), dtype=np.float32
        )
        t2_confidence = np.asarray(
            np.load(artifacts["t2"]["confidence_map"]["file"]), dtype=np.float32
        )
        self.last_evidence = {
            "worker_mode": "initialize",
            "worker_command": command,
            "worker_report": report,
            "worker_artifact_dir": str(call_dir),
            "worker_seed": self.seed,
        }
        return t1_mask, t2_mask, {
            "initializer": "live_sam3_dual_view_text_prompt",
            "sam3_initialization": self.last_evidence,
            "change_confidence": np.maximum(t1_confidence, t2_confidence),
        }


def _uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
