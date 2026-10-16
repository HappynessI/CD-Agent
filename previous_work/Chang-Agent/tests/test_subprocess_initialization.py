import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from change_agent.adapters.subprocess_adapters import SubprocessSAM3Initializer
from tools.segmentation_worker import _sam3_autocast


class Completed:
    returncode = 0
    stdout = "ok"
    stderr = ""


class Failed:
    returncode = 7
    stdout = "worker standard output"
    stderr = "worker traceback"


class SubprocessInitializationTest(unittest.TestCase):
    def test_fresh_initializer_loads_masks_and_persisted_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initializer = SubprocessSAM3Initializer(
                "/usr/bin/python3",
                root / "worker.py",
                root / "artifacts",
                checkpoint=root / "sam3.pt",
                bpe=root / "bpe.gz",
            )

            def fake_run(command, **kwargs):
                self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
                self.assertNotIn("DASHSCOPE_API_KEY", kwargs["env"])
                self.assertNotIn("OTHER_SERVICE_ACCESS_TOKEN", kwargs["env"])
                self.assertEqual(kwargs["env"]["SAFE_TOOL_SETTING"], "kept")
                def value(flag):
                    return Path(command[command.index(flag) + 1])

                t1_mask = np.zeros((8, 8), dtype=np.uint8)
                t2_mask = np.ones((8, 8), dtype=np.uint8)
                np.save(value("--output-mask"), t1_mask)
                np.save(value("--output-mask-t2"), t2_mask)
                evidence_dir = value("--evidence-dir")
                evidence_dir.mkdir(parents=True)
                t1_conf = evidence_dir / "t1_confidence_map.npy"
                t2_conf = evidence_dir / "t2_confidence_map.npy"
                np.save(t1_conf, np.full((8, 8), 0.2, dtype=np.float32))
                np.save(t2_conf, np.full((8, 8), 0.7, dtype=np.float32))
                report = {
                    "status": "success",
                    "intermediate_artifacts": {
                        "t1": {"confidence_map": {"file": str(t1_conf)}},
                        "t2": {"confidence_map": {"file": str(t2_conf)}},
                    },
                }
                value("--report").write_text(json.dumps(report), encoding="utf-8")
                return Completed()

            image = np.zeros((8, 8, 3), dtype=np.uint8)
            with patch.dict(
                os.environ,
                {
                    "DASHSCOPE_API_KEY": "sk-must-not-reach-worker",
                    "OTHER_SERVICE_ACCESS_TOKEN": "must-not-reach-worker",
                    "SAFE_TOOL_SETTING": "kept",
                },
            ):
                with patch(
                    "change_agent.adapters.subprocess_adapters.subprocess.run",
                    fake_run,
                ):
                    t1, t2, evidence = initializer.initialize_masks(
                        image, image, "building"
                    )
            self.assertFalse(t1.any())
            self.assertTrue(t2.all())
            self.assertTrue(np.allclose(evidence["change_confidence"], 0.7))
            self.assertEqual(evidence["initializer"], "live_sam3_dual_view_text_prompt")
            self.assertFalse(any(root.rglob("stdout.log")))
            self.assertFalse(any(root.rglob("stderr.log")))

    def test_failed_worker_retains_bounded_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initializer = SubprocessSAM3Initializer(
                "/usr/bin/python3",
                root / "worker.py",
                root / "artifacts",
                checkpoint=root / "sam3.pt",
                bpe=root / "bpe.gz",
            )
            image = np.zeros((8, 8, 3), dtype=np.uint8)

            with patch(
                "change_agent.adapters.subprocess_adapters.subprocess.run",
                return_value=Failed(),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "initialization failed with exit=7"
                ):
                    initializer.initialize_masks(image, image, "building")

            call_dir = root / "artifacts" / "initialize_000"
            self.assertEqual(
                (call_dir / "stdout.log").read_text(encoding="utf-8"),
                "worker standard output",
            )
            self.assertEqual(
                (call_dir / "stderr.log").read_text(encoding="utf-8"),
                "worker traceback",
            )

    def test_sam3_autocast_is_disabled_for_cpu(self):
        with _sam3_autocast("cpu"):
            pass

    @patch("torch.autocast")
    @patch("torch.cuda.is_bf16_supported", return_value=True)
    def test_sam3_autocast_uses_bfloat16_when_supported(
        self, _is_bf16_supported, autocast
    ):
        _sam3_autocast("cuda")
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    @patch("torch.autocast")
    @patch("torch.cuda.is_bf16_supported", return_value=False)
    def test_sam3_autocast_falls_back_to_float16(
        self, _is_bf16_supported, autocast
    ):
        _sam3_autocast("cuda:0")
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()
