import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from change_agent.trajectory import default_run_metadata
from tools.run_levir_change_agent import _model_identity, _seed_runtime
from tools.seeded_segmentation_worker import _seed_worker


class AuditRuntimeTest(unittest.TestCase):
    def test_parent_and_worker_seed_records_are_explicit(self):
        parent = _seed_runtime(17)
        with patch.dict(os.environ, {"CHANGE_AGENT_SEED": "17"}):
            worker = _seed_worker()

        self.assertEqual(parent["python_random"], 17)
        self.assertEqual(parent["numpy"], 17)
        self.assertEqual(worker["seed"], 17)
        self.assertTrue(worker["python_random"])
        self.assertTrue(worker["numpy"])
        self.assertIn("deterministic_algorithms", worker)

    def test_model_identity_hashes_small_metadata_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
            identity = _model_identity(root)

        expected = hashlib.sha256(b'{"model_type":"qwen3_vl"}').hexdigest()
        self.assertTrue(identity["exists"])
        self.assertEqual(identity["metadata_sha256"]["config.json"], expected)

    def test_trajectory_metadata_resolves_repository_explicitly(self):
        metadata = default_run_metadata()
        self.assertIsNotNone(metadata["git_commit"])
        self.assertIsInstance(metadata["git_dirty"], bool)
        self.assertRegex(metadata["git_worktree_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
