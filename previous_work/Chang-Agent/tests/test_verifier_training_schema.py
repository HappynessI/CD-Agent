import unittest

from tools.train_verifier import make_synthetic_samples
from change_agent.verifier_model import build_verifier_head, verifier_loss


class VerifierTrainingSchemaTest(unittest.TestCase):
    def test_synthetic_samples_do_not_contain_fake_target_view_labels(self):
        samples = make_synthetic_samples(4, 42)
        self.assertNotIn("target_view", samples)
        self.assertEqual(
            set(samples),
            {"features", "quality", "error_map", "error_type"},
        )

    def test_head_and_loss_have_no_target_view_branch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        samples = make_synthetic_samples(2, 42)
        tensors = {key: torch.from_numpy(value) for key, value in samples.items()}
        model = build_verifier_head(tensors["features"].shape[1], hidden_channels=8)
        predictions = model(tensors["features"])
        self.assertFalse(hasattr(predictions, "target_view_logits"))
        self.assertFalse(hasattr(predictions, "action_logits"))
        loss = verifier_loss(predictions, tensors)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
