import unittest

import numpy as np
import torch

from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter


class FakeProcessor:
    def __init__(self):
        self.box = None

    def set_image(self, image):
        return {"shape": (image.height, image.width)}

    def reset_all_prompts(self, state):
        return None

    def set_text_prompt(self, prompt, state):
        height, width = state["shape"]
        logits = np.full((height, width), -4.0, dtype=np.float32)
        logits[2:6, 3:7] = 4.0
        state.update({"semantic_mask_logits": logits, "presence_score": np.array([0.9])})
        return state

    def add_geometric_prompt(self, box, label, state):
        self.box = box
        return state


class SAM3AdapterTest(unittest.TestCase):
    def test_text_and_box_paths_use_public_processor_api(self):
        processor = FakeProcessor()
        adapter = SAM3ProcessorAdapter(processor)
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        t1, t2, evidence = adapter.initialize_masks(image, image, "building")
        self.assertEqual(int(t1.sum()), 16)
        self.assertTrue(np.array_equal(t1, t2))
        self.assertIn("change_confidence", evidence)
        result = adapter.segment_box(image, (0.5, 0.5, 0.2, 0.3), "building")
        self.assertEqual(int(result.sum()), 16)
        self.assertEqual(processor.box, [0.5, 0.5, 0.2, 0.3])

    def test_probability_outputs_use_omniovcd_threshold_and_presence(self):
        adapter = SAM3ProcessorAdapter(FakeProcessor())
        semantic = np.full((1, 1, 8, 8), 0.01, dtype=np.float32)
        semantic[..., 2:6, 2:6] = 0.9
        state = {
            "semantic_mask_logits": semantic,
            "masks_logits": np.empty((0, 1, 8, 8), dtype=np.float32),
            "presence_score": np.asarray(0.8, dtype=np.float32),
        }
        mask, confidence = adapter._mask_from_state(state, (8, 8))
        self.assertEqual(int(mask.sum()), 16)
        self.assertAlmostEqual(float(confidence.max()), 0.72, places=5)

    def test_low_presence_rejects_semantic_false_positive(self):
        adapter = SAM3ProcessorAdapter(FakeProcessor())
        state = {
            "semantic_mask_logits": np.ones((1, 1, 8, 8), dtype=np.float32),
            "masks_logits": np.empty((0, 1, 8, 8), dtype=np.float32),
            "presence_score": np.asarray(0.02, dtype=np.float32),
        }
        mask, _ = adapter._mask_from_state(state, (8, 8))
        self.assertFalse(mask.any())

    def test_empty_detector_outputs_return_an_empty_mask(self):
        adapter = SAM3ProcessorAdapter(FakeProcessor())
        state = {
            "semantic_mask_logits": np.empty((0, 1, 8, 8), dtype=np.float32),
            "masks_logits": np.empty((0, 1, 8, 8), dtype=np.float32),
            "masks": np.empty((0, 8, 8), dtype=np.float32),
            "presence_score": np.empty((0,), dtype=np.float32),
        }
        mask, confidence = adapter._mask_from_state(state, (8, 8))
        self.assertEqual(mask.shape, (8, 8))
        self.assertFalse(mask.any())
        self.assertTrue(
            np.array_equal(confidence, np.zeros((8, 8), dtype=np.float32))
        )

    def test_bfloat16_diagnostics_are_promoted_for_numpy(self):
        value = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
        result = SAM3ProcessorAdapter._numpy(value)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, [0.25, 0.75]))


if __name__ == "__main__":
    unittest.main()
