import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from change_agent.state import ChangeState, VerifierOutput
from change_agent.trajectory import Trajectory, TrajectoryEntry


class TrajectoryVisualizationTest(unittest.TestCase):
    def test_saves_temporal_masks_and_each_matching_instance(self):
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        t1_mask = np.zeros((6, 7), dtype=bool)
        t1_mask[1:3, 1:3] = True
        t2_mask = np.zeros_like(t1_mask)
        t2_mask[1:3, 1:3] = True
        t2_mask[4:6, 5:7] = True
        t1_instances = (t1_mask.copy(),)
        t2_instances = (
            np.logical_and(t2_mask, np.indices(t2_mask.shape)[0] < 3),
            np.logical_and(t2_mask, np.indices(t2_mask.shape)[0] >= 3),
        )
        state = ChangeState(
            t1_image=image,
            t2_image=image,
            query="building",
            t1_mask=t1_mask,
            t2_mask=t2_mask,
            change_mask=np.logical_xor(t1_mask, t2_mask),
            t1_instances=t1_instances,
            t2_instances=t2_instances,
        )
        trajectory = Trajectory(run_metadata={"dataset": "test"})
        trajectory.append(
            TrajectoryEntry(
                step_index=0,
                raw_action=None,
                parsed_action=None,
                verifier=VerifierOutput(comparison="initial"),
                state=state,
                execution={"candidate_accepted": True},
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = trajectory.save(
                root / "trajectory",
                root / "masks",
                root / "visualizations",
            )
            step_dir = root / "visualizations" / "step_000"
            self.assertTrue((step_dir / "t1_mask.png").is_file())
            self.assertTrue((step_dir / "t2_mask.png").is_file())
            self.assertEqual(
                sorted(path.name for path in (step_dir / "t1_instances").glob("*.png")),
                ["instance_000.png"],
            )
            self.assertEqual(
                sorted(path.name for path in (step_dir / "t2_instances").glob("*.png")),
                ["instance_000.png", "instance_001.png"],
            )
            saved_t2 = np.asarray(Image.open(step_dir / "t2_mask.png"))
            self.assertEqual(set(np.unique(saved_t2)), {0, 255})
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["steps"][0]["visualization_dir"],
                "../visualizations/step_000",
            )


if __name__ == "__main__":
    unittest.main()
