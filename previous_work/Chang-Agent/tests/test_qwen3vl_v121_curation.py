import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_qwen3vl_v121_curation import (
    Candidate,
    candidate_rank,
    component_box,
    components,
    select_candidates,
)


class Qwen3VLV121CurationTest(unittest.TestCase):
    def test_components_and_boxes_are_exact(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:5, 3:8] = True
        mask[20:25, 18:24] = True
        found = sorted(components(mask), key=lambda item: int(item.sum()))
        self.assertEqual([int(item.sum()) for item in found], [15, 30])
        self.assertEqual(component_box(found[0]), (3, 2, 8, 5))
        self.assertEqual(component_box(found[1]), (18, 20, 24, 25))

    def test_selection_is_deterministic_and_parent_isolated(self):
        dummy = Path("/unused")
        records = [
            Candidate(
                "train",
                f"parent_{index // 2}",
                index % 4,
                (index // 4) % 4,
                "changed",
                300,
                250,
                (20, 20, 60, 60),
                dummy,
                dummy,
                dummy,
            )
            for index in range(20)
        ]
        first = select_candidates(records, kind="changed", count=6, seed=801)
        second = select_candidates(list(reversed(records)), kind="changed", count=6, seed=801)
        self.assertEqual(first, second)
        self.assertEqual(len({item.parent_id for item in first}), 6)
        self.assertEqual(
            first,
            sorted(first, key=lambda item: candidate_rank(item, 801)),
        )

    def test_test_split_is_not_a_supported_discovery_contract(self):
        # Candidate records deliberately use only the two allowed source split
        # identities; the full builder validates this again in its manifest.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "label.png"
            Image.new("L", (1024, 1024), 0).save(path)
            record = Candidate(
                "val",
                "parent",
                0,
                0,
                "unchanged",
                0,
                0,
                (72, 72, 184, 184),
                path,
                path,
                path,
            )
            self.assertIn(record.split, {"train", "val"})


if __name__ == "__main__":
    unittest.main()
