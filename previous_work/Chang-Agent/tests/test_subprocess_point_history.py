import unittest

import numpy as np

from change_agent.adapters.subprocess_adapters import SubprocessPointBackend


class SubprocessPointHistoryTest(unittest.TestCase):
    def test_point_backend_serializes_accepted_click_history(self):
        backend = SubprocessPointBackend(
            "/usr/bin/python3",
            "/tmp/worker.py",
            "/tmp/artifacts",
            checkpoint="/tmp/simpleclick.pth",
        )
        captured = {}

        def fake_call(mode, image, initial_mask, extra_args):
            captured.update(mode=mode, extra_args=extra_args)
            return np.array(initial_mask, copy=True)

        backend._call = fake_call
        mask = np.zeros((8, 8), dtype=bool)
        backend.refine(
            np.zeros((8, 8, 3), dtype=np.uint8),
            mask,
            (6, 7),
            False,
            (((1, 2), True), ((3, 4), False)),
        )

        self.assertEqual(captured["mode"], "point")
        self.assertEqual(
            captured["extra_args"],
            [
                "--checkpoint",
                "/tmp/simpleclick.pth",
                "--coordinate",
                "6",
                "7",
                "--is-positive",
                "0",
                "--history-click",
                "1",
                "2",
                "1",
                "--history-click",
                "3",
                "4",
                "0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
