import unittest

import numpy as np

from change_agent.adapters.segagent_adapter import SimpleClickAdapter


class FakeClick:
    def __init__(self, is_positive, coords):
        self.is_positive = is_positive
        self.coords = coords


class FakeClicker:
    def __init__(self):
        self.click_indx_offset = 0
        self.clicks = []

    def add_click(self, click):
        self.clicks.append(click)

    def get_clicks(self):
        return list(self.clicks)


class FakePredictor:
    device = "cpu"

    def __init__(self):
        self.images = []
        self.calls = []

    def set_input_image(self, image):
        self.images.append(np.array(image, copy=True))

    def get_prediction(self, clicker, prev_mask=None):
        self.calls.append(
            {
                "clicks": [
                    (click.coords, click.is_positive) for click in clicker.get_clicks()
                ],
                "offset": clicker.click_indx_offset,
                "prev_mask": prev_mask.detach().cpu().numpy().copy(),
            }
        )
        return prev_mask.detach().cpu().numpy()[0, 0]


class TestableSimpleClickAdapter(SimpleClickAdapter):
    def _click_classes(self):
        return FakeClick, FakeClicker


class SimpleClickAdapterTest(unittest.TestCase):
    def test_replays_history_and_forwards_external_mask_to_every_prediction(self):
        predictor = FakePredictor()
        adapter = TestableSimpleClickAdapter(predictor)
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        initial_mask = np.zeros((6, 7), dtype=bool)
        initial_mask[2:4, 3:5] = True

        result = adapter.refine(
            image,
            initial_mask,
            (5, 4),
            False,
            click_history=(((1, 2), True),),
        )

        self.assertEqual(len(predictor.calls), 3)
        self.assertEqual([len(call["clicks"]) for call in predictor.calls], [1, 1, 2])
        self.assertTrue(all(call["offset"] == 1 for call in predictor.calls))
        for call in predictor.calls:
            self.assertEqual(call["prev_mask"].shape, (1, 1, 6, 7))
            self.assertTrue(np.array_equal(call["prev_mask"][0, 0], initial_mask))
        self.assertEqual(
            predictor.calls[-1]["clicks"],
            [((2, 1), True), ((4, 5), False)],
        )
        self.assertTrue(np.array_equal(result, initial_mask))


if __name__ == "__main__":
    unittest.main()
