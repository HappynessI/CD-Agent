import unittest

import numpy as np

from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction


class PointBackend:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        self.calls.append(np.array(initial_mask, dtype=bool, copy=True))
        return self.output.copy()


class BoxBackend:
    def __init__(self, output):
        self.output = output

    def segment_box(self, image, box_cxcywh_normalized, query):
        return self.output.copy()


class ExecutorLocalityTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_positive_point_merges_only_clicked_prediction_component(self):
        initial = np.zeros((16, 16), dtype=bool)
        prediction = np.zeros_like(initial)
        prediction[5:7, 5:7] = True
        prediction[12:14, 12:14] = True
        executor = ActionExecutor(PointBackend(prediction), BoxBackend(prediction))

        result = executor.execute(
            AgentAction("t2", "positive_point", coordinate=(5, 5)),
            self.image,
            initial,
            "building",
        )

        self.assertTrue(result.mask[5:7, 5:7].all())
        self.assertFalse(result.mask[12:14, 12:14].any())
        self.assertEqual(
            result.evidence["locality"]["composition_mode"],
            "merge_clicked_prediction_component",
        )

    def test_negative_point_uses_only_simpleclick_removal_inside_roi(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[2:10, 2:10] = True
        initial[10:12, 10:12] = True
        prediction = initial.copy()
        prediction[3:5, 3:5] = False
        prediction[8:10, 8:10] = False
        executor = ActionExecutor(PointBackend(prediction), BoxBackend(initial))

        result = executor.execute(
            AgentAction("t1", "negative_point", coordinate=(3, 3)),
            self.image,
            initial,
            "building",
        )

        self.assertFalse(result.mask[3:5, 3:5].any())
        self.assertTrue(result.mask[8:10, 8:10].all())
        self.assertTrue(result.mask[10:12, 10:12].all())
        self.assertEqual(result.evidence["locality"]["outside_roi_pixels"], 0)
        self.assertEqual(
            result.evidence["locality"]["composition_mode"],
            "remove_simpleclick_delta_within_roi",
        )
        self.assertEqual(result.evidence["tool_input"]["composed_removed_pixels"], 4)

    def test_box_replaces_only_pixels_inside_box(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[1:3, 1:3] = True
        initial[5:9, 5:9] = True
        prediction = np.zeros_like(initial)
        executor = ActionExecutor(PointBackend(prediction), BoxBackend(prediction))

        result = executor.execute(
            AgentAction("t2", "box", box=(5, 5, 8, 8)),
            self.image,
            initial,
            "building",
        )

        self.assertTrue(result.mask[1:3, 1:3].all())
        self.assertFalse(result.mask[5:9, 5:9].any())
        self.assertEqual(result.evidence["locality"]["outside_roi_pixels"], 0)
        self.assertEqual(
            result.evidence["locality"]["composition_mode"], "replace_box_roi_only"
        )

    def test_proposal_point_localizes_session_and_positive_write_scope(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[1:3, 1:3] = True
        initial[5, 5] = True
        prediction = np.zeros_like(initial)
        prediction[4:10, 4:10] = True
        point = PointBackend(prediction)
        executor = ActionExecutor(
            point,
            BoxBackend(prediction),
            point_roi_min_padding=1,
        )

        result = executor.execute(
            AgentAction("t2", "positive_point", coordinate=(5, 5)),
            self.image,
            initial,
            "building",
            point_session_mask=initial,
            point_region_id="r2",
            point_proposal_box=(4, 4, 7, 7),
            point_proposal_component_area=16,
        )

        self.assertFalse(point.calls[0][1:3, 1:3].any())
        self.assertTrue(point.calls[0][5, 5])
        self.assertTrue(result.mask[4:8, 4:8].all())
        self.assertFalse(result.mask[8:10, 8:10].any())
        self.assertTrue(result.mask[1:3, 1:3].all())
        self.assertEqual(
            result.evidence["locality"]["composition_mode"],
            "merge_clicked_prediction_component_within_proposal_roi",
        )
        self.assertEqual(
            result.evidence["point_authorization"],
            {
                "region_id": "r2",
                "proposal_box_xyxy": [4, 4, 7, 7],
                "proposal_component_area": 16,
                "proposal_change_ratio": 15 / 16,
                "effective_roi_xyxy": [4, 4, 7, 7],
                "model_input_scope_policy": "proposal_roi_mask",
                "write_scope_policy": (
                    "clicked_prediction_component_within_proposal_roi"
                ),
                "scope_policy": "environment_proposal_local",
            },
        )
        self.assertEqual(result.evidence["locality"]["nonlocal_changed_pixels"], 0)

    def test_proposal_negative_is_subtraction_only_inside_adaptive_roi(self):
        initial = np.ones((16, 16), dtype=bool)
        prediction = np.zeros_like(initial)
        executor = ActionExecutor(
            PointBackend(prediction),
            BoxBackend(prediction),
            point_roi_min_padding=1,
            authorized_negative_point_input_scope="proposal_roi",
        )

        result = executor.execute(
            AgentAction("t1", "negative_point", coordinate=(6, 6)),
            self.image,
            initial,
            "building",
            point_region_id="r0",
            point_proposal_box=(5, 5, 8, 8),
            point_proposal_component_area=64,
        )

        self.assertFalse(result.mask[5:9, 5:9].any())
        outside = np.array(result.mask, copy=True)
        outside[5:9, 5:9] = True
        self.assertTrue(outside.all())
        self.assertEqual(result.evidence["tool_input"]["session_initial_mask_pixels"], 16)
        self.assertEqual(result.evidence["tool_input"]["session_source_mask_pixels"], 256)
        self.assertEqual(
            result.evidence["locality"]["composition_mode"],
            "remove_simpleclick_delta_within_clicked_component",
        )
        self.assertEqual(
            result.evidence["point_authorization"]["proposal_change_ratio"],
            0.25,
        )

    def test_full_session_input_and_component_write_are_decoupled(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[2:5, 2:5] = True
        initial[8:11, 8:11] = True
        initial[14, 14] = True
        prediction = np.zeros_like(initial)
        point = PointBackend(prediction)
        executor = ActionExecutor(
            point,
            BoxBackend(prediction),
            point_roi_min_padding=1,
        )

        result = executor.execute(
            AgentAction("t2", "negative_point", coordinate=(3, 3)),
            self.image,
            initial,
            "building",
            point_region_id="r0",
            point_proposal_box=(1, 1, 11, 11),
            point_proposal_component_area=9,
        )

        self.assertEqual(int(point.calls[0].sum()), 19)
        self.assertTrue(point.calls[0][2:5, 2:5].all())
        self.assertTrue(point.calls[0][8:11, 8:11].all())
        self.assertTrue(point.calls[0][14, 14])
        self.assertFalse(result.mask[2:5, 2:5].any())
        self.assertTrue(result.mask[8:11, 8:11].all())
        self.assertTrue(result.mask[14, 14])
        tool_input = result.evidence["tool_input"]
        self.assertEqual(tool_input["proposal_roi_mask_pixels"], 18)
        self.assertEqual(tool_input["session_initial_mask_pixels"], 19)
        self.assertEqual(tool_input["authorized_write_mask_pixels"], 9)
        self.assertEqual(tool_input["discarded_proposal_mask_pixels"], 0)
        self.assertEqual(tool_input["model_input_outside_proposal_pixels"], 1)
        self.assertEqual(
            tool_input["session_localization_policy"],
            "full_session_mask",
        )
        self.assertEqual(
            result.evidence["point_authorization"]["scope_policy"],
            "environment_proposal_clicked_component",
        )
        self.assertEqual(
            result.evidence["point_authorization"]["write_scope_policy"],
            "clicked_component_within_proposal_roi",
        )

    def test_component_input_mode_remains_available_for_replay(self):
        initial = np.zeros((16, 16), dtype=bool)
        initial[2:5, 2:5] = True
        initial[8:11, 8:11] = True
        prediction = np.zeros_like(initial)
        point = PointBackend(prediction)
        executor = ActionExecutor(
            point,
            BoxBackend(prediction),
            point_roi_min_padding=1,
            authorized_negative_point_input_scope="clicked_component",
        )

        executor.execute(
            AgentAction("t2", "negative_point", coordinate=(3, 3)),
            self.image,
            initial,
            "building",
            point_region_id="r0",
            point_proposal_box=(1, 1, 11, 11),
            point_proposal_component_area=9,
        )

        self.assertEqual(int(point.calls[0].sum()), 9)


if __name__ == "__main__":
    unittest.main()
