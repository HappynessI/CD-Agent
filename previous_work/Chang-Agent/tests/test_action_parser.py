import unittest

from change_agent.action_parser import ActionParser, ActionValidationError
from change_agent.coordinates import pixel_box_to_normalized, pixel_point_to_normalized
from change_agent.executor import xyxy_to_normalized_cxcywh


class ActionParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = ActionParser()

    def test_parses_json_fence_and_converts_xy(self):
        action = self.parser.parse(
            '```json\n{"target_view":"t2","action":"positive_point",'
            '"coordinate":[1000,500],"coordinate_frame":"normalized_1000_xy"}\n```',
            (101, 51),
        )
        self.assertEqual(action.coordinate, (100, 25))
        self.assertEqual(action.target_view, "t2")

    def test_parses_box(self):
        action = self.parser.parse(
            '{"target_view":"t1","action":"box","box":[100,200,900,800]}',
            (101, 101),
        )
        self.assertEqual(action.box, (10, 20, 90, 80))

    def test_public_normalized_coordinates_round_trip_to_internal_pixels(self):
        image_size = (256, 256)
        normalized = pixel_point_to_normalized((52, 250), image_size)
        action = self.parser.parse_payload(
            {
                "target_view": "t2",
                "action": "positive_point",
                "coordinate": normalized,
            },
            image_size,
        )
        self.assertEqual(action.coordinate, (52, 250))
        self.assertEqual(
            pixel_box_to_normalized((0, 37, 145, 255), image_size),
            (0, 145, 569, 1000),
        )

    def test_rejects_invalid_payloads(self):
        bad = [
            '{"target_view":"t3","action":"finish"}',
            '{"target_view":"t1","action":"positive_point","coordinate":[-1,4]}',
            '{"target_view":"t1","action":"box","box":[500,0,400,1000],'
            '"coordinate_frame":"normalized_1000_xy"}',
            '{"target_view":"t1","action":"positive_point","coordinate":[1001,2]}',
            '{"target_view":"t1","action":"positive_point","coordinate":[1,2],'
            '"coordinate_frame":"pixel_xy"}',
            '{"target_view":"t1","action":"finish","box":[0,0,1,1]}',
            '{"target_view":"t1","action":"finish","surprise":1}',
            "not json",
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ActionValidationError):
                self.parser.parse(raw, (100, 100))

    def test_system_coordinate_protocol_needs_no_model_field(self):
        point = self.parser.parse(
            '{"target_view":"t2","action":"positive_point","coordinate":[500,500]}',
            (256, 256),
        )
        box = self.parser.parse(
            '{"target_view":"t1","action":"box","box":[0,0,1000,1000]}',
            (256, 256),
        )
        self.assertEqual(point.coordinate, (128, 128))
        self.assertEqual(box.box, (0, 0, 255, 255))

    def test_box_conversion_for_sam3(self):
        result = xyxy_to_normalized_cxcywh((10, 20, 30, 60), (100, 100))
        self.assertEqual(result, (0.2, 0.4, 0.2, 0.4))


if __name__ == "__main__":
    unittest.main()
