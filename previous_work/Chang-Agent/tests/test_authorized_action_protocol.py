import json
import unittest
from types import SimpleNamespace

from tools.run_levir_change_agent import _execute_verifier_action


class AuthorizedEnvironment:
    def __init__(self, evidence):
        self.calls = []
        self.verifier = SimpleNamespace(last_evidence=evidence)
        self.trajectory = SimpleNamespace(
            entries=[SimpleNamespace(execution={})]
        )

    def step(self, raw, **kwargs):
        self.calls.append((raw, kwargs))
        return "next_observation", False


class AuthorizedActionProtocolTest(unittest.TestCase):
    def test_staged_point_passes_region_id_outside_action_json(self):
        environment = AuthorizedEnvironment(
            {"stage_trace": {"plan": {"region_id": "r4"}}}
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="negative_point",
                error_region=(300, 400, 300, 400),
            )
        )

        returned, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=3
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        raw, kwargs = environment.calls[0]
        self.assertEqual(
            json.loads(raw),
            {
                "target_view": "t2",
                "action": "negative_point",
                "coordinate": [300, 400],
            },
        )
        self.assertNotIn("region", raw)
        self.assertEqual(kwargs, {"authorized_region_id": "r4"})

    def test_staged_box_passes_region_id_outside_action_json(self):
        environment = AuthorizedEnvironment(
            {"stage_trace": {"plan": {"region_id": "s3"}}}
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t1",
                suggested_action="box",
                error_region=(100, 200, 400, 500),
            )
        )

        _, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=5
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        raw, kwargs = environment.calls[0]
        self.assertEqual(
            json.loads(raw),
            {
                "target_view": "t1",
                "action": "box",
                "box": [100, 200, 400, 500],
            },
        )
        self.assertNotIn("region", raw)
        self.assertEqual(kwargs, {"authorized_region_id": "s3"})

    def test_replan_region_takes_precedence(self):
        environment = AuthorizedEnvironment(
            {
                "stage_trace": {"plan": {"region_id": "r0"}},
                "replan": {"selected_region_id": "r2"},
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t1",
                suggested_action="positive_point",
                error_region=(100, 200, 100, 200),
            )
        )

        _execute_verifier_action(environment, observation, loop_index=4)

        self.assertEqual(
            environment.calls[0][1], {"authorized_region_id": "r2"}
        )

    def test_live_geometry_recovers_region_for_legacy_hybrid_verifier(self):
        environment = AuthorizedEnvironment({})
        environment.state = SimpleNamespace(
            evidence={
                "verifier_region_proposals": [
                    {
                        "region_id": "r6",
                        "component_seed_normalized": [300, 400],
                        "box_normalized": [250, 350, 450, 550],
                    }
                ]
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="negative_point",
                error_region=(300, 400, 300, 400),
            )
        )

        _, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=6
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        self.assertEqual(
            environment.calls[0][1], {"authorized_region_id": "r6"}
        )


if __name__ == "__main__":
    unittest.main()
