import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np

from change_agent.adapters.stage_backends import BailianQwen3VLStageBackend
from change_agent.adapters.bailian_adapter import BailianGroundingModelQwen3VL
from change_agent.state import AgentObservation, ChangeState


class FakeResponse:
    def read(self):
        return json.dumps(
            {
                "id": "request-test",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "comparison": "initial",
                                        "quality_score": 0.5,
                                        "progress_score": 0.0,
                                        "accept": False,
                                        "stop": False,
                                        "feedback": "test",
                                    }
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
            }
        ).encode("utf-8")


class FakeOpener:
    def __init__(self):
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeResponse()


class SequenceOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request, timeout):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def http_error(status, detail):
    return urllib.error.HTTPError(
        "https://example.invalid/chat/completions",
        status,
        "provider error",
        {},
        io.BytesIO(detail.encode("utf-8")),
    )


class FakeChatClient:
    def __init__(self):
        self.messages = None

    def complete_json(self, messages, *, prompt, call_kind):
        self.messages = messages
        return {
            "target_view": "t2",
            "action": "positive_point",
            "coordinate": [500, 500],
        }


def make_state():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    return ChangeState(image, image, "building", mask, mask, mask)


class BailianBackendTest(unittest.TestCase):
    def test_stage_generation_preserves_legacy_complete_json_override_signature(self):
        class LegacySignatureBackend(BailianQwen3VLStageBackend):
            def __init__(self):
                super().__init__()
                self.call = None

            def complete_json(self, messages, *, prompt, call_kind):
                self.call = (messages, prompt, call_kind)
                return {"decision": {"feedback": "legacy override"}}

        backend = LegacySignatureBackend()
        result = backend.generate_stage(
            "decision", make_state(), {"mode": "initial"}
        )

        self.assertEqual(result["decision"]["feedback"], "legacy override")
        self.assertEqual(backend.call[2], "decision")

    def test_json_mode_request_uses_env_key_without_recording_it(self):
        opener = FakeOpener()
        backend = BailianQwen3VLStageBackend(
            base_url="https://example.invalid/compatible-mode/v1",
            opener=opener,
        )
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret-test-key"}):
            result = backend.generate_stage(
                "decision", make_state(), {"mode": "initial"}
            )

        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertEqual(
            body["response_format"], {"type": "json_object"}
        )
        self.assertEqual(body["model"], "qwen3-vl-plus")
        self.assertIs(body["enable_thinking"], False)
        self.assertNotIn("thinking_budget", body)
        self.assertEqual(
            opener.request.headers["Authorization"], "Bearer secret-test-key"
        )
        self.assertEqual(result["decision"]["comparison"], "initial")
        self.assertNotIn("secret-test-key", json.dumps(backend.last_call))
        self.assertEqual(backend.last_call["request_id"], "request-test")
        self.assertEqual(backend.last_call["attempt_kind"], "initial")
        self.assertEqual(backend.last_call["visual_input_scope"], "standard")

    def test_thinking_mode_uses_explicit_budget(self):
        opener = FakeOpener()
        backend = BailianQwen3VLStageBackend(
            base_url="https://example.invalid/compatible-mode/v1",
            enable_thinking=True,
            thinking_budget=256,
            opener=opener,
        )
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret-test-key"}):
            backend.generate_stage("decision", make_state(), {"mode": "initial"})

        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertIs(body["enable_thinking"], True)
        self.assertEqual(body["thinking_budget"], 256)

    def test_thinking_budget_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "thinking_budget must be positive"):
            BailianQwen3VLStageBackend(thinking_budget=0)

    def test_transient_http_errors_retry_with_bounded_backoff(self):
        opener = SequenceOpener(
            [
                http_error(429, '{"message":"rate limited"}'),
                http_error(503, '{"message":"temporarily unavailable"}'),
                FakeResponse(),
            ]
        )
        delays = []
        backend = BailianQwen3VLStageBackend(
            base_url="https://example.invalid/compatible-mode/v1",
            opener=opener,
            max_network_retries=2,
            retry_backoff_seconds=0.25,
            sleep=delays.append,
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test-secret"}):
            result = backend.generate_stage(
                "decision", make_state(), {"mode": "initial"}
            )

        self.assertEqual(result["decision"]["comparison"], "initial")
        self.assertEqual(opener.calls, 3)
        self.assertEqual(delays, [0.25, 0.5])
        self.assertEqual(
            [item.get("http_status") for item in backend.call_history[:2]],
            [429, 503],
        )
        self.assertTrue(
            all(item["retry_scheduled"] for item in backend.call_history[:2])
        )
        self.assertEqual(backend.last_call["network_attempt"], 3)

    def test_non_retryable_http_error_redacts_complete_credentials(self):
        key = "sk-parent-secret-value"
        leaked = "sk-provider-secret-value"
        opener = SequenceOpener(
            [
                http_error(
                    401,
                    json.dumps(
                        {
                            "message": f"Bearer {key}",
                            "api_key": leaked,
                        }
                    ),
                )
            ]
        )
        backend = BailianQwen3VLStageBackend(
            base_url="https://example.invalid/compatible-mode/v1",
            opener=opener,
            max_network_retries=2,
            sleep=lambda _delay: None,
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": key}):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                backend.generate_stage(
                    "decision", make_state(), {"mode": "initial"}
                )

        serialized = json.dumps(backend.call_history)
        self.assertEqual(opener.calls, 1)
        self.assertNotIn(key, serialized)
        self.assertNotIn(leaked, serialized)
        self.assertIn("[redacted]", serialized)
        self.assertFalse(backend.last_call["retry_scheduled"])

    def test_missing_api_key_fails_before_network(self):
        backend = BailianQwen3VLStageBackend(
            base_url="https://example.invalid/compatible-mode/v1",
            opener=FakeOpener(),
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                backend.generate_stage(
                    "decision", make_state(), {"mode": "initial"}
                )

    def test_hosted_agent_uses_the_same_public_action_protocol(self):
        client = FakeChatClient()
        agent = BailianGroundingModelQwen3VL(client=client)
        state = make_state()
        observation = AgentObservation(
            state.t1_image,
            state.t2_image,
            state.query,
            state.change_mask,
            t1_mask=state.t1_mask,
            t2_mask=state.t2_mask,
        )

        raw, action = agent.act(observation)

        self.assertIn('"coordinate":[500,500]', raw)
        self.assertEqual(action.target_view, "t2")
        self.assertEqual(action.action, "positive_point")
        self.assertEqual(action.coordinate, (2, 2))
        image_items = [
            item
            for message in client.messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "image_url"
        ]
        self.assertEqual(len(image_items), 5)


if __name__ == "__main__":
    unittest.main()
