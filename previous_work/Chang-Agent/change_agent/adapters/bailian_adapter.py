"""BaiLian Qwen3-VL agent adapter sharing the local public action protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..action_parser import ActionParser
from ..state import AgentAction
from ..state import AgentObservation
from .qwen3vl_adapter import GroundingModelQwen3VL
from .stage_backends import BailianQwen3VLStageBackend, _image_data_url, _as_image, _mask_image


class BailianGroundingModelQwen3VL:
    """Hosted drop-in replacement for ``GroundingModelQwen3VL.generate_raw``."""

    def __init__(
        self,
        *,
        client: BailianQwen3VLStageBackend | None = None,
        model: str = "qwen3-vl-plus",
        base_url: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        max_completion_tokens: int = 256,
        seed: int = 42,
        action_parser: ActionParser | None = None,
    ):
        self.client = client or BailianQwen3VLStageBackend(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            max_completion_tokens=max_completion_tokens,
            seed=seed,
        )
        self.model_path = model
        self.action_parser = action_parser or ActionParser()
        self.last_prompt_hash: str | None = None

    def generate_raw(
        self,
        observation: AgentObservation,
        validation_error: str | None = None,
        previous_raw: str | None = None,
    ) -> str:
        prompt = GroundingModelQwen3VL._instruction(
            observation, validation_error, previous_raw
        )
        # JSON mode requires the prompt to explicitly request JSON; the shared
        # action instruction already does so, and this prefix makes it unambiguous.
        prompt = "Return valid JSON only.\n" + prompt
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "T1 earlier RGB image"},
            {"type": "image_url", "image_url": {"url": _image_data_url(_as_image(observation.t1_image))}},
            {"type": "text", "text": "T2 later RGB image"},
            {"type": "image_url", "image_url": {"url": _image_data_url(_as_image(observation.t2_image))}},
        ]
        if observation.t1_mask is not None and observation.t2_mask is not None:
            content.extend(
                [
                    {"type": "text", "text": "Current predicted T1 object mask"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(_mask_image(observation.t1_mask))}},
                    {"type": "text", "text": "Current predicted T2 object mask"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(_mask_image(observation.t2_mask))}},
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Current final change mask"},
                {"type": "image_url", "image_url": {"url": _image_data_url(_mask_image(observation.change_mask))}},
                {"type": "text", "text": prompt},
            ]
        )
        self.last_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        result = self.client.complete_json(
            [
                {"role": "system", "content": "You are a strict change-detection action planner. Return JSON only."},
                {"role": "user", "content": content},
            ],
            prompt=prompt,
            call_kind="agent_action",
        )
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    def act(self, observation: AgentObservation) -> tuple[str, AgentAction]:
        raw = self.generate_raw(observation)
        height, width = observation.t1_image.shape[:2]
        return raw, self.action_parser.parse(raw, (width, height))
