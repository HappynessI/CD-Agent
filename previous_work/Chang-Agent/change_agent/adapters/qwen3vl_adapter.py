"""Qwen3-VL Agent adapter using the modern multimodal chat-template API."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from PIL import Image

from ..action_parser import ActionParser
from ..state import AgentAction, AgentObservation


class GroundingModelQwen3VL:
    DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        max_new_tokens: int = 160,
        device_map: str | None = "auto",
        dtype: str = "auto",
        model: Any | None = None,
        processor: Any | None = None,
        action_parser: ActionParser | None = None,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.action_parser = action_parser or ActionParser()
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        if model is None:
            try:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except (ImportError, AttributeError) as error:
                raise RuntimeError(
                    "Qwen3-VL requires the isolated qwen3vl dependencies "
                    "(transformers>=4.57.0 and accelerate)."
                ) from error
            load_kwargs: dict[str, Any] = {"dtype": dtype}
            # Transformers accepts ``device_map='auto'`` for CUDA placement, while
            # a CPU-only smoke should use the ordinary single-process loader.
            if device_map != "cpu":
                load_kwargs["device_map"] = device_map
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            ).eval()
            processor = AutoProcessor.from_pretrained(model_path)
        self.model = model
        self.processor = processor
        self.last_prompt_hash: str | None = None

    def build_messages(
        self,
        observation: AgentObservation,
        validation_error: str | None = None,
        previous_raw: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._instruction(observation, validation_error, previous_raw)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "T1 image (earlier time):"},
            {"type": "image", "image": self._as_image(observation.t1_image)},
            {"type": "text", "text": "T2 image (later time):"},
            {"type": "image", "image": self._as_image(observation.t2_image)},
        ]
        if observation.t1_mask is not None and observation.t2_mask is not None:
            content.extend(
                [
                    {"type": "text", "text": "Current predicted T1 object mask:"},
                    {"type": "image", "image": self._mask_image(observation.t1_mask)},
                    {"type": "text", "text": "Current predicted T2 object mask:"},
                    {"type": "image", "image": self._mask_image(observation.t2_mask)},
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Current binary change mask:"},
                {"type": "image", "image": self._mask_image(observation.change_mask)},
                {"type": "text", "text": prompt},
            ]
        )
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def generate_raw(
        self,
        observation: AgentObservation,
        validation_error: str | None = None,
        previous_raw: str | None = None,
    ) -> str:
        messages = self.build_messages(observation, validation_error, previous_raw)
        prompt_text = "\n".join(
            item["text"]
            for message in messages
            for item in message["content"]
            if item["type"] == "text"
        )
        self.last_prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated = outputs[:, input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def act(self, observation: AgentObservation) -> tuple[str, AgentAction]:
        raw = self.generate_raw(observation)
        height, width = observation.t1_image.shape[:2]
        return raw, self.action_parser.parse(raw, (width, height))

    @staticmethod
    def _as_image(value: Any) -> Any:
        if isinstance(value, Image.Image) or isinstance(value, str):
            return value
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _mask_image(mask: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")

    @staticmethod
    def _instruction(
        observation: AgentObservation,
        validation_error: str | None = None,
        previous_raw: str | None = None,
    ) -> str:
        feedback = observation.feedback.to_dict() if observation.feedback else None
        history = observation.history_summary or "none"
        has_tool_action = any(
            f"action={name}" in history
            for name in ("positive_point", "negative_point", "box")
        )
        verifier_invalid = observation.feedback is not None and not observation.feedback.verifier_valid
        initial_finish_authorized = bool(
            observation.feedback is not None
            and observation.feedback.verifier_valid
            and observation.feedback.comparison == "initial"
            and observation.feedback.error_type == "none"
            and observation.feedback.stop
        )
        finish_rule = (
            "Verifier feedback is invalid and cannot authorize finish. Do not finish from "
            "this feedback; recheck the visual evidence and choose a segmentation tool action."
            if verifier_invalid
            else
            "At least one segmentation tool action has already run; finish is allowed only "
            "if the current mask is credible."
            if has_tool_action
            else "The initial Verifier found no actionable error; finish is allowed without "
            "a redundant tool action."
            if initial_finish_authorized
            else "No segmentation tool action has run yet. Finish is forbidden: choose a "
            "positive_point, negative_point, or box action now."
        )
        correction = GroundingModelQwen3VL._retry_instruction(
            validation_error, previous_raw
        )
        format_example = GroundingModelQwen3VL._recommended_action_example(
            observation,
            finish_allowed=(has_tool_action or initial_finish_authorized)
            and not verifier_invalid,
        )
        return (
            "You refine a change-detection result. The inputs above are T1, T2, the current "
            "predicted T1/T2 object masks, and the current change mask. The object masks are "
            "model predictions, not GT; inspect them because your action edits one of them. "
            "Do not invent a final mask. Select one "
            "tool action. All public coordinates, including Verifier error_region and your "
            "output, use normalized [0,1000] XY order; they are not image pixels. For a "
            "256x256 image, pixel center (128,128) is approximately (502,502). Return exactly one "
            "JSON object with target_view ('t1' or 't2') and action "
            "('positive_point', 'negative_point', 'box', or 'finish'). The coordinate protocol "
            "is system-defined. Never output coordinate_frame or other configuration fields. "
            "History entries with accepted=false may include rejected_action JSON. Never repeat "
            "that exact target_view/action/coordinate or box while the live mask is unchanged; "
            "choose a different unresolved region or different tool geometry. "
            "Return one JSON object and no explanation.\n"
            f"{correction}"
            f"{format_example}"
            f"{finish_rule}\n"
            f"Query: {observation.query}\n"
            f"Verifier feedback: {json.dumps(feedback, ensure_ascii=False)}\n"
            f"History summary: {history}"
        )

    @staticmethod
    def _recommended_action_example(
        observation: AgentObservation, *, finish_allowed: bool
    ) -> str:
        feedback = observation.feedback
        if feedback is None or not feedback.verifier_valid:
            return (
                "Mandatory syntax: point actions require coordinate:[x,y]; box requires "
                "box:[x1,y1,x2,y2]; finish contains neither coordinate nor box.\n"
            )
        action = feedback.suggested_action
        target_view = feedback.target_view
        if action in {"positive_point", "negative_point"}:
            if feedback.error_region is not None:
                x1, y1, x2, y2 = feedback.error_region
                coordinate = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
            else:
                coordinate = [620, 410]
            example = json.dumps(
                {
                    "target_view": target_view,
                    "action": action,
                    "coordinate": coordinate,
                },
                separators=(",", ":"),
            )
            return (
                "The Verifier recommends a point action. Your output must follow this "
                f"structure:\n{example}\nThe coordinate is the Environment-owned component "
                "anchor inside error_region; copy it exactly so the point hits the audited "
                "component. Never omit or move coordinate.\n"
            )
        if action == "box":
            example = json.dumps(
                {
                    "target_view": target_view,
                    "action": "box",
                    "box": [120, 180, 760, 820],
                },
                separators=(",", ":"),
            )
            return (
                "The Verifier recommends a box action. Your output must follow this "
                f"structure:\n{example}\nChoose the actual box from the visual evidence and "
                "error_region. Never omit box.\n"
            )
        if action == "finish" and finish_allowed:
            example = json.dumps(
                {"target_view": target_view, "action": "finish"},
                separators=(",", ":"),
            )
            return (
                "The Verifier recommends finish. Your output must follow this structure:\n"
                f"{example}\nFinish must contain neither coordinate nor box.\n"
            )
        return (
            "A tool action is required now. Point actions must contain coordinate:[x,y]; "
            "box must contain box:[x1,y1,x2,y2]. Never omit the required parameter.\n"
        )

    @staticmethod
    def _retry_instruction(
        validation_error: str | None, previous_raw: str | None
    ) -> str:
        if not validation_error:
            return ""
        payload = GroundingModelQwen3VL._extract_json_payload(previous_raw)
        target_view = payload.get("target_view")
        action = payload.get("action")
        if "exactly repeats a previously rejected action" in validation_error:
            forbidden = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return (
                f"The exact previous action is forbidden on the unchanged live state: "
                f"{forbidden}. Do not reproduce it. Your next JSON must change the action "
                "type or its coordinate/box and target a different unresolved region. "
                "Return one complete JSON object with all fields required by the new action.\n"
            )
        if target_view in {"t1", "t2"} and action in {
            "positive_point",
            "negative_point",
        } and "coordinate" in validation_error:
            reason = (
                "omitted coordinate"
                if "coordinate" not in payload
                else "used an invalid coordinate"
            )
            return (
                f"Your previous point action {reason}. Return exactly:\n"
                f'{{"target_view":"{target_view}","action":"{action}",'
                '"coordinate":[x,y]}\n'
                "Replace x and y with numeric values in [0,1000]. Never omit coordinate.\n"
            )
        if target_view in {"t1", "t2"} and action == "box" and "box" in validation_error:
            reason = "omitted box" if "box" not in payload else "used an invalid box"
            return (
                f"Your previous box action {reason}. Return exactly:\n"
                f'{{"target_view":"{target_view}","action":"box",'
                '"box":[x1,y1,x2,y2]}\n'
                "Replace x1,y1,x2,y2 with ordered numeric values in [0,1000]. "
                "Never omit box.\n"
            )
        return (
            f"Your previous action was rejected: {validation_error}. Return one corrected "
            "JSON object with every mandatory field and no explanation.\n"
        )

    @staticmethod
    def _extract_json_payload(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {}
