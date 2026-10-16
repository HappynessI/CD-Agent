"""Strict parsing and coordinate conversion for model-produced JSON actions."""

from __future__ import annotations

import json
import math
from typing import Any

from .coordinates import PROTOCOL_COORDINATE_MAX, normalized_point_to_pixel
from .state import AgentAction


class ActionValidationError(ValueError):
    """Raised when an Agent response must not reach a segmentation tool."""


class ActionParser:
    ACTIONS = {"positive_point", "negative_point", "box", "finish"}
    VIEWS = {"t1", "t2"}
    ALLOWED_KEYS = {"target_view", "action", "coordinate", "coordinate_frame", "box"}
    # The coordinate protocol is owned by the runtime. This value is accepted only
    # for compatibility with action payloads produced before the field was removed
    # from the public model schema.
    COORDINATE_FRAME = "normalized_1000_xy"

    def __init__(self, coordinate_max: float = float(PROTOCOL_COORDINATE_MAX)):
        if coordinate_max <= 0:
            raise ValueError("coordinate_max must be positive")
        self.coordinate_max = float(coordinate_max)

    def parse(self, raw_response: str, image_size: tuple[int, int]) -> AgentAction:
        payload = self.extract_payload(raw_response)
        return self.parse_payload(payload, image_size)

    def extract_payload(self, raw_response: str) -> dict[str, Any]:
        """Extract the raw public action payload for auditable trajectory logging."""

        return self._extract_json_object(raw_response)

    def parse_payload(
        self, payload: dict[str, Any], image_size: tuple[int, int]
    ) -> AgentAction:
        if not isinstance(payload, dict):
            raise ActionValidationError("action payload must be a JSON object")
        unknown = set(payload) - self.ALLOWED_KEYS
        if unknown:
            raise ActionValidationError(f"unknown action fields: {sorted(unknown)}")

        target_view = payload.get("target_view")
        action = payload.get("action")
        if target_view not in self.VIEWS:
            raise ActionValidationError("target_view must be 't1' or 't2'")
        if action not in self.ACTIONS:
            raise ActionValidationError(f"unsupported action: {action!r}")

        width, height = self._validate_size(image_size)
        if action in {"positive_point", "negative_point"}:
            if "box" in payload:
                raise ActionValidationError("point action must not include box")
            self._validate_legacy_coordinate_frame(payload)
            coordinate = self._number_list(payload.get("coordinate"), 2, "coordinate")
            x, y = self._point_to_pixels(coordinate, width, height)
            return AgentAction(target_view, action, coordinate=(x, y))

        if action == "box":
            if "coordinate" in payload:
                raise ActionValidationError("box action must not include coordinate")
            self._validate_legacy_coordinate_frame(payload)
            box = self._number_list(payload.get("box"), 4, "box")
            x1, y1 = self._point_to_pixels(box[:2], width, height)
            x2, y2 = self._point_to_pixels(box[2:], width, height)
            if x1 >= x2 or y1 >= y2:
                raise ActionValidationError("box must satisfy x1 < x2 and y1 < y2")
            return AgentAction(target_view, action, box=(x1, y1, x2, y2))

        if "coordinate" in payload or "box" in payload or "coordinate_frame" in payload:
            raise ActionValidationError("finish action must not include coordinate or box")
        return AgentAction(target_view, "finish")

    def _validate_legacy_coordinate_frame(self, payload: dict[str, Any]) -> None:
        """Accept the retired field without making the model declare configuration."""

        if (
            "coordinate_frame" in payload
            and payload["coordinate_frame"] != self.COORDINATE_FRAME
        ):
            raise ActionValidationError(
                "coordinate_frame is system-defined; if supplied by a legacy client, "
                f"it must equal {self.COORDINATE_FRAME!r}"
            )

    def validate_pixel_action(
        self, action: AgentAction, image_size: tuple[int, int]
    ) -> AgentAction:
        """Validate programmatic actions at the same Environment trust boundary."""

        width, height = self._validate_size(image_size)
        if action.target_view not in self.VIEWS or action.action not in self.ACTIONS:
            raise ActionValidationError("invalid target_view or action")
        if action.action in {"positive_point", "negative_point"}:
            if action.coordinate is None or action.box is not None:
                raise ActionValidationError("point action requires only coordinate")
            x, y = action.coordinate
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
                raise ActionValidationError("pixel coordinate values must be integers")
            if not (0 <= x < width and 0 <= y < height):
                raise ActionValidationError("pixel coordinate is outside the target image")
        elif action.action == "box":
            if action.box is None or action.coordinate is not None:
                raise ActionValidationError("box action requires only box")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in action.box):
                raise ActionValidationError("pixel box values must be integers")
            x1, y1, x2, y2 = action.box
            if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                raise ActionValidationError("pixel box must be ordered and inside the image")
        elif action.coordinate is not None or action.box is not None:
            raise ActionValidationError("finish action must not include coordinate or box")
        return action

    @staticmethod
    def _extract_json_object(raw_response: str) -> dict[str, Any]:
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ActionValidationError("model response is empty")
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw_response):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw_response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ActionValidationError("model response does not contain a valid JSON object")

    @staticmethod
    def _validate_size(image_size: tuple[int, int]) -> tuple[int, int]:
        if len(image_size) != 2:
            raise ActionValidationError("image_size must be (width, height)")
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ActionValidationError("image dimensions must be positive")
        return int(width), int(height)

    @staticmethod
    def _number_list(value: Any, length: int, field: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ActionValidationError(f"{field} must contain exactly {length} numbers")
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ActionValidationError(f"{field} values must be numbers")
            item = float(item)
            if not math.isfinite(item):
                raise ActionValidationError(f"{field} values must be finite")
            result.append(item)
        return result

    def _point_to_pixels(
        self, coordinate: list[float], width: int, height: int
    ) -> tuple[int, int]:
        x, y = coordinate
        if not (0 <= x <= self.coordinate_max and 0 <= y <= self.coordinate_max):
            raise ActionValidationError(
                f"coordinates must be normalized to [0, {self.coordinate_max:g}]"
            )
        try:
            return normalized_point_to_pixel(
                coordinate, (width, height), coordinate_max=self.coordinate_max
            )
        except ValueError as error:
            raise ActionValidationError(str(error)) from error
