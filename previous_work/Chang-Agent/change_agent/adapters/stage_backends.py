"""Local and hosted model backends for the staged verifier protocol."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..coordinates import normalized_point_to_pixel
from ..state import ChangeState
from ..terminal_review import (
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    SELECTION_POLICY_VERSION,
)
from ..verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION
from ..verifier_protocol import (
    StageName,
    StageProtocolError,
    is_atomic_evidence_grounding_error,
)
from .omniovcd_adapter import connected_components


_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429})

_PLAN_NULL_PROMPT_VERSION_BY_BRANCH = {
    "next_unreviewed_environment_region": (
        PLAN_NULL_UNREVIEWED_PROMPT_VERSION
    ),
    "first_unresolved_environment_region": (
        PLAN_NULL_ALTERNATIVE_PROMPT_VERSION
    ),
}


class _CompleteStageJsonLogitsProcessor:
    """Disallow local EOS until generated tokens form a complete stage JSON.

    This guard does not repair, clean, or reinterpret model output. It only
    prevents configured end-of-sequence tokens while the generated suffix is
    not yet a complete JSON object for the active stage. The existing parser
    and fail-closed semantic validation remain authoritative.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_length: int,
        stage: str,
        eos_token_ids: tuple[int, ...],
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stage = stage
        self.eos_token_ids = eos_token_ids

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        vocabulary_size = int(scores.shape[-1])
        eos_ids = tuple(
            token_id
            for token_id in self.eos_token_ids
            if 0 <= token_id < vocabulary_size
        )
        if not eos_ids:
            return scores
        for row_index in range(int(input_ids.shape[0])):
            raw = self.tokenizer.decode(
                input_ids[row_index, self.prompt_length :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            try:
                _extract_stage_json(raw, self.stage)
                complete = True
            except StageProtocolError:
                complete = False
            if complete:
                scores[row_index, :] = -float("inf")
                scores[row_index, list(eos_ids)] = 0.0
            else:
                scores[row_index, list(eos_ids)] = -float("inf")
        return scores


def _generation_eos_token_ids(model: Any, processor: Any) -> tuple[int, ...]:
    """Return configured local generation terminators without duplicates."""

    configured = getattr(
        getattr(model, "generation_config", None), "eos_token_id", None
    )
    raw_ids = list(configured) if isinstance(configured, (list, tuple)) else [configured]
    raw_ids.append(getattr(getattr(processor, "tokenizer", None), "eos_token_id", None))
    return tuple(dict.fromkeys(int(item) for item in raw_ids if item is not None))


def _redact_provider_error(detail: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Return a bounded provider diagnostic with credential-shaped values removed."""

    redacted = str(detail)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(
        r"(?i)\bBearer\s+[^\s\"',}\]]+",
        "Bearer [redacted]",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9._-]+\b", "sk-[redacted]", redacted)
    redacted = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|access[_-]?token)["\']?\s*[:=]\s*)'
        r'(["\']?)[^"\'\s,}\]]+\2',
        r"\1[redacted]",
        redacted,
    )
    return " ".join(redacted.split())[:600]


def _retryable_http_status(status: int) -> bool:
    return status in _RETRYABLE_HTTP_STATUSES or 500 <= status <= 599


class LocalQwen3VLStageBackend:
    """Use already-loaded Transformers Qwen weights for staged JSON calls."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.last_call: dict[str, Any] = {}
        self.call_history: list[dict[str, Any]] = []

    def reset_audit(self) -> None:
        self.last_call = {}
        self.call_history = []

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        return self._complete_stage(stage, state, payload, previous_state, None)

    def repair_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        validation_error: str,
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        """Retry one stage with the exact validation failure in the prompt."""

        return self._complete_stage(
            stage, state, payload, previous_state, validation_error
        )

    def _complete_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None,
        validation_error: str | None,
    ) -> Mapping[str, Any]:
        messages = _local_messages(
            stage, state, payload, previous_state, validation_error
        )
        prompt = _stage_prompt(stage, payload, validation_error)
        started = time.monotonic()
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
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            logits_processor=[
                _CompleteStageJsonLogitsProcessor(
                    tokenizer=self.processor.tokenizer,
                    prompt_length=int(input_ids.shape[1]),
                    stage=stage,
                    eos_token_ids=_generation_eos_token_ids(
                        self.model, self.processor
                    ),
                )
            ],
        )
        generated = outputs[:, input_ids.shape[1] :]
        raw = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        audit = {
            "backend": "local_transformers",
            "stage": stage,
            "latency_seconds": round(time.monotonic() - started, 6),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "attempt_kind": "repair" if validation_error else "initial",
            # P1 accounting must be reconstructed from the raw backend audit;
            # it may not infer a zero merely because this is the local backend.
            "network_retry_attempt_count": 0,
            "visual_input_scope": _stage_visual_input_scope(
                stage,
                payload,
                validation_error,
            ),
            "json_completion_guard": True,
            "raw_response": _audit_text(raw),
        }
        try:
            result = _extract_stage_json(raw, stage)
        except StageProtocolError as error:
            audit["parse_error"] = str(error)
            self.last_call = audit
            self.call_history.append(audit)
            raise
        audit["parsed_output"] = result
        self.last_call = audit
        self.call_history.append(audit)
        return result


class BailianQwen3VLStageBackend:
    """Minimal OpenAI-compatible BaiLian client with no credential logging.

    The API key is read only from ``api_key_env``.  ``base_url`` may be a full
    ``.../chat/completions`` URL or an OpenAI-compatible ``.../v1`` base URL.
    Network calls are made only when ``generate_stage`` is explicitly invoked.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        model: str = "qwen3-vl-plus",
        base_url: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_seconds: float = 120.0,
        max_completion_tokens: int = 512,
        seed: int = 42,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        max_network_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        opener: Any | None = None,
        sleep: Any = time.sleep,
    ):
        if thinking_budget is not None and thinking_budget <= 0:
            raise ValueError("thinking_budget must be positive when provided")
        if max_network_retries < 0:
            raise ValueError("max_network_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.model = model
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.max_network_retries = max_network_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleep
        self.last_call: dict[str, Any] = {}
        self.call_history: list[dict[str, Any]] = []
        self._stage_audit_metadata: tuple[str | None, str | None] = (None, None)

    def reset_audit(self) -> None:
        self.last_call = {}
        self.call_history = []

    @property
    def endpoint(self) -> str:
        return (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions"
        )

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        messages = _hosted_messages(stage, state, payload, previous_state, None)
        prompt = _stage_prompt(stage, payload, None)
        return self._complete_stage_json(
            messages,
            prompt=prompt,
            call_kind=stage,
            attempt_kind="initial",
            visual_input_scope=_stage_visual_input_scope(stage, payload, None),
        )

    def repair_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        validation_error: str,
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]:
        messages = _hosted_messages(
            stage, state, payload, previous_state, validation_error
        )
        prompt = _stage_prompt(stage, payload, validation_error)
        return self._complete_stage_json(
            messages,
            prompt=prompt,
            call_kind=stage,
            attempt_kind="repair",
            visual_input_scope=_stage_visual_input_scope(
                stage,
                payload,
                validation_error,
            ),
        )

    def _complete_stage_json(
        self,
        messages: list[dict[str, Any]],
        *,
        prompt: str,
        call_kind: str,
        attempt_kind: str,
        visual_input_scope: str,
    ) -> Mapping[str, Any]:
        """Attach stage audit metadata without changing ``complete_json``'s API."""

        previous = getattr(self, "_stage_audit_metadata", (None, None))
        self._stage_audit_metadata = (attempt_kind, visual_input_scope)
        try:
            return self.complete_json(
                messages,
                prompt=prompt,
                call_kind=call_kind,
            )
        finally:
            self._stage_audit_metadata = previous

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        prompt: str,
        call_kind: str,
    ) -> Mapping[str, Any]:
        """Submit one JSON-mode chat completion using the configured endpoint."""

        attempt_kind, visual_input_scope = self._stage_audit_metadata
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"BaiLian API key environment variable {self.api_key_env!r} is not configured"
            )
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
            "stream": False,
        }
        if self.enable_thinking and self.thinking_budget is not None:
            body["thinking_budget"] = self.thinking_budget
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt_index in range(self.max_network_retries + 1):
            started = time.monotonic()
            try:
                response = self.opener(request, timeout=self.timeout_seconds)
                raw_body = response.read()
                break
            except urllib.error.HTTPError as error:
                try:
                    raw_detail = error.read(4096).decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    raw_detail = ""
                detail = _redact_provider_error(
                    raw_detail, secrets=(api_key,)
                )
                retry_scheduled = bool(
                    _retryable_http_status(error.code)
                    and attempt_index < self.max_network_retries
                )
                audit = {
                    "backend": "bailian_openai_compatible",
                    "model": self.model,
                    "stage": call_kind,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "network_attempt": attempt_index + 1,
                    "http_status": error.code,
                    "provider_error": detail,
                    "retry_scheduled": retry_scheduled,
                }
                if attempt_kind is not None:
                    audit["attempt_kind"] = attempt_kind
                if visual_input_scope is not None:
                    audit["visual_input_scope"] = visual_input_scope
                self.last_call = audit
                self.call_history.append(audit)
                if retry_scheduled:
                    self.sleep(
                        self.retry_backoff_seconds * (2**attempt_index)
                    )
                    continue
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"BaiLian request failed with HTTP {error.code}{suffix}"
                ) from error
            except urllib.error.URLError as error:
                detail = _redact_provider_error(
                    str(error.reason), secrets=(api_key,)
                )
                retry_scheduled = attempt_index < self.max_network_retries
                audit = {
                    "backend": "bailian_openai_compatible",
                    "model": self.model,
                    "stage": call_kind,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "network_attempt": attempt_index + 1,
                    "network_error": detail,
                    "retry_scheduled": retry_scheduled,
                }
                if attempt_kind is not None:
                    audit["attempt_kind"] = attempt_kind
                if visual_input_scope is not None:
                    audit["visual_input_scope"] = visual_input_scope
                self.last_call = audit
                self.call_history.append(audit)
                if retry_scheduled:
                    self.sleep(
                        self.retry_backoff_seconds * (2**attempt_index)
                    )
                    continue
                raise RuntimeError(
                    f"BaiLian request failed after "
                    f"{attempt_index + 1} attempt(s): {detail}"
                ) from error
        else:  # pragma: no cover - the loop either succeeds or raises.
            raise RuntimeError("BaiLian request retry loop ended unexpectedly")
        try:
            envelope = json.loads(raw_body.decode("utf-8"))
            choice = envelope["choices"][0]["message"]["content"]
            if isinstance(choice, list):
                choice = next(
                    (
                        item.get("text")
                        for item in choice
                        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
                    ),
                    None,
                )
            result = (
                _extract_stage_json(choice, call_kind)
                if call_kind in _STAGE_EXPECTED_KEYS
                else _extract_json_object(choice)
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            StageProtocolError,
        ) as error:
            audit = {
                "backend": "bailian_openai_compatible",
                "model": self.model,
                "stage": call_kind,
                "latency_seconds": round(time.monotonic() - started, 6),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "parse_error": str(error),
                "raw_response": _audit_text(locals().get("choice", "")),
            }
            if attempt_kind is not None:
                audit["attempt_kind"] = attempt_kind
            if visual_input_scope is not None:
                audit["visual_input_scope"] = visual_input_scope
            self.last_call = audit
            self.call_history.append(audit)
            raise StageProtocolError("BaiLian response does not contain a valid chat JSON result") from error
        self.last_call = {
            "backend": "bailian_openai_compatible",
            "model": self.model,
            "stage": call_kind,
            "latency_seconds": round(time.monotonic() - started, 6),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "network_attempt": attempt_index + 1,
            "request_id": envelope.get("id"),
            "usage": envelope.get("usage"),
            "raw_response": _audit_text(choice),
            "parsed_output": result,
        }
        if attempt_kind is not None:
            self.last_call["attempt_kind"] = attempt_kind
        if visual_input_scope is not None:
            self.last_call["visual_input_scope"] = visual_input_scope
        self.call_history.append(dict(self.last_call))
        return result


def _local_messages(
    stage: StageName,
    state: ChangeState,
    payload: Mapping[str, Any],
    previous_state: ChangeState | None,
    validation_error: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, image in _stage_images(
        stage,
        state,
        previous_state,
        payload,
        validation_error,
    ):
        content.extend(
            [
                {"type": "text", "text": label},
                {"type": "image", "image": image},
            ]
        )
    content.append(
        {"type": "text", "text": _stage_prompt(stage, payload, validation_error)}
    )
    return [{"role": "user", "content": content}]


def _atomic_evidence_only_repair(
    stage: StageName,
    payload: Mapping[str, Any],
    validation_error: str | None,
) -> bool:
    """Identify a runtime-authorized field-local audit evidence repair."""

    repair_context = payload.get("repair_context")
    origin_error = (
        repair_context.get("origin_validation_error")
        if isinstance(repair_context, Mapping)
        else None
    )
    return bool(
        stage == "audit"
        and isinstance(repair_context, Mapping)
        and repair_context.get("mode") == "atomic_evidence_only"
        and (
            is_atomic_evidence_grounding_error(validation_error or "")
            or is_atomic_evidence_grounding_error(str(origin_error or ""))
        )
    )


def _stage_visual_input_scope(
    stage: StageName,
    payload: Mapping[str, Any],
    validation_error: str | None,
) -> str:
    if _atomic_evidence_only_repair(stage, payload, validation_error):
        return "authoritative_pair_rgb_only_evidence_repair"
    region = payload.get("region")
    if (
        stage == "audit"
        and isinstance(region, Mapping)
        and region.get("audit_kind") == "scan"
    ):
        return "authoritative_pair_rgb_only_scan"
    return "standard"


def _hosted_messages(
    stage: StageName,
    state: ChangeState,
    payload: Mapping[str, Any],
    previous_state: ChangeState | None,
    validation_error: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, image in _stage_images(
        stage,
        state,
        previous_state,
        payload,
        validation_error,
    ):
        content.extend(
            [
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
            ]
        )
    content.append(
        {"type": "text", "text": _stage_prompt(stage, payload, validation_error)}
    )
    return [
        {
            "role": "system",
            "content": "You are a strict change-detection verifier. Return JSON only.",
        },
        {"role": "user", "content": content},
    ]


def _stage_images(
    stage: StageName,
    state: ChangeState,
    previous_state: ChangeState | None,
    payload: Mapping[str, Any],
    validation_error: str | None = None,
) -> list[tuple[str, Image.Image]]:
    if stage == "direct":
        images = [
            ("T1 earlier RGB image", _as_image(state.t1_image)),
            ("T2 later RGB image", _as_image(state.t2_image)),
            ("Current predicted T1 object mask", _mask_image(state.t1_mask)),
            ("Current predicted T2 object mask", _mask_image(state.t2_mask)),
            ("Current final change mask", _mask_image(state.change_mask)),
        ]
        if previous_state is not None:
            rejected = payload.get("mode") == "replan"
            state_label = "Rejected candidate" if rejected else "Previous accepted"
            images.extend(
                [
                    (f"{state_label} T1 object mask", _mask_image(previous_state.t1_mask)),
                    (f"{state_label} T2 object mask", _mask_image(previous_state.t2_mask)),
                    (f"{state_label} final change mask", _mask_image(previous_state.change_mask)),
                ]
            )
            if payload.get("mode") == "candidate":
                added = np.logical_and(
                    state.change_mask, ~previous_state.change_mask
                )
                removed = np.logical_and(
                    previous_state.change_mask, ~state.change_mask
                )
                images.extend(
                    [
                        ("Candidate-added change pixels", _mask_image(added)),
                        ("Candidate-removed change pixels", _mask_image(removed)),
                    ]
                )
                crop_box = _delta_crop_box(np.logical_or(added, removed))
                if crop_box is not None:
                    images.extend(
                        [
                            (
                                "Exact candidate-delta-only T1 RGB crop; cyan contour marks delta",
                                _delta_only_contour(state.t1_image, np.logical_or(added, removed)).crop(crop_box),
                            ),
                            (
                                "Exact candidate-delta-only T2 RGB crop; cyan contour marks delta",
                                _delta_only_contour(state.t2_image, np.logical_or(added, removed)).crop(crop_box),
                            ),
                            ("Exact previous accepted change-mask crop", _mask_image(previous_state.change_mask).crop(crop_box)),
                            ("Exact candidate change-mask crop", _mask_image(state.change_mask).crop(crop_box)),
                            ("Exact candidate-added pixel crop", _mask_image(added).crop(crop_box)),
                            ("Exact candidate-removed pixel crop", _mask_image(removed).crop(crop_box)),
                        ]
                    )
        return images
    if stage == "select":
        catalog = payload.get("proposal_catalog", [])
        if not isinstance(catalog, list):
            raise StageProtocolError("proposal_catalog must be a list")
        return [
            (
                "Numbered global proposal overview: T1 | T2 | current change mask",
                _proposal_overview(state, previous_state, catalog),
            )
        ]
    if stage in {"audit", "evidence", "candidate_evidence", "diagnosis"}:
        visual_context = str(payload.get("visual_context", "hybrid"))
        if visual_context not in {"proposal", "hybrid"}:
            raise StageProtocolError(
                "visual_context must be proposal or hybrid for staged regional calls"
            )
        images: list[tuple[str, Image.Image]] = []
        catalog = payload.get("proposal_catalog", [])
        region = payload.get("region")
        active_id = (
            str(region.get("region_id"))
            if isinstance(region, Mapping) and region.get("region_id")
            else None
        )
        scan_audit = bool(
            stage == "audit"
            and isinstance(region, Mapping)
            and region.get("audit_kind") == "scan"
        )
        evidence_only_repair = _atomic_evidence_only_repair(
            stage,
            payload,
            validation_error,
        )
        if scan_audit:
            box = region.get("box_normalized_1000")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise StageProtocolError(
                    "scan audit region requires an authoritative tile box"
                )
            if (
                state.t1_mask.any()
                or state.t2_mask.any()
                or state.change_mask.any()
            ):
                raise StageProtocolError(
                    "scan audit requires all predicted masks to be empty"
                )
            crop_box = _normalized_exact_crop_box(box, state.image_size)
            # Scan tiles exist only because both predicted object masks are
            # empty. Their all-black crops and the all-white tile-geometry mask
            # repeat Environment facts and can distract a small VLM into using
            # presentation color as scene evidence. The authoritative tile box
            # makes paired RGB-only crops sufficient for semantic audit.
            return [
                (
                    f"Authoritative scan tile {active_id}: T1 RGB-only crop",
                    _provider_safe_exact_crop(
                        _as_image(state.t1_image), crop_box
                    ),
                ),
                (
                    f"Authoritative scan tile {active_id}: T2 RGB-only crop",
                    _provider_safe_exact_crop(
                        _as_image(state.t2_image), crop_box
                    ),
                ),
            ]
        if evidence_only_repair:
            if not isinstance(region, Mapping):
                raise StageProtocolError(
                    "atomic evidence repair requires an authoritative region"
                )
            box = region.get("box_normalized_1000")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise StageProtocolError(
                    "atomic evidence repair requires an authoritative region box"
                )
            crop_box = _normalized_exact_crop_box(box, state.image_size)
            return [
                (
                    f"Authoritative audit region {active_id}: "
                    "T1 unmodified RGB-only crop",
                    _provider_safe_exact_crop(
                        _as_image(state.t1_image), crop_box
                    ),
                ),
                (
                    f"Authoritative audit region {active_id}: "
                    "T2 unmodified RGB-only crop",
                    _provider_safe_exact_crop(
                        _as_image(state.t2_image), crop_box
                    ),
                ),
            ]
        if isinstance(catalog, list) and catalog:
            images.append(
                (
                    f"Active-region marked global overview; yellow is {active_id}",
                    _proposal_overview(
                        state,
                        previous_state,
                        catalog,
                        active_region_id=active_id,
                        include_object_masks=visual_context == "hybrid",
                    ),
                )
            )
        if isinstance(region, Mapping):
            box = region.get("box_normalized_1000")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                crop_box = _normalized_crop_box(box, state.image_size)
                if stage == "candidate_evidence" and previous_state is not None:
                    added = np.logical_and(
                        state.change_mask, ~previous_state.change_mask
                    )
                    removed = np.logical_and(
                        previous_state.change_mask, ~state.change_mask
                    )
                    delta = np.logical_or(added, removed)
                    images.extend(
                        [
                            (
                                "Exact candidate-delta-only T1 RGB crop; cyan contour marks delta",
                                _delta_only_contour(state.t1_image, delta).crop(crop_box),
                            ),
                            (
                                "Exact candidate-delta-only T2 RGB crop; cyan contour marks delta",
                                _delta_only_contour(state.t2_image, delta).crop(crop_box),
                            ),
                            (
                                "Exact proposal T1 object-mask crop",
                                _mask_image(state.t1_mask).crop(crop_box),
                            ),
                            (
                                "Exact proposal T2 object-mask crop",
                                _mask_image(state.t2_mask).crop(crop_box),
                            ),
                            (
                                "Exact proposal change-mask crop",
                                _mask_image(state.change_mask).crop(crop_box),
                            ),
                        ]
                    )
                else:
                    images.extend(
                        [
                            ("Exact T1 proposal crop", _as_image(state.t1_image).crop(crop_box)),
                            ("Exact T2 proposal crop", _as_image(state.t2_image).crop(crop_box)),
                            ("Exact proposal T1 object-mask crop", _mask_image(state.t1_mask).crop(crop_box)),
                            ("Exact proposal T2 object-mask crop", _mask_image(state.t2_mask).crop(crop_box)),
                            ("Exact proposal change-mask crop", _mask_image(state.change_mask).crop(crop_box)),
                        ]
                    )
                    if stage == "audit":
                        images.append(
                            (
                                "Exact audited component geometry crop",
                                _mask_image(
                                    _audited_component_mask(state, region)
                                ).crop(crop_box),
                            )
                        )
                if previous_state is not None:
                    added = np.logical_and(
                        state.change_mask, ~previous_state.change_mask
                    )
                    removed = np.logical_and(
                        previous_state.change_mask, ~state.change_mask
                    )
                    delta = np.logical_or(added, removed)
                    images.extend(
                        [
                            (
                                "Exact previous accepted T1 object-mask crop",
                                _mask_image(previous_state.t1_mask).crop(crop_box),
                            ),
                            (
                                "Exact previous accepted T2 object-mask crop",
                                _mask_image(previous_state.t2_mask).crop(crop_box),
                            ),
                            (
                                "Exact previous accepted change-mask crop",
                                _mask_image(previous_state.change_mask).crop(crop_box),
                            ),
                            (
                                "Exact candidate-added pixel crop",
                                _mask_image(added).crop(crop_box),
                            ),
                            (
                                "Exact candidate-removed pixel crop",
                                _mask_image(removed).crop(crop_box),
                            ),
                        ]
                    )
        return images
    return []


_MARK_COLORS = (
    (255, 70, 70),
    (40, 220, 90),
    (60, 150, 255),
    (255, 80, 220),
    (30, 220, 220),
    (255, 150, 30),
)


def _proposal_overview(
    state: ChangeState,
    previous_state: ChangeState | None,
    catalog: list[Any],
    *,
    active_region_id: str | None = None,
    include_object_masks: bool = False,
) -> Image.Image:
    """Render deterministic Set-of-Mark contours from Environment proposals."""

    panels = [
        _as_image(state.t1_image).convert("RGB"),
        _as_image(state.t2_image).convert("RGB"),
    ]
    if include_object_masks:
        panels.extend(
            [
                _mask_image(state.t1_mask).convert("RGB"),
                _mask_image(state.t2_mask).convert("RGB"),
            ]
        )
    if previous_state is not None:
        panels.append(_mask_image(previous_state.change_mask).convert("RGB"))
    panels.append(_mask_image(state.change_mask).convert("RGB"))
    width, height = state.image_size
    for index, proposal in enumerate(catalog):
        if not isinstance(proposal, Mapping):
            continue
        region_id = str(proposal.get("region_id", ""))
        if not region_id:
            continue
        color = (
            (255, 255, 0)
            if region_id == active_region_id
            else _MARK_COLORS[index % len(_MARK_COLORS)]
        )
        component = _proposal_component(state, previous_state, proposal)
        seed = proposal.get("component_seed_normalized_1000", [0, 0])
        try:
            seed_x, seed_y = normalized_point_to_pixel(seed, state.image_size)
        except (TypeError, ValueError):
            continue
        boundary = _mask_boundary(component) if component is not None else None
        for panel in panels:
            array = np.asarray(panel).copy()
            if boundary is not None and boundary.any():
                array[boundary] = color
                panel.paste(Image.fromarray(array))
            else:
                box = proposal.get("box_normalized_1000")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    left, top, right, bottom = _normalized_crop_box(
                        tuple(int(value) for value in box), state.image_size
                    )
                    ImageDraw.Draw(panel).rectangle(
                        (left, top, max(left, right - 1), max(top, bottom - 1)),
                        outline=color,
                        width=2,
                    )
            draw = ImageDraw.Draw(panel)
            radius = 3
            draw.ellipse(
                (
                    seed_x - radius,
                    seed_y - radius,
                    seed_x + radius,
                    seed_y + radius,
                ),
                fill=color,
                outline=(0, 0, 0),
            )
            label_x = min(max(0, seed_x + 5), max(0, width - 24))
            label_y = min(max(0, seed_y - 8), max(0, height - 12))
            label_box = draw.textbbox((label_x, label_y), region_id)
            draw.rectangle(label_box, fill=(0, 0, 0))
            draw.text((label_x, label_y), region_id, fill=color)
    canvas = Image.new("RGB", (width * len(panels), height), (0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    return canvas


def _proposal_component(
    state: ChangeState,
    previous_state: ChangeState | None,
    proposal: Mapping[str, Any],
) -> np.ndarray | None:
    kind = str(proposal.get("audit_kind", "mixed"))
    temporal_xor = np.logical_xor(state.t1_mask, state.t2_mask)
    if kind == "present":
        source = state.change_mask
    elif kind == "missing":
        source = np.logical_and(temporal_xor, ~state.change_mask)
    elif kind == "scan":
        box = proposal.get("box_normalized_1000")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        left, top = normalized_point_to_pixel(box[:2], state.image_size)
        right, bottom = normalized_point_to_pixel(box[2:], state.image_size)
        component = np.zeros_like(state.change_mask, dtype=bool)
        component[top : bottom + 1, left : right + 1] = True
        return component
    elif kind == "delta_added" and previous_state is not None:
        source = np.logical_and(state.change_mask, ~previous_state.change_mask)
    elif kind == "delta_removed" and previous_state is not None:
        source = np.logical_and(previous_state.change_mask, ~state.change_mask)
    else:
        source = np.logical_or(state.change_mask, temporal_xor)
    seed = proposal.get("component_seed_normalized_1000", [0, 0])
    try:
        seed_x, seed_y = normalized_point_to_pixel(seed, state.image_size)
    except (TypeError, ValueError):
        return None
    for component in connected_components(source, min_area=1):
        if component[seed_y, seed_x]:
            return component
    return None


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return np.logical_and(mask, ~eroded)


def _delta_crop_box(
    mask: np.ndarray, *, padding: int = 8, minimum_side: int = 32
) -> tuple[int, int, int, int] | None:
    """Return a padded PIL crop box around a non-empty binary delta."""

    source = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(source)
    if not len(xs):
        return None
    height, width = source.shape
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + padding + 1)
    bottom = min(height, int(ys.max()) + padding + 1)
    if right - left < minimum_side:
        extra = minimum_side - (right - left)
        left = max(0, left - extra // 2)
        right = min(width, right + extra - extra // 2)
        left = max(0, right - minimum_side)
    if bottom - top < minimum_side:
        extra = minimum_side - (bottom - top)
        top = max(0, top - extra // 2)
        bottom = min(height, bottom + extra - extra // 2)
        top = max(0, bottom - minimum_side)
    return left, top, right, bottom


def _terminal_review_prompt_instruction(
    stage: StageName,
    payload: Mapping[str, Any],
) -> str:
    """Validate and freeze the H1 round-two prompt identity.

    Ordinary/H0 calls have no ``terminal_review`` envelope and therefore keep
    their prompt bytes unchanged.  An envelope is an authenticated protocol
    surface: unknown keys, branch/version mismatches and an unbound synthetic
    scan proposal all fail closed before model execution.
    """

    context = payload.get("terminal_review")
    if context is None:
        return ""
    if stage != "audit" or not isinstance(context, Mapping):
        raise StageProtocolError(
            "terminal-review prompt context is valid only for an audit call"
        )
    expected_keys = {
        "round_index",
        "round_kind",
        "trigger",
        "fresh_backend_context",
        "inherits_hidden_state",
        "copies_round_one_raw_response",
        "selection_policy_version",
        "preferred_region_id",
        "prompt_version",
        "selection_branch",
        "review_region_id",
        "proposal_version",
    }
    if set(context) != expected_keys:
        raise StageProtocolError("terminal-review prompt context keys differ")
    if (
        type(context.get("round_index")) is not int
        or context.get("round_index") != 2
        or context.get("fresh_backend_context") is not True
        or context.get("inherits_hidden_state") is not False
        or context.get("copies_round_one_raw_response") is not False
        or context.get("selection_policy_version")
        != SELECTION_POLICY_VERSION
    ):
        raise StageProtocolError("terminal-review prompt context is malformed")
    region = payload.get("region")
    region_id = region.get("region_id") if isinstance(region, Mapping) else None
    if not isinstance(region_id, str) or context.get("review_region_id") != region_id:
        raise StageProtocolError(
            "terminal-review prompt region differs from Environment proposal"
        )
    proposal_version = (
        region.get("proposal_version") if isinstance(region, Mapping) else None
    )
    if proposal_version is not None and (
        not isinstance(proposal_version, str) or not proposal_version
    ):
        raise StageProtocolError("Environment proposal_version is malformed")
    if context.get("proposal_version") != proposal_version:
        raise StageProtocolError(
            "terminal-review prompt does not bind Environment proposal_version"
        )
    if (
        isinstance(region, Mapping)
        and region.get("audit_kind") == "scan"
        and proposal_version != EMPTY_MASK_SCAN_PROPOSAL_VERSION
    ):
        raise StageProtocolError(
            "terminal-review scan proposal_version differs from frozen contract"
        )
    if (
        isinstance(region, Mapping)
        and region.get("audit_kind") != "scan"
        and proposal_version is not None
    ):
        raise StageProtocolError(
            "terminal-review proposal_version is registered only for scan regions"
        )
    proposal_binding = (
        " The selected synthetic scan proposal is bound to frozen version "
        f"{proposal_version}."
        if proposal_version is not None
        else ""
    )

    trigger = context.get("trigger")
    if trigger == "provisional_finish":
        if (
            context.get("round_kind") != "finish_confirmation"
            or any(
                context.get(key) is not None
                for key in (
                    "preferred_region_id",
                    "prompt_version",
                    "selection_branch",
                )
            )
        ):
            raise StageProtocolError(
                "finish-confirmation prompt context is malformed"
            )
        return (
            " This is frozen H1 terminal review round two. Independently "
            "re-audit this accepted-state Environment region from the supplied "
            "visual evidence; do not inherit a prior answer, assume completion, "
            "or execute an action."
            + proposal_binding
        )
    if (
        trigger != "plan_null_after_allowed_repair"
        or context.get("round_kind") != "plan_null_alternative"
        or context.get("preferred_region_id") != region_id
    ):
        raise StageProtocolError("plan-null prompt context is malformed")
    branch = context.get("selection_branch")
    prompt_version = context.get("prompt_version")
    if _PLAN_NULL_PROMPT_VERSION_BY_BRANCH.get(branch) != prompt_version:
        raise StageProtocolError(
            "plan-null prompt branch/version differs from the frozen mapping"
        )
    if branch == "next_unreviewed_environment_region":
        return (
            " Frozen prompt qwen3vl_plan_null_next_unreviewed_region_v1: "
            "independently audit this next Environment-ordered region, which "
            "round one did not inspect; do not reuse any round-one diagnosis."
            + proposal_binding
        )
    return (
        " Frozen prompt qwen3vl_plan_null_alternative_diagnosis_v1: "
        "independently reconsider this first unresolved Environment-ordered "
        "region and seek an alternative diagnosis from its current visual "
        "evidence; do not repeat or assume the round-one diagnosis."
        + proposal_binding
    )


def _stage_prompt(
    stage: StageName,
    payload: Mapping[str, Any],
    validation_error: str | None = None,
) -> str:
    terminal_review_instruction = _terminal_review_prompt_instruction(
        stage,
        payload,
    )
    evidence_only_repair = _atomic_evidence_only_repair(
        stage,
        payload,
        validation_error,
    )
    context_payload = dict(payload)
    if payload.get("terminal_review") is None:
        ordinary_region = context_payload.get("region")
        if isinstance(ordinary_region, Mapping):
            ordinary_region = dict(ordinary_region)
            ordinary_region.pop("proposal_version", None)
            context_payload["region"] = ordinary_region
        ordinary_catalog = context_payload.get("proposal_catalog")
        if isinstance(ordinary_catalog, list):
            context_payload["proposal_catalog"] = [
                {
                    key: field_value
                    for key, field_value in record.items()
                    if key != "proposal_version"
                }
                if isinstance(record, Mapping)
                else record
                for record in ordinary_catalog
            ]
    context_region = context_payload.get("region")
    if evidence_only_repair:
        context_payload = {
            "target_class": payload.get("target_class"),
            "region": {
                "region_id": context_region.get("region_id"),
            }
            if isinstance(context_region, Mapping)
            else None,
        }
    elif (
        stage == "audit"
        and isinstance(context_region, Mapping)
        and context_region.get("audit_kind") == "scan"
    ):
        # The current scan record contains every authoritative fact needed by
        # this atomic call. The full nine-tile catalog is only a coverage/runtime
        # concern and no scan overview image is supplied.
        context_payload.pop("proposal_catalog", None)
    context = json.dumps(
        {"environment_facts": context_payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    region = payload.get("region")
    region_id = (
        str(region.get("region_id"))
        if isinstance(region, Mapping) and region.get("region_id") is not None
        else "r0"
    )
    target_class = str(payload.get("target_class", "")).strip()
    if stage == "select":
        max_selected = int(payload.get("max_selected_regions", 1))
        target_label = target_class or "the requested target class"
        schema = (
            '{"selection":{"region_ids":["r0"],'
            '"reason":"short observable reason"}}'
        )
        task = (
            "Inspect the numbered global overview and choose the regions most likely to contain "
            f"a material {target_label} change-mask error. Return one or more supplied region IDs, "
            f"at most {max_selected}; never output coordinates. Selection only prioritizes local "
            "inspection and does not assert that unselected regions are correct. State the "
            "strongest observable suspected error because this exact reason will be persisted "
            "into each selected region's atomic audit. For building change, roads, parking, "
            "vehicles, trailers/mobile equipment, vegetation, bare ground, shadows, lighting, "
            "and registration artifacts are non-target."
        )
    elif stage == "audit" and evidence_only_repair:
        if not target_class:
            raise StageProtocolError("audit payload must include target_class")
        schema = (
            '{"region_id":"' + region_id + '","visual_judgment":{'
            '"evidence":"REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT"},'
            '"diagnosis":{}}'
        )
        task = (
            f"Repair only the atomic-audit evidence string for target_class={target_class}. "
            "Inspect exactly the two authoritative unmodified T1/T2 RGB crops. Runtime has "
            "already validated and will mechanically preserve the initial region ID, T1/T2 "
            "semantic states, and all seven checklist statuses; every newly generated field "
            "except visual_judgment.evidence is ignored. Return the compact top-level envelope "
            "shown in the template so JSON completion remains deterministic. Evidence must be "
            "one short ordinary natural-language sentence naming visible physical content and "
            "the T1/T2 timepoint. Use a grammatical sentence with spaces between words and refer "
            "only to physical scene content directly visible in the paired RGB crops."
        )
    elif stage == "audit":
        if not target_class:
            raise StageProtocolError("audit payload must include target_class")
        schema = (
            '{"region_id":"' + region_id + '","visual_judgment":{'
            '"t1_state":"<BUILDING_BACKGROUND_MIXED_OR_UNCERTAIN>",'
            '"t2_state":"<BUILDING_BACKGROUND_MIXED_OR_UNCERTAIN>",'
            '"evidence":"REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT"},'
            '"diagnosis":{"audit_checklist":{'
            '"evidence_sufficient":"<AUDIT_STATUS>",'
            '"target_class_only":"<AUDIT_STATUS>",'
            '"white_pixels_supported":"<AUDIT_STATUS>",'
            '"boundary_alignment":"<AUDIT_STATUS>",'
            '"internal_holes_absent":"<AUDIT_STATUS>",'
            '"changed_object_extent_complete":"<AUDIT_STATUS>",'
            '"fragment_artifacts_absent":"<AUDIT_STATUS>"}}}'
        )
        task = (
            f"Perform one compact atomic semantic audit for target_class={target_class}. "
            "Every Environment proposal is audited in deterministic order. Inspect only the exact "
            "audited component geometry while using unmodified local and full-frame RGB as "
            "context. change_mask_state is an authoritative Environment fact shown as input; "
            "do not copy it into the response. Runtime injects it into the trace. For building "
            "change detection, permanent roofed building structures are target objects; roads, "
            "parking areas, vehicles, trailers, RVs, mobile equipment, vegetation, bare ground, "
            "shadows, illumination, and registration differences are non-target context. "
            "Predicted object masks are fallible aids, never ground truth. t1_state and t2_state "
            "are semantic target-class labels from the unmodified RGB, not pixel colors and not "
            "mask colors. Each must be exactly one of building, background, mixed, or uncertain. "
            "Never output black, white, green, or any other color name for either state. Classify "
            "the exact component pixels in T1 and T2 independently. "
            "A visible object "
            "appearing in T2 is not automatically a building or a correct change. Correct requires "
            "every pixel of the exact connected component to be supported; one real changed building "
            "does not justify attached bare ground, roads, vehicles, or boundary spill. "
            "Fill each checklist item with one status string only: pass, fail, not_applicable, or "
            "uncertain; evidence_sufficient cannot be not_applicable. target_class_only checks "
            "semantic scope; white_pixels_supported checks precision of all material white pixels; "
            "boundary_alignment checks spill or erosion; internal_holes_absent checks missing "
            "interiors; changed_object_extent_complete checks missing target-change extent; and "
            "fragment_artifacts_absent checks isolated noise. Use not_applicable only when the "
            "property is absent, and uncertain rather than pass when the images cannot resolve it. "
            "For Environment change_mask_state=white with a false positive, at least one "
            "precision-side item must fail. For Environment change_mask_state=black with a false "
            "negative, a recall-side item must fail. Simultaneous precision and recall failures "
            "produce a mixed error. A concrete change_mask_state=white proposal with the same "
            "T1/T2 semantic states cannot pass as correct; a concrete change_mask_state=black "
            "proposal with different T1/T2 semantic states cannot pass as correct. At least one "
            "substantive checklist item must be "
            "applicable. Runtime derives evidence quality, mask assessment, "
            "error_type, target view, action, quality, and finish readiness from the checklist, RGB "
            "states, and Environment facts. Do not output those derived fields, per-item prose, "
            "coordinates, confidence, accept, stop, or hidden reasoning. Give one short, concrete "
            "RGB observation specific to this region that names the visible physical content and "
            "the T1/T2 timepoint; color alone is not a semantic target. Proposal or change-mask "
            "color/presence is not evidence for a T1/T2 semantic state. Replace every placeholder "
            "with a JSON "
            "string; never copy placeholder or template prose, and never copy an image marker."
        )
        if isinstance(region, Mapping) and region.get("audit_kind") == "scan":
            task += (
                " This region is one deterministic full-frame scan tile created "
                "only because both predicted T1/T2 object masks are empty. The "
                "tile is not evidence that every pixel is one connected object. "
                "Inspect the complete paired RGB tile for an obvious target-class "
                "appearance or disappearance. Diagnose false_negative only when "
                "a material changed target is visually localized inside this scan "
                "tile; otherwise use correct or uncertain. Unrelated non-target "
                "content elsewhere in the tile does not fail target_class_only; "
                "that check passes when the diagnosis stays scoped to the requested "
                "class. White-pixel precision checks are not_applicable for this "
                "zero-prediction scan tile. Only the paired RGB crops are visual evidence; "
                "never cite predicted-mask state or presentation pixels as evidence of no "
                "RGB change. Write evidence as one ordinary natural-language sentence with "
                "spaces between words; do not join words with underscores. A false-negative scan "
                "is executed with the authoritative tile box, never a model-authored "
                "coordinate."
            )
    elif stage in {"evidence", "candidate_evidence"}:
        schema = (
            f'{{"region_id":"{region_id}","visual_judgment":{{'
            '"t1_state":<RGB_STATE>,"t2_state":<RGB_STATE>,'
            '"evidence_quality":<EVIDENCE_QUALITY>}}}'
        )
        evidence_scope = (
            "the active-region marked full-frame state and exact local proposal crops"
            if payload.get("visual_context") == "hybrid"
            else "the active-region marked RGB/change overview and exact local proposal crops"
        )
        if stage == "candidate_evidence":
            if not target_class:
                raise StageProtocolError(
                    "candidate_evidence payload must include target_class"
                )
            task = (
                f"Read {evidence_scope} and judge only the pixels marked white in the exact "
                "candidate-added/removed mask crops and the original-color pixels inside the "
                "cyan contour in the delta-only T1/T2 RGB crops. Black outside that contour is "
                "presentation padding, not scene evidence; ignore unedited objects elsewhere in "
                "the rectangular crop. T1/T2 states describe the "
                "physical scene under those delta pixels, never white/black mask color. For "
                f"target_class={target_class}; for building, permanent roofed structures are "
                "targets while vehicles, trailers, RVs, mobile equipment, roads, parking areas, "
                "vegetation, and bare ground are non-target. Building means those delta pixels cover a real building, "
                "background means they cover no building, and mixed or uncertain must be used "
                "only when the contoured delta pixels themselves cannot support a clean label. "
                "Do not diagnose false_positive_change or false_negative and do not claim that "
                "removing pixels fixes a false negative. Runtime combines this observation with "
                "the attempted action, persisted initial diagnosis, and delta polarity. If new "
                "candidate evidence conflicts with that diagnosis, use ambiguous or insufficient "
                "instead of silently reversing the semantic conclusion. "
            )
        else:
            task = (
                f"Read {evidence_scope} and classify T1/T2 physical target-class content from "
                "RGB evidence. Do not copy the example or classify mask color. Actively audit "
                "the fallible predicted mask: seeing a real target transition under part of a "
                "white component does not prove that its full extent, boundary, or interior is "
                "correct. "
            )
        task += (
            "Each state must be one of building, background, mixed, uncertain; evidence_quality "
            "must be clear, ambiguous, or insufficient. Use ambiguous or insufficient instead "
            "of inventing a numeric confidence."
        )
    elif stage == "diagnosis":
        schema = (
            '{"region_id":"' + region_id + '","diagnosis":{'
            '"audit_checklist":{'
            '"evidence_sufficient":<AUDIT_STATUS>,'
            '"target_class_only":<AUDIT_STATUS>,'
            '"white_pixels_supported":<AUDIT_STATUS>,'
            '"boundary_alignment":<AUDIT_STATUS>,'
            '"internal_holes_absent":<AUDIT_STATUS>,'
            '"changed_object_extent_complete":<AUDIT_STATUS>,'
            '"fragment_artifacts_absent":<AUDIT_STATUS>},'
            '"target_view":<TARGET_VIEW_OR_NULL>}}'
        )
        task = (
            "Actively audit the current fallible mask from supplied RGB evidence, predicted "
            "masks, proposal crop, and Environment facts. Do not infer correctness from the "
            "change-mask color or from a temporal difference alone. A proposal may contain both "
            "correct and incorrect pixels. Fill every checklist item independently. Every "
            "AUDIT_STATUS must be exactly pass, fail, not_applicable, or uncertain; "
            "evidence_sufficient cannot be not_applicable. target_class_only asks whether all "
            "audited predicted change is confined to the requested semantic class. "
            "white_pixels_supported asks whether every material predicted white pixel is "
            "supported by a real target-class transition. boundary_alignment checks spill or "
            "erosion at object edges. internal_holes_absent checks missing interiors. "
            "changed_object_extent_complete checks missing target-change extent, including a "
            "black missing-change proposal. fragment_artifacts_absent checks isolated noise and "
            "detached fragments. Use not_applicable only when that mask property is genuinely "
            "absent from the audited proposal. If evidence cannot resolve an item, use uncertain, "
            "not pass. Seeing a changed building under any portion of a white component is a hard "
            "negative: it does not prove the component boundary, exterior pixels, holes, or full "
            "extent are correct. Runtime, not the model, deterministically derives error_type and "
            "quality from these statuses. Use target_view null when no correction is indicated or "
            "evidence is insufficient; otherwise use t1 or t2 for the object mask needing "
            "correction.\n"
            "TEXT-ONLY FEW-SHOT CONTRASTS (abstract; never copy blindly):\n"
            "1) Correct white component: a target transition supports its entire white area, "
            "boundary, and interior, with no missing extent or fragments. Mark every applicable "
            "quality item pass; runtime derives none.\n"
            "2) Boundary-spill mixed component: part of a white component covers a real target "
            "transition, but pixels spill onto background and part of the changed target is "
            "missing. Mark white_pixels_supported/boundary_alignment fail and "
            "changed_object_extent_complete fail; runtime derives mixed_error.\n"
            "3) Black missing-change component: RGB shows a real target transition in an "
            "unpredicted black region. Mark changed_object_extent_complete fail, mark white-only "
            "checks not_applicable where appropriate, and runtime derives false_negative. "
            "Replace every schema placeholder with a valid JSON string."
        )
    elif stage == "direct":
        mode = str(payload.get("mode", "initial"))
        candidate_mode = mode == "candidate"
        replan_mode = mode == "replan"
        candidate_effect = (
            '{"intended_error_improved":<BOOLEAN>,'
            '"introduced_false_positive":<BOOLEAN>,'
            '"introduced_false_negative":<BOOLEAN>,'
            '"boundary_or_artifact_worsened":<BOOLEAN>,'
            '"evidence":"short paired evidence"}'
            if candidate_mode
            else "null"
        )
        schema = (
            '{"verdict":{"rubric":{'
            '"evidence_sufficient":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"target_class_only":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_semantic_precision":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_semantic_recall":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"changed_object_extent":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_boundary_alignment":{"pass":<BOOLEAN>,"evidence":"short evidence"},'
            '"change_artifact_control":{"pass":<BOOLEAN>,"evidence":"short evidence"}},'
            '"candidate_effect":' + candidate_effect + ','
            '"error_type":<ERROR_TYPE>,"target_view":<TARGET_VIEW_OR_NULL>,'
            '"suggested_action":<ACTION>,"coordinate_normalized_1000":<POINT_OR_NULL>,'
            '"box_normalized_1000":<BOX_OR_NULL>,"feedback":"short explanation"}}'
        )
        task = (
            "Apply the binary rubric to the complete T1/T2 pair and final change mask without "
            "Proposal geometry. The only semantic target is target_class in Environment facts. "
            "For target_class=building, roads, parking areas, vehicles, vegetation, bare ground, "
            "shadows, illumination, and registration differences are context, never target "
            "changes. Predicted T1/T2 object masks are fallible aids, not ground truth; do not "
            "demand that they segment non-target content or unchanged target objects when the "
            "final change mask is correct. White change pixels are correct only for a real "
            "appearance, disappearance, construction, or demolition of target objects. "
            "Set evidence_sufficient for visual judgeability. target_class_only records whether "
            "your reasoning stayed scoped to target_class; set it true when you correctly call "
            "roads or vehicles non-target false positives, even if the mask contains them. Set "
            "it false only when your reasoning treats non-target content as target change. Set "
            "change_semantic_precision when no "
            "material white region is unsupported, change_semantic_recall when no obvious target "
            "change is missing, changed_object_extent when changed targets have materially "
            "complete coverage, change_boundary_alignment when boundaries follow changed target "
            "objects, and change_artifact_control when no material fragments, holes, or noise "
            "remain. Give one short observable evidence string per item; do not provide hidden "
            "chain-of-thought. Runtime computes quality, progress, comparison, and acceptance, "
            "so never output those fields. error_type must be exactly one of none, "
            "false_positive_change, false_negative, mixed_error, or uncertain_region. Every "
            "rubric item passing requires none; none requires every quality item and target scope "
            "to pass. A failed evidence_sufficient gate requires uncertain_region; a failed "
            "target_class_only item is an auditable scope error and may still carry an actionable "
            "false-positive or mixed diagnosis. For none, "
            "use target_view null, suggested_action finish, and null geometry. "
            "For an error, choose t1 or t2 and one executable positive_point, negative_point, "
            "or box; suggested_action must be exactly one of positive_point, negative_point, "
            "box, or finish (never use revise/correct/edit). A point needs normalized [0,1000] coordinate; a box needs normalized "
            "[0,1000] XYXY box. For a negative_point, the selected coordinate must be white "
            "in the selected current object mask. A positive_point may be on either mask state "
            "because its predicted component can still expand the current mask. "
            "For a candidate, compare actual candidate versus previous "
            "accepted masks: intended_error_improved means the attempted target-class error was "
            "materially reduced; the three harm flags report newly introduced semantic or "
            "shape damage."
        )
        if replan_mode:
            task += (
                " This is a rollback replan: current images and masks are the accepted state; "
                "the additional rejected-candidate masks and Environment facts explain why the "
                "previous action failed. Do not repeat the rejected action or geometry. Choose a "
                "materially different executable correction, or finish only if no error remains. "
                "Use the bounded rejection_history to avoid all recently failed actions. "
                "This is action replanning, not candidate evaluation; candidate_effect must be "
                "JSON null. Runtime comparison is uncertain until the replacement action is "
                "executed; comparison must be uncertain in this replan response. "
                ""
            )
    else:
        schema = (
            '{"decision":{"feedback":"short complete-state assessment"}}'
        )
        task = (
            "Summarize the complete initial state using the audited diagnoses and describe the "
            "most material remaining error, if any. Do not output a numeric quality or confidence, "
            "comparison, accept, stop, or next action; runtime derives quality from the discrete "
            "audit checklist; runtime derives final readiness from the independent state diagnosis and "
            "executable plan. Candidate transition decisions never use this stage; runtime derives "
            "them from candidate_evidence."
        )
    task += terminal_review_instruction
    repair = ""
    if evidence_only_repair:
        repair = (
            "\nREPAIR: Produce a new region-specific evidence sentence directly from "
            "the paired unmodified RGB crops."
        )
    elif validation_error:
        repair = (
            "\nREPAIR: The previous response was rejected for this exact reason: "
            f"{validation_error[:500]} Do not repeat the rejected structure."
        )
        if stage == "audit" and "invalid RGB state" in validation_error:
            repair += (
                " Correct t1_state and t2_state using semantic target presence in the "
                "unmodified RGB. Each must be exactly building, background, mixed, or "
                "uncertain; black, white, green, and all other color names are invalid."
            )
        if stage == "audit" and "schema boilerplate" in validation_error:
            repair += (
                " Replace evidence with one region-specific visible fact from the paired "
                "unmodified RGB; never copy template prose or cite mask color as RGB evidence."
            )
        if stage == "audit" and "image/schema placeholder" in validation_error:
            repair += (
                " Replace the image/schema marker with one short natural-language visible "
                "fact from the paired unmodified RGB."
            )
        if stage == "audit" and "underscore-delimited label" in validation_error:
            repair += (
                " Rewrite evidence as one short natural-language sentence about visible content "
                "in the paired unmodified RGB. Use spaces between words and do not join words "
                "with underscores."
            )
        if stage == "audit" and "mask/presentation color" in validation_error:
            repair += (
                " Replace evidence with visible target-class content from paired RGB. "
                "Do not cite change_mask_state or any mask, presentation pixels, or added marks."
            )
        if (
            stage == "audit"
            and "JSON/numeric/coordinate placeholder" in validation_error
        ):
            repair += (
                " Replace evidence with one short natural-language observation from the "
                "paired unmodified RGB; a bare JSON value/list, number, RGB tuple, box, "
                "or coordinate list is not visual evidence."
            )
        if stage == "direct" and isinstance(payload.get("repair_context"), Mapping):
            repair += (
                " Preserve the previous response's rubric pass booleans, error_type, "
                "and candidate_effect flags. Change only the invalid target/action/geometry "
                "fields needed to satisfy the tool contract; do not turn an actionable error "
                "into none/finish merely because the first geometry was invalid."
            )
    output_contract = (
        "OUTPUT CONTRACT: Return exactly one JSON object, no markdown and no explanation. "
        "The top-level keys must exactly match this template:\n"
        f"{schema}\n"
        f"TASK: {task}"
        f"{repair}\n"
    )
    if evidence_only_repair:
        return output_contract + "T1 is earlier and T2 is later."
    return output_contract + (
        "Environment region IDs and normalized [0,1000] geometry are authoritative; never alter "
        "them. T1 is earlier and T2 is later. Do not copy the Environment envelope into the "
        "answer.\n<ENVIRONMENT_FACTS>\n"
        f"{context}\n"
        "</ENVIRONMENT_FACTS>"
    )


def _audited_component_mask(
    state: ChangeState, region: Mapping[str, Any]
) -> np.ndarray:
    """Rebuild the exact Environment-owned component from its canonical seed."""

    kind = str(region.get("audit_kind", "mixed"))
    change = np.asarray(state.change_mask, dtype=bool)
    temporal = np.logical_xor(state.t1_mask, state.t2_mask)
    source = (
        change
        if kind == "present"
        else np.logical_and(temporal, ~change)
        if kind == "missing"
        else np.logical_or(change, temporal)
    )
    if kind == "scan":
        box = region.get("box_normalized_1000")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise StageProtocolError("scan region lacks a canonical box")
        left, top = normalized_point_to_pixel(box[:2], state.image_size)
        right, bottom = normalized_point_to_pixel(box[2:], state.image_size)
        result = np.zeros_like(change, dtype=bool)
        result[top : bottom + 1, left : right + 1] = True
        return result
    seed = region.get("component_seed_normalized_1000")
    if not isinstance(seed, (list, tuple)) or len(seed) != 2:
        raise StageProtocolError("audit region lacks a canonical component seed")
    x, y = normalized_point_to_pixel(seed, state.image_size)
    for component in connected_components(source):
        if component[y, x]:
            return component
    raise StageProtocolError(
        "audit seed does not belong to the authoritative source mask"
    )


_STAGE_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "audit": frozenset({"region_id", "visual_judgment", "diagnosis"}),
    "select": frozenset({"selection"}),
    "evidence": frozenset({"region_id", "visual_judgment"}),
    "candidate_evidence": frozenset({"region_id", "visual_judgment"}),
    "diagnosis": frozenset({"region_id", "diagnosis"}),
    "decision": frozenset({"decision"}),
    "direct": frozenset({"verdict"}),
}


def _extract_stage_json(raw: Any, stage: str) -> Mapping[str, Any]:
    """Select the JSON envelope for ``stage`` instead of the first object."""

    expected = _STAGE_EXPECTED_KEYS.get(stage)
    if expected is None:
        raise StageProtocolError(f"unsupported staged JSON call: {stage!r}")
    candidates = [raw] if isinstance(raw, Mapping) else list(_json_objects(raw))
    for candidate in candidates:
        if isinstance(candidate, Mapping) and frozenset(candidate) == expected:
            return candidate
    keys = [sorted(str(key) for key in item) for item in candidates if isinstance(item, Mapping)]
    raise StageProtocolError(
        f"{stage} response contains no JSON object with exact keys {sorted(expected)}; "
        f"candidate_keys={keys}"
    )


def _extract_json_object(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise StageProtocolError("model response must be JSON text or an object")
    start = raw.find("{")
    if start < 0:
        raise StageProtocolError("model response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as error:
        raise StageProtocolError("model response contains incomplete JSON") from error
    if not isinstance(value, Mapping):
        raise StageProtocolError("model response JSON must be an object")
    return value


def _json_objects(raw: Any):
    if not isinstance(raw, str):
        raise StageProtocolError("model response must be JSON text or an object")
    decoder = json.JSONDecoder()
    cursor = 0
    found = False
    while True:
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if isinstance(value, Mapping):
            found = True
            yield value
    if not found:
        raise StageProtocolError("model response contains no complete JSON object")


def _audit_text(raw: Any, limit: int = 12000) -> str:
    value = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _as_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")


def _delta_only_contour(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    contour_color: tuple[int, int, int] = (0, 255, 255),
) -> Image.Image:
    """Keep original RGB only on the candidate delta and mark its outer contour.

    Pixels outside the authoritative delta are black presentation padding, not
    scene evidence.  The one-pixel cyan ring is drawn outside the delta so the
    original RGB values at every delta pixel remain unchanged.
    """

    source = np.asarray(_as_image(image).convert("RGB"), dtype=np.uint8)
    active = np.asarray(mask, dtype=bool)
    if active.shape != source.shape[:2]:
        raise StageProtocolError("delta-only mask must match RGB image dimensions")
    result = np.zeros_like(source)
    result[active] = source[active]
    if active.any():
        padded = np.pad(active, 1, constant_values=False)
        dilated = np.zeros_like(active, dtype=bool)
        for offset_y in range(3):
            for offset_x in range(3):
                dilated |= padded[
                    offset_y : offset_y + active.shape[0],
                    offset_x : offset_x + active.shape[1],
                ]
        contour = np.logical_and(dilated, ~active)
        result[contour] = np.asarray(contour_color, dtype=np.uint8)
    return Image.fromarray(result, mode="RGB")


def _highlight_mask(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Overlay an authoritative binary delta in yellow without hiding RGB context."""

    result = np.asarray(_as_image(image).convert("RGB"), dtype=np.uint8).copy()
    active = np.asarray(mask, dtype=bool)
    if active.shape != result.shape[:2]:
        raise StageProtocolError("highlight mask must match RGB image dimensions")
    if active.any():
        yellow = np.array([255, 255, 0], dtype=np.float32)
        blended = result[active].astype(np.float32) * 0.35 + yellow * 0.65
        result[active] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def _normalized_crop_box(
    box: list[int] | tuple[int, ...], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert a normalized inclusive box to a provider-safe PIL crop box.

    BaiLian's vision endpoint rejects images whose height or width is below
    11 pixels. Region proposals can legitimately be smaller than that after
    connected-component extraction, so expand only the crop sent to the
    provider while keeping it inside the original image. The proposal's
    normalized coordinates and mask facts remain unchanged.
    """

    return _expand_crop_min_side(
        _normalized_exact_crop_box(box, image_size),
        image_size,
        min_side=11,
    )


def _normalized_exact_crop_box(
    box: list[int] | tuple[int, ...], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert a normalized inclusive box without importing neighboring pixels."""

    width, height = image_size
    x1, y1, x2, y2 = (int(value) for value in box)
    left = max(0, min(width - 1, round(x1 * (width - 1) / 1000)))
    top = max(0, min(height - 1, round(y1 * (height - 1) / 1000)))
    right = max(left + 1, min(width, round(x2 * (width - 1) / 1000) + 1))
    bottom = max(top + 1, min(height, round(y2 * (height - 1) / 1000) + 1))
    return left, top, right, bottom


def _provider_safe_exact_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    *,
    min_side: int = 11,
) -> Image.Image:
    """Keep only authoritative pixels, replicating them if a provider needs 11px."""

    crop = image.crop(crop_box)
    if crop.width >= min_side and crop.height >= min_side:
        return crop
    return crop.resize(
        (max(min_side, crop.width), max(min_side, crop.height)),
        resample=Image.Resampling.NEAREST,
    )


def _expand_crop_min_side(
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_side: int,
) -> tuple[int, int, int, int]:
    """Expand a PIL crop box to a minimum side length without leaving the image."""

    if min_side < 1:
        raise ValueError("min_side must be positive")
    width, height = image_size
    left, top, right, bottom = crop_box
    target_width = min(width, min_side)
    target_height = min(height, min_side)

    if right - left < target_width:
        right = min(width, left + target_width)
        left = max(0, right - target_width)
    if bottom - top < target_height:
        bottom = min(height, top + target_height)
        top = max(0, bottom - target_height)
    return left, top, right, bottom
