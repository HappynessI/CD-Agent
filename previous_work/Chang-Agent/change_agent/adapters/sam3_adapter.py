"""Concrete adapter for OmniOVCD's ``SAM3ImageProcessor`` public API."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class SAM3ProcessorAdapter:
    """Use SAM3 text prompts for initialization and geometric prompts for boxes.

    The heavy SAM3 model/processor is constructed in the OmniOVCD environment and
    injected here. No private encoder state is exposed to the Agent.
    """

    def __init__(self, processor: Any, mask_threshold: float = 0.4):
        self.processor = processor
        self.mask_threshold = mask_threshold

    def initialize_masks(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        t1_mask, t1_evidence = self.segment_text(t1_image, query)
        t2_mask, t2_evidence = self.segment_text(t2_image, query)
        evidence = {
            "sam3_t1": t1_evidence,
            "sam3_t2": t2_evidence,
        }
        confidence1 = t1_evidence.get("confidence_map")
        confidence2 = t2_evidence.get("confidence_map")
        if confidence1 is not None and confidence2 is not None:
            evidence["change_confidence"] = np.maximum(confidence1, confidence2)
        return t1_mask, t2_mask, evidence

    def segment_text(
        self, image: np.ndarray, query: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one public text-prompt pass for staged/low-memory runtimes."""

        return self._text_segment(image, query)

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        state = self.processor.set_image(self._to_pil(image))
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=query, state=state)
        state = self.processor.add_geometric_prompt(
            box=list(box_cxcywh_normalized), label=True, state=state
        )
        return self._mask_from_state(state, image.shape[:2])[0]

    def _text_segment(
        self, image: np.ndarray, query: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.processor.set_image(self._to_pil(image))
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=query, state=state)
        mask, confidence = self._mask_from_state(state, image.shape[:2])
        evidence: dict[str, Any] = {"confidence_map": confidence}
        for key in (
            "semantic_mask_logits",
            "masks_logits",
            "masks",
            "backbone_fpn",
            "fpn_features",
        ):
            if key in state:
                # These are selected diagnostics only; full transformer activations
                # and attention maps intentionally remain transient.
                value = state[key]
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        array = self._numpy(subvalue)
                        if array.dtype != object:
                            evidence[f"{key}.{subkey}"] = array
                else:
                    array = self._numpy(value)
                    if array.dtype != object:
                        evidence[key] = array
        for key in ("presence_score", "object_score", "scores"):
            if key in state:
                evidence[key] = self._numpy(state[key])
        return mask, evidence

    def _mask_from_state(
        self, state: dict[str, Any], shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = shape
        score_maps: list[np.ndarray] = []
        has_mask_output = False
        semantic = state.get("semantic_mask_logits")
        if semantic is not None:
            has_mask_output = True
            semantic_array = self._numpy(semantic)
            if semantic_array.size:
                score_maps.append(self._collapse_score_map(semantic_array))

        instances = state.get("masks_logits")
        if instances is not None:
            has_mask_output = True
            instance_array = self._numpy(instances).astype(float)
            if instance_array.size:
                while instance_array.ndim > 3:
                    instance_array = instance_array.squeeze(1)
                instance_array = self._as_probability(instance_array)
                object_scores = state.get("object_score")
                if object_scores is not None:
                    weights = self._numpy(object_scores).astype(float).reshape(-1, 1, 1)
                    if len(weights) == len(instance_array):
                        instance_array = instance_array * weights
                score_maps.append(instance_array.max(axis=0))

        if not score_maps:
            masks = state.get("masks")
            if masks is not None:
                has_mask_output = True
                masks_array = self._numpy(masks).astype(float)
                if masks_array.size:
                    score_maps.append(self._collapse_score_map(masks_array))
            if not has_mask_output:
                raise KeyError("SAM3 state contains no semantic/instance mask output")

        array = (
            np.maximum.reduce(score_maps)
            if score_maps
            else np.zeros((height, width), dtype=float)
        )
        presence = state.get("presence_score")
        if presence is not None:
            presence_value = float(np.asarray(self._numpy(presence), dtype=float).max(initial=0))
            array = array * np.clip(presence_value, 0, 1)
        if array.shape != (height, width):
            image = Image.fromarray(array.astype(np.float32), mode="F")
            array = np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR))
        mask = array > self.mask_threshold
        confidence = np.clip(array, 0, 1)
        return mask.astype(bool), confidence.astype(np.float32)

    @classmethod
    def _collapse_score_map(cls, value: np.ndarray) -> np.ndarray:
        array = cls._as_probability(np.asarray(value, dtype=float))
        while array.ndim > 2:
            array = array.max(axis=0)
        return array

    @staticmethod
    def _as_probability(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.size and (float(array.min()) < 0 or float(array.max()) > 1):
            return 1.0 / (1.0 + np.exp(-np.clip(array, -30, 30)))
        return np.clip(array, 0, 1)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            # NumPy has no native PyTorch BF16 conversion path. Evidence is
            # diagnostic-only, so promote at this serialization boundary.
            value = value.float()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
