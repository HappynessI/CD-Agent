"""Thin adapter around SegAgent's SimpleClick wrapper."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


class SimpleClickAdapter:
    """Replay an interactive click session from an external initial mask."""

    def __init__(self, segmentation_model: Any, segagent_root: str | Path | None = None):
        self.segmentation_model = segmentation_model
        self.segagent_root = Path(segagent_root).resolve() if segagent_root else None

    def _click_classes(self) -> tuple[type[Any], type[Any]]:
        if self.segagent_root is not None:
            root = str(self.segagent_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        module = importlib.import_module(
            "third_party.SimpleClick.isegm.inference.clicker"
        )
        return module.Click, module.Clicker

    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
        click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
    ) -> np.ndarray:
        Click, Clicker = self._click_classes()
        clicker = Clicker()
        # SimpleClick reserves click index zero for an externally supplied mask.
        clicker.click_indx_offset = 1
        predictor = getattr(self.segmentation_model, "predictor", self.segmentation_model)
        predictor.set_input_image(np.asarray(image))
        prev_mask = self._previous_mask_tensor(predictor, initial_mask)

        prediction: np.ndarray | None = None
        clicks = (*click_history, (coordinate, is_positive))
        for click_index, (click_coordinate, click_is_positive) in enumerate(clicks):
            x, y = click_coordinate
            clicker.add_click(
                Click(is_positive=click_is_positive, coords=(int(y), int(x)))
            )
            prediction = predictor.get_prediction(clicker, prev_mask=prev_mask)
            # Match SimpleClick's interactive controller for its first external-mask click.
            if click_index == 0:
                prediction = predictor.get_prediction(clicker, prev_mask=prev_mask)

        if prediction is None:
            raise RuntimeError("SimpleClick session must contain at least the current click")
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = np.asarray(prediction)
        if prediction.dtype == np.bool_:
            return np.array(prediction, copy=True)
        return prediction > 0.49

    @staticmethod
    def _previous_mask_tensor(predictor: Any, initial_mask: np.ndarray) -> Any:
        import torch

        mask = np.asarray(initial_mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError("SimpleClick initial mask must be two-dimensional")
        return torch.as_tensor(
            mask,
            dtype=torch.float32,
            device=getattr(predictor, "device", None),
        ).unsqueeze(0).unsqueeze(0)
