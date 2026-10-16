"""Small trainable verifier head intended for frozen visual feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VerifierPredictions:
    quality: Any
    error_map_logits: Any
    error_type_logits: Any


def build_verifier_head(
    input_channels: int,
    hidden_channels: int = 128,
    error_types: int = 5,
) -> Any:
    """Build lazily so NumPy-only inference does not require PyTorch."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Verifier training requires the 'train' dependencies") from error

    class FrozenFeatureVerifierHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
            )
            self.error_map = nn.Conv2d(hidden_channels, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.quality = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())
            self.error_type = nn.Linear(hidden_channels, error_types)

        def forward(self, frozen_features: Any) -> VerifierPredictions:
            features = self.encoder(frozen_features)
            pooled = self.pool(features).flatten(1)
            return VerifierPredictions(
                quality=self.quality(pooled).squeeze(1),
                error_map_logits=self.error_map(features),
                error_type_logits=self.error_type(pooled),
            )

    return FrozenFeatureVerifierHead()


def verifier_loss(predictions: VerifierPredictions, targets: dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as functional

    quality_loss = functional.mse_loss(predictions.quality, targets["quality"].float())
    error_target = targets["error_map"].float()
    bce = functional.binary_cross_entropy_with_logits(
        predictions.error_map_logits, error_target
    )
    probability = torch.sigmoid(predictions.error_map_logits)
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * error_target).sum(dim=reduce_dims)
    dice_loss = 1 - ((2 * intersection + 1) / (
        probability.sum(dim=reduce_dims) + error_target.sum(dim=reduce_dims) + 1
    )).mean()
    classification = (
        functional.cross_entropy(predictions.error_type_logits, targets["error_type"])
    )
    return quality_loss + bce + dice_loss + classification
