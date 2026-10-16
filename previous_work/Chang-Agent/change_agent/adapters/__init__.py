"""Adapters for the independently installed upstream model environments."""

from .omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from .qwen3vl_adapter import GroundingModelQwen3VL
from .qwen3vl_verifier import Qwen3VLZeroShotVerifier
from .staged_verifier import StagedQwenVerifier
from .direct_verifier import DirectQwenVerifier
from .stage_backends import BailianQwen3VLStageBackend, LocalQwen3VLStageBackend
from .bailian_adapter import BailianGroundingModelQwen3VL
from .sam3_adapter import SAM3ProcessorAdapter
from .segagent_adapter import SimpleClickAdapter
from .subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
    SubprocessSAM3Initializer,
)

__all__ = [
    "GroundingModelQwen3VL",
    "Qwen3VLZeroShotVerifier",
    "StagedQwenVerifier",
    "DirectQwenVerifier",
    "LocalQwen3VLStageBackend",
    "BailianQwen3VLStageBackend",
    "BailianGroundingModelQwen3VL",
    "MaskPairProcessor",
    "OmniOVCDAdapter",
    "SAM3ProcessorAdapter",
    "SimpleClickAdapter",
    "SubprocessBoxBackend",
    "SubprocessPointBackend",
    "SubprocessSAM3Initializer",
]
