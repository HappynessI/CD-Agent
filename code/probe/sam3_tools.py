"""SAM3 construction shared by the zero-shot baseline and the oracle experiment.

Run in conda_envs/omniovcd-env. Mirrors tools/segmentation_worker.py (same checkpoint,
BPE vocabulary, resolution 1008 and autocast) so the baseline matches the old prototype.
"""
from __future__ import annotations

import sys
from contextlib import nullcontext

import torch

CHANG_AGENT = "/guisongxia01/pangchao/wangyihan/Chang-Agent"
CHECKPOINT = "/guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt"
BPE = f"{CHANG_AGENT}/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
RESOLUTION = 1008


def build_processor(inst_interactivity: bool = False):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(bpe_path=BPE, checkpoint_path=CHECKPOINT, device="cuda",
                                   load_from_HF=False, enable_inst_interactivity=inst_interactivity)
    model.to("cuda")
    model._device = torch.device("cuda")
    return Sam3Processor(model, resolution=RESOLUTION, device="cuda")


def autocast():
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def text_adapter(processor):
    sys.path.insert(0, CHANG_AGENT)
    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    # Default mask threshold 0.4, as in the prototype runner.
    return SAM3ProcessorAdapter(processor)
