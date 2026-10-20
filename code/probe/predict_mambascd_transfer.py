"""Transfer the official SECOND MambaSCD binary head to lsSCD/SCSCD7 test.

No training, target-domain normalization fitting, resizing, TTA or semantic-head
remapping. Run in vlm-vs-onejev-env on a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import PRED, WORKSPACE, load_label, load_rgb, names, pixel_metrics, save_mask, write_json
from exp6_common import CFG, VENDOR

CHECKPOINT = WORKSPACE / "Vlm_vs_OneJev/checkpoints/MambaSCD_Base_SECOND_SeK_0.2292.pth"
EXPECTED = {"lsscd": 480, "scscd7": 322}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; requires a separate --output")
    args = parser.parse_args()
    formal = PRED / args.dataset / "mambascd_base"
    out = args.output or formal
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit and out.resolve() == formal.resolve():
        parser.error("Smoke runs require --output outside the formal prediction directory")
    items = names(args.dataset)
    if len(items) != EXPECTED[args.dataset]:
        raise ValueError(f"{args.dataset}: expected {EXPECTED[args.dataset]} test pairs, found {len(items)}")
    items = items[:args.limit or None]
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    sys.path.insert(0, str(VENDOR))
    from changedetection.configs.config import get_config
    from changedetection.datasets.imutils import normalize_img, to_channel_first
    from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
    from changedetection.script.script_utils import get_vssm_kwargs, load_checkpoint

    cfg = get_config(SimpleNamespace(cfg=str(CFG), opts=None))
    model = ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None, **get_vssm_kwargs(cfg))
    # Reuse the loader and completeness rule of Vlm_vs_OneJev/experiment/detectors.py.
    load_info = load_checkpoint(model, str(args.checkpoint))
    missing_parameters = set(load_info["missing_keys"]) & set(dict(model.named_parameters()))
    if missing_parameters or load_info["mismatched_keys"] or load_info["unexpected_keys"]:
        raise RuntimeError(f"Incomplete or incompatible MambaSCD checkpoint: {load_info}")
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    (out / "prob").mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report_path.unlink(missing_ok=True)
    counts = np.zeros(3, np.int64)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, name in enumerate(items):
            pair = []
            for view in ("t1", "t2"):
                image = load_rgb(args.dataset, name, view)
                if image.shape != (512, 512, 3):
                    raise ValueError(f"{name}/{view}: expected RGB 512x512, got {image.shape}")
                array = np.ascontiguousarray(to_channel_first(normalize_img(image)))
                pair.append(torch.from_numpy(array)[None].cuda())
            cd, _, _ = model(*pair)
            prob = cd.softmax(1)[0, 1].float().cpu().numpy()
            pred = prob > 0.5
            np.save(out / "prob" / (Path(name).stem + ".npy"), prob.astype(np.float16))
            save_mask(out / "mask" / name, pred)
            truth = load_label(args.dataset, name)
            counts += [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]
            if (index + 1) % 50 == 0:
                print(f"{index + 1}/{len(items)}", flush=True)
    report = {
        "dataset": args.dataset, "split": "test", "detector": "mambascd_base",
        "checkpoint": str(args.checkpoint), "checkpoint_training_dataset": "SECOND train",
        "checkpoint_loading": load_info, "cfg": str(CFG),
        "images": len(items), "image_names": items, "full_test": len(items) == EXPECTED[args.dataset],
        "threshold": 0.5, "threshold_before_float16": True, "probability": "softmax(cd)[:,1]",
        "normalization": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]},
        "precision": "float32", "tta": False, "target_training": False,
        "metrics": pixel_metrics(*map(int, counts)), "seconds": time.perf_counter() - started,
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
