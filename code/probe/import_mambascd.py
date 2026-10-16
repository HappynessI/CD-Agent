"""SECOND detector: import MambaSCD-Base binary-change probabilities from Vlm_vs_OneJev.

Those were produced by the official checkpoint (Vlm_vs_OneJev/checkpoints/MambaSCD_Base_SECOND_
SeK_0.2292.pth) on all 1694 SECOND test pairs; the binary softmax change head is the official
prediction and its mask equals prob > 0.5. Re-saved here in the probe layout (float16 prob + mask).
Reference micro F1 from Vlm_vs_OneJev/reports/实验报告.md: 0.741652.
"""
from __future__ import annotations

import numpy as np

from common import PRED, WORKSPACE, load_label, names, pixel_metrics, save_mask, write_json

SOURCE = WORKSPACE / "Vlm_vs_OneJev/outputs/formal/SECOND/probabilities"


def main():
    out = PRED / "second" / "mambascd_base"
    (out / "prob").mkdir(parents=True, exist_ok=True)
    counts = np.zeros(3, np.int64)
    items = names("second")
    for name in items:
        prob = np.load(SOURCE / name.replace(".png", ".npy")).astype(np.float32)
        np.save(out / "prob" / name.replace(".png", ".npy"), prob.astype(np.float16))
        pred = prob > 0.5
        save_mask(out / "mask" / name, pred)
        truth = load_label("second", name)
        counts += [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]
    report = {"dataset": "second", "detector": "mambascd_base", "images": len(items), "threshold": 0.5,
              "source": str(SOURCE), "metrics": pixel_metrics(*map(int, counts)), "reference_micro_f1": 0.741652}
    write_json(out / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()
