"""Plot V1 learning curves (weighted BCE and AUC, train subset vs validation) from train_log.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp6b_common import ROOT  # noqa: E402

NAMES = {"none": "none (metadata only)", "sam_vith": "SAM ViT-H", "dinov3": "DINOv3",
         "remoteclip": "RemoteCLIP"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, default=ROOT / "verifier/curve_runs")
    parser.add_argument("--runs", nargs="+", default=list(NAMES))
    parser.add_argument("--output", type=Path, default=ROOT / "verifier/curve_runs/v1_learning_curves.png")
    args = parser.parse_args()
    fig, axes = plt.subplots(2, len(args.runs), figsize=(4.2 * len(args.runs), 7), squeeze=False)
    summary = {}
    for col, run in enumerate(args.runs):
        log = [json.loads(line) for line in (args.runs_dir / run / "train_log.jsonl").open()]
        epoch = [r["epoch"] for r in log]
        series = {
            "train (dropout, epoch mean)": [r["loss"] for r in log],
            "train subset (eval mode)": [r["train_eval"]["loss"] for r in log],
            "validation": [r["val"]["loss"] for r in log],
        }
        ax = axes[0, col]
        for label, values in series.items():
            ax.plot(epoch, values, marker=".", label=label)
        ax.set_title(NAMES.get(run, run))
        ax.set_ylabel("weighted BCE")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.plot(epoch, [r["train_eval"]["auc"] for r in log], marker=".", label="train subset AUC")
        ax.plot(epoch, [r["val"]["auc"] for r in log], marker=".", label="validation AUC")
        best = max(log, key=lambda r: r["val"]["ap"])["epoch"]
        for a in axes[:, col]:
            a.axvline(best, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("AUC")
        ax.grid(alpha=0.3)
        summary[run] = [dict(epoch=r["epoch"], train_loss=r["loss"], train_eval_loss=r["train_eval"]["loss"],
                             val_loss=r["val"]["loss"], train_eval_auc=r["train_eval"]["auc"],
                             val_auc=r["val"]["auc"], val_ap=r["val"]["ap"]) for r in log]
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("V1 verifier learning curves on second_cv (dashed: best validation AP epoch)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
