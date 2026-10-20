"""Experiment 6a check: are out-of-fold SECOND train predictions distributed like test predictions?

For the cross-fitted MambaSCD (preds/second_cv, one held-out fold per model) and the official
MambaSCD on SECOND test (preds/second), report micro pixel F1/P/R and the no-signal miss share
(GT objects split by class transition, detector coverage < 10%, area >= 20 px; the exp1/exp4 rule).
Writes exp6/second_cv/heldout_metrics.json. Run in conda_envs/exp5-env (CPU).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

from common import OUT, PRED, load_label, names, pixel_metrics, write_json
from exp6_common import CV
from objects import gt_label, match


def stats(name, pred_dir):
    gt = load_label("second", name)
    pred = np.asarray(Image.open(pred_dir / "mask" / name)) > 0
    _, _, gobjs, _, _ = match(gt, pred, gt_label("second", name, gt))
    miss = [o["area"] for o in gobjs if o["coverage"] < 0.1 and o["area"] >= 20]
    return [int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()),
            int(gt.sum()), sum(miss), len(miss), int(pred.sum())]


def summarise(rows):
    tp, fp, fn, gt_px, miss_px, miss_n, pred_px = np.sum(rows, axis=0).tolist()
    m = pixel_metrics(tp, fp, fn)
    return {"images": len(rows), "f1": round(100 * m["f1"], 2), "precision": round(100 * m["precision"], 2),
            "recall": round(100 * m["recall"], 2), "no_signal_miss_px_share": round(100 * miss_px / gt_px, 2),
            "no_signal_misses_per_image": round(miss_n / len(rows), 2),
            "changed_px_share_gt": round(100 * gt_px / (len(rows) * 512 * 512), 2),
            "changed_px_share_pred": round(100 * pred_px / (len(rows) * 512 * 512), 2)}


def main():
    manifest = json.loads((CV / "manifest.json").read_text())
    out = {}
    with ProcessPoolExecutor(8) as pool:
        cv_dir = PRED / "second_cv/mambascd_base"
        for fold in manifest["folds"]:
            imgs = fold["heldout_images"]
            out[f"fold{fold['fold']}_heldout"] = summarise(list(pool.map(partial(stats, pred_dir=cv_dir), imgs, chunksize=16)))
        all_cv = [i for f in manifest["folds"] for i in f["heldout_images"]]
        out["train_crossfit_all"] = summarise(list(pool.map(partial(stats, pred_dir=cv_dir), all_cv, chunksize=16)))
        test = names("second")
        out["test_official"] = summarise(list(pool.map(partial(stats, pred_dir=PRED / "second/mambascd_base"), test, chunksize=16)))
    write_json(CV / "heldout_metrics.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
