"""Train a weighted-BCE MLP on the image-disjoint second_cv verifier split."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid

import numpy as np
import torch
from torch import nn

from exp6b_common import (
    BACKBONES, SEED, SOURCES, Features, Pool, binary_metrics, classifier, grouped_metrics,
    normalization, predict, run_dir, verifier_split, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--include_pred", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0
            or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        parser.error("Invalid training hyperparameters")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(args.seed)
    train_images, val_images = verifier_split()
    pool = Pool("second_cv")
    features = Features(pool, args.backbone)
    eligible = pool.label >= 0
    if not args.include_pred:
        eligible &= pool.source != SOURCES.index("pred")
    train = eligible & pool.image_mask(train_images)
    val = eligible & pool.image_mask(val_images)
    positive, negative = int((pool.label[train] == 1).sum()), int((pool.label[train] == 0).sum())
    if min(positive, negative) == 0 or len(np.unique(pool.label[val])) != 2:
        raise ValueError("Training and validation must each contain positive and negative pieces")
    out = run_dir(args.backbone, args.include_pred)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("model.pt", "train_log.jsonl", "val_metrics.json")):
        raise FileExistsError(f"{out}: move the previous run before training")
    mean, std = normalization(features, train, args.batch_size)
    model = classifier(features.input_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = negative / positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
    best_ap, best_epoch, best_state, best_scores = -1.0, 0, None, None
    start = time.monotonic()
    run_id = str(uuid.uuid4())
    with (out / "train_log.jsonl").open("w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for idx, x in features.batches(train, args.batch_size, rng):
                tensor = torch.from_numpy((x - mean) / std).to(args.device)
                labels = torch.from_numpy(pool.label[idx].astype(np.float32)).to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tensor).squeeze(1), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is non-finite")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(idx)
                count += len(idx)
            scores = predict(model, features, val, mean, std, args.batch_size, args.device)
            metrics = binary_metrics(pool.label[val], scores[val])
            if metrics["ap"] > best_ap:
                best_ap, best_epoch = metrics["ap"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_scores = scores.copy()
            record = dict(epoch=epoch, loss=loss_sum / count, train_samples=count,
                          val=metrics, best_epoch=best_epoch,
                          seconds=time.monotonic() - start)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(f"epoch {epoch}: loss={record['loss']:.5f} val AP={metrics['ap']:.5f} "
                  f"best={best_epoch}", flush=True)
            if epoch - best_epoch >= args.patience:
                break
    checkpoint = dict(
        run_id=run_id, backbone=args.backbone, include_pred=args.include_pred,
        state_dict=best_state, mean=torch.from_numpy(mean), std=torch.from_numpy(std),
        input_dim=features.input_dim, feature_config=features.config,
        config=vars(args), best_epoch=best_epoch, best_val_ap=best_ap,
        pos_weight=pos_weight, train_images=train_images, val_images=val_images,
    )
    temporary = out / "model.tmp.pt"
    torch.save(checkpoint, temporary)
    temporary.replace(out / "model.pt")
    report = dict(
        run_id=run_id, backbone=args.backbone, include_pred=args.include_pred,
        best_epoch=best_epoch, epochs_run=epoch, selection="validation_AP_on_training_population",
        train=dict(images=len(train_images), n=int(train.sum()), positive=positive,
                   negative=negative, positive_rate=positive / (positive + negative),
                   pos_weight=pos_weight),
        validation_images=len(val_images),
        selection_population=binary_metrics(pool.label[val], best_scores[val]),
        recall=grouped_metrics(pool, best_scores, val),
        feature_config=features.config, input_dim=features.input_dim,
        seconds=time.monotonic() - start,
    )
    if args.include_pred:
        pred = val & (pool.source == SOURCES.index("pred"))
        report["pred"] = binary_metrics(pool.label[pred], best_scores[pred])
    write_json(out / "val_metrics.json", report)
    print(f"saved {out / 'model.pt'} (best epoch {best_epoch})", flush=True)


if __name__ == "__main__":
    main()
