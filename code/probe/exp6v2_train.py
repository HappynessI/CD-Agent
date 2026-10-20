"""Single-GPU, bf16 language-LoRA SFT; select an adapter by validation AUC."""
from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from exp6b_common import ROOT, SEED, verifier_split, write_json
from exp6v2_common import (
    MODEL, input_batch, load_model, read_rows, response_logps, save_adapter,
    target_ids, validation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a", "b"), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "v2/data")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=500, help="Optimizer steps")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0, help="Smoke only; requires --no-save")
    parser.add_argument("--no-save", action="store_true", help="Smoke: never write an adapter")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if min(args.rank, args.alpha, args.epochs, args.grad_accum, args.eval_every, args.max_new_tokens) < 1:
        parser.error("Integer hyperparameters must be positive")
    if args.lr <= 0 or args.max_steps < 0 or args.val_limit < 0:
        parser.error("Invalid learning rate or limits")
    if args.val_limit and not args.no_save:
        parser.error("--val-limit is only for --no-save smoke runs")
    out = args.out or ROOT / f"v2/{args.variant}"
    out.mkdir(parents=True, exist_ok=False)
    train = read_rows(args.data / "train.jsonl", "train")
    val = read_rows(args.data / "val.jsonl", "val")
    if args.val_limit:
        val = random.Random(args.seed).sample(val, min(args.val_limit, len(val)))
    train_images, val_images = verifier_split()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    processor, model = load_model(args.model, trainable=True, rank=args.rank, alpha=args.alpha)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, round(0.03 * total_steps))
    run_id = f"v2{args.variant}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = dict(run_id=run_id, variant=args.variant, model=str(args.model.resolve()),
                    train_images=train_images, val_images=val_images, config=config,
                    scoring=("answer_first_token_logit_difference" if args.variant == "a"
                             else "generated_reason_then_answer_first_token_logit_difference"),
                    trainable_parameters=sum(p.numel() for p in parameters),
                    train_samples=len(train), validation_samples=len(val))
    write_json(out / "config.json", metadata)
    start = time.monotonic()
    step, best_auc, best_step = 0, -1.0, None
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for offset in range(0, len(order), args.grad_accum):
            model.train()
            model.get_base_model().model.visual.eval()
            optimizer.zero_grad(set_to_none=True)
            token_count, loss_sum = 0, 0.0
            for index in order[offset:offset + args.grad_accum]:
                row = train[index]
                batch = input_batch(processor, row, args.variant)
                response = target_ids(processor, row, args.variant)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = -response_logps(model, batch, response).sum()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                token_count += len(response)
                loss_sum += float(loss.detach())
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(token_count)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            step += 1
            scale = step / warmup if step <= warmup else max(0.0, (total_steps - step + 1) / max(1, total_steps - warmup))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.step()
            print(f"step={step}/{total_steps} loss={loss_sum / token_count:.5f}", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                metrics, _ = validation(processor, model, val, args.variant, args.max_new_tokens)
                history.append(dict(step=step, epoch=epoch + 1, loss=loss_sum / token_count, metrics=metrics))
                if metrics["auc"] > best_auc:
                    best_auc, best_step = metrics["auc"], step
                    if not args.no_save:
                        save_adapter(model, processor, out, {
                            **metadata, "best_step": step, "validation": metrics,
                            "elapsed_seconds": time.monotonic() - start})
                write_json(out / "validation_history.json", history)
            if step == total_steps:
                break
        if step == total_steps:
            break
    torch.cuda.synchronize()
    write_json(out / "training_report.json", dict(
        **metadata, optimizer_steps=step, best_step=best_step, best_validation_auc=best_auc,
        saved_adapter=not args.no_save, elapsed_seconds=time.monotonic() - start,
        peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3))


if __name__ == "__main__":
    main()
