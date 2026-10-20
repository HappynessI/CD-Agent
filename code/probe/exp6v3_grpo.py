"""Optional single-GPU GRPO from V2b, gated by V2a/V2b validation AUC."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from exp6b_common import ROOT, SEED, verifier_split, write_json
from exp6v2_common import (
    input_batch, load_model, read_adapter, read_rows, response_logps, save_adapter, validation,
)


def reward(text, label, format_bonus):
    match = re.search(r"(?:^|\n)答案：([是否])\s*$", text)
    correct = bool(match and (match.group(1) == "是") == bool(label))
    formatted = bool(match and text.count("答案：") == 1 and text[:match.start()].strip())
    return float(correct) + format_bonus * float(formatted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2a-adapter", type=Path, required=True)
    parser.add_argument("--v2b-adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "v2/data")
    parser.add_argument("--out", type=Path, default=ROOT / "v3")
    parser.add_argument("--min-auc-gain", type=float, default=0.02)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200, help="Sample groups, including zero-advantage groups")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--format-bonus", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-hours", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-save", action="store_true", help="Smoke: no adapter checkpoints")
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.steps, args.ppo_epochs, args.eval_every, args.max_new_tokens) < 1:
        parser.error("Sizes and step counts must be positive")
    if (args.group_size < 2 or not 0 < args.clip < 1 or args.beta < 0 or args.lr <= 0
            or args.format_bonus < 0 or args.min_auc_gain <= 0 or not 0 < args.max_hours <= 6):
        parser.error("Invalid GRPO hyperparameters or time budget")
    a, b = read_adapter(args.v2a_adapter, "a"), read_adapter(args.v2b_adapter, "b")
    if (a["config"]["data"] != b["config"]["data"] or a["validation_samples"] != b["validation_samples"]
            or Path(b["config"]["data"]).resolve() != args.data.resolve()):
        raise ValueError("Compare V2a/V2b on the same frozen data sample")
    gain = b["validation"]["auc"] - a["validation"]["auc"]
    if gain < args.min_auc_gain:
        raise ValueError(f"V2b validation AUC gain {gain:.4f} < {args.min_auc_gain}; V3 not enabled")
    if args.max_new_tokens != b["config"]["max_new_tokens"]:
        parser.error("Keep V2b's generation limit")
    args.out.mkdir(parents=True, exist_ok=False)
    train = read_rows(args.data / "train.jsonl", "train")
    val = read_rows(args.data / "val.jsonl", "val")
    rng = random.Random(args.seed)
    train = rng.sample(train, min(args.train_size, len(train)))
    val = random.Random(args.seed + 1).sample(val, min(args.val_size, len(val)))
    torch.manual_seed(args.seed)
    start = time.monotonic()
    deadline = start + args.max_hours * 3600
    processor, policy = load_model(b["model"], args.v2b_adapter, trainable=True)
    _, reference = load_model(b["model"], args.v2b_adapter)
    reference.requires_grad_(False)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    run_id = f"v3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    train_images, val_images = verifier_split()
    metadata = dict(
        run_id=run_id, variant="b", model=b["model"], parent_run_id=b["run_id"],
        train_images=train_images, val_images=val_images, validation_samples=len(val),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        validation_sample=[dict(image=r["image"], piece_id=r["piece_id"]) for r in val],
        training_sample=[dict(image=r["image"], piece_id=r["piece_id"]) for r in train],
        gate=dict(v2a_run=a["run_id"], v2b_run=b["run_id"], validation_auc_gain=gain,
                  min_auc_gain=args.min_auc_gain))
    write_json(args.out / "config.json", metadata)
    initial, _ = validation(processor, policy, val, "b", args.max_new_tokens)
    best_auc = initial["auc"]
    best_step, updates, completed_groups, zero_advantage_groups = 0, 0, 0, 0
    validation_seconds = initial["seconds"]
    if not args.no_save:
        save_adapter(policy, processor, args.out, dict(
            **metadata, best_step=0, validation=initial, retained_initial_v2b=True))
    max_group_seconds, last_evaluated = 0.0, 0
    stop_reason = "step_limit"
    with (args.out / "training.jsonl").open("x") as handle:
        for step in range(1, args.steps + 1):
            reserve = 1.5 * (validation_seconds + max_group_seconds) + 120
            if time.monotonic() + reserve >= deadline:
                stop_reason = "time_budget"
                break
            group_start = time.monotonic()
            row = rng.choice(train)
            batch = input_batch(processor, row, "b")
            prompt_length = batch["input_ids"].shape[1]
            responses, rewards, texts = [], [], []
            old_logps, reference_logps = [], []
            policy.eval()
            with torch.no_grad():
                for _ in range(args.group_size):
                    # Sequential rollouts avoid G-fold vision activations. No top-k/p truncation:
                    # sampled distribution is the same one used for the PPO likelihood ratio.
                    generated = policy.generate(
                        **batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                        repetition_penalty=1.0, max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id)
                    response = generated[0, prompt_length:].tolist()
                    text = processor.tokenizer.decode(response, skip_special_tokens=True)
                    texts.append(text)
                    responses.append(response)
                    rewards.append(reward(text, row["label"], args.format_bonus))
                    old_logps.append(response_logps(policy, batch, response).detach())
                    reference_logps.append(response_logps(reference, batch, response).detach())
            values = torch.tensor(rewards, device="cuda:0")
            std = values.std(unbiased=False)
            advantages = (values - values.mean()) / (std + 1e-6)
            group_loss, group_kl = 0.0, 0.0
            if float(std) == 0:
                zero_advantage_groups += 1
            else:
                for _ in range(args.ppo_epochs):
                    policy.train()
                    policy.get_base_model().model.visual.eval()
                    optimizer.zero_grad(set_to_none=True)
                    for i, response in enumerate(responses):
                        current = response_logps(policy, batch, response)
                        ratio = (current - old_logps[i]).exp()
                        unclipped = ratio * advantages[i]
                        clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantages[i]
                        # Nonnegative sampled KL estimator exp(log ref - log pi) - 1 - log ref + log pi.
                        delta = reference_logps[i] - current
                        kl = delta.exp() - 1 - delta
                        loss = (-torch.minimum(unclipped, clipped) + args.beta * kl).mean() / args.group_size
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite GRPO objective")
                        loss.backward()
                        group_loss += float(loss.detach())
                        group_kl += float(kl.detach().mean()) / args.group_size
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
            torch.cuda.synchronize()
            max_group_seconds = max(max_group_seconds, time.monotonic() - group_start)
            completed_groups = step
            record = dict(step=step, image=row["image"], piece_id=row["piece_id"], rewards=rewards,
                          responses=texts, response_tokens=[len(r) for r in responses],
                          loss=group_loss, kl=group_kl, updates=updates,
                          elapsed_seconds=time.monotonic() - start)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _ = validation(processor, policy, val, "b", args.max_new_tokens)
                validation_seconds = max(validation_seconds, metrics["seconds"])
                record["validation"] = metrics
                last_evaluated = step
                if metrics["auc"] > best_auc:
                    best_auc, best_step = metrics["auc"], step
                    if not args.no_save:
                        save_adapter(policy, processor, args.out, dict(
                            **metadata, best_step=step, validation=metrics, retained_initial_v2b=False))
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"group={step} rewards={rewards} updates={updates}", flush=True)
        if completed_groups > last_evaluated and time.monotonic() + 1.25 * validation_seconds + 60 < deadline:
            metrics, _ = validation(processor, policy, val, "b", args.max_new_tokens)
            if metrics["auc"] > best_auc:
                best_auc, best_step = metrics["auc"], completed_groups
                if not args.no_save:
                    save_adapter(policy, processor, args.out, dict(
                        **metadata, best_step=best_step, validation=metrics, retained_initial_v2b=False))
            handle.write(json.dumps(dict(final_validation=metrics, step=completed_groups)) + "\n")
    write_json(args.out / "training_report.json", dict(
        run_id=run_id, completed_groups=completed_groups, optimizer_updates=updates,
        zero_advantage_groups=zero_advantage_groups, best_step=best_step, best_validation_auc=best_auc,
        initial_validation_auc=initial["auc"], retained_initial_v2b=best_step == 0,
        stop_reason=stop_reason, saved_adapter=not args.no_save,
        elapsed_seconds=time.monotonic() - start,
        peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3))


if __name__ == "__main__":
    main()
