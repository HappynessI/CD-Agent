"""Time real batch-16 SCD updates and one internal validation; never save weights."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exp6_common import budget_seconds, write_json
from exp6_train_fold import FoldTrainer, add_args, evaluate, optimize_step, setup


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1 or args.max_iters < args.warmup + args.steps:
        parser.error('Need positive warmup/steps and max_iters >= warmup + steps')
    manifest, fold, official = setup(args)
    if args.report.exists():
        raise FileExistsError(args.report)
    trainer = FoldTrainer(official)
    trainer.model.train()
    torch.cuda.synchronize()
    iterator = iter(trainer.train_loader)
    startup = time.perf_counter() - started
    waits, step_times = [], []
    warmup_seconds = 0.0
    complete = True
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup + args.steps):
        # Leave time for reporting/notification before the 30-minute Slurm limit.
        if time.perf_counter() - started > 1500:
            complete = False
            break
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        batch = next(iterator)
        loaded = time.perf_counter()
        optimize_step(trainer, batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        if i >= args.warmup:
            waits.append(loaded - step_start)
            step_times.append(elapsed)
        else:
            warmup_seconds += elapsed
        if (i + 1) % 20 == 0:
            print(f'updates={i + 1} last_step_seconds={elapsed:.4f}', flush=True)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    val_seconds, val_score = None, None
    if complete and time.perf_counter() - started < 1500:
        torch.cuda.synchronize()
        val_start = time.perf_counter()
        val_score, _ = evaluate(trainer)
        torch.cuda.synchronize()
        val_seconds = time.perf_counter() - val_start
    else:
        complete = False
    report = dict(
        complete=complete, manifest=official.manifest, fold=args.fold,
        gpu=torch.cuda.get_device_name(), batch_size=16, crop_size=512, dtype='float32',
        optimizer='AdamW', learning_rate=1e-4, weight_decay=5e-3,
        cfg=official.cfg, encoder_pretrained_path=official.encoder_pretrained_path,
        encoder_load=trainer.encoder_load_info, train_num_workers=16,
        max_iters=args.max_iters, lr_step_size=args.lr_step_size,
        warmup_steps=args.warmup, timed_steps=len(step_times), requested_timed_steps=args.steps,
        warmup_seconds=warmup_seconds,
        mean_step_seconds=sum(step_times) / len(step_times) if step_times else None,
        mean_data_wait_seconds=sum(waits) / len(waits) if waits else None,
        mean_compute_seconds=(sum(step_times) - sum(waits)) / len(step_times) if step_times else None,
        startup_seconds=startup, peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
        validation_images=len(fold['val_images']), validation_seconds=val_seconds,
        validation_seconds_per_image=val_seconds / len(fold['val_images']) if val_seconds else None,
        validation_sek=val_score, elapsed_seconds=time.perf_counter() - started,
        budget_reserve_fraction=0.20, checkpoint_saved=False,
    )
    if complete:
        report['budget_gpu_days'] = {str(n): budget_seconds(manifest, report, n) / 86400
                                    for n in sorted({args.max_iters, 20000, 50000})}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    if not complete:
        raise RuntimeError('Timing exceeded its soft deadline; incomplete report cannot authorize training')


if __name__ == '__main__':
    main()
