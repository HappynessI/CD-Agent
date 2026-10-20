"""Wrap the local official SCD trainer with fold paths, early stopping and final saves."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext, redirect_stdout
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from exp6_common import CFG, CV, INIT, VENDOR, budget_seconds, read_fold, write_json

sys.path.insert(0, str(VENDOR))
from changedetection.checkpoints import load_encoder_pretrained_weights
from changedetection.datasets import build_dataset
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from changedetection.script.script_utils import get_vssm_kwargs, populate_name_lists
from changedetection.tasks.scd import SCDTrainer


def add_args(parser):
    parser.add_argument('--cv_root', type=Path, default=CV)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--max_iters', type=int, required=True)
    parser.add_argument('--lr_step_size', type=int, default=10000)


def setup(args):
    if args.max_iters < 1 or args.lr_step_size < 1:
        raise ValueError('max_iters and lr_step_size must be positive')
    manifest, fold = read_fold(args.cv_root, args.fold)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size not in (1, 2) or torch.cuda.device_count() != world_size:
        raise RuntimeError('Use one GPU or torchrun with two local GPUs')
    torch.cuda.set_device(local_rank)
    gpu = torch.cuda.get_device_properties(local_rank)
    if 'A800' not in gpu.name or gpu.total_memory < 75 * 2**30:
        raise RuntimeError('Request A800 80GB GPUs for the training/timing job')
    seed = manifest['seed'] + args.fold
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    root = Path(fold['root'])
    official = SimpleNamespace(
        cfg=str(CFG), opts=None, dataset='SECOND', type='train', model_type='MambaSCD_Base',
        train_dataset_path=str(root / 'train'), train_data_list_path=str(root / 'train.txt'),
        test_dataset_path=str(root / 'val'), test_data_list_path=str(root / 'val.txt'),
        encoder_pretrained_path=str(INIT), model_checkpoint_path=None, resume_training_path=None,
        model_param_path=str(root / 'ckpt'), shuffle=True, batch_size=16 // world_size, crop_size=512,
        start_iter=0, cuda=True, max_iters=args.max_iters, learning_rate=1e-4,
        momentum=0.9, weight_decay=5e-3, lr_step_size=args.lr_step_size,
        fold=args.fold, seed=seed, manifest=str(args.cv_root.absolute() / 'manifest.json'),
        rank=rank, local_rank=local_rank, world_size=world_size, global_batch_size=16,
        data_seed=seed + rank,
    )
    populate_name_lists(official, {'train_data_list_path': 'train_data_name_list',
                                   'test_data_list_path': 'test_data_name_list'})
    if official.train_data_name_list != [Path(n).stem for n in fold['train_images']]:
        raise ValueError('Train list differs from manifest')
    if official.test_data_name_list != fold['val_images']:
        raise ValueError('Validation list differs from manifest')
    return manifest, fold, official


class FoldTrainer(SCDTrainer):
    def __init__(self, args):
        # The vendor config loader prints directly; only rank 0 emits setup logs.
        with redirect_stdout(io.StringIO()) if args.rank else nullcontext():
            super().__init__(args)
        if args.world_size > 1:
            self.model = DDP(self.model, device_ids=[args.local_rank], output_device=args.local_rank)

    def _resolve_model_save_path(self):
        return self.args.model_param_path

    def build_train_loader(self):
        if self.args.world_size == 1:
            return super().build_train_loader()
        # Expand to max_iters * GLOBAL batch before partitioning, so both ranks
        # have exactly max_iters batches. Reuse the official dataset/augmentation.
        dataset = build_dataset(
            self.args.dataset, dataset_path=self.args.train_dataset_path,
            data_list=self.args.train_data_name_list, crop_size=self.args.crop_size,
            max_iters=self.args.max_iters, batch_size=self.args.global_batch_size,
            split=self.args.type,
        )
        # The permutation seed must match across ranks for disjoint index slices.
        # Worker augmentation seeds use seed + rank through the loader generator.
        sampler = DistributedSampler(dataset, num_replicas=self.args.world_size,
                                     rank=self.args.rank, shuffle=True, seed=self.args.seed)
        return DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler,
                          num_workers=8, multiprocessing_context='spawn',
                          generator=torch.Generator().manual_seed(self.args.data_seed))

    def build_eval_loaders(self):
        loaders = super().build_eval_loaders() if self.args.rank == 0 else {}
        if self.args.world_size > 1:
            for loader in loaders.values():
                loader.multiprocessing_context = 'spawn'
        return loaders

    def build_model(self, config):
        # The vendor Backbone_VSSM.load_pretrained catches all errors. Load explicitly
        # so an absent/incompatible ImageNet initialization cannot become a random run.
        model = ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None, **get_vssm_kwargs(config))
        info = load_encoder_pretrained_weights(model.encoder, self.args.encoder_pretrained_path)
        missing = [k for k in info['missing_keys'] if not k.startswith('outnorm')]
        unexpected = [k for k in info['unexpected_keys'] if not k.startswith('classifier.')]
        if missing or unexpected or info['mismatched_keys'] or not info['loaded_keys']:
            raise RuntimeError(f'Incompatible ImageNet encoder initialization: {info}')
        self.encoder_load_info = info
        print('ImageNet encoder load:', json.dumps(info), flush=True)
        return model

    def build_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step_size, gamma=0.5)


MICRO_BATCH = 8   # One GPU: 2 x 8 accumulation. Two GPUs: one batch of 8 per rank.


def optimize_step(trainer, batch):
    # Same ordering and full float32 computation as BaseTrainer.training(), with accumulation.
    n = batch[0].shape[0]
    chunks = max(1, (n + MICRO_BATCH - 1) // MICRO_BATCH)
    trainer.optimizer.zero_grad()
    for i in range(chunks):
        part = [x[i * MICRO_BATCH:(i + 1) * MICRO_BATCH] if torch.is_tensor(x) else x for x in batch]
        result = trainer.train_step(part)
        loss = result['loss']
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite training loss')
        (loss / chunks).backward()
    trainer.optimizer.step()
    trainer.after_optimizer_step()
    return result['log_items']


def evaluate(trainer):
    # Rank 0 validates without DDP forward/buffer collectives; rank 1 waits below.
    model = trainer.model
    trainer.model = model.module if isinstance(model, DDP) else model
    trainer.model.eval()
    try:
        results = {name: trainer.evaluate_loader(name, loader) for name, loader in trainer.eval_loaders.items()}
    finally:
        trainer.model = model
        trainer.model.train()
    score = float(trainer.selection_metric(results))
    if not math.isfinite(score):
        raise RuntimeError(f'Nonfinite validation SeK: {results}')
    return score, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    parser.add_argument('--timing_report', type=Path, required=True)
    parser.add_argument('--patience_evals', type=int, default=0,
                        help='Stop after this many validations without SeK improvement; 0 disables')
    args = parser.parse_args()
    if args.patience_evals < 0:
        parser.error('patience_evals must be nonnegative')
    manifest, fold, official = setup(args)
    distributed = official.world_size > 1
    if distributed:
        # Includes time spent waiting for rank-0 validation and checkpoint I/O.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    try:
        train(args, manifest, fold, official)
    finally:
        if distributed:
            dist.destroy_process_group()


def train(args, manifest, fold, official):
    primary = official.rank == 0
    distributed = official.world_size > 1
    timing = json.loads(args.timing_report.read_text())
    if not timing['complete'] or timing['manifest'] != official.manifest:
        raise ValueError('Need a completed timing run on this fold manifest')
    estimate = budget_seconds(manifest, timing, args.max_iters)
    if primary:
        print(f'Single-GPU timing reference including 20% reserve: {estimate / 86400:.3f} GPU days; '
              'not a DDP estimate or budget limit', flush=True)
    root = Path(fold['root'])
    ckpt = root / 'ckpt'
    if ckpt.exists() and any(ckpt.iterdir()):
        raise FileExistsError(f'Review existing run before starting a new one: {ckpt}')
    if distributed:
        dist.barrier(device_ids=[official.local_rank])
    official.patience_evals = args.patience_evals
    official.timing_report = str(args.timing_report.absolute())
    trainer = FoldTrainer(official)
    if primary:
        print(f'{"DDP ready" if distributed else "Single GPU ready"}: '
              f'world_size={official.world_size} backend={"nccl" if distributed else "none"} '
              f'local_batch={official.batch_size} global_batch=16 '
              f'accumulation_steps={official.batch_size // MICRO_BATCH} '
              f'steps_per_rank={len(trainer.train_loader)} '
              f'sampler={type(trainer.train_loader.sampler).__name__} '
              f'data_seeds={[official.seed + r for r in range(official.world_size)]} '
              'batchnorm=plain', flush=True)
        write_json(root / 'train_config.json', dict(args=vars(official),
                   encoder_load=trainer.encoder_load_info, single_gpu_reference_days=estimate / 86400))
        (root / 'model_config.yaml').write_text(trainer.config.dump())
    started = time.perf_counter()
    best_score, best_record, stale = None, None, 0
    trainer.model.train()
    stop = torch.zeros(1, dtype=torch.int32, device=trainer.device)
    with (root / 'training.jsonl').open('w') if primary else nullcontext() as log:
        for step, batch in enumerate(trainer.train_loader, 1):
            losses = optimize_step(trainer, batch)
            if primary and step % 10 == 0:
                print(trainer.format_train_log(step, args.max_iters, losses), flush=True)
            if step % 500 and step != args.max_iters:
                continue
            if primary:
                score, results = evaluate(trainer)
                improved = best_score is None or score > best_score
                if improved:
                    best_score, stale = score, 0
                    best_record = dict(iteration=step, score=score, results=results,
                                       checkpoint_path=str(ckpt / 'best_model.pth'))
                    # Vendor save_training_checkpoint unwraps model.module.state_dict().
                    trainer.save_checkpoint('best_model.pth', step, best_score, best_record)
                else:
                    stale += 1
                trainer.save_checkpoint('latest.pth', step, best_score, best_record)
                row = dict(iteration=step, lr=trainer.optimizer.param_groups[0]['lr'],
                           losses=losses, results=results, best_iteration=best_record['iteration'],
                           stale_evals=stale, elapsed_seconds=time.perf_counter() - started)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
                stop.fill_(bool(args.patience_evals and stale >= args.patience_evals))
            if distributed:
                dist.broadcast(stop, src=0)
            if stop.item():
                break
    if not primary:
        return
    # The final inference artifact is the best internal-validation model.
    (ckpt / 'final.pth').symlink_to('best_model.pth')
    summary = dict(fold=args.fold, completed=True, stopped_iteration=step,
                   max_iters=args.max_iters, early_stopped=step < args.max_iters,
                   best_iteration=best_record['iteration'], best_sek=best_score,
                   final_checkpoint=str(ckpt / 'final.pth'), latest_checkpoint=str(ckpt / 'latest.pth'),
                   selection='maximum internal validation SeK', manifest=official.manifest,
                   elapsed_seconds=time.perf_counter() - started)
    write_json(root / 'training_complete.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
