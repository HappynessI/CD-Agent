"""Prepare image-level SECOND cross-fitting lists and file symlinks; no pixel copies."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from exp6_common import CV, DIRS, SECOND, write_json


def write_list(path, images, stems=False):
    path.write_text(''.join((Path(n).stem if stems else n) + '\n' for n in images))


def link_split(root, images):
    for official, local in DIRS.items():
        target = root / official
        target.mkdir(parents=True)
        for name in images:
            (target / name).symlink_to(SECOND / local / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CV)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1016)
    parser.add_argument('--verifier_val_fraction', type=float, default=0.10)
    parser.add_argument('--detector_val_fraction', type=float, default=0.05)
    args = parser.parse_args()
    if not 2 <= args.k <= 10 or not all(0 < p < 1 for p in (
            args.verifier_val_fraction, args.detector_val_fraction)):
        parser.error('Require 2 <= k <= 10 and validation fractions strictly between 0 and 1')

    images = sorted(p.name for p in (SECOND / 'img1').glob('train_*.png'))
    if len(images) != 2968:
        raise ValueError(f'Expected 2968 SECOND train images, found {len(images)}')
    for local in DIRS.values():
        if {p.name for p in (SECOND / local).glob('train_*.png')} != set(images):
            raise ValueError(f'Unpaired SECOND train files in {local}')

    shuffled = images.copy()
    random.Random(args.seed).shuffle(shuffled)
    heldout_folds = [sorted(shuffled[k::args.k]) for k in range(args.k)]
    verifier_val = sorted(random.Random(args.seed + 1).sample(
        images, round(len(images) * args.verifier_val_fraction)))
    verifier_train = sorted(set(images) - set(verifier_val))
    # Refuse to mix a fresh split with any existing training or predictions.
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    write_list(root / 'all_train.txt', images)
    write_list(root / 'verifier_train.txt', verifier_train)
    write_list(root / 'verifier_val.txt', verifier_val)
    manifest = dict(dataset='second_cv', source=str(SECOND), k=args.k, seed=args.seed,
                    verifier_val_fraction=args.verifier_val_fraction,
                    detector_val_fraction=args.detector_val_fraction,
                    all_images=images, verifier_train_images=verifier_train,
                    verifier_val_images=verifier_val, folds=[])
    for k, heldout in enumerate(heldout_folds):
        pool = sorted(set(images) - set(heldout))
        val = sorted(random.Random(args.seed + 100 + k).sample(
            pool, max(1, round(len(pool) * args.detector_val_fraction))))
        train = sorted(set(pool) - set(val))
        if not train:
            raise ValueError('Detector training split is empty')
        fold_root = root / f'fold{k}'
        for split, names in [('train', train), ('val', val), ('heldout', heldout)]:
            link_split(fold_root / split, names)
            write_list(fold_root / f'{split}.txt', names, stems=(split == 'train'))
        record = dict(fold=k, root=str(fold_root), train_images=train,
                      val_images=val, heldout_images=heldout)
        manifest['folds'].append(record)
        print(f'fold{k}: train={len(train)} val={len(val)} heldout={len(heldout)}', flush=True)
    write_json(root / 'manifest.json', manifest)
    print(f'manifest={root / "manifest.json"}; verifier_val={len(verifier_val)}', flush=True)


if __name__ == '__main__':
    main()
