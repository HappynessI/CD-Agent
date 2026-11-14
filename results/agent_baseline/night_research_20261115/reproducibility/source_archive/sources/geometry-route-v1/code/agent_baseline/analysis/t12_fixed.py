"""GPU route: per-class temporal XOR, then union; freeze on 60 dev images."""
import json
from pathlib import Path
import numpy as np
from common import components, save_mask_png, write_json
from agent_baseline.backends_t13 import WorkerPool
from .probe_t13_common import RUN, parser, selected, inventories, pair_backend, backend_config
from .second_metrics import counts, score, mask
from .make_t12_tasks import t12_truth, TARGETS
from .second_labels import load_labels


def semantic(backend, time, text):
    """Same binary argmax as ProposalTools.segment_semantic; reuse worker cache."""
    return backend('classify', time=time, classes=[text]).argmax(axis=0) == 1


def filter_small(pred, min_area):
    labels, comps = components(pred)
    return np.isin(labels, [c['id'] for c in comps if c['area'] >= min_area])


def route(backend, min_area=0, sam3_building=None):
    masks = []
    for text in TARGETS:
        if text == 'building' and sam3_building is not None:
            masks.append(np.asarray(sam3_building, bool))
        else:
            masks.append(semantic(backend, 't1', text) ^ semantic(backend, 't2', text))
    return filter_small(np.logical_or.reduce(masks), min_area)


def bind(path, settings):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f'Output settings differ: {path}; choose a new directory')
    if not path.exists():
        write_json(path, settings)


def save_prediction(path, prediction):
    path = Path(path)
    temporary = path.with_suffix('.tmp.png')
    save_mask_png(temporary, prediction)
    temporary.replace(path)


def configurations(a, kind):
    if any(x < 0 for x in a.min_areas):
        raise ValueError('min-area must be nonnegative')
    if kind == 't12':
        sources = ['semantic', 'sam3'] if a.sam3_mask_dir else ['semantic']
        return {f'{s}_min{n}': dict(building=s, min_area=n) for s in sources for n in a.min_areas}
    if any(not 0 < x <= 1 for x in a.overlaps):
        raise ValueError('Component overlap must be in (0,1]')
    result = {'unintersected': dict(mode='none', overlap=0., min_area=0)}
    for n in a.min_areas:
        result[f'pixel_min{n}'] = dict(mode='pixel', overlap=0., min_area=n)
        for overlap in a.overlaps:
            result[f'component{overlap:g}_min{n}'] = dict(mode='component', overlap=overlap, min_area=n)
    return result


def main(kind='t12'):
    from .c1t_fixed import intersect, source_mask
    from .make_c1t_tasks import conversion_truth
    p = parser(__doc__)
    p.set_defaults(output_root=RUN)
    p.add_argument('--settings', type=Path, help='Complete dev report; required for val')
    p.add_argument('--min-areas', nargs='+', type=int, default=[0, 16, 64])
    p.add_argument('--sam3-mask-dir', type=Path, required=kind == 'c1t',
                   help='Frozen SECOND any masks for t12; appeared masks for c1t')
    p.add_argument('--overlaps', nargs='+', type=float, default=[.25, .5])
    p.add_argument('--merge', nargs='+', type=Path)
    a = p.parse_args()
    names = selected(a)
    inventory = inventories(a.run_root)
    frozen = json.loads(a.settings.read_text()) if a.settings else None
    if a.phase == 'val' and (frozen is None or frozen.get('phase') != 'dev'
            or not frozen.get('complete') or frozen['names'] != inventory['dev'] or frozen['kind'] != kind):
        p.error('val requires complete matching 60-image dev settings')
    config = backend_config(a, frozen)
    setups = frozen['setups'] if a.phase == 'val' else configurations(a, kind)
    if a.phase == 'val':
        keys = [frozen['selected']]
        if kind == 'c1t':
            keys += ['unintersected']
        setups = {k: setups[k] for k in dict.fromkeys(keys)}
    output = a.output_root/kind/'fixed'
    # Bind the entire split, so all shards and restarts share one immutable route.
    policy = dict(kind=kind, phase=a.phase, backend_config=config, setups=setups,
        names=inventory[a.phase], dataset=str(a.dataset_root.resolve()),
        sam3_mask_dir=str(a.sam3_mask_dir.resolve()) if a.sam3_mask_dir else None,
        semantic_names=['building', 'low vegetation', 'non-vegetated ground'])
    if frozen is not None and a.phase == 'val' and policy['sam3_mask_dir'] != frozen['policy']['sam3_mask_dir']:
        raise ValueError('Use the same SAM3 mask directory as development (both inventories)')
    bind(output/a.phase/'settings.json', policy)
    if a.merge:
        rows = {}
        for path in a.merge:
            report = json.loads(path.read_text())
            if report['policy'] != policy or set(rows) & set(report['per_image']):
                raise ValueError('Mismatched settings or duplicate shard images')
            rows.update(report['per_image'])
        if set(rows) != set(inventory[a.phase]):
            raise ValueError('Incomplete shards')
        names, complete = inventory[a.phase], True
    else:
        rows = {}
        pool = WorkerPool(config, a.cache)
        try:
            for name in names:
                paths = {key: output/key/a.phase/name for key in setups}
                missing = [key for key, path in paths.items() if not path.exists()]
                if missing:
                    backend = pair_backend(pool, a, name)
                    supplied = mask(a.sam3_mask_dir/name) if a.sam3_mask_dir else None
                    if kind == 'c1t':
                        source = source_mask(backend)
                    for key in missing:
                        setting = setups[key]
                        pred = (route(backend, setting['min_area'], supplied if setting['building'] == 'sam3' else None)
                            if kind == 't12' else filter_small(intersect(supplied, source, setting['mode'], setting['overlap']), setting['min_area']))
                        save_prediction(paths[key], pred)
                # Truth is accessed only after prediction and never passed to a backend.
                labels = load_labels(a.dataset_root, name)
                truths = conversion_truth(*labels) if kind == 'c1t' else {'new': t12_truth(*labels)}
                truth = truths['new']
                rows[name] = {}
                before = mask(a.sam3_mask_dir/name, truth.shape) if kind == 'c1t' else None
                for key, path in paths.items():
                    pred = mask(path, truth.shape)
                    diagnostics = dict(predicted=int(pred.sum()), truth_pixels=int(truth.sum()))
                    if before is not None:
                        diagnostics.update(before_pixels=int(before.sum()), before_tp=int((before & truth).sum()),
                            after_tp=int((pred & truth).sum()), all_candidate_tp_cut=bool((before & truth).any() and not (pred & truth).any()))
                    rows[name][key] = dict(counts=counts(pred, truth, truths.get('demolished')), diagnostics=diagnostics)
                print(name, flush=True)
        finally:
            pool.close()
        complete = a.shards == 1
    scores = {key: score([rows[n][key]['counts'] for n in names]) for key in setups}
    report = dict(kind=kind, phase=a.phase, complete=complete, names=names, backend_config=config,
        setups=setups, policy=policy, per_image=rows, scores=scores,
        selection='Maximum pooled dev F1; ties prefer first declared setting; no-intersection is reference only')
    report['diagnostics'] = {key: {field: sum(int(rows[n][key]['diagnostics'].get(field, 0)) for n in names)
        for field in ('predicted', 'truth_pixels', 'before_pixels', 'before_tp', 'after_tp', 'all_candidate_tp_cut')}
        for key in setups}
    if complete and a.phase == 'dev':
        report['selected'] = max((k for k in setups if k != 'unintersected'), key=lambda k: scores[k]['f1_c'])
    elif a.phase == 'val':
        report['selected'] = frozen['selected']
    write_json(output/a.phase/f'report_{a.shard}of{a.shards}{"_merged" if a.merge else ""}.json', report)


if __name__ == '__main__':
    main()
