"""Fixed controls execute the same atomic tools as the agent."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from common import names, load_rgb, evaluate, save_mask_png, write_json
from .tools import Toolbox
from . import tools
from .state import Mask, Session
from .tasks import read_tasks, load_pair
from .backends import SegmentBackend

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))


def imc(toolbox, score_thr=.5, direction='any'):
    if direction not in ('any', 'appeared', 'disappeared'):
        raise ValueError('Invalid change direction')
    def execute(name, **args):
        result = toolbox.execute(name, args)
        if not result.public['ok']:
            raise RuntimeError(result.text)
        return result.public
    t1 = execute('segment_text', time='t1', text='building', tile=512, score_thr=score_thr)['inst']
    t2 = execute('segment_text', time='t2', text='building', tile=512, score_thr=score_thr)['inst']
    matched = execute('match_instances', inst_t1=t1, inst_t2=t2)
    sets = [matched['appeared'], matched['disappeared']] if direction == 'any' else [matched[direction]]
    key = execute('to_mask', sets=sets)['mask']
    return toolbox.session.get(key, Mask).data.copy()


def one(arguments):
    split, name, threshold, save, pixel_thr, init_name, direction = arguments
    toolbox = Toolbox(load_rgb(split, name), RUN / 'fixed', segment=SegmentBackend(split, name, pixel_thr=pixel_thr))
    mask = imc(toolbox, threshold, direction)
    if save:
        save_mask_png(RUN / 'init' / init_name / split / (Path(name).stem + '.png'), mask)
    return name, mask


def run(split, procs, threshold=.5, save=False, pixel_thr=.5, init_name='sam3_imc', listed=None, direction='any'):
    todo = [(split, name, threshold, save, pixel_thr, init_name, direction) for name in (listed or names(split))]
    with ProcessPoolExecutor(max_workers=procs) as pool:
        preds = dict(pool.map(one, todo))
    metrics, per = evaluate(preds, split)
    return dict(metrics, per_image_f1=per, score_thr=threshold, pixel_thr=pixel_thr, tile=512, overlap_thr=.25, match_basis=tools.MATCH_BASIS)


def run_manifest(rows, output, direction=None, scope=None, worker=None, pixel_thr=.85, score_thr=.5):
    """Fixed atomic route for explicit images; labels are read only after prediction."""
    import numpy as np
    from PIL import Image
    from .analysis.second_labels import pixel_counts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per, total, seen = {}, np.zeros(3, np.int64), set()
    policies = {(direction or row['change_direction'], scope or row.get('scope')) for row in rows}
    if len(policies) != 1:
        raise ValueError('Use one direction and scope per output directory')
    policy = dict(direction=next(iter(policies))[0], scope=next(iter(policies))[1],
                  pixel_thr=pixel_thr, score_thr=score_thr, tile=512, overlap_thr=.25)
    settings = output / 'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != policy:
        raise ValueError('Output directory has a different frozen configuration')
    write_json(settings, policy)
    for row in rows:
        stem = Path(row['id']).stem
        if stem in seen:
            raise ValueError('Duplicate image: use one subtask manifest')
        seen.add(stem)
        rgb = load_pair(row)
        session = Session(rgb[0].shape[:2], target=row['target'],
            change_direction=direction or row['change_direction'], scope=scope or row.get('scope'))
        segment = SegmentBackend('val', Path(row['image_t1']).name, worker, pixel_thr,
            target=session.target, image_paths=[row['image_t1'], row['image_t2']])
        box = Toolbox(rgb, output / 'tools', segment=segment, session=session)
        mask = imc(box, score_thr, session.change_direction)
        save_mask_png(output / (stem + '.png'), mask)
        if row.get('gt_mask'):
            from .evaluation import load_reference
            truth, valid = load_reference(row['gt_mask'], row.get('gt_ignore_mask'))
            tp, fp, fn = pixel_counts(mask & valid, truth)
            per[stem] = dict(tp=tp, fp=fp, fn=fn)
            total += [tp, fp, fn]
    result = dict(policy, per_image=per, tp=int(total[0]), fp=int(total[1]), fn=int(total[2]))
    write_json(output / 'counts.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('route', choices=['imc', 'tune'])
    parser.add_argument('--split', choices=['val','test','train'], default='val')
    parser.add_argument('--names-file')
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--pixel-thr', type=float, default=.5)
    parser.add_argument('--init-name', default='sam3_imc')
    parser.add_argument('--apply-test', action='store_true', help='Apply frozen val selection, without retuning.')
    parser.add_argument('--direction', choices=['any', 'appeared', 'disappeared'])
    parser.add_argument('--scope', help='Neutral scope card name, without path.')
    parser.add_argument('--tasks', help='One task/direction manifest for explicit source images.')
    parser.add_argument('--image-t1')
    parser.add_argument('--image-t2')
    parser.add_argument('--gt-mask', help='Optional reference for per-image counts, never sent to tools.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sam3-worker')
    args = parser.parse_args()
    if args.tasks or args.image_t1 or args.image_t2:
        if args.route != 'imc' or not args.output_dir:
            parser.error('Explicit images require route imc and --output-dir')
        if args.tasks:
            if args.image_t1 or args.image_t2:
                parser.error('Choose --tasks or an explicit image pair')
            rows = read_tasks(args.tasks)
        else:
            if not (args.image_t1 and args.image_t2):
                parser.error('Both image paths are required')
            rows = [dict(id=Path(args.image_t1).name, image_t1=args.image_t1, image_t2=args.image_t2,
                         target='building', change_direction=args.direction or 'any', scope=args.scope, gt_mask=args.gt_mask)]
        result = run_manifest(rows, args.output_dir, args.direction, args.scope, args.sam3_worker, args.pixel_thr)
        print(json.dumps(result))
        return
    if args.route == 'imc':
        listed = Path(args.names_file).read_text().split() if args.names_file else None
        result = run(args.split, args.procs, save=True, pixel_thr=args.pixel_thr, init_name=args.init_name, listed=listed, direction=args.direction or 'any')
        tag = '' if args.init_name == 'sam3_imc' else '_' + args.init_name
        write_json(RUN / 'fixed' / f'imc{tag}_{args.split}.json', result)
    elif args.direction not in (None, 'any'):
        parser.error('Direction experiments use frozen thresholds; tune only supports the legacy any route.')
    elif args.apply_test:
        frozen = json.loads((RUN / 'fixed' / 'tune_val.json').read_text())
        if frozen['match_basis'] != tools.MATCH_BASIS:
            parser.error('Matching basis differs from the frozen validation configuration.')
        result = run('test', args.procs, frozen['selected_score_thr'])
        write_json(RUN / 'fixed' / 'tune_test.json', result)
    else:
        if args.split != 'val':
            parser.error('Threshold selection is restricted to val; --apply-test uses the frozen selection.')
        scan = [run('val', args.procs, n / 100) for n in range(30, 81, 5)]
        best = max(scan, key=lambda row: row['f1'])
        result = dict(best, selected_score_thr=best['score_thr'], scan=scan)
        write_json(RUN / 'fixed' / 'tune_val.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
