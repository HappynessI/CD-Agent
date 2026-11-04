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
from .state import Mask
from .backends import SegmentBackend

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))


def imc(toolbox, score_thr=.5):
    def execute(name, **args):
        result = toolbox.execute(name, args)
        if not result.public['ok']:
            raise RuntimeError(result.text)
        return result.public
    t1 = execute('segment_text', time='t1', text='building', tile=512, score_thr=score_thr)['inst']
    t2 = execute('segment_text', time='t2', text='building', tile=512, score_thr=score_thr)['inst']
    matched = execute('match_instances', inst_t1=t1, inst_t2=t2)
    key = execute('to_mask', sets=[matched['appeared'], matched['disappeared']])['mask']
    return toolbox.session.get(key, Mask).data.copy()


def one(arguments):
    split, name, threshold, save, pixel_thr, init_name = arguments
    toolbox = Toolbox(load_rgb(split, name), RUN / 'fixed', segment=SegmentBackend(split, name, pixel_thr=pixel_thr))
    mask = imc(toolbox, threshold)
    if save:
        save_mask_png(RUN / 'init' / init_name / split / (Path(name).stem + '.png'), mask)
    return name, mask


def run(split, procs, threshold=.5, save=False, pixel_thr=.5, init_name='sam3_imc', listed=None):
    todo = [(split, name, threshold, save, pixel_thr, init_name) for name in (listed or names(split))]
    with ProcessPoolExecutor(max_workers=procs) as pool:
        preds = dict(pool.map(one, todo))
    metrics, per = evaluate(preds, split)
    return dict(metrics, per_image_f1=per, score_thr=threshold, pixel_thr=pixel_thr, tile=512, overlap_thr=.25, match_basis=tools.MATCH_BASIS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('route', choices=['imc', 'tune'])
    parser.add_argument('--split', choices=['val','test','train'], default='val')
    parser.add_argument('--names-file')
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--pixel-thr', type=float, default=.5)
    parser.add_argument('--init-name', default='sam3_imc')
    parser.add_argument('--apply-test', action='store_true', help='Apply frozen val selection, without retuning.')
    args = parser.parse_args()
    if args.route == 'imc':
        listed = Path(args.names_file).read_text().split() if args.names_file else None
        result = run(args.split, args.procs, save=True, pixel_thr=args.pixel_thr, init_name=args.init_name, listed=listed)
        tag = '' if args.init_name == 'sam3_imc' else '_' + args.init_name
        write_json(RUN / 'fixed' / f'imc{tag}_{args.split}.json', result)
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
