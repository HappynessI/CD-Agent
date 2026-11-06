"""Line A: SECOND segmentation zoom test on the 60 dev images (not the 128 reported images).

prep  (exp5-env + HTTP SAM3 worker with run root D): write 2x bilinear images to D/zoom/img{1,2}/, then ask the
      worker for 'building', tile 512, pixel .85, conf .1 (same call as second_prepare).
score (exp5-env): baseline (512 cache) and zoom (1024 cache, masks reduced to 512 by 2x2 mean >= .5):
      directed fixed route F1 (new, demolished) and SAM3 building P/R/IoU per date in labelled pixels only.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

R = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
D = R / 'second_dev'
DATA = Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/SECOND')
names = (D / 'names.txt').read_text().split()


def zoom_path(part, name):
    return D / 'zoom' / part / name


def prep(shard, shards, worker):
    from agent_baseline.backends import SegmentBackend
    for name in names[shard::shards]:
        paths = [str(zoom_path('img1', name)), str(zoom_path('img2', name))]
        for part, time in (('img1', 't1'), ('img2', 't2')):
            out = zoom_path(part, name)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                im = Image.open(DATA / part / name).convert('RGB')
                im.resize((im.width * 2, im.height * 2), Image.BILINEAR).save(out)
        backend = SegmentBackend('val', name, worker, .85, image_paths=paths)
        for time in ('t1', 't2'):
            _, meta = backend(time, 'building', 512, .5)
        print(name, flush=True)


def down(mask):
    h, w = mask.shape
    return mask.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)) >= .5


def f1(c):
    tp, fp, fn = c
    return dict(f1=200 * tp / max(1, 2 * tp + fp + fn), p=100 * tp / max(1, tp + fp), r=100 * tp / max(1, tp + fn),
                iou=100 * tp / max(1, tp + fp + fn), tp=int(tp), fp=int(fp), fn=int(fn))


def score(output):
    os.environ['AGENT_RUN'] = str(D)
    from agent_baseline.fixed_routes import imc
    from agent_baseline.tools import Toolbox
    from agent_baseline.state import Session
    from agent_baseline.backends import SegmentBackend
    from agent_baseline.analysis.second_labels import load_labels, building_truth, pixel_counts
    from agent_baseline import instances
    tot = {m: {k: np.zeros(3, np.int64) for k in ('appeared', 'disappeared', 'seg_t1', 'seg_t2')} for m in ('base', 'zoom')}
    per = {}
    for name in names:
        l1, l2 = load_labels(DATA, name)
        truth = building_truth(l1, l2)
        labelled = l1 > 0
        per[name] = {}
        for mode in ('base', 'zoom'):
            paths = [str(DATA / 'img1' / name), str(DATA / 'img2' / name)] if mode == 'base' else \
                    [str(zoom_path('img1', name)), str(zoom_path('img2', name))]
            rgb = [np.asarray(Image.open(p).convert('RGB')) for p in paths]
            fix = (lambda m: m) if mode == 'base' else down
            segment = SegmentBackend('val', name, None, .85, image_paths=paths)
            row = {}
            for d, key in (('appeared', 'new'), ('disappeared', 'demolished')):
                session = Session(rgb[0].shape[:2], target='building', change_direction=d, scope='building_rebuild_excluded')
                box = Toolbox(rgb, R / 'scratch_zoom' / mode, segment=segment, session=session)
                c = pixel_counts(fix(imc(box, .5, d)), truth[key])
                tot[mode][d] += c
                row[d] = c
            for time, lab in (('t1', l1), ('t2', l2)):
                rows, _ = segment(time, 'building', 512, .5)
                union = np.zeros(rgb[0].shape[:2], bool)
                for r in rows:
                    union.flat[r['idx']] = True
                union = fix(union)
                c = pixel_counts(union & labelled, (lab == 5) & labelled)
                tot[mode]['seg_' + time] += c
                row['seg_' + time] = c
            per[name][mode] = row
        print(name, {m: per[name][m]['appeared'] for m in per[name]}, flush=True)
    summary = {m: {k: f1(v) for k, v in tot[m].items()} for m in tot}
    for m in summary:
        summary[m]['route_mean_f1'] = (summary[m]['appeared']['f1'] + summary[m]['disappeared']['f1']) / 2
    summary['gain_route_mean_f1'] = summary['zoom']['route_mean_f1'] - summary['base']['route_mean_f1']
    summary['adopt'] = bool(summary['gain_route_mean_f1'] >= 3)
    Path(output).write_text(json.dumps(dict(summary=summary, per_image=per, images=len(names)), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['prep', 'score'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--worker')
    p.add_argument('--output', default=str(D / 'zoom_test.json'))
    a = p.parse_args()
    prep(a.shard, a.shards, a.worker) if a.step == 'prep' else score(a.output)
