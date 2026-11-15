"""GPU stage: T2 zoom + T3 verify for every queued item, and T5 box for stage-B items.

The orchestrated v0 loop is deterministic given the MLLM answers, so every answer the loop could
ask for is computed once here and cached; agent.py then replays the loop under each budget k and
threshold, consuming these answers in queue order and charging one call per T3/T5 it uses.

    python calls.py --tag r1 --split val --shard 0 --shards 4 [--max-b 50]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from common import QWEN, RUN, load_rgb, names, read_jsonl
import items as I
import render


def crop_paths(tag, split, name, uid):
    base = RUN / 'crops' / tag / split / Path(name).stem
    ext = 'jpg' if tag == 'r1' else 'png'  # r1 saved q90 JPEG display copies; later tags lossless
    return base / f'{uid}_t1.{ext}', base / f'{uid}_t2.{ext}'


def main(a):
    from mllm import Qwen
    q = Qwen(QWEN)
    for name in names(a.split)[a.shard::a.shards]:
        out = RUN / 'calls' / a.tag / a.split / (Path(name).stem + '.jsonl')
        done = {r['uid'] for r in read_jsonl(out)}
        st = I.Store(a.tag, a.split, name)
        todo = [(it, 'A') for it in st.meta['A']] + [(it, 'B') for it in st.meta['B'][:a.max_b]]
        todo = [(it, k) for it, k in todo if it['uid'] not in done]
        if not todo:
            continue
        rgb = load_rgb(a.split, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a') as f:
            for i in range(0, len(todo), a.batch):
                chunk, rows, pairs = todo[i:i + a.batch], [], []
                for it, kind in chunk:
                    crop = render.crop_box(it['bbox'], st.shape)
                    t1, t2 = render.render_pair(rgb, st.mask(it['uid']), crop)
                    p1, p2 = crop_paths(a.tag, a.split, name, it['uid'])
                    p1.parent.mkdir(parents=True, exist_ok=True)
                    for im, path in ((t1, p1), (t2, p2)):
                        im.save(path, **({'quality': 90} if path.suffix == '.jpg' else {}))
                    pairs.append((t1, t2))
                    rows.append(dict(uid=it['uid'], kind=kind, crop=crop, images=[str(p1), str(p2)]))
                for row, v in zip(rows, q.verify(pairs)):
                    row['verify'] = v
                bi = [j for j, r in enumerate(rows) if r['kind'] == 'B' and not a.no_box]
                if bi:
                    for j, b in zip(bi, q.box([pairs[j] for j in bi])):
                        rows[j]['box'] = b
                for row in rows:
                    f.write(json.dumps(row) + '\n')
                f.flush()
        print(name, len(todo), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--max-b', type=int, default=50)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--no-box', action='store_true', help='skip T5 (rounds without box drawing)')
    main(p.parse_args())
