"""Compute-node preparation of frozen C2/C4 boxes, region PNGs and clipped GT."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from agent_baseline.analysis.second_labels import SCOPE, validation_names
from agent_baseline.analysis.second_metrics import mask
from agent_baseline.fixed_routes import RUN
from agent_baseline.tasks import BOX_INSTRUCTIONS, read_tasks

SUBTASKS = [('c2', 'T1.1xC2', 'any', 'c0'), ('c4', 'T1.1xC4', 'appeared', 'new')]
RULE = dict(size='independent uniform integer width/height in [ceil(.4*side), floor(.7*side)]',
            position='uniform integer top-left among positions that fit',
            intersection='one shared half-image selection; reject and resample the whole box until GT intersects; empty GT waives requirement',
            coordinates='integer pixels [x0,y0,x1,y1]; x1/y1 exclusive',
            order='sorted image names, then c2 and c4; NumPy default_rng')


def region_mask(shape, box):
    h, w = shape
    x0, y0, x1, y1 = box
    if any(type(v) is not int for v in box) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Box must have integer exclusive bounds inside the image')
    result = np.zeros(shape, bool)
    result[y0:y1, x0:x1] = True
    return result


def sample_box(shape, rng):
    h, w = shape
    width = int(rng.integers(int(np.ceil(.4 * w)), int(np.floor(.7 * w)) + 1))
    height = int(rng.integers(int(np.ceil(.4 * h)), int(np.floor(.7 * h)) + 1))
    x = int(rng.integers(w - width + 1))
    y = int(rng.integers(h - height + 1))
    return [x, y, x + width, y + height]


def make_boxes(names, truth_loader, seed=1016):
    names = sorted(names)
    if not names or len(names) != len(set(names)) or len(names) % 2:
        raise ValueError('Use a nonempty even number of distinct images')
    rng = np.random.default_rng(seed)
    required = set(rng.choice(names, len(names) // 2, replace=False).tolist())
    images = {}
    for name in names:
        truths = truth_loader(name)
        item = {}
        for kind, _, _, truth_kind in SUBTASKS:
            truth = np.asarray(truths[truth_kind], bool)
            if truth.ndim != 2 or min(truth.shape) < 3:
                raise ValueError('Expected a 2D truth at least 3 pixels on each side')
            requested, empty = name in required, not bool(truth.any())
            require = requested and not empty
            attempts = 0
            while True:
                box = sample_box(truth.shape, rng)
                x0, y0, x1, y1 = box
                intersects = bool(truth[y0:y1, x0:x1].any())
                attempts += 1
                if not require or intersects:
                    break
            item[kind] = dict(box=box, shape=list(truth.shape), selected_for_intersection=requested,
                require_intersection=require, empty_truth_waiver=requested and empty,
                intersects_truth=intersects, attempts=attempts)
        if item['c2']['shape'] != item['c4']['shape']:
            raise ValueError('C0 and new truth shapes differ')
        images[name] = item
    return dict(seed=seed, rules=RULE, count=len(names), selected_for_intersection=sorted(required), images=images)


def frozen_write(path, content):
    """Keep equal artifacts; reject changed content before writing any replacement."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing output differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def png_bytes(data):
    output = io.BytesIO()
    Image.fromarray(np.asarray(data, np.uint8) * 255).save(output, format='PNG')
    return output.getvalue()


def generate(run, output, names_file=None, seed=1016, expected_count=128, source_run=None):
    run, output = Path(run).resolve(), Path(output)
    source_run = Path(source_run).resolve() if source_run is not None else run
    names = validation_names(names_file or source_run / 'second/second_val128.txt')
    if len(names) != expected_count:
        raise ValueError(f'Expected {expected_count} images, got {len(names)}')
    sources = read_tasks(source_run / 'second/tasks/T1.1xC0_any.jsonl')
    by_name = {row['id']: row for row in sources}
    if set(by_name) != set(names):
        raise ValueError('C0 manifest and frozen image list differ')
    def load(name):
        return {key: mask(source_run / 'second/gt' / key / name) for key in ('c0', 'new')}
    document = make_boxes(names, load, seed)
    writes = [(output, (json.dumps(document, indent=2) + '\n').encode())]
    for kind, task, direction, truth_kind in SUBTASKS:
        rows = []
        for name in names:
            info = document['images'][name][kind]
            region = region_mask(tuple(info['shape']), info['box'])
            region_path = run / 'second/regions' / kind / name
            gt_path = run / 'second/gt' / kind / name
            writes.extend([(region_path, png_bytes(region)), (gt_path, png_bytes(load(name)[truth_kind] & region))])
            row = dict(by_name[name], instruction=BOX_INSTRUCTIONS[task].format(box=info['box']),
                task_type=task, change_direction=direction, target='building', scope=SCOPE,
                constraint=None, init_mask=None, region_mask=str(region_path), gt_mask=str(gt_path), meta={})
            rows.append(row)
        content = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        writes.append((run / 'second/tasks' / f'{task}_{direction}.jsonl', content))
    # Preflight all existing artifacts so a conflict cannot leave a partly changed run.
    for path, content in writes:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Existing output differs: {path}')
    for path, content in writes:
        frozen_write(path, content)
    return document


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--source-run-root', type=Path, help='Read existing C0 manifest/truth from this root; write only to --run-root and --output')
    p.add_argument('--names-file', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3] / 'results/agent_baseline/data/second_boxes.json')
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    document = generate(a.run_root, a.output, a.names_file, a.seed, source_run=a.source_run_root)
    print(json.dumps(dict(count=document['count'], seed=document['seed'], output=str(a.output))))


if __name__ == '__main__':
    main()
