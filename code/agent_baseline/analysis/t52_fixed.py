"""Fixed point response without an Agent. Reuses LEVIR SAM3/SAM embedding caches."""
import argparse
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from common import components, load_mask_png, save_mask_png, write_json
from agent_baseline.backends import SharedPrompter, read_soft
from agent_baseline.tasks import read_tasks, levir_cache_name


def correct(initial, feedback, prompter, t2_instances):
    """Delete original components, then union independent positive-point segments.

    Deletions always refer to the given mask, even after an addition touches it.
    All feedback receives the same original mask; positive segments are unioned.
    """
    initial = np.asarray(initial, bool)
    labels, _ = components(initial)
    result = initial.copy()
    additions = np.zeros_like(initial)
    trace = []
    for item in feedback:
        x, y = item['point']
        if item['type'] == 'false_positive':
            component_id = int(labels[y, x])
            if component_id == 0:
                raise ValueError('False-positive point must lie in the given mask')
            result[labels == component_id] = False
            trace.append(dict(point=[x, y], type=item['type'], removed_component=component_id))
        elif item['type'] == 'missed':
            flat = y * initial.shape[1] + x
            time = 't2' if any(np.any(r['idx'] == flat) for r in t2_instances) else 't1'
            mask, quality, meta = prompter(time, points=[[x, y]], point_labels=[1])
            mask = np.asarray(mask, bool)
            if mask.shape != initial.shape:
                raise ValueError('SAM prediction shape differs from the given mask')
            additions |= mask
            trace.append(dict(point=[x, y], type=item['type'], time=time, quality=float(quality), backend=meta))
        else:
            raise ValueError(f'Unknown feedback type: {item["type"]}')
    return result | additions, trace


def run(manifest, output, device='cuda'):
    rows = read_tasks(manifest, 'T5.2')
    if not rows:
        raise ValueError('Empty T5.2 manifest')
    output = Path(output)
    def factory():
        from sam_tools import Prompter
        return Prompter(device=device)
    shared = SharedPrompter(factory=factory)
    log = {}
    for row in rows:
        name = levir_cache_name(row, 'val')
        initial = load_mask_png(row['init_mask'])
        feedback = row['meta']['feedback']
        t2 = read_soft('val', name, 1, .85, .5) if any(e['type'] == 'missed' for e in feedback) else []
        final, trace = correct(initial, feedback, shared.for_image('val', name), t2)
        save_mask_png(output / (Path(row['id']).stem + '.png'), final)
        log[row['id']] = trace
    write_json(output / 'responses.json', dict(device=device, connectivity=8, pixel_thr=.85,
        score_thr=.5, tile=512, composition='delete given components then union additions', images=log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    run(a.tasks, a.output_dir, a.device)


if __name__ == '__main__':
    main()
