"""T4.1 size attribute on LEVIR-CD: "building changes whose footprint exceeds X square meters".

LEVIR is 0.5 m/pixel (one pixel = 0.25 m^2). Truth = 8-connected components of the change truth with
area_px * 0.25 > X. X is set on dev2 only (200 LEVIR training images): the 2/3 quantile of dev2 change
component areas, rounded to 10 m^2, so about a third of components qualify; the 64 validation images
are not looked at. Adjacent houses merge into one component; that is part of the definition.
Baselines with the same area filter: the fixed route (SAM3 I-M-C, pixel threshold 0.85, the T1.1 given
mask) and the rule (fixed route minus components the flash correspondence verifier answered no).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import load_gt, components, save_mask_png, write_json, names

RUN = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
LEVIR = Path('/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD')
M2_PER_PX = 0.25
INSTRUCTION = ('Find building changes whose footprint is larger than {x} square meters between T1 and T2. '
               'The images have 0.5 m resolution, so one pixel is 0.25 square meters. Output a binary change mask.')


def keep_large(mask, min_m2):
    labels, comps = components(mask)
    keep = [c['id'] for c in comps if c['area'] * M2_PER_PX > min_m2]
    return np.isin(labels, keep) if keep else np.zeros_like(mask, bool)


def threshold(run):
    dev2 = (run / 'dev2/names.txt').read_text().split()
    areas = [c['area'] for n in dev2 for c in components(load_gt('train', n))[1]]
    q = float(np.quantile(areas, 2 / 3)) * M2_PER_PX
    x = int(round(q / 10.0) * 10)
    share = float(np.mean([a * M2_PER_PX > x for a in areas]))
    return x, dict(dev2_images=len(dev2), components=len(areas), quantile_m2=q, share_above=share)


def rule_mask(init, stem, answers):
    labels, comps = components(init)
    no = [c['id'] for c in comps if answers.get(f'{stem}:{c["id"]}', {}).get('answer') == 'no']
    return init & ~np.isin(labels, no)


def generate(run=RUN):
    run = Path(run)
    x, info = threshold(run)
    answers = {json.loads(l)['uid']: json.loads(l) for l in (run / 'ask_eval/val_sam3_imc_p85/f37_corr.jsonl').read_text().splitlines() if l.strip()}
    out = run / 't41'
    rows, stats = [], dict(val_components=0, val_kept=0)
    for name in names('val'):
        stem = Path(name).stem
        gt = load_gt('val', name)
        big = keep_large(gt, x)
        stats['val_components'] += len(components(gt)[1]); stats['val_kept'] += len(components(big)[1])
        save_mask_png(out / 'gt' / name, big)
        from PIL import Image
        with Image.open(run / 'init/sam3_imc_p85/val' / name) as im:
            init = np.asarray(im) > 0
        save_mask_png(out / 'fixed' / name, keep_large(init, x))
        save_mask_png(out / 'rule' / name, keep_large(rule_mask(init, stem, answers), x))
        rows.append(dict(id=name, image_t1=str(LEVIR / 'val_original/A' / name), image_t2=str(LEVIR / 'val_original/B' / name),
            task_type='T4.1', instruction=INSTRUCTION.format(x=x), constraint=None, target='building',
            change_direction='any', scope=None, init_mask=None, gt_mask=str(out / 'gt' / name), meta=dict(min_m2=x)))
    (out / 'tasks').mkdir(parents=True, exist_ok=True)
    (out / 'tasks/T4.1_any.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = dict(min_m2=x, threshold=info, **stats, instruction=INSTRUCTION.format(x=x),
                  rule_answers='ask_eval/val_sam3_imc_p85/f37_corr.jsonl')
    write_json(out / 'preparation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=RUN)
    print(json.dumps(generate(p.parse_args().run_root)))


if __name__ == '__main__':
    main()
