"""T3.1 development diagnostics. GPU inference only when main() creates real backends.

Select one task per image with t3_subset.pick before applying --names. The SAM3
routes retrieve source/target class instances on both dates and intersect directed
source loss with target gain. The visual route uses one fixed reference exemplar
per class in a reference/query mosaic; no class text is sent to the visual model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from common import components
from agent_baseline.backends_t13 import WorkerPool
from agent_baseline.tools_t13 import Features
from agent_baseline.tasks import read_tasks
from .probe_t3 import task_scores
from .t3_fixed import f1
from .t3_subset import pick
from .probe_t13_common import CLASSES
from .io_retry import read_text, read_image, write_json, write_text, retry_io

THRESHOLDS = [round(x/20, 2) for x in range(20)]


def transition(row):
    suffix = row['id'].rsplit('_', 1)[-1]
    if 'to' in suffix:
        source, target = map(int, suffix.split('to'))
        return CLASSES[source], CLASSES[target]
    return tuple(row['meta']['transition'].split(' -> '))


def select_rows(rows, names=None):
    chosen, _ = pick(rows)
    if names is not None:
        wanted = {Path(n).stem for n in names}
        chosen = [r for r in chosen if Path(r['image_t1']).stem in wanted]
        if {Path(r['image_t1']).stem for r in chosen} != wanted:
            raise ValueError('Requested image is outside development tasks')
    return chosen


def mask_stack(masks, shape):
    return np.asarray(masks, bool).reshape(-1, *shape)


def proposal_metrics(masks, truth, exemplar):
    """Independent best IoU for each GT instance, and optimistic pixel coverage.

    No greedy assignment: one proposal may cover multiple truths. Exemplar pixels
    are removed before IoU and pixel scores, including from every proposal.
    """
    masks = mask_stack(masks, truth.shape) & ~exemplar
    truth = np.asarray(truth, bool) & ~exemplar
    labels, comps = components(truth)
    hits = 0
    for c in comps:
        region = labels == c['id']
        inter = (masks & region).sum(axis=(1, 2))
        union = masks.sum(axis=(1, 2)) + c['area'] - inter
        hits += bool(np.any(inter / np.maximum(union, 1) >= .5))
    union = masks.any(axis=0) if len(masks) else np.zeros(truth.shape, bool)
    total, covered = int(truth.sum()), int((union & truth).sum())
    return dict(gt_instances=len(comps), aligned_instances=int(hits), truth_pixels=total,
                covered_pixels=covered, proposals=len(masks),
                instance_recall_upper=hits/len(comps) if comps else None,
                pixel_coverage_upper=covered/total if total else None,
                pixel_f1_upper=200*covered/(total+covered) if total else None)


def pixel_counts(pred, truth, exemplar):
    pred, truth = np.asarray(pred, bool) & ~exemplar, np.asarray(truth, bool) & ~exemplar
    return [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]


def directed_proposals(source1, source2, target1, target2, shape):
    """Class transition requires source loss AND target gain, same direction as exemplar."""
    source1, source2, target1, target2 = [mask_stack(x, shape) for x in (source1, source2, target1, target2)]
    union = lambda masks: masks.any(axis=0) if len(masks) else np.zeros(shape, bool)
    loss, gain = ~union(source2), ~union(target1)
    source_candidates = source1 & loss & union(target2) & gain
    target_candidates = target2 & gain & union(source1) & loss
    candidates = np.concatenate([source_candidates, target_candidates], axis=0)
    return candidates[np.any(candidates, axis=(1, 2))]


def evaluate_task(row, dynamic, sam3, thresholds=THRESHOLDS):
    exemplar = read_image(row['exemplar_mask']) > 0
    shape = exemplar.shape
    if not exemplar.any():
        raise ValueError('Exemplar must be nonempty')
    masks = np.concatenate([mask_stack(dynamic('segment_everything', time=t), shape) for t in ('t1', 't2')], axis=0)
    features = [Features(dynamic('extract_features', time=t), shape, t) for t in ('t1', 't2')]
    task = dict(instances=[np.flatnonzero(exemplar).tolist()], exemplar_index=0)
    similarities = task_scores(masks, *features, task) if len(masks) else np.zeros(0)
    predictions = {'sam_dino': {str(t): (masks[similarities >= t].any(axis=0) if np.any(similarities >= t)
                                              else np.zeros(shape, bool)) for t in thresholds}}
    candidates = {'sam_dino': masks}
    yy, xx = np.nonzero(exemplar)
    box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    source, target = transition(row)
    for mode in ('sam3_visual', 'sam3_text'):
        found = []
        for category, reference_time in ((source, 't1'), (target, 't2')):
            for time in ('t1', 't2'):
                found.append(sam3(mode=mode, time=time, category=category,
                                   reference_time=reference_time, box=box))
        candidates[mode] = directed_proposals(*found, shape)
        predictions[mode] = {'default': candidates[mode].any(axis=0) if len(candidates[mode]) else np.zeros(shape, bool)}
    # Label only enters the scoring stage, after all three predictions are fixed.
    truth = read_image(row['gt_mask']) > 0
    if truth.shape != shape:
        raise ValueError('Truth and exemplar shape differ')
    result = {}
    for mode, proposed in candidates.items():
        result[mode] = dict(proposals=proposal_metrics(proposed, truth, exemplar),
            scores={t: dict(counts=pixel_counts(pred, truth, exemplar),
                             **f1(pixel_counts(pred, truth, exemplar))) for t, pred in predictions[mode].items()})
    return dict(id=row['id'], transition=f'{source} -> {target}', methods=result)


def aggregate(tasks):
    result = {}
    for mode in ('sam_dino', 'sam3_visual', 'sam3_text'):
        rows = [t['methods'][mode] for t in tasks]
        keys = ('gt_instances', 'aligned_instances', 'truth_pixels', 'covered_pixels', 'proposals')
        totals = {key: sum(r['proposals'][key] for r in rows) for key in keys}
        totals.update(instance_recall_upper=totals['aligned_instances']/totals['gt_instances'] if totals['gt_instances'] else None,
                      pixel_coverage_upper=totals['covered_pixels']/totals['truth_pixels'] if totals['truth_pixels'] else None,
                      pixel_f1_upper=200*totals['covered_pixels']/(totals['truth_pixels']+totals['covered_pixels']) if totals['truth_pixels'] else None)
        scores = {t: f1(np.asarray([r['scores'][t]['counts'] for r in rows], np.int64).sum(axis=0)) for t in rows[0]['scores']}
        selected = max(scores, key=lambda t: (round(scores[t]['f1_c'], 10), -float(t))) if mode == 'sam_dino' else 'default'
        result[mode] = dict(proposals=totals, scores=scores, diagnostic_best_threshold=selected)
    return result


def conclusions(summary):
    base = summary['sam_dino']
    upper = base['proposals']
    fixed = base['scores']['0.6']['f1_c']
    best = base['scores'][base['diagnostic_best_threshold']]['f1_c']
    visual, text = [summary[k]['scores']['default']['f1_c'] for k in ('sam3_visual', 'sam3_text')]
    ir = upper['instance_recall_upper']
    coverage = upper['pixel_coverage_upper']
    cap = upper['pixel_f1_upper']
    lines = [f'SAM+DINO 固定 0.6 F1={fixed:.2f}；开发门限扫描最高={best:.2f}。',
             f'SAM 提议实例对齐召回上限={ir:.3f}，像素覆盖上限={coverage:.3f}，像素裁剪理想 F1 上限={cap:.2f}。'
             if ir is not None and coverage is not None else '没有可评分真值实例，无法判断提议瓶颈。',
             f'SAM3 示例框 F1={visual:.2f}；类别文字参照 F1={text:.2f}。']
    if cap is not None:
        if coverage < .5:
            lines.append('提议遗漏限制明显：超过一半真值像素未被提议覆盖。')
        if ir is not None and ir < .5:
            lines.append('提议粒度对齐较差：不足一半真值实例被单提议以 IoU≥0.5 覆盖。')
        if cap-best > 10:
            lines.append(f'匹配/整组件选择仍有损失：扫描最高分距乐观像素上限 {cap-best:.2f} 分；该差距也含提议误检，不能全部归给 DINO。')
    lines.append('本诊断没有运行 Agent，不能单凭这些数值确认工作流归因；需与同一开发子集的 Agent 轨迹及分数配对。')
    return '\n'.join(lines)


class Sam3Backend:
    """Persistent dedicated worker, with readable request metadata and packed mask cache."""
    def __init__(self, args):
        self.args, self.proc, self.log = args, None, None

    def for_row(self, row):
        def infer(mode, time, category, reference_time, box):
            folder = self.args.cache / 't3_sam3' / mode / row['id']
            path = folder / f'{reference_time}_{time}.npz'
            request = dict(query=row['image_t1' if time == 't1' else 'image_t2'], output=str(path))
            if mode == 'sam3_visual':
                request.update(reference=row['image_t1' if reference_time == 't1' else 'image_t2'], box=box)
            else:
                request['text'] = category
            identity = dict(request=request, device=self.args.device,
                checkpoint=str(self.args.sam3_checkpoint), bpe=str(self.args.sam3_bpe),
                root=str(self.args.sam3_root), pixel_thr=.85, score_thr=.5, resolution=1008)
            identity['sources'] = [{k: str(p), 'size': retry_io(lambda p=Path(p): p.stat()).st_size,
                                    'mtime_ns': retry_io(lambda p=Path(p): p.stat()).st_mtime_ns}
                                   for k, p in request.items() if k in ('query', 'reference')]
            meta = path.with_suffix('.json')
            if path.exists() and (not meta.exists() or json.loads(read_text(meta)) != identity):
                raise ValueError('SAM3 cache settings differ; use a fresh --cache')
            if not path.exists():
                retry_io(lambda: folder.mkdir(parents=True, exist_ok=True))
                if self.proc is None:
                    self.log = retry_io(lambda: (self.args.cache / 'sam3.stderr.log').open('a'))
                    cmd = [str(self.args.sam3_python), '-B', '-m', 'agent_baseline.workers.t3_sam3_worker',
                           '--device', self.args.device, '--checkpoint', str(self.args.sam3_checkpoint),
                           '--bpe', str(self.args.sam3_bpe), '--sam3-root', str(self.args.sam3_root)]
                    self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
                self.proc.stdin.write(json.dumps(request)+'\n')
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError('SAM3 worker exited; see sam3.stderr.log')
                result = json.loads(line)
                if not result['ok']:
                    raise RuntimeError(result['error'])
                write_json(meta, identity)
            def load():
                with np.load(path, allow_pickle=False) as z:
                    h, w = map(int, z['shape'])
                    return np.unpackbits(z['masks'], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
            return retry_io(load)
        return infer

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait()
            self.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--names', type=Path, help='Optional subset, image stems or filenames, whitespace separated')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--vision-config', type=Path, required=True, help='Existing t3_fixed WorkerPool configuration')
    p.add_argument('--sam3-python', type=Path, required=True)
    p.add_argument('--sam3-root', type=Path, required=True, help='Parent directory of installed sam3 package')
    p.add_argument('--sam3-checkpoint', type=Path, required=True)
    p.add_argument('--sam3-bpe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    a = p.parse_args()
    if not a.device.startswith('cuda'):
        p.error('Real diagnostic backends require a Slurm GPU; fake backends are tested on CPU')
    if .6 not in a.thresholds:
        p.error('Threshold sweep must include the frozen 0.6 baseline')
    rows = select_rows(retry_io(lambda: read_tasks(a.tasks)), read_text(a.names).split() if a.names else None)
    if not rows or any(r['task_type'] != 'T3.1' for r in rows):
        p.error('Require nonempty T3.1 development tasks')
    if not a.names and len(rows) != 60:
        p.error('Full development diagnostic requires 60 images; use --names for a subset')
    retry_io(lambda: a.cache.mkdir(parents=True, exist_ok=True))
    config = json.loads(read_text(a.vision_config))
    # DynamicBackend uses cuda:0 internally. Bind its subprocess to the selected
    # visible GPU, while SAM3 accepts the original cuda:N device directly.
    if ':' in a.device:
        index = int(a.device.split(':')[1])
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        selected_device = visible[index] if visible != [''] else str(index)
        config['dynamic'].setdefault('env', {})['CUDA_VISIBLE_DEVICES'] = selected_device
    dynamic, sam = WorkerPool(config, a.cache / 'vision'), Sam3Backend(a)
    tasks = []
    try:
        for row in rows:
            backend = dynamic.for_pair([row['image_t1'], row['image_t2']])
            tasks.append(evaluate_task(row, backend, sam.for_row(row), a.thresholds))
            write_json(a.output.with_suffix('.partial.json'), dict(status='incomplete', tasks=tasks))
            print(row['id'], flush=True)
    finally:
        dynamic.close()
        sam.close()
    summary = aggregate(tasks)
    text = conclusions(summary)
    report = dict(images=len(rows), task_ids=[r['id'] for r in rows], thresholds=a.thresholds,
                  methods=summary, tasks=tasks, conclusion=text,
                  settings=dict(vision_config=config, device=a.device, exemplar_excluded=True,
                                sam3='resolution=1008; pixel=.85; score=.5; visual=reference/query mosaic',
                                transition='source T1 loss AND target T2 gain'))
    write_json(a.output, report)
    lines = ['| 方法 | F1 | P | R | 实例对齐召回上限 | 像素覆盖上限 |',
             '|---|---:|---:|---:|---:|---:|']
    for mode, data in summary.items():
        setting = '0.6' if mode == 'sam_dino' else 'default'
        metric, upper = data['scores'][setting], data['proposals']
        value = lambda x: 'NA' if x is None else f'{100*x:.2f}'
        lines.append(f"| {mode} | {metric['f1_c']:.2f} | {metric['precision']:.2f} | {metric['recall']:.2f} | "
                     f"{value(upper['instance_recall_upper'])} | {value(upper['pixel_coverage_upper'])} |")
    write_text(a.output.with_suffix('.md'), '\n'.join(lines)+'\n\n'+text+'\n')
    print(text)


if __name__ == '__main__':
    main()
