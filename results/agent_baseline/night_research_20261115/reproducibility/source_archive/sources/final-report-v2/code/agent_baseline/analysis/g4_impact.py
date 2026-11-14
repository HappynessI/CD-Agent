"""G4 step 0.2: how often the current answer is not the latest mask, and how many sessions G4 would change.

Reads existing trajectories only (steps.jsonl, final.json, masks/). Definitions:
- latest mask: the most recent `mask#` handle created by a tool call (mask#0 and region#0 excluded);
- current answer: the step's `current_mask` after the reply;
- review window: replies whose remaining budget after the reply is 1..REVIEW_REPLIES (limit 30, so
  replies 27-29; a G2 budget grant extends the limit, approximated by the session's last reply);
- G4 changes the request if, inside the window and before submission, current != latest (the G2 note
  already covers current = none, reported separately), or for region tasks the submitted mask has pixels
  outside region#0 (submission-time clipping).
Diagnostic only: F1 with the latest mask as the answer for fallback sessions. Labels only score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

LIMIT, REVIEW = 30, 3


def load(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


def session(d, gt=None, region=None):
    steps = [json.loads(l) for l in (d / 'steps.jsonl').read_text().splitlines() if l.strip()]
    final = json.loads((d / 'final.json').read_text())
    latest, rows = None, []
    for s in steps:
        for c in s.get('calls', []):
            for h in c.get('handles_created') or []:
                if h.startswith('mask#') and h != 'mask#0':
                    latest = h
        rows.append((s['step'], s.get('current_mask'), latest))
    last = steps[-1]['step'] if steps else 0
    limit = max(LIMIT, last)
    window = [(n, cur, lat) for n, cur, lat in rows if 1 <= limit - n <= REVIEW]
    differs = [(n, cur, lat) for n, cur, lat in window if lat is not None and cur != lat]
    out = dict(fallback=final['submitted_by'] != 'model', submitted_by=final['submitted_by'],
               end_current=rows[-1][1] if rows else None, end_latest=latest,
               end_differs=bool(rows) and latest is not None and rows[-1][1] != latest,
               reminder=any(cur is not None for _, cur, _ in differs), g2_note_case=any(cur is None for _, cur, _ in differs))
    pred = load(d / final['mask_path']) if final.get('mask_path') else None
    out['clip'] = bool(region is not None and pred is not None and (pred & ~region).any())
    out['changes_request'] = out['reminder'] or out['clip']
    if gt is not None and out['fallback'] and out['end_differs'] and latest:
        lm = load(d / 'masks' / (latest.replace('#', '_') + '.png'))
        p = pred if pred is not None else np.zeros_like(gt)
        out['diag'] = dict(current=[int((p & gt).sum()), int((p & ~gt).sum()), int((~p & gt).sum())],
                           latest=[int((lm & gt).sum()), int((lm & ~gt).sum()), int((~lm & gt).sum())])
    return out


def f1(c):
    tp, fp, fn = np.asarray(c, float).sum(axis=0)
    return 200 * tp / max(1, 2 * tp + fp + fn)


def summarize(name, runs, manifest=None):
    rows = {}
    if manifest:
        for l in Path(manifest).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[Path(r['id']).stem] = r
    res = []
    for run in runs:
        for d in sorted(p for p in Path(run).iterdir() if (p / 'final.json').exists()):
            r = rows.get(d.name)
            gt = load(r['gt_mask']) if r else None
            region = load(r['region_mask']) if r and r.get('region_mask') else None
            res.append(session(d, gt, region))
    n = len(res)
    fb = [r for r in res if r['fallback']]
    diag = [r['diag'] for r in fb if 'diag' in r]
    return dict(group=name, sessions=n, fallback=len(fb),
                fallback_end_differs=sum(r['end_differs'] for r in fb),
                changes_request=sum(r['changes_request'] for r in res), reminder=sum(r['reminder'] for r in res),
                clip=sum(r['clip'] for r in res), g2_note_case=sum(r['g2_note_case'] for r in res),
                changes_request_share=sum(r['changes_request'] for r in res) / max(1, n),
                diag_fallback_differs=dict(sessions=len(diag), f1_current=f1([d['current'] for d in diag]) if diag else None,
                                           f1_latest=f1([d['latest'] for d in diag]) if diag else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True, help='JSON list of {name, runs:[...], manifest?}')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = [summarize(g['name'], g['runs'], g.get('manifest')) for g in json.loads(a.spec.read_text())]
    a.output.write_text(json.dumps(out, indent=1))
    lines = ['| 组 | 会话 | 兜底 | 兜底中当前≠最新 | G4 改动请求（提醒 / 求交） | 比例 | 诊断 F1 当前 → 最新（兜底且不同） |',
             '|---|---:|---:|---:|---|---:|---|']
    for g in out:
        d = g['diag_fallback_differs']
        diag = f"{d['f1_current']:.2f} → {d['f1_latest']:.2f}（{d['sessions']}）" if d['sessions'] else '—'
        lines.append(f"| {g['group']} | {g['sessions']} | {g['fallback']} | {g['fallback_end_differs']} | {g['changes_request']}（{g['reminder']} / {g['clip']}） | {100*g['changes_request_share']:.1f}% | {diag} |")
    a.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
