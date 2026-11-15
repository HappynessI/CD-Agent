"""Route B ablation: same DINOv2 top-k candidates with every answer treated as yes (no verifier).
Also the yes/no/unparsed rate by group. Reads route_b outputs only; writes route_b/noverify_scores.json."""
import json, sys, collections
from pathlib import Path
from agent_baseline.analysis import t3_routes_runtime as rt
run_root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = rt.load_dev(run_root); folder = out / 'route_b'
scores = {str(k): [] for k in rt.GRID}; rate = collections.Counter()
for row in rows:
    indices, masks = rt.candidates(folder, row['id'])
    real = {j: json.loads(rt.read_text(folder / 'answers' / f"{row['id']}__{i}.json")) for j, i in enumerate(indices)}
    for j in range(min(16, len(indices))):
        rate[(rt.group(row), real[j]['answer'])] += 1
    yes = {j: dict(answer='yes') for j in real}
    ex = rt.read_image(row['exemplar_mask']) > 0; truth = rt.read_image(row['gt_mask']) > 0
    for k in rt.GRID:
        pred = rt.filtered_union(masks, range(min(k, len(indices))), yes, ex)
        scores[str(k)].append(dict(id=row['id'], counts=rt.pixel_counts(pred, truth, ex)))
summary = {}
for k, tasks in scores.items():
    by = {t['id']: t for t in tasks}; summary[k] = {}
    for g in ('全部', *rt.GROUPS):
        sel = [by[r['id']] for r in rows if g == '全部' or rt.group(r) == g]
        summary[k][g] = rt.pooled(sel)
res = dict(noverify=summary, answers_top16={f'{g}|{a}': n for (g, a), n in rate.items()})
rt.write_json(folder / 'noverify_scores.json', res)
for k, v in summary.items(): print(k, {g: '%.2f/%.2f/%.2f' % (s['f1_c'], s['precision'], s['recall']) for g, s in v.items()})
print(res['answers_top16'])
