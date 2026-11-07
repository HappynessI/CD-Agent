"""T5.3 10-image trial: given, largest, verifier, Agent on the same images. F1, repair, outside edits, localization."""
import json, sys, numpy as np
from PIL import Image
sys.path[:0] = ['code_p', 'code_p/ovcd_v0']
from agent_baseline.analysis.t52_metrics import repair_and_outside, session_prediction
from agent_baseline.analysis.t53_metrics import localization, localization_summary
R = '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/'
fb = json.load(open(R + 'results_n/t53_feedback.json'))['images']
run = R + (sys.argv[1] if len(sys.argv) > 1 else 'json_qwen37flash_g3_t53trial_fb_s1') + '/T5.3/val'
stems = sorted(__import__('os').listdir(run))
g = lambda p: np.asarray(Image.open(p)) > 0
f1 = lambda c: 200 * c[0] / max(1, 2 * c[0] + c[1] + c[2])
M = ('given', 'largest', 'verifier', 'agent')
T = {m: np.zeros(3) for m in M}; rep = {m: [0, 0] for m in M}; out = {m: [0, 0] for m in M}; loc = {m: [] for m in M}
for stem in stems:
    v = fb[stem + '.png']; items = v if isinstance(v, list) else v.get('feedback', v.get('items', []))
    init = g(R + f'init/sam3_imc_p85/val/{stem}.png'); gt = g(f'/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/val_original/label/{stem}.png')
    fin = json.load(open(f'{run}/{stem}/final.json'))
    for m in M:
        p = init if m == 'given' else session_prediction(run, stem, init.shape)[0] if m == 'agent' else g(R + f't53/fixed/{m}/masks/{stem}.png')
        T[m] += [(p & gt).sum(), (p & ~gt).sum(), (~p & gt).sum()]
        r = repair_and_outside(p, init, gt, items)
        rep[m][0] += sum(r['repaired'].values()); rep[m][1] += sum(r['pointed'].values())
        out[m][0] += r['outside_added_px']; out[m][1] += r['outside_removed_px']
        loc[m].append(localization(p, init, gt, items))
    a = loc['agent'][-1]
    print(stem, (items[0]['type'], items[0].get('where')) if items else 'none', 'by', fin['submitted_by'], 'cost', round(fin['cost_cny'], 3),
          'agent changed', a['changed_components'], 'hit', a['matching_components'], 'edited', a['edited_regions'], '/', a['described_regions'])
for m in M:
    print(m, 'F1', round(f1(T[m]), 2), 'repaired', rep[m], 'outside add/remove px', out[m], localization_summary(loc[m]))
