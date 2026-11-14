"""Per-call segment_prompt box statistics: box area share vs. precision of the returned mask (recorder candidate_eval)."""
import json,glob,collections
R='/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/'
groups={'SECOND C2/C4':['second_g2box_any_s*/T1.1xC2','second_g2box_appeared_s*/T1.1xC4'],
        'SECOND C0/C1/T5':['second_g2_any_s*/T1.1xC0','second_g2_appeared_s*/T1.1xC1','second_g2_disappeared_s*/T1.1xC1','second_g2_appeared_s*/T5.1xC1','second_g2_disappeared_s*/T5.1xC1'],
        'LEVIR G2':['json_qwen37flash_g2_s*/T1.1xC0','json_qwen37flash_g2_s*/T5.1']}
rows=[]
for g,pats in groups.items():
    for pat in pats:
        for d in glob.glob(R+pat+'/val/*/'):
            side=1024 if 'json_qwen' in pat else 512
            for l in open(d+'steps.jsonl'):
                for c in json.loads(l).get('calls') or []:
                    b=(c.get('args') or {}).get('box') if c['tool']=='segment_prompt' else None
                    if not (b and len(b)==4 and c.get('ok')): continue
                    ev=(c.get('candidate_eval') or [None])[0]
                    if not ev: continue
                    share=max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))/side/side
                    area=ev['tp']+ev['fp']
                    rows.append((g,share,ev['tp'],ev['fp'],area/side/side))
print('calls',collections.Counter(r[0] for r in rows))
# failure: returned mask is mostly not change (precision<.5) and not tiny (>=0.5% of image)
fail=lambda r: r[3]>r[2] and r[4]>=.005
for g in list(groups)+['all']:
    sel=[r for r in rows if g=='all' or r[0]==g]
    F=[r for r in sel if fail(r)]; G=[r for r in sel if not fail(r)]
    print(f'\n{g}: calls {len(sel)}, failures {len(F)} (FP px {sum(r[3] for r in F)}), ok {len(G)} (TP px {sum(r[2] for r in G)})')
    for th in (.05,.08,.10,.12,.15,.20,.30):
        bf=[r for r in F if r[1]>th]; bg=[r for r in G if r[1]>th]
        print(f'  box>{th:.0%}: blocks failures {len(bf)}/{len(F)} ({sum(r[3] for r in bf)}/{sum(r[3] for r in F)} FP px), blocks ok calls {len(bg)}/{len(G)} ({sum(r[2] for r in bg)}/{sum(r[2] for r in G)} TP px)')
    b=collections.defaultdict(lambda:[0,0])
    for r in sel:
        k=min(int(r[1]*20),10); b[k][fail(r)]+=1
    print('  share bucket (5% steps): ok/fail', {f'{k*5}-{k*5+5}%':v for k,v in sorted(b.items())})
