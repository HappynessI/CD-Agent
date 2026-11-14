# Fixed R1 patch-set matching: completed tune failure review

All 30 fixed tune cases completed with zero API calls; the three contract tests passed and original delta Top8 order reproduced in 30/30 cases. This review reads saved JSON only. It does not recompute masks, inspect feature arrays, or change existing results. This is an offline retrieval experiment, not an online Agent evaluation.

On the 11 positive tasks, F1 falls from 11.27688 to 8.01359: TP 9826 → 6627 (−3199), FP 124127 → 118452 (−5675). On the 19 empty tasks, FP increases from 191988 to 205178 (+13190). The lower positive-group FP does not offset the lost true coverage and increased empty-task errors. Do not promote confirmation or search patch weights.

## Three principal cases

Counts are official task-union TP / FP / FN after excluding the exemplar. IDs index the same fixed prepared R1 candidate pool; Top8 lists are ordered.

| Task | Remaining GT | Delta counts | Patch-set counts | ΔTP / ΔFP / ΔFN |
|---|---:|---|---|---|
| train_1538_1to3 | 5243 | 5121 / 4863 / 122 | 1505 / 11734 / 3738 | −3616 / +6871 / +3616 |
| train_2848_1to4 | 0 | 0 / 24510 / 0 | 0 / 33806 / 0 | 0 / +9296 / 0 |
| train_2928_3to1 | 0 | 0 / 10138 / 0 | 0 / 19046 / 0 | 0 / +8908 / 0 |

**1538: largest true-positive loss.** Delta Top8 is `[53,26,35,11,41,43,70,34]`; patch-set Top8 is `[53,26,7,17,32,35,11,23]`. The top two candidates remain unchanged, but four lower selections change: IDs 41/43/70/34 leave and 7/17/32/23 enter. IDs 35 and 11 drop from ranks 3/4 to 6/7. Thus retaining the highest matches does not preserve true coverage of the union. The pool has 71 candidates; the example has 29 supported patches on a 36×36 grid, with nonzero delta mass 1.0.

**2848: largest empty-task FP increase.** Delta Top8 is `[5,9,7,4,2,0,6,8]`; patch-set is `[5,3,4,2,8,1,7,0]`. IDs 9/6 leave and 3/1 enter; ID 3 rises into second position. Only 10 candidates exist in this pool, so selecting eight retains most of the pool under either ranking. Changing the two excluded candidates nevertheless increases union FP by 9296. The example has 70 supported patches and nonzero delta mass 1.0.

**2928: second-largest empty-task FP increase.** Delta Top8 is `[17,37,40,56,36,59,9,3]`; patch-set is `[37,3,17,7,16,59,14,42]`. IDs 40/56/36/9 leave and 7/16/14/42 enter; ID 3 moves from rank 8 to rank 2. Pool size is 64, example support is 37 patches, and nonzero delta mass is 1.0. Union FP grows by 8908 without any possible TP on this empty task.

These two empty cases contribute +18204 FP before changes elsewhere; other empty cases offset 5014 of that increase. The largest positive loss is not the entire result: train_1745_5to2 loses another 211 TP, while train_1754_5to1 gains 241 and train_2912_5to2 gains 387. Together, −3616−211+241+387 = −3199 TP. Benefits in a few cases therefore coexist with an adverse fixed-cohort result.

## What the evidence does and does not establish

The same cached proposal pool and reproduced baseline order isolate a ranking-method difference, not candidate-generation drift. The saved IDs show which selected sets change; union counts show the resulting coverage errors. They do not establish the marginal pixel contribution of each entering/leaving candidate, since candidates can overlap.

The fixed score allows one patch to be the best match for many patches in the other region. That permits a hypothesis that heterogeneous or mixed support can receive favorable matches. This review has not inspected the patch-to-patch match assignments or demonstrated mixed-region preference; the three cases do not prove that mechanism. Likewise, bilinear support may include grid receptive fields spanning neighboring content, but the present JSON does not quantify that contamination. Nonzero example delta mass 1.0 in these cases rules out suppression by the example-side near-zero policy as their immediate explanation; it says nothing about candidate-side mass or visual semantics.

Example-date swapping measures the scoring formula's direction sensitivity. It is not independent evidence of semantic understanding, and the original labels are not new truth for the swapped example.

## Evidence and decision

Relative to `CD-Agent/results/agent_baseline/night_t3_20261114/`, aggregate evidence is `r1_patchset/tune/summary.json`. Each named case is in `r1_patchset/tune/tasks/<task>/metrics.json` with ranked selected IDs and counts; `ranking_provenance.json` records baseline reproduction, pool size and example support. Original geometry baseline records remain under `retrieval/B0/tasks/<task>/metrics.json`.

Reject this predefined patch-set score as an improvement on the current fixed tune cohort. Do not run confirmation or scan patch weights. Preserve the negative result and the limited case-level gains without treating either as new configuration-selection data. The night's T3 computational line is complete; no actual online three-seed Agent comparison was run.
