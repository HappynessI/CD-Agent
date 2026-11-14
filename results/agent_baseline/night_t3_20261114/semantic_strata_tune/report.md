# Fixed tune semantic label-agreement strata

Original fixed tune30 only; confirmation records are ignored
Before/after/pair agreement with existing GT labels is not independently audited grounding accuracy. Remaining GT only stratifies offline diagnostics; original parses, labels and task cohort remain unchanged.
Original make_t3_tasks.MIN_AREA=16 is a component construction cutoff; remaining total>=16 is sensitivity only and is not the original definition.

| Remaining GT | Group | Variant | N | Before count/rate | After count/rate | Pair count/rate |
|---|---|---|---:|---|---|---|
| all | 全部 | B0 | 30 | 16/30 (0.5333) | 15/30 (0.5000) | 8/30 (0.2667) |
| all | 全部 | B1 | 30 | 14/30 (0.4667) | 10/30 (0.3333) | 4/30 (0.1333) |
| all | 全部 | B2 | 30 | 16/30 (0.5333) | 12/30 (0.4000) | 6/30 (0.2000) |
| all | 全部 | B3 | 30 | 17/30 (0.5667) | 13/30 (0.4333) | 6/30 (0.2000) |
| all | 目标类 | B0 | 15 | 9/15 (0.6000) | 9/15 (0.6000) | 5/15 (0.3333) |
| all | 目标类 | B1 | 15 | 8/15 (0.5333) | 8/15 (0.5333) | 3/15 (0.2000) |
| all | 目标类 | B2 | 15 | 8/15 (0.5333) | 9/15 (0.6000) | 4/15 (0.2667) |
| all | 目标类 | B3 | 15 | 11/15 (0.7333) | 7/15 (0.4667) | 3/15 (0.2000) |
| all | 地表类 | B0 | 15 | 7/15 (0.4667) | 6/15 (0.4000) | 3/15 (0.2000) |
| all | 地表类 | B1 | 15 | 6/15 (0.4000) | 2/15 (0.1333) | 1/15 (0.0667) |
| all | 地表类 | B2 | 15 | 8/15 (0.5333) | 3/15 (0.2000) | 2/15 (0.1333) |
| all | 地表类 | B3 | 15 | 6/15 (0.4000) | 6/15 (0.4000) | 3/15 (0.2000) |
| positive | 全部 | B0 | 11 | 7/11 (0.6364) | 4/11 (0.3636) | 1/11 (0.0909) |
| positive | 全部 | B1 | 11 | 8/11 (0.7273) | 3/11 (0.2727) | 2/11 (0.1818) |
| positive | 全部 | B2 | 11 | 9/11 (0.8182) | 4/11 (0.3636) | 4/11 (0.3636) |
| positive | 全部 | B3 | 11 | 8/11 (0.7273) | 3/11 (0.2727) | 2/11 (0.1818) |
| positive | 目标类 | B0 | 6 | 5/6 (0.8333) | 2/6 (0.3333) | 1/6 (0.1667) |
| positive | 目标类 | B1 | 6 | 6/6 (1.0000) | 2/6 (0.3333) | 2/6 (0.3333) |
| positive | 目标类 | B2 | 6 | 6/6 (1.0000) | 3/6 (0.5000) | 3/6 (0.5000) |
| positive | 目标类 | B3 | 6 | 6/6 (1.0000) | 0/6 (0.0000) | 0/6 (0.0000) |
| positive | 地表类 | B0 | 5 | 2/5 (0.4000) | 2/5 (0.4000) | 0/5 (0.0000) |
| positive | 地表类 | B1 | 5 | 2/5 (0.4000) | 1/5 (0.2000) | 0/5 (0.0000) |
| positive | 地表类 | B2 | 5 | 3/5 (0.6000) | 1/5 (0.2000) | 1/5 (0.2000) |
| positive | 地表类 | B3 | 5 | 2/5 (0.4000) | 3/5 (0.6000) | 2/5 (0.4000) |
| remaining_gt>=16_diagnostic | 全部 | B0 | 8 | 6/8 (0.7500) | 1/8 (0.1250) | 0/8 (0.0000) |
| remaining_gt>=16_diagnostic | 全部 | B1 | 8 | 6/8 (0.7500) | 1/8 (0.1250) | 0/8 (0.0000) |
| remaining_gt>=16_diagnostic | 全部 | B2 | 8 | 7/8 (0.8750) | 2/8 (0.2500) | 2/8 (0.2500) |
| remaining_gt>=16_diagnostic | 全部 | B3 | 8 | 5/8 (0.6250) | 2/8 (0.2500) | 1/8 (0.1250) |
| remaining_gt>=16_diagnostic | 目标类 | B0 | 4 | 4/4 (1.0000) | 0/4 (0.0000) | 0/4 (0.0000) |
| remaining_gt>=16_diagnostic | 目标类 | B1 | 4 | 4/4 (1.0000) | 0/4 (0.0000) | 0/4 (0.0000) |
| remaining_gt>=16_diagnostic | 目标类 | B2 | 4 | 4/4 (1.0000) | 1/4 (0.2500) | 1/4 (0.2500) |
| remaining_gt>=16_diagnostic | 目标类 | B3 | 4 | 4/4 (1.0000) | 0/4 (0.0000) | 0/4 (0.0000) |
| remaining_gt>=16_diagnostic | 地表类 | B0 | 4 | 2/4 (0.5000) | 1/4 (0.2500) | 0/4 (0.0000) |
| remaining_gt>=16_diagnostic | 地表类 | B1 | 4 | 2/4 (0.5000) | 1/4 (0.2500) | 0/4 (0.0000) |
| remaining_gt>=16_diagnostic | 地表类 | B2 | 4 | 3/4 (0.7500) | 1/4 (0.2500) | 1/4 (0.2500) |
| remaining_gt>=16_diagnostic | 地表类 | B3 | 4 | 1/4 (0.2500) | 2/4 (0.5000) | 1/4 (0.2500) |
| empty | 全部 | B0 | 19 | 9/19 (0.4737) | 11/19 (0.5789) | 7/19 (0.3684) |
| empty | 全部 | B1 | 19 | 6/19 (0.3158) | 7/19 (0.3684) | 2/19 (0.1053) |
| empty | 全部 | B2 | 19 | 7/19 (0.3684) | 8/19 (0.4211) | 2/19 (0.1053) |
| empty | 全部 | B3 | 19 | 9/19 (0.4737) | 10/19 (0.5263) | 4/19 (0.2105) |
| empty | 目标类 | B0 | 9 | 4/9 (0.4444) | 7/9 (0.7778) | 4/9 (0.4444) |
| empty | 目标类 | B1 | 9 | 2/9 (0.2222) | 6/9 (0.6667) | 1/9 (0.1111) |
| empty | 目标类 | B2 | 9 | 2/9 (0.2222) | 6/9 (0.6667) | 1/9 (0.1111) |
| empty | 目标类 | B3 | 9 | 5/9 (0.5556) | 7/9 (0.7778) | 3/9 (0.3333) |
| empty | 地表类 | B0 | 10 | 5/10 (0.5000) | 4/10 (0.4000) | 3/10 (0.3000) |
| empty | 地表类 | B1 | 10 | 4/10 (0.4000) | 1/10 (0.1000) | 1/10 (0.1000) |
| empty | 地表类 | B2 | 10 | 5/10 (0.5000) | 2/10 (0.2000) | 1/10 (0.1000) |
| empty | 地表类 | B3 | 10 | 4/10 (0.4000) | 3/10 (0.3000) | 1/10 (0.1000) |

## Correct pair identities and changes from B0

- all / B0: correct=['train_1065_2to5', 'train_1136_5to1', 'train_127_3to2', 'train_147_1to5', 'train_2848_1to4', 'train_385_5to6', 'train_555_4to2', 'train_786_3to5']; gained=[]; lost=[]
- all / B1: correct=['train_1136_5to1', 'train_2571_4to5', 'train_385_5to6', 'train_555_4to2']; gained=['train_2571_4to5']; lost=['train_1065_2to5', 'train_127_3to2', 'train_147_1to5', 'train_2848_1to4', 'train_786_3to5']
- all / B2: correct=['train_1136_5to1', 'train_127_3to2', 'train_1745_5to2', 'train_1808_2to1', 'train_2571_4to5', 'train_385_5to6']; gained=['train_1745_5to2', 'train_1808_2to1', 'train_2571_4to5']; lost=['train_1065_2to5', 'train_147_1to5', 'train_2848_1to4', 'train_555_4to2', 'train_786_3to5']
- all / B3: correct=['train_1065_2to5', 'train_1166_3to5', 'train_127_3to2', 'train_1808_2to1', 'train_786_3to5', 'train_915_3to4']; gained=['train_1166_3to5', 'train_1808_2to1', 'train_915_3to4']; lost=['train_1136_5to1', 'train_147_1to5', 'train_2848_1to4', 'train_385_5to6', 'train_555_4to2']
- positive / B0: correct=['train_1136_5to1']; gained=[]; lost=[]
- positive / B1: correct=['train_1136_5to1', 'train_2571_4to5']; gained=['train_2571_4to5']; lost=[]
- positive / B2: correct=['train_1136_5to1', 'train_1745_5to2', 'train_1808_2to1', 'train_2571_4to5']; gained=['train_1745_5to2', 'train_1808_2to1', 'train_2571_4to5']; lost=[]
- positive / B3: correct=['train_1808_2to1', 'train_915_3to4']; gained=['train_1808_2to1', 'train_915_3to4']; lost=['train_1136_5to1']
- remaining_gt>=16_diagnostic / B0: correct=[]; gained=[]; lost=[]
- remaining_gt>=16_diagnostic / B1: correct=[]; gained=[]; lost=[]
- remaining_gt>=16_diagnostic / B2: correct=['train_1745_5to2', 'train_1808_2to1']; gained=['train_1745_5to2', 'train_1808_2to1']; lost=[]
- remaining_gt>=16_diagnostic / B3: correct=['train_1808_2to1']; gained=['train_1808_2to1']; lost=[]
- empty / B0: correct=['train_1065_2to5', 'train_127_3to2', 'train_147_1to5', 'train_2848_1to4', 'train_385_5to6', 'train_555_4to2', 'train_786_3to5']; gained=[]; lost=[]
- empty / B1: correct=['train_385_5to6', 'train_555_4to2']; gained=[]; lost=['train_1065_2to5', 'train_127_3to2', 'train_147_1to5', 'train_2848_1to4', 'train_786_3to5']
- empty / B2: correct=['train_127_3to2', 'train_385_5to6']; gained=[]; lost=['train_1065_2to5', 'train_147_1to5', 'train_2848_1to4', 'train_555_4to2', 'train_786_3to5']
- empty / B3: correct=['train_1065_2to5', 'train_1166_3to5', 'train_127_3to2', 'train_786_3to5']; gained=['train_1166_3to5']; lost=['train_147_1to5', 'train_2848_1to4', 'train_385_5to6', 'train_555_4to2']

## Interpretation of the earlier B3 decision

Earlier aggregate pair agreement (B3 6/30 versus B0 8/30) alone is insufficient to characterize agreement on retrievable positive tasks. This audit gives B3 2/11 versus B0 1/11 on remaining-GT positives. Use the subgroup identities above to qualify the earlier rationale; do not claim all strata degraded solely from the aggregate.
This is a retrospective diagnosis of the recorded decision, not an automatic promotion, a new parsing run, a retrieval-benefit result, or confirmation-based winner selection. Correct label pairs still require support-grounding review; later oracle text-proposal coverage limitations remain separate evidence.
