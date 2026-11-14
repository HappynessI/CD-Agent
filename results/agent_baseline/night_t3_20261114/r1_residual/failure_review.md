# Residual candidate support: completed tune failure review

This is a Main Agent-team review of recorded answers and rendered inputs, not independent human truth. All 30 fixed tune tasks completed: 48 new residual requests, 180 exact answer reuses, new cost CNY 0.01144. No confirmation promotion is justified. Original direct4 is the strict paired control; the original R1 Top8 ranking, Example, prompt and model settings remained fixed. Candidate support changed from C to C minus E, with the corresponding crop change allowed by the predefined intervention.

Positive-task F1 changes from 10.8147 to 10.7389. TP remains 5560; FP increases by 726. Empty-task FP remains 70057. These are task-union results, not sums of individual candidate areas.

## Exactly three answer flips

Of the 48 re-asked overlapping candidates, 45 retain the same yes/no decision. The 180 nonoverlapping decisions were deliberately reused, so their agreement is not an independent model observation. Overlapping accepted candidates change from 44 to 43. No unparsed decision is involved in the three flips.

| Task / candidate | Answer | Candidate / overlap / residual pixels | Residual TP / FP | Task-union ΔTP / ΔFP / ΔFN |
|---|---|---|---|---|
| train_1136_5to1 / 19 | yes → no | 342 / 145 / 197 | 0 / 197 | 0 / −197 / 0 |
| train_147_1to5 / 61 | yes → no | 603 / 345 / 258 | 0 / 258 | 0 / 0 / 0 |
| train_2890_5to1 / 6 | no → yes | 949 / 2 / 947 | 0 / 947 | 0 / +923 / 0 |

The net increase is exactly 923 − 197 = 726 FP pixels. Each listed task has only one answer flip; the remaining 27 task unions have identical counts. Therefore the union records and candidate counts establish:

- For 1136, rejecting candidate 19 removes all 197 residual FP pixels: none remain covered by the other accepted candidates.
- For 147, rejecting candidate 61 removes no union pixels: all 258 residual FP pixels remain covered by other accepted candidates. This is an actual answer change hidden by union redundancy, not an unchanged answer.
- For 2890, accepting candidate 6 adds 923 of its 947 residual FP pixels; 24 were already covered by other accepted candidates. It adds no TP.

These deductions use existing JSON counts, with no image-mask recomputation. They do not identify which other candidate supplies the redundant coverage.

## Recorded reasoning and visual interpretation

**1136 / 19:** Originally the model describes both regions as impervious to vegetation and accepts. After removing the exemplar support, it still assigns both that broad transition but rejects based on blue-roof versus white-object appearance, differing coverage, and possible remnants. The answer introduces a finer object/color distinction despite a broad class match. The FP reduction under the labels is real, but does not by itself validate this reasoning: this task has only three remaining GT pixels.

**147 / 61:** Originally both regions are described as bare ground to construction. The residual answer instead describes an unchanged blue tarp/roof in Candidate T1/T2 and rejects. This is consistent with the intervention's intended mechanism at the answer level, yet has zero final-union benefit because the same residual pixels remain accepted elsewhere. No independent visual truth assessment is claimed for this case.

**2890 / 6:** Originally Candidate is described as bare ground at both dates and rejected. The residual answer instead calls Candidate T1 impervious surface and T2 bare soil, matching the Example, and accepts. Only two original pixels overlap E (0.21% of this candidate); almost the entire support is unchanged. Visual inspection of both recorded Candidate T1/T2 input pairs shows nearly the same crop and a small contour-notch difference. The candidate T1 interior is mottled dark/gray while prominent light roofs lie adjacent to the contour; residual reasoning about structured light-gray impervious surfaces may be influenced by that context, but this review does not establish that cause. The original and residual land-cover readings conflict despite this small intervention. Fixed seed and temperature do not make that flip proof of a specific grounding mechanism, nor establish independent repeatability.

## Evidence paths

All paths below are relative to `CD-Agent/results/agent_baseline/night_t3_20261114/`:

- Aggregate and paired task counts: `r1_residual/tune/summary.json`; `r1_residual/tune/tasks/<task>/metrics.json`.
- Original decisions and counts: `r1_direct4/tune/tasks/<task>/metrics.json`.
- Per-candidate support and original remaining TP/FP: `r1_overlap_tune/tasks/<task>/diagnostic.json`.
- Full original/residual response: `{r1_direct4,r1_residual}/tune/tasks/<task>/verification/<candidate>/answer.json`.
- Exact request and actual four images: the same verification directory's `input.json`, `00.png`–`03.png`; Candidate T1/T2 are `02.png`/`03.png`. Residual `support.json` records support sizes and source paths.

## Decision

The residual intervention does not improve the fixed tune result and does not recover lost TP. The earlier association between overlap and acceptance is insufficient to claim self-matching caused most FP: most new decisions do not change, one favorable rejection is completely redundant at union level, and a tiny support change introduces a harmful acceptance. Preserve this negative result; do not promote confirmation or infer that 73889 overlap-associated FP pixels were individually removable. Further configuration selection is not justified by these outcomes.
