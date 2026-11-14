# T3 night actual evidence

Units: pixel F1/precision/recall are percentages; label agreement, coverage and empty-prediction rates are fractions in [0,1]. Label agreement is not manual semantic or support-grounding accuracy. The existing dev60 was historically inspected; the new image split isolates new method selection, not unseen data. All original tasks remain, including empty remaining GT.

B1/B2 change support rendering and prompts jointly; B2 also adds both dates. B3 uses four aligned RGB/binary-support panels. B3 tune did not improve B0, so confirmation was not promoted.

B0 R1/R2/R3 were predeclared fixed full60 diagnostics, reported by tune/confirm; confirmation was not triggered by a tuning win. No B1/B2/B3 promotion or confirm-based winner selection is implied.

R1 geometry-DINO changes the pool relative to unfiltered historical DINO. R2 uses predicted classes, including wrong/unknown predictions; native order is filtered SAM query order with source proposals before target, not confidence ranking or random ranking. R3 local verification is a joint input/prompt change. Unparsed answers follow historical keep policy; drop-unparsed results are sensitivity diagnostics.

Raw-contour tune30 was prepared but not submitted: the completed B0 R2 pool covers zero GT pixels in fixed tune positives, so a verifier cannot recover TP from this pool. Historical GT-text oracle is reused offline in oracle_cache/report.md to separate semantic and proposal-constraint limits, without new GT-fed model requests.

Observed local-verifier negative case: train_1136_5to1 candidates0/1 are rejected because the model says the same rectangular building shape persists. Actual inputs show a blue roof then green vegetation; the fixed spotlight support creates the same bright silhouette on both dates. This is evidence of a possible support-shape/object-shape confusion, not proof of causal attribution. This task has only3 remaining GT pixels, so a beneficial rejection score would not establish correct visual reasoning. Inputs/responses are archived under retrieval/B0/tasks/train_1136_5to1/verification.

R1-direct4 is a distinct planned tune-first experiment: same fixed R1 Top8 versus no verification, four local raw-RGB Example/Candidate date views, no predicted class pair. R1 eligibility retains candidate overlap with the example; only scoring removes exemplar pixels. No hidden overlap filter is introduced. Old eight-view original-DINO and new two-view text-pool routes are non-paired contextual references.

Endpoint methods share the original SAM pool with fixed equal weights. Swapping exemplar dates measures ranking sensitivity only; it does not define new GT.

T3.3 visual-review uncertainty: the fixed first tune example train_2130_T33_2to4 is clipped at the source crop bottom; close/wide views nearly coincide. Its post-date tan/gray-green support can resemble bare ground or turbid water. Treat the water-label mismatch as requiring visual/label review, not automatically a pure model error. No labels were changed.

| Stage | Completed/expected | Status |
|---|---:|---|
| B0 | 60/60 | complete |
| B1 | 60/60 | complete |
| B2 | 60/60 | complete |
| B3 | 30/30 | complete |
| t33/B0 | 12/12 | complete |
| t33/B1 | 12/12 | complete |
| t33/B2 | 12/12 | complete |
| B3/confirm | —/— | not_promoted |
| retrieval/B0/tune | 30/30 | complete |
| retrieval/B0/confirm | 30/30 | complete |
| t33/retrieval/B0/tune | 6/6 | complete |
| t33/retrieval/B0/confirm | 6/6 | complete |
| r1_direct4/tune | 30/30 | complete |
| r1_direct4/confirm | 0/30 | not_promoted_or_partial |
| r1_endpoint/tune | 0/30 | pending |
| r1_endpoint/confirm | 0/30 | not_promoted_or_partial |
| local_raw/tune | 0/30 | not_submitted_zero_TP_headroom |
| local_raw/confirm | 0/30 | not_requested_without_tune_evidence |
| oracle_cache | 60/60 | complete |
| endpoint | 60/60 | complete |

## Decisions from tuning evidence only

Predeclared geometry filtering changes tune Top8 F1 from 2.753 to 5.366, TP from 9908 to 9826, and FP from 669503 to 316115. Recall remains 24.373%; this supports keeping it as a low-cost baseline, not a solved-task claim.

Predicted text proposals cover 0 tuning GT pixels. Local verification changes FP from 4255 to 3717, but its empty-prediction rate on positive tasks is 1.000, versus negative accuracy 0.947 on empty tasks. Rejection alone is not reliable recovery.

Do not promote the text bridge or run the planned whole tune raw-view arm when its proposal pool has zero TP headroom. First reuse historical oracle/cache evidence to separate parsing from proposal constraints; evaluate the fixed endpoint alternatives using tune positive-target evidence before considering any method choice. Never select weights or methods from confirmation outcomes.

## Other completed diagnostics and pending evidence

T3.3 fixed12 retrieval is complete. Tuning6 Top8 F1: R1_geometry_dino@8=14.401, R2_native_order@8=0.000, R3_verified@8=0.000. These use different proposal pools; only R3 before/after verification shares its fixed text pool. Cross-image label agreement and ambiguous source crops remain separate from achieved pixel scores.
Endpoint fixed-pool tune all Top8 F1: delta=2.753, endpoint=2.061, equal_joint=4.818.
Endpoint fixed-pool tune positive Top8 F1: delta=6.536, endpoint=4.727, equal_joint=10.765.
Endpoint fixed-pool tune empty Top8 F1: delta=0.000, endpoint=0.000, equal_joint=0.000.
Endpoint comparisons alone share the full original SAM pool; R1 geometry-filtered values are not a same-pool scoring contrast. Example-date swaps report ranking sensitivity, not new truth or a causal mechanism claim. No coefficient was fitted on confirmation.
Historical oracle tune positives cover 4/40315 pixels (F1 0.0198), versus predicted text 0/40315. Even GT classes do not resolve coverage. Of 11 cache-complete positive cases, 10 have zero sourceT1/targetT2 intersection; intersection coverage 3479 drops to 4 after strict exclusions. Per-task evidence identifies missing endpoint proposals separately from loss/gain suppression. These oracle masks are historical, not new GT-fed API calls.
R1-direct4 tune positive F1 11.277 → 10.815; TP 9826 → 5560, FP 124127 → 56948. Judge positive-target preservation, empty-task behavior and visual-review evidence together before any confirmation promotion; no single mechanical F1 cutoff decides it.
R1 same-pool joint-score experiment is pending/partial; confirmation is not automatically promoted.

## Label agreement only

| Variant | Split | Group | N | Before | After | Pair |
|---|---|---|---:|---:|---:|---:|
| B0 | all | 全部 | 60 | 0.5333 | 0.5000 | 0.2167 |
| B0 | all | 目标类 | 27 | 0.5185 | 0.6667 | 0.2593 |
| B0 | all | 地表类 | 33 | 0.5455 | 0.3636 | 0.1818 |
| B0 | tune | 全部 | 30 | 0.5333 | 0.5000 | 0.2667 |
| B0 | tune | 目标类 | 15 | 0.6000 | 0.6000 | 0.3333 |
| B0 | tune | 地表类 | 15 | 0.4667 | 0.4000 | 0.2000 |
| B0 | confirm | 全部 | 30 | 0.5333 | 0.5000 | 0.1667 |
| B0 | confirm | 目标类 | 12 | 0.4167 | 0.7500 | 0.1667 |
| B0 | confirm | 地表类 | 18 | 0.6111 | 0.3333 | 0.1667 |
| B1 | all | 全部 | 60 | 0.3667 | 0.3500 | 0.0833 |
| B1 | all | 目标类 | 27 | 0.4074 | 0.5926 | 0.1111 |
| B1 | all | 地表类 | 33 | 0.3333 | 0.1515 | 0.0606 |
| B1 | tune | 全部 | 30 | 0.4667 | 0.3333 | 0.1333 |
| B1 | tune | 目标类 | 15 | 0.5333 | 0.5333 | 0.2000 |
| B1 | tune | 地表类 | 15 | 0.4000 | 0.1333 | 0.0667 |
| B1 | confirm | 全部 | 30 | 0.2667 | 0.3667 | 0.0333 |
| B1 | confirm | 目标类 | 12 | 0.2500 | 0.6667 | 0.0000 |
| B1 | confirm | 地表类 | 18 | 0.2778 | 0.1667 | 0.0556 |
| B2 | all | 全部 | 60 | 0.4333 | 0.3833 | 0.1167 |
| B2 | all | 目标类 | 27 | 0.4074 | 0.6296 | 0.1481 |
| B2 | all | 地表类 | 33 | 0.4545 | 0.1818 | 0.0909 |
| B2 | tune | 全部 | 30 | 0.5333 | 0.4000 | 0.2000 |
| B2 | tune | 目标类 | 15 | 0.5333 | 0.6000 | 0.2667 |
| B2 | tune | 地表类 | 15 | 0.5333 | 0.2000 | 0.1333 |
| B2 | confirm | 全部 | 30 | 0.3333 | 0.3667 | 0.0333 |
| B2 | confirm | 目标类 | 12 | 0.2500 | 0.6667 | 0.0000 |
| B2 | confirm | 地表类 | 18 | 0.3889 | 0.1667 | 0.0556 |
| B3 | all | 全部 | 30 | 0.5667 | 0.4333 | 0.2000 |
| B3 | all | 目标类 | 15 | 0.7333 | 0.4667 | 0.2000 |
| B3 | all | 地表类 | 15 | 0.4000 | 0.4000 | 0.2000 |
| B3 | tune | 全部 | 30 | 0.5667 | 0.4333 | 0.2000 |
| B3 | tune | 目标类 | 15 | 0.7333 | 0.4667 | 0.2000 |
| B3 | tune | 地表类 | 15 | 0.4000 | 0.4000 | 0.2000 |
| B3 | confirm | 全部 | 0 | NA | NA | NA |
| B3 | confirm | 目标类 | 0 | NA | NA | NA |
| B3 | confirm | 地表类 | 0 | NA | NA | NA |
| t33/B0 | all | 全部 | 12 | 0.5833 | 0.3333 | 0.1667 |
| t33/B0 | all | 目标类 | 8 | 0.6250 | 0.3750 | 0.2500 |
| t33/B0 | all | 地表类 | 4 | 0.5000 | 0.2500 | 0.0000 |
| t33/B0 | tune | 全部 | 6 | 0.8333 | 0.1667 | 0.0000 |
| t33/B0 | tune | 目标类 | 3 | 1.0000 | 0.0000 | 0.0000 |
| t33/B0 | tune | 地表类 | 3 | 0.6667 | 0.3333 | 0.0000 |
| t33/B0 | confirm | 全部 | 6 | 0.3333 | 0.5000 | 0.3333 |
| t33/B0 | confirm | 目标类 | 5 | 0.4000 | 0.6000 | 0.4000 |
| t33/B0 | confirm | 地表类 | 1 | 0.0000 | 0.0000 | 0.0000 |
| t33/B1 | all | 全部 | 12 | 0.5833 | 0.3333 | 0.0833 |
| t33/B1 | all | 目标类 | 8 | 0.6250 | 0.5000 | 0.1250 |
| t33/B1 | all | 地表类 | 4 | 0.5000 | 0.0000 | 0.0000 |
| t33/B1 | tune | 全部 | 6 | 0.8333 | 0.0000 | 0.0000 |
| t33/B1 | tune | 目标类 | 3 | 1.0000 | 0.0000 | 0.0000 |
| t33/B1 | tune | 地表类 | 3 | 0.6667 | 0.0000 | 0.0000 |
| t33/B1 | confirm | 全部 | 6 | 0.3333 | 0.6667 | 0.1667 |
| t33/B1 | confirm | 目标类 | 5 | 0.4000 | 0.8000 | 0.2000 |
| t33/B1 | confirm | 地表类 | 1 | 0.0000 | 0.0000 | 0.0000 |
| t33/B2 | all | 全部 | 12 | 0.4167 | 0.3333 | 0.0833 |
| t33/B2 | all | 目标类 | 8 | 0.6250 | 0.5000 | 0.1250 |
| t33/B2 | all | 地表类 | 4 | 0.0000 | 0.0000 | 0.0000 |
| t33/B2 | tune | 全部 | 6 | 0.5000 | 0.0000 | 0.0000 |
| t33/B2 | tune | 目标类 | 3 | 1.0000 | 0.0000 | 0.0000 |
| t33/B2 | tune | 地表类 | 3 | 0.0000 | 0.0000 | 0.0000 |
| t33/B2 | confirm | 全部 | 6 | 0.3333 | 0.6667 | 0.1667 |
| t33/B2 | confirm | 目标类 | 5 | 0.4000 | 0.8000 | 0.2000 |
| t33/B2 | confirm | 地表类 | 1 | 0.0000 | 0.0000 | 0.0000 |

## Top8 pixel results

Only completed task artifacts are pooled below; stage counts above identify partial runs. Full TopK and per-task counts remain in report.json. Empty-GT F1 is not a rejection measure; empty prediction rate equals negative accuracy on that stratum.

| Stage | Split | GT | Group | N | Method | F1 | P | R | Empty prediction | Pixel coverage | Instance coverage |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| old_dev60 | all | all | 全部 | 60 | bridge@8 | 3.9360 | 3.5584 | 4.4033 | 0.4167 | NA | NA |
| old_dev60 | all | all | 全部 | 60 | bridge_verified@8 | 4.2158 | 4.0436 | 4.4033 | 0.4167 | NA | NA |
| old_dev60 | all | all | 全部 | 60 | dino@8 | 5.3603 | 3.1621 | 17.5861 | 0.0000 | NA | NA |
| old_dev60 | all | all | 全部 | 60 | dino_verified@8 | 5.7562 | 3.4474 | 17.4287 | 0.0000 | NA | NA |
| old_dev60 | all | all | 全部 | 60 | oracle_class_diagnostic@8 | 11.2158 | 7.4259 | 22.9066 | 0.0000 | NA | NA |
| old_dev60 | all | all | 目标类 | 27 | bridge@8 | 9.5683 | 15.5575 | 6.9086 | 0.5556 | NA | NA |
| old_dev60 | all | all | 目标类 | 27 | bridge_verified@8 | 9.8488 | 17.1454 | 6.9086 | 0.5556 | NA | NA |
| old_dev60 | all | all | 目标类 | 27 | dino@8 | 12.8770 | 9.0678 | 22.2045 | 0.0000 | NA | NA |
| old_dev60 | all | all | 目标类 | 27 | dino_verified@8 | 15.3407 | 11.7193 | 22.2011 | 0.0000 | NA | NA |
| old_dev60 | all | all | 目标类 | 27 | oracle_class_diagnostic@8 | 24.3892 | 20.1397 | 30.9116 | 0.0000 | NA | NA |
| old_dev60 | all | all | 地表类 | 33 | bridge@8 | 0.4441 | 0.3148 | 0.7535 | 0.3030 | NA | NA |
| old_dev60 | all | all | 地表类 | 33 | bridge_verified@8 | 0.4880 | 0.3608 | 0.7535 | 0.3030 | NA | NA |
| old_dev60 | all | all | 地表类 | 33 | dino@8 | 1.9570 | 1.0754 | 10.8578 | 0.0000 | NA | NA |
| old_dev60 | all | all | 地表类 | 33 | dino_verified@8 | 1.9653 | 1.0844 | 10.4760 | 0.0000 | NA | NA |
| old_dev60 | all | all | 地表类 | 33 | oracle_class_diagnostic@8 | 3.5458 | 2.1047 | 11.2447 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 全部 | 30 | bridge@8 | 5.6675 | 7.9498 | 4.4033 | 0.4667 | NA | NA |
| old_dev60 | all | positive | 全部 | 30 | bridge_verified@8 | 6.0907 | 9.8749 | 4.4033 | 0.4667 | NA | NA |
| old_dev60 | all | positive | 全部 | 30 | dino@8 | 8.7760 | 5.8469 | 17.5861 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 全部 | 30 | dino_verified@8 | 9.6461 | 6.6684 | 17.4287 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 全部 | 30 | oracle_class_diagnostic@8 | 17.5718 | 14.2524 | 22.9066 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 目标类 | 16 | bridge@8 | 11.2684 | 30.5428 | 6.9086 | 0.6250 | NA | NA |
| old_dev60 | all | positive | 目标类 | 16 | bridge_verified@8 | 11.3970 | 32.5324 | 6.9086 | 0.6250 | NA | NA |
| old_dev60 | all | positive | 目标类 | 16 | dino@8 | 15.9513 | 12.4462 | 22.2045 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 目标类 | 16 | dino_verified@8 | 19.2662 | 17.0167 | 22.2011 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 目标类 | 16 | oracle_class_diagnostic@8 | 32.3232 | 33.8698 | 30.9116 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 地表类 | 14 | bridge@8 | 0.7419 | 0.7306 | 0.7535 | 0.2857 | NA | NA |
| old_dev60 | all | positive | 地表类 | 14 | bridge_verified@8 | 0.8437 | 0.9585 | 0.7535 | 0.2857 | NA | NA |
| old_dev60 | all | positive | 地表类 | 14 | dino@8 | 3.7502 | 2.2665 | 10.8578 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 地表类 | 14 | dino_verified@8 | 3.7953 | 2.3174 | 10.4760 | 0.0000 | NA | NA |
| old_dev60 | all | positive | 地表类 | 14 | oracle_class_diagnostic@8 | 6.2141 | 4.2934 | 11.2447 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 全部 | 30 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3667 | NA | NA |
| old_dev60 | all | empty | 全部 | 30 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3667 | NA | NA |
| old_dev60 | all | empty | 全部 | 30 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 全部 | 30 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 全部 | 30 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 目标类 | 11 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.4545 | NA | NA |
| old_dev60 | all | empty | 目标类 | 11 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.4545 | NA | NA |
| old_dev60 | all | empty | 目标类 | 11 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 目标类 | 11 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 目标类 | 11 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 地表类 | 19 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3158 | NA | NA |
| old_dev60 | all | empty | 地表类 | 19 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3158 | NA | NA |
| old_dev60 | all | empty | 地表类 | 19 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 地表类 | 19 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | all | empty | 地表类 | 19 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 全部 | 30 | bridge@8 | 0.4398 | 0.2675 | 1.2353 | 0.4333 | NA | NA |
| old_dev60 | tune | all | 全部 | 30 | bridge_verified@8 | 0.4793 | 0.2973 | 1.2353 | 0.4333 | NA | NA |
| old_dev60 | tune | all | 全部 | 30 | dino@8 | 2.7533 | 1.4583 | 24.5765 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 全部 | 30 | dino_verified@8 | 2.8866 | 1.5334 | 24.5541 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 全部 | 30 | oracle_class_diagnostic@8 | 3.2657 | 1.7699 | 21.0864 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 目标类 | 15 | bridge@8 | 1.8491 | 1.4486 | 2.5555 | 0.4667 | NA | NA |
| old_dev60 | tune | all | 目标类 | 15 | bridge_verified@8 | 2.0769 | 1.7493 | 2.5555 | 0.4667 | NA | NA |
| old_dev60 | tune | all | 目标类 | 15 | dino@8 | 1.5463 | 0.8803 | 6.3503 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 目标类 | 15 | dino_verified@8 | 1.6507 | 0.9492 | 6.3246 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 目标类 | 15 | oracle_class_diagnostic@8 | 0.7621 | 0.4457 | 2.6275 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 地表类 | 15 | bridge@8 | 0.0012 | 0.0007 | 0.0048 | 0.4000 | NA | NA |
| old_dev60 | tune | all | 地表类 | 15 | bridge_verified@8 | 0.0013 | 0.0007 | 0.0048 | 0.4000 | NA | NA |
| old_dev60 | tune | all | 地表类 | 15 | dino@8 | 3.0976 | 1.6087 | 41.5632 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 地表类 | 15 | dino_verified@8 | 3.2297 | 1.6801 | 41.5441 | 0.0000 | NA | NA |
| old_dev60 | tune | all | 地表类 | 15 | oracle_class_diagnostic@8 | 4.1343 | 2.1851 | 38.2901 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 全部 | 11 | bridge@8 | 0.8669 | 0.6677 | 1.2353 | 0.5455 | NA | NA |
| old_dev60 | tune | positive | 全部 | 11 | bridge_verified@8 | 0.9776 | 0.8089 | 1.2353 | 0.5455 | NA | NA |
| old_dev60 | tune | positive | 全部 | 11 | dino@8 | 6.5355 | 3.7689 | 24.5765 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 全部 | 11 | dino_verified@8 | 6.9204 | 4.0278 | 24.5541 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 全部 | 11 | oracle_class_diagnostic@8 | 6.4773 | 3.8263 | 21.0864 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 目标类 | 6 | bridge@8 | 2.9497 | 3.4877 | 2.5555 | 0.5000 | NA | NA |
| old_dev60 | tune | positive | 目标类 | 6 | bridge_verified@8 | 3.1330 | 4.0476 | 2.5555 | 0.5000 | NA | NA |
| old_dev60 | tune | positive | 目标类 | 6 | dino@8 | 2.7942 | 1.7912 | 6.3503 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 目标类 | 6 | dino_verified@8 | 2.8908 | 1.8736 | 6.3246 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 目标类 | 6 | oracle_class_diagnostic@8 | 1.7076 | 1.2648 | 2.6275 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 地表类 | 5 | bridge@8 | 0.0025 | 0.0017 | 0.0048 | 0.6000 | NA | NA |
| old_dev60 | tune | positive | 地表类 | 5 | bridge_verified@8 | 0.0029 | 0.0020 | 0.0048 | 0.6000 | NA | NA |
| old_dev60 | tune | positive | 地表类 | 5 | dino@8 | 8.0751 | 4.4720 | 41.5632 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 地表类 | 5 | dino_verified@8 | 8.6266 | 4.8130 | 41.5441 | 0.0000 | NA | NA |
| old_dev60 | tune | positive | 地表类 | 5 | oracle_class_diagnostic@8 | 7.8860 | 4.3957 | 38.2901 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 全部 | 19 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3684 | NA | NA |
| old_dev60 | tune | empty | 全部 | 19 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3684 | NA | NA |
| old_dev60 | tune | empty | 全部 | 19 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 全部 | 19 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 全部 | 19 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 目标类 | 9 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.4444 | NA | NA |
| old_dev60 | tune | empty | 目标类 | 9 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.4444 | NA | NA |
| old_dev60 | tune | empty | 目标类 | 9 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 目标类 | 9 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 目标类 | 9 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 地表类 | 10 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | NA | NA |
| old_dev60 | tune | empty | 地表类 | 10 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | NA | NA |
| old_dev60 | tune | empty | 地表类 | 10 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 地表类 | 10 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | tune | empty | 地表类 | 10 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 全部 | 30 | bridge@8 | 6.4393 | 8.9307 | 5.0348 | 0.4000 | NA | NA |
| old_dev60 | confirm | all | 全部 | 30 | bridge_verified@8 | 6.8134 | 10.5351 | 5.0348 | 0.4000 | NA | NA |
| old_dev60 | confirm | all | 全部 | 30 | dino@8 | 7.5120 | 4.8904 | 16.1929 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 全部 | 30 | dino_verified@8 | 8.2693 | 5.5744 | 16.0085 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 全部 | 30 | oracle_class_diagnostic@8 | 20.0170 | 17.5622 | 23.2694 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 目标类 | 12 | bridge@8 | 12.2633 | 31.9276 | 7.5892 | 0.6667 | NA | NA |
| old_dev60 | confirm | all | 目标类 | 12 | bridge_verified@8 | 12.2649 | 31.9482 | 7.5892 | 0.6667 | NA | NA |
| old_dev60 | confirm | all | 目标类 | 12 | dino@8 | 18.2580 | 14.4870 | 24.6831 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 目标类 | 12 | dino_verified@8 | 22.9726 | 21.4838 | 24.6831 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 目标类 | 12 | oracle_class_diagnostic@8 | 38.1339 | 41.4166 | 35.3333 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 地表类 | 18 | bridge@8 | 0.9154 | 0.8797 | 0.9541 | 0.2222 | NA | NA |
| old_dev60 | confirm | all | 地表类 | 18 | bridge_verified@8 | 1.0249 | 1.1070 | 0.9541 | 0.2222 | NA | NA |
| old_dev60 | confirm | all | 地表类 | 18 | dino@8 | 0.7646 | 0.4473 | 2.6299 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 地表类 | 18 | dino_verified@8 | 0.6494 | 0.3825 | 2.1509 | 0.0000 | NA | NA |
| old_dev60 | confirm | all | 地表类 | 18 | oracle_class_diagnostic@8 | 2.5969 | 1.9231 | 3.9975 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 全部 | 19 | bridge@8 | 7.7723 | 17.0341 | 5.0348 | 0.4211 | NA | NA |
| old_dev60 | confirm | positive | 全部 | 19 | bridge_verified@8 | 8.1837 | 21.8494 | 5.0348 | 0.4211 | NA | NA |
| old_dev60 | confirm | positive | 全部 | 19 | dino@8 | 9.7914 | 7.0173 | 16.1929 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 全部 | 19 | dino_verified@8 | 10.9665 | 8.3398 | 16.0085 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 全部 | 19 | oracle_class_diagnostic@8 | 25.4426 | 28.0636 | 23.2694 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 目标类 | 10 | bridge@8 | 13.2330 | 51.6240 | 7.5892 | 0.7000 | NA | NA |
| old_dev60 | confirm | positive | 目标类 | 10 | bridge_verified@8 | 13.2347 | 51.6777 | 7.5892 | 0.7000 | NA | NA |
| old_dev60 | confirm | positive | 目标类 | 10 | dino@8 | 19.6780 | 16.3605 | 24.6831 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 目标类 | 10 | dino_verified@8 | 24.9211 | 25.1637 | 24.6831 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 目标类 | 10 | oracle_class_diagnostic@8 | 40.8344 | 48.3644 | 35.3333 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 地表类 | 9 | bridge@8 | 1.2449 | 1.7904 | 0.9541 | 0.1111 | NA | NA |
| old_dev60 | confirm | positive | 地表类 | 9 | bridge_verified@8 | 1.3991 | 2.6216 | 0.9541 | 0.1111 | NA | NA |
| old_dev60 | confirm | positive | 地表类 | 9 | dino@8 | 1.1475 | 0.7338 | 2.6299 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 地表类 | 9 | dino_verified@8 | 0.9735 | 0.6291 | 2.1509 | 0.0000 | NA | NA |
| old_dev60 | confirm | positive | 地表类 | 9 | oracle_class_diagnostic@8 | 4.0243 | 4.0515 | 3.9975 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 全部 | 11 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3636 | NA | NA |
| old_dev60 | confirm | empty | 全部 | 11 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3636 | NA | NA |
| old_dev60 | confirm | empty | 全部 | 11 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 全部 | 11 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 全部 | 11 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 目标类 | 2 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | NA | NA |
| old_dev60 | confirm | empty | 目标类 | 2 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | NA | NA |
| old_dev60 | confirm | empty | 目标类 | 2 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 目标类 | 2 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 目标类 | 2 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 地表类 | 9 | bridge@8 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | NA | NA |
| old_dev60 | confirm | empty | 地表类 | 9 | bridge_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | NA | NA |
| old_dev60 | confirm | empty | 地表类 | 9 | dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 地表类 | 9 | dino_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| old_dev60 | confirm | empty | 地表类 | 9 | oracle_class_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | all | 全部 | 60 | delta@8 | 5.3603 | 3.1621 | 17.5861 | 0.0000 | 0.1759 | 0.0985 |
| endpoint | all | all | 全部 | 60 | delta_example_swapped@8 | 0.1875 | 0.1430 | 0.2725 | 0.0000 | 0.0027 | 0.0076 |
| endpoint | all | all | 全部 | 60 | endpoint@8 | 4.6349 | 2.6738 | 17.3862 | 0.0000 | 0.1739 | 0.0833 |
| endpoint | all | all | 全部 | 60 | endpoint_example_swapped@8 | 1.7877 | 1.0007 | 8.3693 | 0.0000 | 0.0837 | 0.0379 |
| endpoint | all | all | 全部 | 60 | equal_joint@8 | 5.8079 | 3.4243 | 19.1093 | 0.0000 | 0.1911 | 0.0758 |
| endpoint | all | all | 全部 | 60 | equal_joint_example_swapped@8 | 0.2084 | 0.1513 | 0.3347 | 0.0000 | 0.0033 | 0.0076 |
| endpoint | all | all | 目标类 | 27 | delta@8 | 12.8770 | 9.0678 | 22.2045 | 0.0000 | 0.2220 | 0.0805 |
| endpoint | all | all | 目标类 | 27 | delta_example_swapped@8 | 0.3207 | 0.2909 | 0.3573 | 0.0000 | 0.0036 | 0.0115 |
| endpoint | all | all | 目标类 | 27 | endpoint@8 | 8.9338 | 5.6010 | 22.0606 | 0.0000 | 0.2206 | 0.0575 |
| endpoint | all | all | 目标类 | 27 | endpoint_example_swapped@8 | 0.6651 | 0.3798 | 2.6743 | 0.0000 | 0.0267 | 0.0000 |
| endpoint | all | all | 目标类 | 27 | equal_joint@8 | 14.2148 | 9.9774 | 24.7085 | 0.0000 | 0.2471 | 0.0575 |
| endpoint | all | all | 目标类 | 27 | equal_joint_example_swapped@8 | 0.1341 | 0.1148 | 0.1613 | 0.0000 | 0.0016 | 0.0000 |
| endpoint | all | all | 地表类 | 33 | delta@8 | 1.9570 | 1.0754 | 10.8578 | 0.0000 | 0.1086 | 0.1333 |
| endpoint | all | all | 地表类 | 33 | delta_example_swapped@8 | 0.0765 | 0.0515 | 0.1489 | 0.0000 | 0.0015 | 0.0000 |
| endpoint | all | all | 地表类 | 33 | endpoint@8 | 1.8824 | 1.0331 | 10.5763 | 0.0000 | 0.1058 | 0.1333 |
| endpoint | all | all | 地表类 | 33 | endpoint_example_swapped@8 | 2.9528 | 1.6199 | 16.6660 | 0.0000 | 0.1667 | 0.1111 |
| endpoint | all | all | 地表类 | 33 | equal_joint@8 | 1.9729 | 1.0841 | 10.9520 | 0.0000 | 0.1095 | 0.1111 |
| endpoint | all | all | 地表类 | 33 | equal_joint_example_swapped@8 | 0.2676 | 0.1733 | 0.5874 | 0.0000 | 0.0059 | 0.0222 |
| endpoint | all | positive | 全部 | 30 | delta@8 | 8.7760 | 5.8469 | 17.5861 | 0.0000 | 0.1759 | 0.0985 |
| endpoint | all | positive | 全部 | 30 | delta_example_swapped@8 | 0.2524 | 0.2351 | 0.2725 | 0.0000 | 0.0027 | 0.0076 |
| endpoint | all | positive | 全部 | 30 | endpoint@8 | 8.3899 | 5.5290 | 17.3862 | 0.0000 | 0.1739 | 0.0833 |
| endpoint | all | positive | 全部 | 30 | endpoint_example_swapped@8 | 2.9786 | 1.8117 | 8.3693 | 0.0000 | 0.0837 | 0.0379 |
| endpoint | all | positive | 全部 | 30 | equal_joint@8 | 9.1672 | 6.0299 | 19.1093 | 0.0000 | 0.1911 | 0.0758 |
| endpoint | all | positive | 全部 | 30 | equal_joint_example_swapped@8 | 0.2974 | 0.2676 | 0.3347 | 0.0000 | 0.0033 | 0.0076 |
| endpoint | all | positive | 目标类 | 16 | delta@8 | 15.9513 | 12.4462 | 22.2045 | 0.0000 | 0.2220 | 0.0805 |
| endpoint | all | positive | 目标类 | 16 | delta_example_swapped@8 | 0.3981 | 0.4494 | 0.3573 | 0.0000 | 0.0036 | 0.0115 |
| endpoint | all | positive | 目标类 | 16 | endpoint@8 | 15.5779 | 12.0398 | 22.0606 | 0.0000 | 0.2206 | 0.0575 |
| endpoint | all | positive | 目标类 | 16 | endpoint_example_swapped@8 | 1.1983 | 0.7721 | 2.6743 | 0.0000 | 0.0267 | 0.0000 |
| endpoint | all | positive | 目标类 | 16 | equal_joint@8 | 16.8540 | 12.7887 | 24.7085 | 0.0000 | 0.2471 | 0.0575 |
| endpoint | all | positive | 目标类 | 16 | equal_joint_example_swapped@8 | 0.1789 | 0.2007 | 0.1613 | 0.0000 | 0.0016 | 0.0000 |
| endpoint | all | positive | 地表类 | 14 | delta@8 | 3.7502 | 2.2665 | 10.8578 | 0.0000 | 0.1086 | 0.1333 |
| endpoint | all | positive | 地表类 | 14 | delta_example_swapped@8 | 0.1107 | 0.0882 | 0.1489 | 0.0000 | 0.0015 | 0.0000 |
| endpoint | all | positive | 地表类 | 14 | endpoint@8 | 3.4927 | 2.0917 | 10.5763 | 0.0000 | 0.1058 | 0.1333 |
| endpoint | all | positive | 地表类 | 14 | endpoint_example_swapped@8 | 4.5637 | 2.6439 | 16.6660 | 0.0000 | 0.1667 | 0.1111 |
| endpoint | all | positive | 地表类 | 14 | equal_joint@8 | 3.6683 | 2.2031 | 10.9520 | 0.0000 | 0.1095 | 0.1111 |
| endpoint | all | positive | 地表类 | 14 | equal_joint_example_swapped@8 | 0.4048 | 0.3088 | 0.5874 | 0.0000 | 0.0059 | 0.0222 |
| endpoint | all | empty | 全部 | 30 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 全部 | 30 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 全部 | 30 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 全部 | 30 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 全部 | 30 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 全部 | 30 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 目标类 | 11 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | all | empty | 地表类 | 19 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | all | 全部 | 30 | delta@8 | 2.7533 | 1.4583 | 24.5765 | 0.0000 | 0.2458 | 0.2121 |
| endpoint | tune | all | 全部 | 30 | delta_example_swapped@8 | 0.0977 | 0.0564 | 0.3646 | 0.0000 | 0.0036 | 0.0000 |
| endpoint | tune | all | 全部 | 30 | endpoint@8 | 2.0611 | 1.0810 | 22.0687 | 0.0000 | 0.2207 | 0.1515 |
| endpoint | tune | all | 全部 | 30 | endpoint_example_swapped@8 | 2.3852 | 1.2325 | 36.8126 | 0.0000 | 0.3681 | 0.0909 |
| endpoint | tune | all | 全部 | 30 | equal_joint@8 | 4.8179 | 2.5491 | 43.8249 | 0.0000 | 0.4382 | 0.1515 |
| endpoint | tune | all | 全部 | 30 | equal_joint_example_swapped@8 | 0.0783 | 0.0439 | 0.3646 | 0.0000 | 0.0036 | 0.0000 |
| endpoint | tune | all | 目标类 | 15 | delta@8 | 1.5463 | 0.8803 | 6.3503 | 0.0000 | 0.0635 | 0.1111 |
| endpoint | tune | all | 目标类 | 15 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | tune | all | 目标类 | 15 | endpoint@8 | 0.5316 | 0.2851 | 3.9233 | 0.0000 | 0.0392 | 0.0556 |
| endpoint | tune | all | 目标类 | 15 | endpoint_example_swapped@8 | 0.2304 | 0.1187 | 3.8924 | 0.0000 | 0.0389 | 0.0000 |
| endpoint | tune | all | 目标类 | 15 | equal_joint@8 | 11.4393 | 6.4729 | 49.1516 | 0.0000 | 0.4915 | 0.0556 |
| endpoint | tune | all | 目标类 | 15 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | tune | all | 地表类 | 15 | delta@8 | 3.0976 | 1.6087 | 41.5632 | 0.0000 | 0.4156 | 0.3333 |
| endpoint | tune | all | 地表类 | 15 | delta_example_swapped@8 | 0.1676 | 0.0951 | 0.7045 | 0.0000 | 0.0070 | 0.0000 |
| endpoint | tune | all | 地表类 | 15 | endpoint@8 | 2.8231 | 1.4646 | 38.9802 | 0.0000 | 0.3898 | 0.2667 |
| endpoint | tune | all | 地表类 | 15 | endpoint_example_swapped@8 | 4.7957 | 2.4862 | 67.4941 | 0.0000 | 0.6749 | 0.2000 |
| endpoint | tune | all | 地表类 | 15 | equal_joint@8 | 2.8638 | 1.4867 | 38.8604 | 0.0000 | 0.3886 | 0.2667 |
| endpoint | tune | all | 地表类 | 15 | equal_joint_example_swapped@8 | 0.1207 | 0.0660 | 0.7045 | 0.0000 | 0.0070 | 0.0000 |
| endpoint | tune | positive | 全部 | 11 | delta@8 | 6.5355 | 3.7689 | 24.5765 | 0.0000 | 0.2458 | 0.2121 |
| endpoint | tune | positive | 全部 | 11 | delta_example_swapped@8 | 0.1716 | 0.1122 | 0.3646 | 0.0000 | 0.0036 | 0.0000 |
| endpoint | tune | positive | 全部 | 11 | endpoint@8 | 4.7270 | 2.6470 | 22.0687 | 0.0000 | 0.2207 | 0.1515 |
| endpoint | tune | positive | 全部 | 11 | endpoint_example_swapped@8 | 4.4629 | 2.3755 | 36.8126 | 0.0000 | 0.3681 | 0.0909 |
| endpoint | tune | positive | 全部 | 11 | equal_joint@8 | 10.7655 | 6.1364 | 43.8249 | 0.0000 | 0.4382 | 0.1515 |
| endpoint | tune | positive | 全部 | 11 | equal_joint_example_swapped@8 | 0.1274 | 0.0772 | 0.3646 | 0.0000 | 0.0036 | 0.0000 |
| endpoint | tune | positive | 目标类 | 6 | delta@8 | 2.7942 | 1.7912 | 6.3503 | 0.0000 | 0.0635 | 0.1111 |
| endpoint | tune | positive | 目标类 | 6 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | tune | positive | 目标类 | 6 | endpoint@8 | 1.0582 | 0.6116 | 3.9233 | 0.0000 | 0.0392 | 0.0556 |
| endpoint | tune | positive | 目标类 | 6 | endpoint_example_swapped@8 | 0.4522 | 0.2401 | 3.8924 | 0.0000 | 0.0389 | 0.0000 |
| endpoint | tune | positive | 目标类 | 6 | equal_joint@8 | 16.9555 | 10.2448 | 49.1516 | 0.0000 | 0.4915 | 0.0556 |
| endpoint | tune | positive | 目标类 | 6 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | tune | positive | 地表类 | 5 | delta@8 | 8.0751 | 4.4720 | 41.5632 | 0.0000 | 0.4156 | 0.3333 |
| endpoint | tune | positive | 地表类 | 5 | delta_example_swapped@8 | 0.3004 | 0.1909 | 0.7045 | 0.0000 | 0.0070 | 0.0000 |
| endpoint | tune | positive | 地表类 | 5 | endpoint@8 | 7.0051 | 3.8483 | 38.9802 | 0.0000 | 0.3898 | 0.2667 |
| endpoint | tune | positive | 地表类 | 5 | endpoint_example_swapped@8 | 8.5285 | 4.5518 | 67.4941 | 0.0000 | 0.6749 | 0.2000 |
| endpoint | tune | positive | 地表类 | 5 | equal_joint@8 | 7.5265 | 4.1667 | 38.8604 | 0.0000 | 0.3886 | 0.2667 |
| endpoint | tune | positive | 地表类 | 5 | equal_joint_example_swapped@8 | 0.1895 | 0.1095 | 0.7045 | 0.0000 | 0.0070 | 0.0000 |
| endpoint | tune | empty | 全部 | 19 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 全部 | 19 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 全部 | 19 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 全部 | 19 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 全部 | 19 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 全部 | 19 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 目标类 | 9 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | tune | empty | 地表类 | 10 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | all | 全部 | 30 | delta@8 | 7.5120 | 4.8904 | 16.1929 | 0.0000 | 0.1619 | 0.0606 |
| endpoint | confirm | all | 全部 | 30 | delta_example_swapped@8 | 0.2544 | 0.2547 | 0.2541 | 0.0000 | 0.0025 | 0.0101 |
| endpoint | confirm | all | 全部 | 30 | endpoint@8 | 6.9575 | 4.4115 | 16.4529 | 0.0000 | 0.1645 | 0.0606 |
| endpoint | confirm | all | 全部 | 30 | endpoint_example_swapped@8 | 1.0637 | 0.6623 | 2.7003 | 0.0000 | 0.0270 | 0.0202 |
| endpoint | confirm | all | 全部 | 30 | equal_joint@8 | 6.6494 | 4.3427 | 14.1832 | 0.0000 | 0.1418 | 0.0505 |
| endpoint | confirm | all | 全部 | 30 | equal_joint_example_swapped@8 | 0.3293 | 0.3298 | 0.3288 | 0.0000 | 0.0033 | 0.0101 |
| endpoint | confirm | all | 目标类 | 12 | delta@8 | 18.2580 | 14.4870 | 24.6831 | 0.0000 | 0.2468 | 0.0725 |
| endpoint | confirm | all | 目标类 | 12 | delta_example_swapped@8 | 0.5268 | 0.7266 | 0.4132 | 0.0000 | 0.0041 | 0.0145 |
| endpoint | confirm | all | 目标类 | 12 | endpoint@8 | 14.6309 | 10.3595 | 24.8961 | 0.0000 | 0.2490 | 0.0580 |
| endpoint | confirm | all | 目标类 | 12 | endpoint_example_swapped@8 | 1.2367 | 0.8233 | 2.4839 | 0.0000 | 0.0248 | 0.0000 |
| endpoint | confirm | all | 目标类 | 12 | equal_joint@8 | 15.6079 | 12.4589 | 20.8873 | 0.0000 | 0.2089 | 0.0580 |
| endpoint | confirm | all | 目标类 | 12 | equal_joint_example_swapped@8 | 0.2168 | 0.2588 | 0.1865 | 0.0000 | 0.0019 | 0.0000 |
| endpoint | confirm | all | 地表类 | 18 | delta@8 | 0.7646 | 0.4473 | 2.6299 | 0.0000 | 0.0263 | 0.0333 |
| endpoint | confirm | all | 地表类 | 18 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | confirm | all | 地表类 | 18 | endpoint@8 | 0.8659 | 0.5070 | 2.9651 | 0.0000 | 0.0297 | 0.0667 |
| endpoint | confirm | all | 地表类 | 18 | endpoint_example_swapped@8 | 0.8998 | 0.5279 | 3.0460 | 0.0000 | 0.0305 | 0.0667 |
| endpoint | confirm | all | 地表类 | 18 | equal_joint@8 | 1.0209 | 0.5984 | 3.4736 | 0.0000 | 0.0347 | 0.0333 |
| endpoint | confirm | all | 地表类 | 18 | equal_joint_example_swapped@8 | 0.4562 | 0.3867 | 0.5560 | 0.0000 | 0.0056 | 0.0333 |
| endpoint | confirm | positive | 全部 | 19 | delta@8 | 9.7914 | 7.0173 | 16.1929 | 0.0000 | 0.1619 | 0.0606 |
| endpoint | confirm | positive | 全部 | 19 | delta_example_swapped@8 | 0.2917 | 0.3423 | 0.2541 | 0.0000 | 0.0025 | 0.0101 |
| endpoint | confirm | positive | 全部 | 19 | endpoint@8 | 10.5822 | 7.7992 | 16.4529 | 0.0000 | 0.1645 | 0.0606 |
| endpoint | confirm | positive | 全部 | 19 | endpoint_example_swapped@8 | 1.5646 | 1.1014 | 2.7003 | 0.0000 | 0.0270 | 0.0202 |
| endpoint | confirm | positive | 全部 | 19 | equal_joint@8 | 8.3992 | 5.9661 | 14.1832 | 0.0000 | 0.1418 | 0.0505 |
| endpoint | confirm | positive | 全部 | 19 | equal_joint_example_swapped@8 | 0.4219 | 0.5885 | 0.3288 | 0.0000 | 0.0033 | 0.0101 |
| endpoint | confirm | positive | 目标类 | 10 | delta@8 | 19.6780 | 16.3605 | 24.6831 | 0.0000 | 0.2468 | 0.0725 |
| endpoint | confirm | positive | 目标类 | 10 | delta_example_swapped@8 | 0.5563 | 0.8513 | 0.4132 | 0.0000 | 0.0041 | 0.0145 |
| endpoint | confirm | positive | 目标类 | 10 | endpoint@8 | 23.5325 | 22.3105 | 24.8961 | 0.0000 | 0.2490 | 0.0580 |
| endpoint | confirm | positive | 目标类 | 10 | endpoint_example_swapped@8 | 2.0111 | 1.6895 | 2.4839 | 0.0000 | 0.0248 | 0.0000 |
| endpoint | confirm | positive | 目标类 | 10 | equal_joint@8 | 16.8170 | 14.0744 | 20.8873 | 0.0000 | 0.2089 | 0.0580 |
| endpoint | confirm | positive | 目标类 | 10 | equal_joint_example_swapped@8 | 0.2524 | 0.3903 | 0.1865 | 0.0000 | 0.0019 | 0.0000 |
| endpoint | confirm | positive | 地表类 | 9 | delta@8 | 1.1475 | 0.7338 | 2.6299 | 0.0000 | 0.0263 | 0.0333 |
| endpoint | confirm | positive | 地表类 | 9 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| endpoint | confirm | positive | 地表类 | 9 | endpoint@8 | 1.2626 | 0.8020 | 2.9651 | 0.0000 | 0.0297 | 0.0667 |
| endpoint | confirm | positive | 地表类 | 9 | endpoint_example_swapped@8 | 1.2137 | 0.7578 | 3.0460 | 0.0000 | 0.0305 | 0.0667 |
| endpoint | confirm | positive | 地表类 | 9 | equal_joint@8 | 1.4461 | 0.9131 | 3.4736 | 0.0000 | 0.0347 | 0.0333 |
| endpoint | confirm | positive | 地表类 | 9 | equal_joint_example_swapped@8 | 0.6590 | 0.8087 | 0.5560 | 0.0000 | 0.0056 | 0.0333 |
| endpoint | confirm | empty | 全部 | 11 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 全部 | 11 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 全部 | 11 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 全部 | 11 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 全部 | 11 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 全部 | 11 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 目标类 | 2 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | delta@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | delta_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | endpoint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | endpoint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | equal_joint@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| endpoint | confirm | empty | 地表类 | 9 | equal_joint_example_swapped@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | all | all | 全部 | 60 | R1_geometry_dino@8 | 9.8282 | 6.5419 | 19.7490 | 0.0000 | 0.1975 | 0.1136 |
| retrieval/B0 | all | all | 全部 | 60 | R2_native_order@8 | 9.3130 | 54.2882 | 5.0934 | 0.8500 | 0.0509 | 0.0152 |
| retrieval/B0 | all | all | 全部 | 60 | R3_dino_noverify@8 | 6.9473 | 46.6920 | 3.7528 | 0.8500 | 0.0375 | 0.0152 |
| retrieval/B0 | all | all | 全部 | 60 | R3_drop_unparsed_diagnostic@8 | 6.9650 | 48.3458 | 3.7528 | 0.9500 | 0.0375 | 0.0152 |
| retrieval/B0 | all | all | 全部 | 60 | R3_verified@8 | 6.9650 | 48.3458 | 3.7528 | 0.9500 | 0.0375 | 0.0152 |
| retrieval/B0 | all | all | 目标类 | 27 | R1_geometry_dino@8 | 18.0623 | 14.2769 | 24.5792 | 0.0000 | 0.2458 | 0.0920 |
| retrieval/B0 | all | all | 目标类 | 27 | R2_native_order@8 | 15.3793 | 73.3991 | 8.5896 | 0.8148 | 0.0859 | 0.0230 |
| retrieval/B0 | all | all | 目标类 | 27 | R3_dino_noverify@8 | 11.5664 | 67.0793 | 6.3289 | 0.8148 | 0.0633 | 0.0230 |
| retrieval/B0 | all | all | 目标类 | 27 | R3_drop_unparsed_diagnostic@8 | 11.6062 | 69.8534 | 6.3289 | 0.9259 | 0.0633 | 0.0230 |
| retrieval/B0 | all | all | 目标类 | 27 | R3_verified@8 | 11.6062 | 69.8534 | 6.3289 | 0.9259 | 0.0633 | 0.0230 |
| retrieval/B0 | all | all | 地表类 | 33 | R1_geometry_dino@8 | 4.3029 | 2.5897 | 12.7122 | 0.0000 | 0.1271 | 0.1556 |
| retrieval/B0 | all | all | 地表类 | 33 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8788 | 0.0000 | 0.0000 |
| retrieval/B0 | all | all | 地表类 | 33 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8788 | 0.0000 | 0.0000 |
| retrieval/B0 | all | all | 地表类 | 33 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9697 | 0.0000 | 0.0000 |
| retrieval/B0 | all | all | 地表类 | 33 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9697 | 0.0000 | 0.0000 |
| retrieval/B0 | all | positive | 全部 | 30 | R1_geometry_dino@8 | 14.7637 | 11.7881 | 19.7490 | 0.0000 | 0.1975 | 0.1136 |
| retrieval/B0 | all | positive | 全部 | 30 | R2_native_order@8 | 9.4505 | 65.3722 | 5.0934 | 0.8000 | 0.0509 | 0.0152 |
| retrieval/B0 | all | positive | 全部 | 30 | R3_dino_noverify@8 | 7.0511 | 58.2134 | 3.7528 | 0.8000 | 0.0375 | 0.0152 |
| retrieval/B0 | all | positive | 全部 | 30 | R3_drop_unparsed_diagnostic@8 | 7.0655 | 60.2355 | 3.7528 | 0.9333 | 0.0375 | 0.0152 |
| retrieval/B0 | all | positive | 全部 | 30 | R3_verified@8 | 7.0655 | 60.2355 | 3.7528 | 0.9333 | 0.0375 | 0.0152 |
| retrieval/B0 | all | positive | 目标类 | 16 | R1_geometry_dino@8 | 25.7142 | 26.9590 | 24.5792 | 0.0000 | 0.2458 | 0.0920 |
| retrieval/B0 | all | positive | 目标类 | 16 | R2_native_order@8 | 15.7469 | 94.4432 | 8.5896 | 0.8125 | 0.0859 | 0.0230 |
| retrieval/B0 | all | positive | 目标类 | 16 | R3_dino_noverify@8 | 11.8488 | 92.6993 | 6.3289 | 0.8125 | 0.0633 | 0.0230 |
| retrieval/B0 | all | positive | 目标类 | 16 | R3_drop_unparsed_diagnostic@8 | 11.8878 | 97.7243 | 6.3289 | 0.9375 | 0.0633 | 0.0230 |
| retrieval/B0 | all | positive | 目标类 | 16 | R3_verified@8 | 11.8878 | 97.7243 | 6.3289 | 0.9375 | 0.0633 | 0.0230 |
| retrieval/B0 | all | positive | 地表类 | 14 | R1_geometry_dino@8 | 6.7121 | 4.5599 | 12.7122 | 0.0000 | 0.1271 | 0.1556 |
| retrieval/B0 | all | positive | 地表类 | 14 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.7857 | 0.0000 | 0.0000 |
| retrieval/B0 | all | positive | 地表类 | 14 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.7857 | 0.0000 | 0.0000 |
| retrieval/B0 | all | positive | 地表类 | 14 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9286 | 0.0000 | 0.0000 |
| retrieval/B0 | all | positive | 地表类 | 14 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9286 | 0.0000 | 0.0000 |
| retrieval/B0 | all | empty | 全部 | 30 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | all | empty | 全部 | 30 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | NA | NA |
| retrieval/B0 | all | empty | 全部 | 30 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | NA | NA |
| retrieval/B0 | all | empty | 全部 | 30 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9667 | NA | NA |
| retrieval/B0 | all | empty | 全部 | 30 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9667 | NA | NA |
| retrieval/B0 | all | empty | 目标类 | 11 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | all | empty | 目标类 | 11 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | NA | NA |
| retrieval/B0 | all | empty | 目标类 | 11 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | NA | NA |
| retrieval/B0 | all | empty | 目标类 | 11 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9091 | NA | NA |
| retrieval/B0 | all | empty | 目标类 | 11 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9091 | NA | NA |
| retrieval/B0 | all | empty | 地表类 | 19 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | all | empty | 地表类 | 19 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | all | empty | 地表类 | 19 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | all | empty | 地表类 | 19 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | all | empty | 地表类 | 19 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | tune | all | 全部 | 30 | R1_geometry_dino@8 | 5.3656 | 3.0147 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| retrieval/B0 | tune | all | 全部 | 30 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 全部 | 30 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 全部 | 30 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9667 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 全部 | 30 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9667 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 目标类 | 15 | R1_geometry_dino@8 | 0.8638 | 0.4929 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| retrieval/B0 | tune | all | 目标类 | 15 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 目标类 | 15 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 目标类 | 15 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9333 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 目标类 | 15 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9333 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 地表类 | 15 | R1_geometry_dino@8 | 8.7429 | 4.8556 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| retrieval/B0 | tune | all | 地表类 | 15 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 地表类 | 15 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 地表类 | 15 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | all | 地表类 | 15 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 全部 | 11 | R1_geometry_dino@8 | 11.2769 | 7.3354 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| retrieval/B0 | tune | positive | 全部 | 11 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 全部 | 11 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 全部 | 11 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 全部 | 11 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 目标类 | 6 | R1_geometry_dino@8 | 2.2476 | 1.6584 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| retrieval/B0 | tune | positive | 目标类 | 6 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 目标类 | 6 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 目标类 | 6 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 目标类 | 6 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 地表类 | 5 | R1_geometry_dino@8 | 16.0581 | 9.8293 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| retrieval/B0 | tune | positive | 地表类 | 5 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 地表类 | 5 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 地表类 | 5 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | positive | 地表类 | 5 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| retrieval/B0 | tune | empty | 全部 | 19 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | tune | empty | 全部 | 19 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | tune | empty | 全部 | 19 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | tune | empty | 全部 | 19 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | tune | empty | 全部 | 19 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9474 | NA | NA |
| retrieval/B0 | tune | empty | 目标类 | 9 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | tune | empty | 目标类 | 9 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | tune | empty | 目标类 | 9 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | tune | empty | 目标类 | 9 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | tune | empty | 目标类 | 9 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | tune | empty | 地表类 | 10 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | tune | empty | 地表类 | 10 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | tune | empty | 地表类 | 10 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | tune | empty | 地表类 | 10 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | tune | empty | 地表类 | 10 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | all | 全部 | 30 | R1_geometry_dino@8 | 12.5135 | 9.3709 | 18.8274 | 0.0000 | 0.1883 | 0.0808 |
| retrieval/B0 | confirm | all | 全部 | 30 | R2_native_order@8 | 11.1931 | 66.7711 | 6.1085 | 0.8000 | 0.0611 | 0.0202 |
| retrieval/B0 | confirm | all | 全部 | 30 | R3_dino_noverify@8 | 8.3708 | 59.7258 | 4.5008 | 0.8000 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | all | 全部 | 30 | R3_drop_unparsed_diagnostic@8 | 8.3758 | 60.2355 | 4.5008 | 0.9333 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | all | 全部 | 30 | R3_verified@8 | 8.3758 | 60.2355 | 4.5008 | 0.9333 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | all | 目标类 | 12 | R1_geometry_dino@8 | 29.5753 | 31.4943 | 27.8768 | 0.0000 | 0.2788 | 0.0870 |
| retrieval/B0 | confirm | all | 目标类 | 12 | R2_native_order@8 | 18.0406 | 98.2272 | 9.9324 | 0.8333 | 0.0993 | 0.0290 |
| retrieval/B0 | confirm | all | 目标类 | 12 | R3_dino_noverify@8 | 13.6167 | 97.7139 | 7.3183 | 0.8333 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | all | 目标类 | 12 | R3_drop_unparsed_diagnostic@8 | 13.6168 | 97.7243 | 7.3183 | 0.9167 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | all | 目标类 | 12 | R3_verified@8 | 13.6168 | 97.7243 | 7.3183 | 0.9167 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | all | 地表类 | 18 | R1_geometry_dino@8 | 1.8195 | 1.1489 | 4.3712 | 0.0000 | 0.0437 | 0.0667 |
| retrieval/B0 | confirm | all | 地表类 | 18 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.7778 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | all | 地表类 | 18 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.7778 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | all | 地表类 | 18 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.9444 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | all | 地表类 | 18 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.9444 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | positive | 全部 | 19 | R1_geometry_dino@8 | 16.0437 | 13.9771 | 18.8274 | 0.0000 | 0.1883 | 0.0808 |
| retrieval/B0 | confirm | positive | 全部 | 19 | R2_native_order@8 | 11.1986 | 67.1668 | 6.1085 | 0.7895 | 0.0611 | 0.0202 |
| retrieval/B0 | confirm | positive | 全部 | 19 | R3_dino_noverify@8 | 8.3750 | 60.1559 | 4.5008 | 0.7895 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | positive | 全部 | 19 | R3_drop_unparsed_diagnostic@8 | 8.3758 | 60.2355 | 4.5008 | 0.8947 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | positive | 全部 | 19 | R3_verified@8 | 8.3758 | 60.2355 | 4.5008 | 0.8947 | 0.0450 | 0.0202 |
| retrieval/B0 | confirm | positive | 目标类 | 10 | R1_geometry_dino@8 | 32.3094 | 38.4182 | 27.8768 | 0.0000 | 0.2788 | 0.0870 |
| retrieval/B0 | confirm | positive | 目标类 | 10 | R2_native_order@8 | 18.0407 | 98.2350 | 9.9324 | 0.9000 | 0.0993 | 0.0290 |
| retrieval/B0 | confirm | positive | 目标类 | 10 | R3_dino_noverify@8 | 13.6168 | 97.7243 | 7.3183 | 0.9000 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | positive | 目标类 | 10 | R3_drop_unparsed_diagnostic@8 | 13.6168 | 97.7243 | 7.3183 | 0.9000 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | positive | 目标类 | 10 | R3_verified@8 | 13.6168 | 97.7243 | 7.3183 | 0.9000 | 0.0732 | 0.0290 |
| retrieval/B0 | confirm | positive | 地表类 | 9 | R1_geometry_dino@8 | 2.6177 | 1.8683 | 4.3712 | 0.0000 | 0.0437 | 0.0667 |
| retrieval/B0 | confirm | positive | 地表类 | 9 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | positive | 地表类 | 9 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | positive | 地表类 | 9 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | positive | 地表类 | 9 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | 0.0000 | 0.0000 |
| retrieval/B0 | confirm | empty | 全部 | 11 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 全部 | 11 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | NA | NA |
| retrieval/B0 | confirm | empty | 全部 | 11 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8182 | NA | NA |
| retrieval/B0 | confirm | empty | 全部 | 11 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 全部 | 11 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 目标类 | 2 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 目标类 | 2 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | NA | NA |
| retrieval/B0 | confirm | empty | 目标类 | 2 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | NA | NA |
| retrieval/B0 | confirm | empty | 目标类 | 2 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 目标类 | 2 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 地表类 | 9 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 地表类 | 9 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | confirm | empty | 地表类 | 9 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 0.8889 | NA | NA |
| retrieval/B0 | confirm | empty | 地表类 | 9 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| retrieval/B0 | confirm | empty | 地表类 | 9 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | NA | NA |
| t33/retrieval/B0 | all | all | 全部 | 12 | R1_geometry_dino@8 | 8.2893 | 15.8909 | 5.6071 | 0.0000 | 0.0561 | 0.0727 |
| t33/retrieval/B0 | all | all | 全部 | 12 | R2_native_order@8 | 7.9128 | 95.0068 | 4.1283 | 0.9167 | 0.0413 | 0.0545 |
| t33/retrieval/B0 | all | all | 全部 | 12 | R3_dino_noverify@8 | 7.9128 | 95.0068 | 4.1283 | 0.9167 | 0.0413 | 0.0545 |
| t33/retrieval/B0 | all | all | 全部 | 12 | R3_drop_unparsed_diagnostic@8 | 3.2594 | 93.4298 | 1.6587 | 0.9167 | 0.0166 | 0.0000 |
| t33/retrieval/B0 | all | all | 全部 | 12 | R3_verified@8 | 3.2594 | 93.4298 | 1.6587 | 0.9167 | 0.0166 | 0.0000 |
| t33/retrieval/B0 | all | all | 目标类 | 8 | R1_geometry_dino@8 | 8.0351 | 18.2741 | 5.1497 | 0.0000 | 0.0515 | 0.0816 |
| t33/retrieval/B0 | all | all | 目标类 | 8 | R2_native_order@8 | 9.8752 | 95.0068 | 5.2083 | 0.8750 | 0.0521 | 0.0612 |
| t33/retrieval/B0 | all | all | 目标类 | 8 | R3_dino_noverify@8 | 9.8752 | 95.0068 | 5.2083 | 0.8750 | 0.0521 | 0.0612 |
| t33/retrieval/B0 | all | all | 目标类 | 8 | R3_drop_unparsed_diagnostic@8 | 4.0934 | 93.4298 | 2.0925 | 0.8750 | 0.0209 | 0.0000 |
| t33/retrieval/B0 | all | all | 目标类 | 8 | R3_verified@8 | 4.0934 | 93.4298 | 2.0925 | 0.8750 | 0.0209 | 0.0000 |
| t33/retrieval/B0 | all | all | 地表类 | 4 | R1_geometry_dino@8 | 9.0562 | 11.7795 | 7.3557 | 0.0000 | 0.0736 | 0.0000 |
| t33/retrieval/B0 | all | all | 地表类 | 4 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | all | 地表类 | 4 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | all | 地表类 | 4 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | all | 地表类 | 4 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | positive | 全部 | 12 | R1_geometry_dino@8 | 8.2893 | 15.8909 | 5.6071 | 0.0000 | 0.0561 | 0.0727 |
| t33/retrieval/B0 | all | positive | 全部 | 12 | R2_native_order@8 | 7.9128 | 95.0068 | 4.1283 | 0.9167 | 0.0413 | 0.0545 |
| t33/retrieval/B0 | all | positive | 全部 | 12 | R3_dino_noverify@8 | 7.9128 | 95.0068 | 4.1283 | 0.9167 | 0.0413 | 0.0545 |
| t33/retrieval/B0 | all | positive | 全部 | 12 | R3_drop_unparsed_diagnostic@8 | 3.2594 | 93.4298 | 1.6587 | 0.9167 | 0.0166 | 0.0000 |
| t33/retrieval/B0 | all | positive | 全部 | 12 | R3_verified@8 | 3.2594 | 93.4298 | 1.6587 | 0.9167 | 0.0166 | 0.0000 |
| t33/retrieval/B0 | all | positive | 目标类 | 8 | R1_geometry_dino@8 | 8.0351 | 18.2741 | 5.1497 | 0.0000 | 0.0515 | 0.0816 |
| t33/retrieval/B0 | all | positive | 目标类 | 8 | R2_native_order@8 | 9.8752 | 95.0068 | 5.2083 | 0.8750 | 0.0521 | 0.0612 |
| t33/retrieval/B0 | all | positive | 目标类 | 8 | R3_dino_noverify@8 | 9.8752 | 95.0068 | 5.2083 | 0.8750 | 0.0521 | 0.0612 |
| t33/retrieval/B0 | all | positive | 目标类 | 8 | R3_drop_unparsed_diagnostic@8 | 4.0934 | 93.4298 | 2.0925 | 0.8750 | 0.0209 | 0.0000 |
| t33/retrieval/B0 | all | positive | 目标类 | 8 | R3_verified@8 | 4.0934 | 93.4298 | 2.0925 | 0.8750 | 0.0209 | 0.0000 |
| t33/retrieval/B0 | all | positive | 地表类 | 4 | R1_geometry_dino@8 | 9.0562 | 11.7795 | 7.3557 | 0.0000 | 0.0736 | 0.0000 |
| t33/retrieval/B0 | all | positive | 地表类 | 4 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | positive | 地表类 | 4 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | positive | 地表类 | 4 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | all | positive | 地表类 | 4 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 全部 | 6 | R1_geometry_dino@8 | 14.4007 | 10.8084 | 21.5697 | 0.0000 | 0.2157 | 0.0000 |
| t33/retrieval/B0 | tune | all | 全部 | 6 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 全部 | 6 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 全部 | 6 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 全部 | 6 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 目标类 | 3 | R1_geometry_dino@8 | 9.6047 | 10.4864 | 8.8598 | 0.0000 | 0.0886 | 0.0000 |
| t33/retrieval/B0 | tune | all | 目标类 | 3 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 目标类 | 3 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 目标类 | 3 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 目标类 | 3 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 地表类 | 3 | R1_geometry_dino@8 | 19.0366 | 10.9728 | 71.8091 | 0.0000 | 0.7181 | 0.0000 |
| t33/retrieval/B0 | tune | all | 地表类 | 3 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 地表类 | 3 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 地表类 | 3 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | all | 地表类 | 3 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 全部 | 6 | R1_geometry_dino@8 | 14.4007 | 10.8084 | 21.5697 | 0.0000 | 0.2157 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 全部 | 6 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 全部 | 6 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 全部 | 6 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 全部 | 6 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 目标类 | 3 | R1_geometry_dino@8 | 9.6047 | 10.4864 | 8.8598 | 0.0000 | 0.0886 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 目标类 | 3 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 目标类 | 3 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 目标类 | 3 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 目标类 | 3 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 地表类 | 3 | R1_geometry_dino@8 | 19.0366 | 10.9728 | 71.8091 | 0.0000 | 0.7181 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 地表类 | 3 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 地表类 | 3 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 地表类 | 3 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | tune | positive | 地表类 | 3 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 全部 | 6 | R1_geometry_dino@8 | 6.7082 | 21.5080 | 3.9738 | 0.0000 | 0.0397 | 0.1176 |
| t33/retrieval/B0 | confirm | all | 全部 | 6 | R2_native_order@8 | 8.6855 | 95.0068 | 4.5508 | 0.8333 | 0.0455 | 0.0882 |
| t33/retrieval/B0 | confirm | all | 全部 | 6 | R3_dino_noverify@8 | 8.6855 | 95.0068 | 4.5508 | 0.8333 | 0.0455 | 0.0882 |
| t33/retrieval/B0 | confirm | all | 全部 | 6 | R3_drop_unparsed_diagnostic@8 | 3.5865 | 93.4298 | 1.8284 | 0.8333 | 0.0183 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 全部 | 6 | R3_verified@8 | 3.5865 | 93.4298 | 1.8284 | 0.8333 | 0.0183 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 目标类 | 5 | R1_geometry_dino@8 | 7.7911 | 21.3058 | 4.7672 | 0.0000 | 0.0477 | 0.1212 |
| t33/retrieval/B0 | confirm | all | 目标类 | 5 | R2_native_order@8 | 10.8352 | 95.0068 | 5.7452 | 0.8000 | 0.0575 | 0.0909 |
| t33/retrieval/B0 | confirm | all | 目标类 | 5 | R3_dino_noverify@8 | 10.8352 | 95.0068 | 5.7452 | 0.8000 | 0.0575 | 0.0909 |
| t33/retrieval/B0 | confirm | all | 目标类 | 5 | R3_drop_unparsed_diagnostic@8 | 4.5052 | 93.4298 | 2.3083 | 0.8000 | 0.0231 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 目标类 | 5 | R3_verified@8 | 4.5052 | 93.4298 | 2.3083 | 0.8000 | 0.0231 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 地表类 | 1 | R1_geometry_dino@8 | 1.8359 | 26.2663 | 0.9512 | 0.0000 | 0.0095 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 地表类 | 1 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 地表类 | 1 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 地表类 | 1 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | all | 地表类 | 1 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 全部 | 6 | R1_geometry_dino@8 | 6.7082 | 21.5080 | 3.9738 | 0.0000 | 0.0397 | 0.1176 |
| t33/retrieval/B0 | confirm | positive | 全部 | 6 | R2_native_order@8 | 8.6855 | 95.0068 | 4.5508 | 0.8333 | 0.0455 | 0.0882 |
| t33/retrieval/B0 | confirm | positive | 全部 | 6 | R3_dino_noverify@8 | 8.6855 | 95.0068 | 4.5508 | 0.8333 | 0.0455 | 0.0882 |
| t33/retrieval/B0 | confirm | positive | 全部 | 6 | R3_drop_unparsed_diagnostic@8 | 3.5865 | 93.4298 | 1.8284 | 0.8333 | 0.0183 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 全部 | 6 | R3_verified@8 | 3.5865 | 93.4298 | 1.8284 | 0.8333 | 0.0183 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 目标类 | 5 | R1_geometry_dino@8 | 7.7911 | 21.3058 | 4.7672 | 0.0000 | 0.0477 | 0.1212 |
| t33/retrieval/B0 | confirm | positive | 目标类 | 5 | R2_native_order@8 | 10.8352 | 95.0068 | 5.7452 | 0.8000 | 0.0575 | 0.0909 |
| t33/retrieval/B0 | confirm | positive | 目标类 | 5 | R3_dino_noverify@8 | 10.8352 | 95.0068 | 5.7452 | 0.8000 | 0.0575 | 0.0909 |
| t33/retrieval/B0 | confirm | positive | 目标类 | 5 | R3_drop_unparsed_diagnostic@8 | 4.5052 | 93.4298 | 2.3083 | 0.8000 | 0.0231 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 目标类 | 5 | R3_verified@8 | 4.5052 | 93.4298 | 2.3083 | 0.8000 | 0.0231 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 地表类 | 1 | R1_geometry_dino@8 | 1.8359 | 26.2663 | 0.9512 | 0.0000 | 0.0095 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 地表类 | 1 | R2_native_order@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 地表类 | 1 | R3_dino_noverify@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 地表类 | 1 | R3_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| t33/retrieval/B0 | confirm | positive | 地表类 | 1 | R3_verified@8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| r1_direct4 | all | all | 全部 | 30 | R1_direct4_drop_unparsed_diagnostic@8 | 6.4322 | 4.1942 | 13.7914 | 0.0333 | 0.1379 | 0.1515 |
| r1_direct4 | all | all | 全部 | 30 | R1_direct4_verified@8 | 6.4322 | 4.1942 | 13.7914 | 0.0333 | 0.1379 | 0.1515 |
| r1_direct4 | all | all | 全部 | 30 | R1_geometry_dino@8 | 5.3656 | 3.0147 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| r1_direct4 | all | all | 目标类 | 15 | R1_direct4_drop_unparsed_diagnostic@8 | 1.2716 | 0.8351 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | all | all | 目标类 | 15 | R1_direct4_verified@8 | 1.2716 | 0.8351 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | all | all | 目标类 | 15 | R1_geometry_dino@8 | 0.8638 | 0.4929 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| r1_direct4 | all | all | 地表类 | 15 | R1_direct4_drop_unparsed_diagnostic@8 | 11.0320 | 7.1477 | 24.1626 | 0.0667 | 0.2416 | 0.2667 |
| r1_direct4 | all | all | 地表类 | 15 | R1_direct4_verified@8 | 11.0320 | 7.1477 | 24.1626 | 0.0667 | 0.2416 | 0.2667 |
| r1_direct4 | all | all | 地表类 | 15 | R1_geometry_dino@8 | 8.7429 | 4.8556 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| r1_direct4 | all | positive | 全部 | 11 | R1_direct4_drop_unparsed_diagnostic@8 | 10.8147 | 8.8949 | 13.7914 | 0.0909 | 0.1379 | 0.1515 |
| r1_direct4 | all | positive | 全部 | 11 | R1_direct4_verified@8 | 10.8147 | 8.8949 | 13.7914 | 0.0909 | 0.1379 | 0.1515 |
| r1_direct4 | all | positive | 全部 | 11 | R1_geometry_dino@8 | 11.2769 | 7.3354 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| r1_direct4 | all | positive | 目标类 | 6 | R1_direct4_drop_unparsed_diagnostic@8 | 2.5320 | 2.4129 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | all | positive | 目标类 | 6 | R1_direct4_verified@8 | 2.5320 | 2.4129 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | all | positive | 目标类 | 6 | R1_geometry_dino@8 | 2.2476 | 1.6584 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| r1_direct4 | all | positive | 地表类 | 5 | R1_direct4_drop_unparsed_diagnostic@8 | 16.2889 | 12.2856 | 24.1626 | 0.2000 | 0.2416 | 0.2667 |
| r1_direct4 | all | positive | 地表类 | 5 | R1_direct4_verified@8 | 16.2889 | 12.2856 | 24.1626 | 0.2000 | 0.2416 | 0.2667 |
| r1_direct4 | all | positive | 地表类 | 5 | R1_geometry_dino@8 | 16.0581 | 9.8293 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| r1_direct4 | all | empty | 全部 | 19 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 全部 | 19 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 全部 | 19 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 目标类 | 9 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 目标类 | 9 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 目标类 | 9 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 地表类 | 10 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 地表类 | 10 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | all | empty | 地表类 | 10 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | all | 全部 | 30 | R1_direct4_drop_unparsed_diagnostic@8 | 6.4322 | 4.1942 | 13.7914 | 0.0333 | 0.1379 | 0.1515 |
| r1_direct4 | tune | all | 全部 | 30 | R1_direct4_verified@8 | 6.4322 | 4.1942 | 13.7914 | 0.0333 | 0.1379 | 0.1515 |
| r1_direct4 | tune | all | 全部 | 30 | R1_geometry_dino@8 | 5.3656 | 3.0147 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| r1_direct4 | tune | all | 目标类 | 15 | R1_direct4_drop_unparsed_diagnostic@8 | 1.2716 | 0.8351 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | tune | all | 目标类 | 15 | R1_direct4_verified@8 | 1.2716 | 0.8351 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | tune | all | 目标类 | 15 | R1_geometry_dino@8 | 0.8638 | 0.4929 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| r1_direct4 | tune | all | 地表类 | 15 | R1_direct4_drop_unparsed_diagnostic@8 | 11.0320 | 7.1477 | 24.1626 | 0.0667 | 0.2416 | 0.2667 |
| r1_direct4 | tune | all | 地表类 | 15 | R1_direct4_verified@8 | 11.0320 | 7.1477 | 24.1626 | 0.0667 | 0.2416 | 0.2667 |
| r1_direct4 | tune | all | 地表类 | 15 | R1_geometry_dino@8 | 8.7429 | 4.8556 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| r1_direct4 | tune | positive | 全部 | 11 | R1_direct4_drop_unparsed_diagnostic@8 | 10.8147 | 8.8949 | 13.7914 | 0.0909 | 0.1379 | 0.1515 |
| r1_direct4 | tune | positive | 全部 | 11 | R1_direct4_verified@8 | 10.8147 | 8.8949 | 13.7914 | 0.0909 | 0.1379 | 0.1515 |
| r1_direct4 | tune | positive | 全部 | 11 | R1_geometry_dino@8 | 11.2769 | 7.3354 | 24.3731 | 0.0000 | 0.2437 | 0.2121 |
| r1_direct4 | tune | positive | 目标类 | 6 | R1_direct4_drop_unparsed_diagnostic@8 | 2.5320 | 2.4129 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | tune | positive | 目标类 | 6 | R1_direct4_verified@8 | 2.5320 | 2.4129 | 2.6635 | 0.0000 | 0.0266 | 0.0556 |
| r1_direct4 | tune | positive | 目标类 | 6 | R1_geometry_dino@8 | 2.2476 | 1.6584 | 3.4862 | 0.0000 | 0.0349 | 0.1111 |
| r1_direct4 | tune | positive | 地表类 | 5 | R1_direct4_drop_unparsed_diagnostic@8 | 16.2889 | 12.2856 | 24.1626 | 0.2000 | 0.2416 | 0.2667 |
| r1_direct4 | tune | positive | 地表类 | 5 | R1_direct4_verified@8 | 16.2889 | 12.2856 | 24.1626 | 0.2000 | 0.2416 | 0.2667 |
| r1_direct4 | tune | positive | 地表类 | 5 | R1_geometry_dino@8 | 16.0581 | 9.8293 | 43.8396 | 0.0000 | 0.4384 | 0.3333 |
| r1_direct4 | tune | empty | 全部 | 19 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 全部 | 19 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 全部 | 19 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 目标类 | 9 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 目标类 | 9 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 目标类 | 9 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 地表类 | 10 | R1_direct4_drop_unparsed_diagnostic@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 地表类 | 10 | R1_direct4_verified@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| r1_direct4 | tune | empty | 地表类 | 10 | R1_geometry_dino@8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA |

## Candidate-pool upper bounds

| Stage/pool | Split | GT | Group | N | Pixel coverage | Instance coverage |
|---|---|---|---|---:|---:|---:|
| endpoint/fixed_pool_upper | all | all | 全部 | 60 | 0.9830 | 0.4545 |
| endpoint/fixed_pool_upper | all | all | 目标类 | 27 | 0.9797 | 0.4943 |
| endpoint/fixed_pool_upper | all | all | 地表类 | 33 | 0.9879 | 0.3778 |
| endpoint/fixed_pool_upper | all | positive | 全部 | 30 | 0.9830 | 0.4545 |
| endpoint/fixed_pool_upper | all | positive | 目标类 | 16 | 0.9797 | 0.4943 |
| endpoint/fixed_pool_upper | all | positive | 地表类 | 14 | 0.9879 | 0.3778 |
| endpoint/fixed_pool_upper | all | empty | 全部 | 30 | NA | NA |
| endpoint/fixed_pool_upper | all | empty | 目标类 | 11 | NA | NA |
| endpoint/fixed_pool_upper | all | empty | 地表类 | 19 | NA | NA |
| endpoint/fixed_pool_upper | tune | all | 全部 | 30 | 0.9891 | 0.6970 |
| endpoint/fixed_pool_upper | tune | all | 目标类 | 15 | 0.9812 | 0.7778 |
| endpoint/fixed_pool_upper | tune | all | 地表类 | 15 | 0.9964 | 0.6000 |
| endpoint/fixed_pool_upper | tune | positive | 全部 | 11 | 0.9891 | 0.6970 |
| endpoint/fixed_pool_upper | tune | positive | 目标类 | 6 | 0.9812 | 0.7778 |
| endpoint/fixed_pool_upper | tune | positive | 地表类 | 5 | 0.9964 | 0.6000 |
| endpoint/fixed_pool_upper | tune | empty | 全部 | 19 | NA | NA |
| endpoint/fixed_pool_upper | tune | empty | 目标类 | 9 | NA | NA |
| endpoint/fixed_pool_upper | tune | empty | 地表类 | 10 | NA | NA |
| endpoint/fixed_pool_upper | confirm | all | 全部 | 30 | 0.9818 | 0.3737 |
| endpoint/fixed_pool_upper | confirm | all | 目标类 | 12 | 0.9795 | 0.4203 |
| endpoint/fixed_pool_upper | confirm | all | 地表类 | 18 | 0.9857 | 0.2667 |
| endpoint/fixed_pool_upper | confirm | positive | 全部 | 19 | 0.9818 | 0.3737 |
| endpoint/fixed_pool_upper | confirm | positive | 目标类 | 10 | 0.9795 | 0.4203 |
| endpoint/fixed_pool_upper | confirm | positive | 地表类 | 9 | 0.9857 | 0.2667 |
| endpoint/fixed_pool_upper | confirm | empty | 全部 | 11 | NA | NA |
| endpoint/fixed_pool_upper | confirm | empty | 目标类 | 2 | NA | NA |
| endpoint/fixed_pool_upper | confirm | empty | 地表类 | 9 | NA | NA |
| retrieval/B0/text_proposal_upper | all | all | 全部 | 60 | 0.0509 | 0.0227 |
| retrieval/B0/geometry_proposal_upper | all | all | 全部 | 60 | 0.2953 | 0.1894 |
| retrieval/B0/text_proposal_upper | all | all | 目标类 | 27 | 0.0859 | 0.0345 |
| retrieval/B0/geometry_proposal_upper | all | all | 目标类 | 27 | 0.3515 | 0.1954 |
| retrieval/B0/text_proposal_upper | all | all | 地表类 | 33 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | all | all | 地表类 | 33 | 0.2134 | 0.1778 |
| retrieval/B0/text_proposal_upper | all | positive | 全部 | 30 | 0.0509 | 0.0227 |
| retrieval/B0/geometry_proposal_upper | all | positive | 全部 | 30 | 0.2953 | 0.1894 |
| retrieval/B0/text_proposal_upper | all | positive | 目标类 | 16 | 0.0859 | 0.0345 |
| retrieval/B0/geometry_proposal_upper | all | positive | 目标类 | 16 | 0.3515 | 0.1954 |
| retrieval/B0/text_proposal_upper | all | positive | 地表类 | 14 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | all | positive | 地表类 | 14 | 0.2134 | 0.1778 |
| retrieval/B0/text_proposal_upper | all | empty | 全部 | 30 | NA | NA |
| retrieval/B0/geometry_proposal_upper | all | empty | 全部 | 30 | NA | NA |
| retrieval/B0/text_proposal_upper | all | empty | 目标类 | 11 | NA | NA |
| retrieval/B0/geometry_proposal_upper | all | empty | 目标类 | 11 | NA | NA |
| retrieval/B0/text_proposal_upper | all | empty | 地表类 | 19 | NA | NA |
| retrieval/B0/geometry_proposal_upper | all | empty | 地表类 | 19 | NA | NA |
| retrieval/B0/text_proposal_upper | tune | all | 全部 | 30 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | all | 全部 | 30 | 0.3205 | 0.3030 |
| retrieval/B0/text_proposal_upper | tune | all | 目标类 | 15 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | all | 目标类 | 15 | 0.1125 | 0.2778 |
| retrieval/B0/text_proposal_upper | tune | all | 地表类 | 15 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | all | 地表类 | 15 | 0.5144 | 0.3333 |
| retrieval/B0/text_proposal_upper | tune | positive | 全部 | 11 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | positive | 全部 | 11 | 0.3205 | 0.3030 |
| retrieval/B0/text_proposal_upper | tune | positive | 目标类 | 6 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | positive | 目标类 | 6 | 0.1125 | 0.2778 |
| retrieval/B0/text_proposal_upper | tune | positive | 地表类 | 5 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | tune | positive | 地表类 | 5 | 0.5144 | 0.3333 |
| retrieval/B0/text_proposal_upper | tune | empty | 全部 | 19 | NA | NA |
| retrieval/B0/geometry_proposal_upper | tune | empty | 全部 | 19 | NA | NA |
| retrieval/B0/text_proposal_upper | tune | empty | 目标类 | 9 | NA | NA |
| retrieval/B0/geometry_proposal_upper | tune | empty | 目标类 | 9 | NA | NA |
| retrieval/B0/text_proposal_upper | tune | empty | 地表类 | 10 | NA | NA |
| retrieval/B0/geometry_proposal_upper | tune | empty | 地表类 | 10 | NA | NA |
| retrieval/B0/text_proposal_upper | confirm | all | 全部 | 30 | 0.0611 | 0.0303 |
| retrieval/B0/geometry_proposal_upper | confirm | all | 全部 | 30 | 0.2903 | 0.1515 |
| retrieval/B0/text_proposal_upper | confirm | all | 目标类 | 12 | 0.0993 | 0.0435 |
| retrieval/B0/geometry_proposal_upper | confirm | all | 目标类 | 12 | 0.3889 | 0.1739 |
| retrieval/B0/text_proposal_upper | confirm | all | 地表类 | 18 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | confirm | all | 地表类 | 18 | 0.1327 | 0.1000 |
| retrieval/B0/text_proposal_upper | confirm | positive | 全部 | 19 | 0.0611 | 0.0303 |
| retrieval/B0/geometry_proposal_upper | confirm | positive | 全部 | 19 | 0.2903 | 0.1515 |
| retrieval/B0/text_proposal_upper | confirm | positive | 目标类 | 10 | 0.0993 | 0.0435 |
| retrieval/B0/geometry_proposal_upper | confirm | positive | 目标类 | 10 | 0.3889 | 0.1739 |
| retrieval/B0/text_proposal_upper | confirm | positive | 地表类 | 9 | 0.0000 | 0.0000 |
| retrieval/B0/geometry_proposal_upper | confirm | positive | 地表类 | 9 | 0.1327 | 0.1000 |
| retrieval/B0/text_proposal_upper | confirm | empty | 全部 | 11 | NA | NA |
| retrieval/B0/geometry_proposal_upper | confirm | empty | 全部 | 11 | NA | NA |
| retrieval/B0/text_proposal_upper | confirm | empty | 目标类 | 2 | NA | NA |
| retrieval/B0/geometry_proposal_upper | confirm | empty | 目标类 | 2 | NA | NA |
| retrieval/B0/text_proposal_upper | confirm | empty | 地表类 | 9 | NA | NA |
| retrieval/B0/geometry_proposal_upper | confirm | empty | 地表类 | 9 | NA | NA |
| t33/retrieval/B0/text_proposal_upper | all | all | 全部 | 12 | 0.0413 | 0.0545 |
| t33/retrieval/B0/geometry_proposal_upper | all | all | 全部 | 12 | 0.1203 | 0.1636 |
| t33/retrieval/B0/text_proposal_upper | all | all | 目标类 | 8 | 0.0521 | 0.0612 |
| t33/retrieval/B0/geometry_proposal_upper | all | all | 目标类 | 8 | 0.1231 | 0.1633 |
| t33/retrieval/B0/text_proposal_upper | all | all | 地表类 | 4 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | all | all | 地表类 | 4 | 0.1096 | 0.1667 |
| t33/retrieval/B0/text_proposal_upper | all | positive | 全部 | 12 | 0.0413 | 0.0545 |
| t33/retrieval/B0/geometry_proposal_upper | all | positive | 全部 | 12 | 0.1203 | 0.1636 |
| t33/retrieval/B0/text_proposal_upper | all | positive | 目标类 | 8 | 0.0521 | 0.0612 |
| t33/retrieval/B0/geometry_proposal_upper | all | positive | 目标类 | 8 | 0.1231 | 0.1633 |
| t33/retrieval/B0/text_proposal_upper | all | positive | 地表类 | 4 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | all | positive | 地表类 | 4 | 0.1096 | 0.1667 |
| t33/retrieval/B0/text_proposal_upper | tune | all | 全部 | 6 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | all | 全部 | 6 | 0.4363 | 0.2381 |
| t33/retrieval/B0/text_proposal_upper | tune | all | 目标类 | 3 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | all | 目标类 | 3 | 0.3071 | 0.2500 |
| t33/retrieval/B0/text_proposal_upper | tune | all | 地表类 | 3 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | all | 地表类 | 3 | 0.9473 | 0.2000 |
| t33/retrieval/B0/text_proposal_upper | tune | positive | 全部 | 6 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | positive | 全部 | 6 | 0.4363 | 0.2381 |
| t33/retrieval/B0/text_proposal_upper | tune | positive | 目标类 | 3 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | positive | 目标类 | 3 | 0.3071 | 0.2500 |
| t33/retrieval/B0/text_proposal_upper | tune | positive | 地表类 | 3 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | tune | positive | 地表类 | 3 | 0.9473 | 0.2000 |
| t33/retrieval/B0/text_proposal_upper | confirm | all | 全部 | 6 | 0.0455 | 0.0882 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | all | 全部 | 6 | 0.0880 | 0.1176 |
| t33/retrieval/B0/text_proposal_upper | confirm | all | 目标类 | 5 | 0.0575 | 0.0909 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | all | 目标类 | 5 | 0.1041 | 0.1212 |
| t33/retrieval/B0/text_proposal_upper | confirm | all | 地表类 | 1 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | all | 地表类 | 1 | 0.0264 | 0.0000 |
| t33/retrieval/B0/text_proposal_upper | confirm | positive | 全部 | 6 | 0.0455 | 0.0882 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | positive | 全部 | 6 | 0.0880 | 0.1176 |
| t33/retrieval/B0/text_proposal_upper | confirm | positive | 目标类 | 5 | 0.0575 | 0.0909 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | positive | 目标类 | 5 | 0.1041 | 0.1212 |
| t33/retrieval/B0/text_proposal_upper | confirm | positive | 地表类 | 1 | 0.0000 | 0.0000 |
| t33/retrieval/B0/geometry_proposal_upper | confirm | positive | 地表类 | 1 | 0.0264 | 0.0000 |
| r1_direct4/geometry_proposal_upper | all | all | 全部 | 30 | 0.3205 | 0.3030 |
| r1_direct4/geometry_proposal_upper | all | all | 目标类 | 15 | 0.1125 | 0.2778 |
| r1_direct4/geometry_proposal_upper | all | all | 地表类 | 15 | 0.5144 | 0.3333 |
| r1_direct4/geometry_proposal_upper | all | positive | 全部 | 11 | 0.3205 | 0.3030 |
| r1_direct4/geometry_proposal_upper | all | positive | 目标类 | 6 | 0.1125 | 0.2778 |
| r1_direct4/geometry_proposal_upper | all | positive | 地表类 | 5 | 0.5144 | 0.3333 |
| r1_direct4/geometry_proposal_upper | all | empty | 全部 | 19 | NA | NA |
| r1_direct4/geometry_proposal_upper | all | empty | 目标类 | 9 | NA | NA |
| r1_direct4/geometry_proposal_upper | all | empty | 地表类 | 10 | NA | NA |
| r1_direct4/geometry_proposal_upper | tune | all | 全部 | 30 | 0.3205 | 0.3030 |
| r1_direct4/geometry_proposal_upper | tune | all | 目标类 | 15 | 0.1125 | 0.2778 |
| r1_direct4/geometry_proposal_upper | tune | all | 地表类 | 15 | 0.5144 | 0.3333 |
| r1_direct4/geometry_proposal_upper | tune | positive | 全部 | 11 | 0.3205 | 0.3030 |
| r1_direct4/geometry_proposal_upper | tune | positive | 目标类 | 6 | 0.1125 | 0.2778 |
| r1_direct4/geometry_proposal_upper | tune | positive | 地表类 | 5 | 0.5144 | 0.3333 |
| r1_direct4/geometry_proposal_upper | tune | empty | 全部 | 19 | NA | NA |
| r1_direct4/geometry_proposal_upper | tune | empty | 目标类 | 9 | NA | NA |
| r1_direct4/geometry_proposal_upper | tune | empty | 地表类 | 10 | NA | NA |
| old_dev60/proposal_upper | all | all | 全部 | 60 | 0.9830 | 0.4545 |
| old_dev60/proposal_upper | all | all | 目标类 | 27 | 0.9797 | 0.4943 |
| old_dev60/proposal_upper | all | all | 地表类 | 33 | 0.9879 | 0.3778 |
| old_dev60/proposal_upper | all | positive | 全部 | 30 | 0.9830 | 0.4545 |
| old_dev60/proposal_upper | all | positive | 目标类 | 16 | 0.9797 | 0.4943 |
| old_dev60/proposal_upper | all | positive | 地表类 | 14 | 0.9879 | 0.3778 |
| old_dev60/proposal_upper | all | empty | 全部 | 30 | NA | NA |
| old_dev60/proposal_upper | all | empty | 目标类 | 11 | NA | NA |
| old_dev60/proposal_upper | all | empty | 地表类 | 19 | NA | NA |
| old_dev60/proposal_upper | tune | all | 全部 | 30 | 0.9891 | 0.6970 |
| old_dev60/proposal_upper | tune | all | 目标类 | 15 | 0.9812 | 0.7778 |
| old_dev60/proposal_upper | tune | all | 地表类 | 15 | 0.9964 | 0.6000 |
| old_dev60/proposal_upper | tune | positive | 全部 | 11 | 0.9891 | 0.6970 |
| old_dev60/proposal_upper | tune | positive | 目标类 | 6 | 0.9812 | 0.7778 |
| old_dev60/proposal_upper | tune | positive | 地表类 | 5 | 0.9964 | 0.6000 |
| old_dev60/proposal_upper | tune | empty | 全部 | 19 | NA | NA |
| old_dev60/proposal_upper | tune | empty | 目标类 | 9 | NA | NA |
| old_dev60/proposal_upper | tune | empty | 地表类 | 10 | NA | NA |
| old_dev60/proposal_upper | confirm | all | 全部 | 30 | 0.9818 | 0.3737 |
| old_dev60/proposal_upper | confirm | all | 目标类 | 12 | 0.9795 | 0.4203 |
| old_dev60/proposal_upper | confirm | all | 地表类 | 18 | 0.9857 | 0.2667 |
| old_dev60/proposal_upper | confirm | positive | 全部 | 19 | 0.9818 | 0.3737 |
| old_dev60/proposal_upper | confirm | positive | 目标类 | 10 | 0.9795 | 0.4203 |
| old_dev60/proposal_upper | confirm | positive | 地表类 | 9 | 0.9857 | 0.2667 |
| old_dev60/proposal_upper | confirm | empty | 全部 | 11 | NA | NA |
| old_dev60/proposal_upper | confirm | empty | 目标类 | 2 | NA | NA |
| old_dev60/proposal_upper | confirm | empty | 地表类 | 9 | NA | NA |

## Historical GT-text oracle (offline only)

| Split | GT | Group | N | Method | F1 | P | R | Covered/GT pixels |
|---|---|---|---:|---|---:|---:|---:|---|
| all | all | 全部 | 60 | historical_oracle | 21.2867 | 76.4743 | 12.3641 | 29994/242589 |
| all | all | 全部 | 60 | predicted | 9.3130 | 54.2882 | 5.0934 | 12356/242589 |
| all | all | 目标类 | 27 | historical_oracle | 32.9522 | 78.5752 | 20.8476 | 29989/143849 |
| all | all | 目标类 | 27 | predicted | 15.3793 | 73.3991 | 8.5896 | 12356/143849 |
| all | all | 地表类 | 33 | historical_oracle | 0.0100 | 0.4739 | 0.0051 | 5/98740 |
| all | all | 地表类 | 33 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/98740 |
| all | positive | 全部 | 30 | historical_oracle | 21.5745 | 84.5831 | 12.3641 | 29994/242589 |
| all | positive | 全部 | 30 | predicted | 9.4505 | 65.3722 | 5.0934 | 12356/242589 |
| all | positive | 目标类 | 16 | historical_oracle | 33.6454 | 87.1368 | 20.8476 | 29989/143849 |
| all | positive | 目标类 | 16 | predicted | 15.7469 | 94.4432 | 8.5896 | 12356/143849 |
| all | positive | 地表类 | 14 | historical_oracle | 0.0100 | 0.4785 | 0.0051 | 5/98740 |
| all | positive | 地表类 | 14 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/98740 |
| all | empty | 全部 | 30 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| all | empty | 全部 | 30 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| all | empty | 目标类 | 11 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| all | empty | 目标类 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| all | empty | 地表类 | 19 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| all | empty | 地表类 | 19 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | all | 全部 | 30 | historical_oracle | 0.0181 | 0.1040 | 0.0099 | 4/40315 |
| tune | all | 全部 | 30 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/40315 |
| tune | all | 目标类 | 15 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/19448 |
| tune | all | 目标类 | 15 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/19448 |
| tune | all | 地表类 | 15 | historical_oracle | 0.0383 | 44.4444 | 0.0192 | 4/20867 |
| tune | all | 地表类 | 15 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/20867 |
| tune | positive | 全部 | 11 | historical_oracle | 0.0198 | 4.1667 | 0.0099 | 4/40315 |
| tune | positive | 全部 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/40315 |
| tune | positive | 目标类 | 6 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/19448 |
| tune | positive | 目标类 | 6 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/19448 |
| tune | positive | 地表类 | 5 | historical_oracle | 0.0383 | 44.4444 | 0.0192 | 4/20867 |
| tune | positive | 地表类 | 5 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/20867 |
| tune | empty | 全部 | 19 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | empty | 全部 | 19 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | empty | 目标类 | 9 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | empty | 目标类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | empty | 地表类 | 10 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| tune | empty | 地表类 | 10 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | all | 全部 | 30 | historical_oracle | 25.2389 | 84.7774 | 14.8264 | 29990/202274 |
| confirm | all | 全部 | 30 | predicted | 11.1931 | 66.7711 | 6.1085 | 12356/202274 |
| confirm | all | 目标类 | 12 | historical_oracle | 37.7862 | 87.3576 | 24.1067 | 29989/124401 |
| confirm | all | 目标类 | 12 | predicted | 18.0406 | 98.2272 | 9.9324 | 12356/124401 |
| confirm | all | 地表类 | 18 | historical_oracle | 0.0025 | 0.0956 | 0.0013 | 1/77873 |
| confirm | all | 地表类 | 18 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/77873 |
| confirm | positive | 全部 | 19 | historical_oracle | 25.2400 | 84.8014 | 14.8264 | 29990/202274 |
| confirm | positive | 全部 | 19 | predicted | 11.1986 | 67.1668 | 6.1085 | 12356/202274 |
| confirm | positive | 目标类 | 10 | historical_oracle | 37.7862 | 87.3576 | 24.1067 | 29989/124401 |
| confirm | positive | 目标类 | 10 | predicted | 18.0407 | 98.2350 | 9.9324 | 12356/124401 |
| confirm | positive | 地表类 | 9 | historical_oracle | 0.0025 | 0.0965 | 0.0013 | 1/77873 |
| confirm | positive | 地表类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/77873 |
| confirm | empty | 全部 | 11 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | empty | 全部 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | empty | 目标类 | 2 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | empty | 目标类 | 2 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | empty | 地表类 | 9 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| confirm | empty | 地表类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 |

## Reproduction, manual evidence and accounting

Historical DINO reproduction: 60 completed tasks, 240 TopK comparisons; all available counts agree: True; mismatches: 0. This is not a full60 claim until all60 finish.

New cumulative API cost: CNY 0.1435194, from 645 unique request artifacts. Historical B0 is reused without rebilling; summary totals are not added.

The Main Agent visual review, not independent human truth, is embedded in report.json and remains in grounding_review.json/md. It distinguishes wrong support localization, uncertain visual states and class-label disagreement. Fixed-pool upper bounds, per-task coverage, counterfactual rank overlap and unparsed diagnostics remain separately recorded; none are actual achieved Agent performance.
