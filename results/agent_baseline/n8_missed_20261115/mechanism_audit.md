# Fixed-confirm mechanism audit

Paired request inputs: 199/199 identical images, seeds and parameters; only evidence text differs.

| group | stage | candidate TP | candidate FP |
|---|---|---|---|
| all | original_all | 71338 | 1203216 |
| all | s5_dino020_all | 69898 | 984045 |
| all | direction_routed | 57373 | 967491 |
| all | dilation_only | 51506 | 942547 |
| all | area25 | 51454 | 940356 |
| all | semantic | 22509 | 235390 |
| all | residual_evidence | 20610 | 229492 |
| levir_any | original_all | 8933 | 995680 |
| levir_any | s5_dino020_all | 8491 | 789760 |
| levir_any | direction_routed | 8491 | 789760 |
| levir_any | dilation_only | 3715 | 765881 |
| levir_any | area25 | 3705 | 764063 |
| levir_any | semantic | 3481 | 224601 |
| levir_any | residual_evidence | 3481 | 224601 |
| second_appeared | original_all | 39599 | 82125 |
| second_appeared | s5_dino020_all | 38603 | 76861 |
| second_appeared | direction_routed | 38603 | 76861 |
| second_appeared | dilation_only | 37678 | 75853 |
| second_appeared | area25 | 37671 | 75605 |
| second_appeared | semantic | 10744 | 4304 |
| second_appeared | residual_evidence | 8845 | 3153 |
| second_disappeared | original_all | 22806 | 125411 |
| second_disappeared | s5_dino020_all | 22804 | 117424 |
| second_disappeared | direction_routed | 10279 | 100870 |
| second_disappeared | dilation_only | 10113 | 100813 |
| second_disappeared | area25 | 10078 | 100688 |
| second_disappeared | semantic | 8284 | 6485 |
| second_disappeared | residual_evidence | 8284 | 1738 |

## Verifier response switches
{'top10_union': {'candidates': 80, 'no_to_yes': 6, 'yes_to_no': 2, 'no_to_yes_zero_tp': 6, 'yes_to_no_zero_tp': 1, 'V0_accepted_zero_tp': 44, 'evidence_accepted_zero_tp': 49}, 'primary_top5': {'candidates': 41, 'no_to_yes': 4, 'yes_to_no': 0, 'no_to_yes_zero_tp': 4, 'yes_to_no_zero_tp': 0, 'V0_accepted_zero_tp': 27, 'evidence_accepted_zero_tp': 31}}

## Per-case fixed arms
| group/image | arm | F1 | Δ initial | added TP/FP | TopK candidate TP/FP |
|---|---|---|---|---|---|
| second_appeared/train_1177 | alpha0.0_res1024_V0_k5 | 82.1933 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_1177 | alpha0.0_res1024_V0_k10 | 82.1933 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_1177 | alpha0.0_res1024_V0_evidence_k5 | 82.1933 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_1177 | alpha0.0_res1024_V0_evidence_k10 | 82.1933 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_119 | alpha0.0_res1024_V0_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_119 | alpha0.0_res1024_V0_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_119 | alpha0.0_res1024_V0_evidence_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_119 | alpha0.0_res1024_V0_evidence_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_appeared/train_127 | alpha0.0_res1024_V0_k5 | 30.5702 | +7.2799 | 1984/125 | 1984/125 |
| second_appeared/train_127 | alpha0.0_res1024_V0_k10 | 42.6612 | +19.3710 | 5742/449 | 6513/449 |
| second_appeared/train_127 | alpha0.0_res1024_V0_evidence_k5 | 30.5702 | +7.2799 | 1984/125 | 1984/125 |
| second_appeared/train_127 | alpha0.0_res1024_V0_evidence_k10 | 42.8937 | +19.6035 | 5742/168 | 6513/449 |
| second_appeared/train_1322 | alpha0.0_res1024_V0_k5 | 61.3817 | -3.2404 | 0/1260 | 0/1512 |
| second_appeared/train_1322 | alpha0.0_res1024_V0_k10 | 60.2429 | -4.3791 | 0/1735 | 0/2175 |
| second_appeared/train_1322 | alpha0.0_res1024_V0_evidence_k5 | 60.7723 | -3.8498 | 0/1512 | 0/1512 |
| second_appeared/train_1322 | alpha0.0_res1024_V0_evidence_k10 | 59.4740 | -5.1480 | 0/2066 | 0/2175 |
| second_appeared/train_1370 | alpha0.0_res1024_V0_k5 | 56.2288 | +17.2895 | 4231/0 | 4231/0 |
| second_appeared/train_1370 | alpha0.0_res1024_V0_k10 | 56.2288 | +17.2895 | 4231/0 | 4231/0 |
| second_appeared/train_1370 | alpha0.0_res1024_V0_evidence_k5 | 56.2288 | +17.2895 | 4231/0 | 4231/0 |
| second_appeared/train_1370 | alpha0.0_res1024_V0_evidence_k10 | 56.2288 | +17.2895 | 4231/0 | 4231/0 |
| second_disappeared/train_1177 | alpha0.0_res1024_V0_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1177 | alpha0.0_res1024_V0_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1177 | alpha0.0_res1024_V0_evidence_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1177 | alpha0.0_res1024_V0_evidence_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_119 | alpha0.0_res1024_V0_k5 | 32.1933 | -0.0977 | 0/107 | 1838/258 |
| second_disappeared/train_119 | alpha0.0_res1024_V0_k10 | 35.3469 | +3.0559 | 686/146 | 2556/547 |
| second_disappeared/train_119 | alpha0.0_res1024_V0_evidence_k5 | 32.1933 | -0.0977 | 0/107 | 1838/258 |
| second_disappeared/train_119 | alpha0.0_res1024_V0_evidence_k10 | 32.1933 | -0.0977 | 0/107 | 2556/547 |
| second_disappeared/train_127 | alpha0.0_res1024_V0_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_127 | alpha0.0_res1024_V0_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_127 | alpha0.0_res1024_V0_evidence_k5 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_127 | alpha0.0_res1024_V0_evidence_k10 | 0.0000 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1322 | alpha0.0_res1024_V0_k5 | 54.1691 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1322 | alpha0.0_res1024_V0_k10 | 54.1691 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1322 | alpha0.0_res1024_V0_evidence_k5 | 54.1691 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1322 | alpha0.0_res1024_V0_evidence_k10 | 54.1691 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1370 | alpha0.0_res1024_V0_k5 | 16.5250 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1370 | alpha0.0_res1024_V0_k10 | 16.5250 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1370 | alpha0.0_res1024_V0_evidence_k5 | 16.5250 | +0.0000 | 0/0 | 0/0 |
| second_disappeared/train_1370 | alpha0.0_res1024_V0_evidence_k10 | 16.5250 | +0.0000 | 0/0 | 0/0 |
| levir_any/train_118 | alpha0.0_res1024_V0_k5 | 85.0463 | -2.6170 | 0/5649 | 0/5649 |
| levir_any/train_118 | alpha0.0_res1024_V0_k10 | 83.7404 | -3.9229 | 0/8600 | 0/8600 |
| levir_any/train_118 | alpha0.0_res1024_V0_evidence_k5 | 85.0463 | -2.6170 | 0/5649 | 0/5649 |
| levir_any/train_118 | alpha0.0_res1024_V0_evidence_k10 | 83.7404 | -3.9229 | 0/8600 | 0/8600 |
| levir_any/train_119 | alpha0.0_res1024_V0_k5 | 87.4043 | -0.1963 | 0/635 | 0/635 |
| levir_any/train_119 | alpha0.0_res1024_V0_k10 | 86.5335 | -1.0671 | 58/3562 | 58/3562 |
| levir_any/train_119 | alpha0.0_res1024_V0_evidence_k5 | 87.4043 | -0.1963 | 0/635 | 0/635 |
| levir_any/train_119 | alpha0.0_res1024_V0_evidence_k10 | 86.5335 | -1.0671 | 58/3562 | 58/3562 |
| levir_any/train_121 | alpha0.0_res1024_V0_k5 | 84.1344 | -2.6924 | 0/2893 | 0/4298 |
| levir_any/train_121 | alpha0.0_res1024_V0_k10 | 83.1015 | -3.7253 | 124/4227 | 124/5926 |
| levir_any/train_121 | alpha0.0_res1024_V0_evidence_k5 | 82.8861 | -3.9406 | 0/4298 | 0/4298 |
| levir_any/train_121 | alpha0.0_res1024_V0_evidence_k10 | 81.8737 | -4.9531 | 124/5648 | 124/5926 |
| levir_any/train_122 | alpha0.0_res1024_V0_k5 | 79.6046 | -0.9875 | 0/1863 | 0/1863 |
| levir_any/train_122 | alpha0.0_res1024_V0_k10 | 79.9778 | -0.6143 | 1393/3244 | 1393/3244 |
| levir_any/train_122 | alpha0.0_res1024_V0_evidence_k5 | 79.6046 | -0.9875 | 0/1863 | 0/1863 |
| levir_any/train_122 | alpha0.0_res1024_V0_evidence_k10 | 79.9778 | -0.6143 | 1393/3244 | 1393/3244 |
| levir_any/train_124 | alpha0.0_res1024_V0_k5 | 86.6896 | -0.3007 | 0/926 | 0/926 |
| levir_any/train_124 | alpha0.0_res1024_V0_k10 | 86.4952 | -0.4952 | 63/1611 | 63/1611 |
| levir_any/train_124 | alpha0.0_res1024_V0_evidence_k5 | 86.6896 | -0.3007 | 0/926 | 0/926 |
| levir_any/train_124 | alpha0.0_res1024_V0_evidence_k10 | 86.4952 | -0.4952 | 63/1611 | 63/1611 |

## Candidate answer disagreements

| candidate | TP/FP | ΔP / DINO | V0 → evidence | V0 reason | evidence reason |
|---|---|---|---|---|---|
| second_appeared__train_127__n8_0013 | 0/281 | 0.2578/0.0809 | yes → no | The T1 image shows a building with a distinct dark shadow on its left side, while the T2 image shows an expanded footprint with additional roof structure in that same area, indicating new construction rather than just a shadow or misregistration. | The same building with a white roof and dark shadow is present in both dates, with consistent shape and position relative to the canal; the increased building probability in T2 likely reflects improved visibility or lighting rather than construction. |
| second_appeared__train_1322__n8_0001 | 0/182 | 0.5328/0.3595 | no → yes | The red contour marks the same linear ground feature (likely a path edge or boundary) in both dates, with no building structure present inside the contour in either image. | The region shows bare ground in T1 and a distinct building structure adjacent to a new blue roof in T2, indicating new construction. |
| second_appeared__train_1322__n8_0002 | 0/43 | 0.4164/0.4700 | no → yes | The red contour marks the same linear feature in both dates; while T2 shows a large new blue building nearby, the object within the specific contour remains unchanged, indicating no new construction at that precise location. | The region shows a small, indistinct feature in T1 but clearly corresponds to the roof edge of a large blue building constructed in T2, representing new construction on previously undeveloped land. |
| second_appeared__train_1322__n8_0003 | 0/27 | 0.2832/0.4213 | no → yes | The red contour marks the same linear feature in both dates, likely a fence or boundary line on the ground, which has not changed into or from a building. | The area was bare ground in T1 and shows a distinct white linear structure (likely a building or wall) adjacent to a new large blue roof in T2, indicating new construction. |
| second_appeared__train_1322__n8_0009 | 0/79 | 0.2157/0.3736 | no → yes | The red contour marks the same physical location where a small structure or feature is visible in both dates, with no evidence of new construction or demolition at that specific spot. | The region was bare ground with a grid pattern in T1 and contains a new small structure in T2, representing new construction rather than a replacement. |
| second_disappeared__train_119__n8_0026 | 686/39 | 0.3333/0.5932 | yes → no | The building present in T1 within the red contour is completely absent in T2, where the area appears as dark vegetation or water, indicating demolition. | The object inside the contour is a building present in T1 that is absent in T2, but the wider view shows the entire industrial complex has been replaced by agricultural ponds, indicating the change is due to non-building land cover alteration rather than simple demolition of the specific structure. |
| levir_any__train_121__n8_0016 | 0/1418 | 0.4172/0.6317 | no → yes | The object inside the contour is a pile of sand or construction material present in T2, not a building, and was likely vegetation or bare ground in T1. | The region was covered by vegetation in T1 and shows a distinct building structure with a roof in T2, indicating new construction. |
| levir_any__train_121__n8_0021 | 0/1405 | 0.4176/0.6317 | no → yes | The object inside the contour is a temporary pile of materials or earth visible in T2, which falls under excluded items, and there is no evidence of a permanent building structure being constructed, demolished, or replaced. | The area was covered by dense vegetation in T1 and shows a distinct building structure with a roof in T2, indicating new construction. |
