# Top1 and exemplar-overlap association audit

Post hoc association only. All masks/GT/scores unchanged; overlap and majority-overlap groups are descriptive, not inference gates. Raw candidate area is exactly the single-candidate scored area plus exemplar intersection; it is not exemplar area. Overlap does not establish same-instance self-matching, nor prove that removing an overlapping candidate restores recall.

| Split | Rank | Support | Overlap group | N | Top1 TP | Top8 TP | Top1 FP | Nonzero-TP tasks | Overlap pixels | Post-exemplar predicted pixels |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tune | delta | all | all | 30 | 210 | 9826 | 61112 | 1 | 44332 | 61322 |
| tune | delta | all | no_overlap | 7 | 0 | 0 | 6710 | 0 | 0 | 6710 |
| tune | delta | all | overlap | 23 | 210 | 9826 | 54402 | 1 | 44332 | 54612 |
| tune | delta | all | majority_overlap | 17 | 0 | 7726 | 5155 | 0 | 32582 | 5155 |
| tune | delta | positive | all | 11 | 210 | 9826 | 36599 | 1 | 16605 | 36809 |
| tune | delta | positive | no_overlap | 3 | 0 | 0 | 5043 | 0 | 0 | 5043 |
| tune | delta | positive | overlap | 8 | 210 | 9826 | 31556 | 1 | 16605 | 31766 |
| tune | delta | positive | majority_overlap | 6 | 0 | 7726 | 1084 | 0 | 5644 | 1084 |
| tune | delta | empty | all | 19 | 0 | 0 | 24513 | 0 | 27727 | 24513 |
| tune | delta | empty | no_overlap | 4 | 0 | 0 | 1667 | 0 | 0 | 1667 |
| tune | delta | empty | overlap | 15 | 0 | 0 | 22846 | 0 | 27727 | 22846 |
| tune | delta | empty | majority_overlap | 11 | 0 | 0 | 4071 | 0 | 26938 | 4071 |
| tune | joint | all | all | 30 | 210 | 9859 | 59502 | 1 | 44332 | 59712 |
| tune | joint | all | no_overlap | 7 | 0 | 0 | 5100 | 0 | 0 | 5100 |
| tune | joint | all | overlap | 23 | 210 | 9859 | 54402 | 1 | 44332 | 54612 |
| tune | joint | all | majority_overlap | 17 | 0 | 7759 | 5155 | 0 | 32582 | 5155 |
| tune | joint | positive | all | 11 | 210 | 9859 | 34697 | 1 | 16605 | 34907 |
| tune | joint | positive | no_overlap | 3 | 0 | 0 | 3141 | 0 | 0 | 3141 |
| tune | joint | positive | overlap | 8 | 210 | 9859 | 31556 | 1 | 16605 | 31766 |
| tune | joint | positive | majority_overlap | 6 | 0 | 7759 | 1084 | 0 | 5644 | 1084 |
| tune | joint | empty | all | 19 | 0 | 0 | 24805 | 0 | 27727 | 24805 |
| tune | joint | empty | no_overlap | 4 | 0 | 0 | 1959 | 0 | 0 | 1959 |
| tune | joint | empty | overlap | 15 | 0 | 0 | 22846 | 0 | 27727 | 22846 |
| tune | joint | empty | majority_overlap | 11 | 0 | 0 | 4071 | 0 | 26938 | 4071 |
| confirm | delta | all | all | 30 | 51 | 38083 | 111186 | 4 | 116664 | 111237 |
| confirm | delta | all | no_overlap | 7 | 29 | 1313 | 6575 | 1 | 0 | 6604 |
| confirm | delta | all | overlap | 23 | 22 | 36770 | 104611 | 3 | 116664 | 104633 |
| confirm | delta | all | majority_overlap | 16 | 0 | 36628 | 9182 | 0 | 57971 | 9182 |
| confirm | delta | positive | all | 19 | 51 | 38083 | 97882 | 4 | 85103 | 97933 |
| confirm | delta | positive | no_overlap | 5 | 29 | 1313 | 3928 | 1 | 0 | 3957 |
| confirm | delta | positive | overlap | 14 | 22 | 36770 | 93954 | 3 | 85103 | 93976 |
| confirm | delta | positive | majority_overlap | 11 | 0 | 36628 | 2634 | 0 | 27740 | 2634 |
| confirm | delta | empty | all | 11 | 0 | 0 | 13304 | 0 | 31561 | 13304 |
| confirm | delta | empty | no_overlap | 2 | 0 | 0 | 2647 | 0 | 0 | 2647 |
| confirm | delta | empty | overlap | 9 | 0 | 0 | 10657 | 0 | 31561 | 10657 |
| confirm | delta | empty | majority_overlap | 5 | 0 | 0 | 6548 | 0 | 30231 | 6548 |
| confirm | joint | all | all | 30 | 51 | 35505 | 111186 | 4 | 116664 | 111237 |
| confirm | joint | all | no_overlap | 7 | 29 | 760 | 6575 | 1 | 0 | 6604 |
| confirm | joint | all | overlap | 23 | 22 | 34745 | 104611 | 3 | 116664 | 104633 |
| confirm | joint | all | majority_overlap | 16 | 0 | 34498 | 9182 | 0 | 57971 | 9182 |
| confirm | joint | positive | all | 19 | 51 | 35505 | 97882 | 4 | 85103 | 97933 |
| confirm | joint | positive | no_overlap | 5 | 29 | 760 | 3928 | 1 | 0 | 3957 |
| confirm | joint | positive | overlap | 14 | 22 | 34745 | 93954 | 3 | 85103 | 93976 |
| confirm | joint | positive | majority_overlap | 11 | 0 | 34498 | 2634 | 0 | 27740 | 2634 |
| confirm | joint | empty | all | 11 | 0 | 0 | 13304 | 0 | 31561 | 13304 |
| confirm | joint | empty | no_overlap | 2 | 0 | 0 | 2647 | 0 | 0 | 2647 |
| confirm | joint | empty | overlap | 9 | 0 | 0 | 10657 | 0 | 31561 | 10657 |
| confirm | joint | empty | majority_overlap | 5 | 0 | 0 | 6548 | 0 | 30231 | 6548 |
| all | delta | all | all | 60 | 261 | 47909 | 172298 | 5 | 160996 | 172559 |
| all | delta | all | no_overlap | 14 | 29 | 1313 | 13285 | 1 | 0 | 13314 |
| all | delta | all | overlap | 46 | 232 | 46596 | 159013 | 4 | 160996 | 159245 |
| all | delta | all | majority_overlap | 33 | 0 | 44354 | 14337 | 0 | 90553 | 14337 |
| all | delta | positive | all | 30 | 261 | 47909 | 134481 | 5 | 101708 | 134742 |
| all | delta | positive | no_overlap | 8 | 29 | 1313 | 8971 | 1 | 0 | 9000 |
| all | delta | positive | overlap | 22 | 232 | 46596 | 125510 | 4 | 101708 | 125742 |
| all | delta | positive | majority_overlap | 17 | 0 | 44354 | 3718 | 0 | 33384 | 3718 |
| all | delta | empty | all | 30 | 0 | 0 | 37817 | 0 | 59288 | 37817 |
| all | delta | empty | no_overlap | 6 | 0 | 0 | 4314 | 0 | 0 | 4314 |
| all | delta | empty | overlap | 24 | 0 | 0 | 33503 | 0 | 59288 | 33503 |
| all | delta | empty | majority_overlap | 16 | 0 | 0 | 10619 | 0 | 57169 | 10619 |
| all | joint | all | all | 60 | 261 | 45364 | 170688 | 5 | 160996 | 170949 |
| all | joint | all | no_overlap | 14 | 29 | 760 | 11675 | 1 | 0 | 11704 |
| all | joint | all | overlap | 46 | 232 | 44604 | 159013 | 4 | 160996 | 159245 |
| all | joint | all | majority_overlap | 33 | 0 | 42257 | 14337 | 0 | 90553 | 14337 |
| all | joint | positive | all | 30 | 261 | 45364 | 132579 | 5 | 101708 | 132840 |
| all | joint | positive | no_overlap | 8 | 29 | 760 | 7069 | 1 | 0 | 7098 |
| all | joint | positive | overlap | 22 | 232 | 44604 | 125510 | 4 | 101708 | 125742 |
| all | joint | positive | majority_overlap | 17 | 0 | 42257 | 3718 | 0 | 33384 | 3718 |
| all | joint | empty | all | 30 | 0 | 0 | 38109 | 0 | 59288 | 38109 |
| all | joint | empty | no_overlap | 6 | 0 | 0 | 4606 | 0 | 0 | 4606 |
| all | joint | empty | overlap | 24 | 0 | 0 | 33503 | 0 | 59288 | 33503 |
| all | joint | empty | majority_overlap | 16 | 0 | 0 | 10619 | 0 | 57169 | 10619 |

## Largest Top8-to-Top1 TP losses

| Split | Task | Rank | TP lost | Top8 TP | Top1 TP | Exemplar overlap pixels | Fraction of raw candidate overlapping exemplar | Remaining prediction pixels |
|---|---|---|---:|---:|---:|---:|---:|---:|
| confirm | train_2192_1to5 | joint | 21111 | 21111 | 0 | 6993 | 0.9773584905660377 | 162 |
| confirm | train_2192_1to5 | delta | 17170 | 17170 | 0 | 6993 | 0.9773584905660377 | 162 |
| confirm | train_2383_2to5 | delta | 11776 | 11776 | 0 | 1318 | 0.9726937269372694 | 37 |
| confirm | train_2383_2to5 | joint | 5447 | 5447 | 0 | 1318 | 0.9726937269372694 | 37 |
| tune | train_1538_1to3 | delta | 5121 | 5121 | 0 | 1389 | 0.8835877862595419 | 183 |
| tune | train_1538_1to3 | joint | 5121 | 5121 | 0 | 1389 | 0.8835877862595419 | 183 |
| confirm | train_2255_2to5 | delta | 3031 | 3031 | 0 | 15127 | 0.9574656623836951 | 672 |
| confirm | train_2255_2to5 | joint | 3031 | 3031 | 0 | 15127 | 0.9574656623836951 | 672 |
| confirm | train_23_2to1 | delta | 1973 | 1973 | 0 | 682 | 0.740499457111835 | 239 |
| confirm | train_23_2to1 | joint | 1973 | 1973 | 0 | 682 | 0.740499457111835 | 239 |
| tune | train_1659_1to2 | joint | 1960 | 1960 | 0 | 330 | 0.6776180698151951 | 157 |
| tune | train_1659_1to2 | delta | 1927 | 1927 | 0 | 330 | 0.6776180698151951 | 157 |
| tune | train_1808_2to1 | delta | 1890 | 2100 | 210 | 10918 | 0.26298294633394353 | 30598 |
| tune | train_1808_2to1 | joint | 1890 | 2100 | 210 | 10918 | 0.26298294633394353 | 30598 |
| confirm | train_2438_5to1 | delta | 1284 | 1284 | 0 | 0 | 0.0 | 223 |
| confirm | train_2828_1to5 | delta | 1107 | 1107 | 0 | 128 | 0.8311688311688312 | 26 |
| confirm | train_2828_1to5 | joint | 1107 | 1107 | 0 | 128 | 0.8311688311688312 | 26 |
| confirm | train_2438_5to1 | joint | 731 | 731 | 0 | 0 | 0.0 | 223 |
| confirm | train_368_1to2 | delta | 714 | 714 | 0 | 229 | 0.5044052863436124 | 225 |
| confirm | train_368_1to2 | joint | 714 | 714 | 0 | 229 | 0.5044052863436124 | 225 |
