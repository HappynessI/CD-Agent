# Cached SegEarth dense parsed-pair comparison

routeA and B0 dense are historical methods reproduced from cached probabilities; new parsed-pair masks: B2 only. Offline fixed confirm30, no model input GT, no online Agent run.
B2 fixed primary arm; multiply exposed exploratory confirmation for tool-mechanism diagnosis, not online promotion. SAM text confirmation does not determine dense semantic confirmation outcomes.
Status: complete, 30/30. Reproduction: {'routeA': 30, 'B0': 30}.
Common seven-channel argmax retains background. No threshold/class/model changes. Unknown or same-class VLM pairs give empty predictions. Exemplar is removed from prediction and truth. The >=16 remaining-total-GT subset is sensitivity only; the original MIN_AREA=16 was a component construction threshold. B2 sees both dates, so differences cannot be called a pure spotlight intervention.

| GT | Group | N | Method | F1 | P | R | TP | FP | FN | Empty prediction rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| all | 全部 | 30 | routeA_historical | 13.687877598138892 | 9.726021693419108 | 23.095899621305755 | 46717 | 433613 | 155557 | 0.36666666666666664 |
| all | 全部 | 30 | B0_historical | 4.05064460200916 | 3.519683086261235 | 4.770262119699022 | 9649 | 264495 | 192625 | 0.5 |
| all | 全部 | 30 | B2_dense | 4.769174697275158 | 4.9221639423608075 | 4.6254090985494924 | 9356 | 180723 | 192918 | 0.6666666666666666 |
| all | 目标类 | 12 | routeA_historical | 23.94205908421047 | 17.766673282662587 | 36.69745420052894 | 45652 | 211301 | 78749 | 0.3333333333333333 |
| all | 目标类 | 12 | B0_historical | 7.257897179161181 | 6.838378524406023 | 7.732252956165947 | 9619 | 131043 | 114782 | 0.75 |
| all | 目标类 | 12 | B2_dense | 8.751719555676171 | 10.470571111882396 | 7.517624456395045 | 9352 | 79965 | 115049 | 0.75 |
| all | 地表类 | 18 | routeA_historical | 0.7070539419087137 | 0.47677245195342405 | 1.3676113672261245 | 1065 | 222312 | 76808 | 0.3888888888888889 |
| all | 地表类 | 18 | B0_historical | 0.028388256724468312 | 0.02247494044140783 | 0.038524263865524634 | 30 | 133452 | 77843 | 0.3333333333333333 |
| all | 地表类 | 18 | B2_dense | 0.004478405687575223 | 0.0039697505011810005 | 0.0051365685154032845 | 4 | 100758 | 77869 | 0.6111111111111112 |
| positive | 全部 | 19 | routeA_historical | 18.01185959717467 | 14.762277935423526 | 23.095899621305755 | 46717 | 269745 | 155557 | 0.3684210526315789 |
| positive | 全部 | 19 | B0_historical | 5.3016629166563645 | 5.966300819292008 | 4.770262119699022 | 9649 | 152076 | 192625 | 0.5263157894736842 |
| positive | 全部 | 19 | B2_dense | 5.757007045503492 | 7.621623382971097 | 4.6254090985494924 | 9356 | 113400 | 192918 | 0.8421052631578947 |
| positive | 目标类 | 10 | routeA_historical | 30.404262404262404 | 25.95353015082519 | 36.69745420052894 | 45652 | 130247 | 78749 | 0.3 |
| positive | 目标类 | 10 | B0_historical | 10.454923400485846 | 16.137095691853443 | 7.732252956165947 | 9619 | 49989 | 114782 | 0.8 |
| positive | 目标类 | 10 | B2_dense | 10.483658518813302 | 17.3153119792631 | 7.517624456395045 | 9352 | 44658 | 115049 | 0.9 |
| positive | 地表类 | 9 | routeA_historical | 0.975113992199088 | 0.757667380462853 | 1.3676113672261245 | 1065 | 139498 | 76808 | 0.4444444444444444 |
| positive | 地表类 | 9 | B0_historical | 0.03333518528807156 | 0.029378066335673785 | 0.038524263865524634 | 30 | 102087 | 77843 | 0.2222222222222222 |
| positive | 地表类 | 9 | B2_dense | 0.0054563187581418504 | 0.005818520350274925 | 0.0051365685154032845 | 4 | 68742 | 77869 | 0.7777777777777778 |
| empty | 全部 | 11 | routeA_historical | 0.0 | 0.0 | 0.0 | 0 | 163868 | 0 | 0.36363636363636365 |
| empty | 全部 | 11 | B0_historical | 0.0 | 0.0 | 0.0 | 0 | 112419 | 0 | 0.45454545454545453 |
| empty | 全部 | 11 | B2_dense | 0.0 | 0.0 | 0.0 | 0 | 67323 | 0 | 0.36363636363636365 |
| empty | 目标类 | 2 | routeA_historical | 0.0 | 0.0 | 0.0 | 0 | 81054 | 0 | 0.5 |
| empty | 目标类 | 2 | B0_historical | 0.0 | 0.0 | 0.0 | 0 | 81054 | 0 | 0.5 |
| empty | 目标类 | 2 | B2_dense | 0.0 | 0.0 | 0.0 | 0 | 35307 | 0 | 0.0 |
| empty | 地表类 | 9 | routeA_historical | 0.0 | 0.0 | 0.0 | 0 | 82814 | 0 | 0.3333333333333333 |
| empty | 地表类 | 9 | B0_historical | 0.0 | 0.0 | 0.0 | 0 | 31365 | 0 | 0.4444444444444444 |
| empty | 地表类 | 9 | B2_dense | 0.0 | 0.0 | 0.0 | 0 | 32016 | 0 | 0.4444444444444444 |
| remaining_gt>=16_diagnostic | 全部 | 17 | routeA_historical | 20.761333333333333 | 18.85535068457763 | 23.095977375207656 | 46713 | 201031 | 155543 | 0.35294117647058826 |
| remaining_gt>=16_diagnostic | 全部 | 17 | B0_historical | 6.5331585738815905 | 10.370187190211489 | 4.768708962898505 | 9645 | 83362 | 192611 | 0.5294117647058824 |
| remaining_gt>=16_diagnostic | 全部 | 17 | B2_dense | 7.297868853738285 | 17.306339982974944 | 4.623843050391583 | 9352 | 44686 | 192904 | 0.8823529411764706 |
| remaining_gt>=16_diagnostic | 目标类 | 10 | routeA_historical | 30.404262404262404 | 25.95353015082519 | 36.69745420052894 | 45652 | 130247 | 78749 | 0.3 |
| remaining_gt>=16_diagnostic | 目标类 | 10 | B0_historical | 10.454923400485846 | 16.137095691853443 | 7.732252956165947 | 9619 | 49989 | 114782 | 0.8 |
| remaining_gt>=16_diagnostic | 目标类 | 10 | B2_dense | 10.483658518813302 | 17.3153119792631 | 7.517624456395045 | 9352 | 44658 | 115049 | 0.9 |
| remaining_gt>=16_diagnostic | 地表类 | 7 | routeA_historical | 1.41750167000668 | 1.4767903124782518 | 1.3627898015541713 | 1061 | 70784 | 76794 | 0.42857142857142855 |
| remaining_gt>=16_diagnostic | 地表类 | 7 | B0_historical | 0.046739892498247254 | 0.07784664211503338 | 0.03339541455269411 | 26 | 33373 | 77829 | 0.14285714285714285 |
| remaining_gt>=16_diagnostic | 地表类 | 7 | B2_dense | 0.0 | 0.0 | 0.0 | 0 | 28 | 77855 | 0.8571428571428571 |

Per-task masks, pairs, counts, source paths and exact old/new reproduction counts are retained under tasks/. Missing caches are errors, never zero predictions or omitted cases. No configuration is selected by this report.
