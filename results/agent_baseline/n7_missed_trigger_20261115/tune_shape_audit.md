# Initial-mask mechanism audit

No gate is selected in this audit. Shapes below are obtainable from current Mask components; no candidate generation or API call.

| split | group | image | coverage | components | largest fraction | tiny area fraction <=16px | baseline F1 | fixed ΔF1 | first gate failure |
|---|---|---|---|---|---|---|---|---|---|
| tune | second_appeared | train_1065 | 0.0604 | 3 | 0.7237 | 0.0000 | 69.1992 | 8.1316 | skipped_benefit |
| tune | second_appeared | train_1129 | 0.2268 | 8 | 0.2624 | 0.0000 | 90.4325 | -1.3779 | None |
| tune | second_appeared | train_1136 | 0.1135 | 5 | 0.2772 | 0.0000 | 97.4021 | -28.5248 | None |
| tune | second_appeared | train_1166 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| tune | second_appeared | train_1344 | 0.0200 | 3 | 0.9992 | 0.0008 | 43.2998 | 28.8452 | None |
| tune | second_disappeared | train_1065 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| tune | second_disappeared | train_1129 | 0.0039 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| tune | second_disappeared | train_1136 | 0.0022 | 1 | 1.0000 | 0.0000 | 32.8748 | 52.8165 | None |
| tune | second_disappeared | train_1166 | 0.0017 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| tune | second_disappeared | train_1344 | 0.0187 | 4 | 0.5279 | 0.0006 | 0.0000 | 0.0000 | None |
| tune | levir_any | train_106 | 0.0654 | 50 | 0.0881 | 0.0000 | 85.5608 | -8.2257 | None |
| tune | levir_any | train_109 | 0.0778 | 47 | 0.0396 | 0.0001 | 87.7697 | -9.3465 | None |
| tune | levir_any | train_11 | 0.0560 | 49 | 0.0290 | 0.0001 | 89.3751 | -1.4357 | None |
| tune | levir_any | train_116 | 0.1756 | 110 | 0.0209 | 0.0001 | 89.5807 | -19.5865 | None |
| tune | levir_any | train_123 | 0.0743 | 74 | 0.0447 | 0.0000 | 91.9895 | -7.9994 | None |
