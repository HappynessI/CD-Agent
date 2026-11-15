# First-round primary coverage failure cases

| split | group | image | coverage | called | ΔF1 | extra TP | extra FP | failure |
|---|---|---|---|---|---|---|---|---|
| tune | second_appeared | train_1065 | 0.06036 | False | 0.0000 | 0 | 0 | skipped_benefit |
| confirm | second_appeared | train_1177 | 0.04162 | True | -9.6638 | 194 | 3857 | called_harm |
| confirm | second_disappeared | train_1322 | 0.00475 | True | -52.1093 | 0 | 43994 | called_harm |
| confirm | levir_any | train_121 | 0.04718 | True | -37.7997 | 245 | 70455 | called_harm |
| confirm | levir_any | train_122 | 0.08030 | False | 0.0000 | 0 | 0 | skipped_benefit |
