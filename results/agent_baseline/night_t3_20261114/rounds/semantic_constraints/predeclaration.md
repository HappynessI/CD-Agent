# Predicted positive-endpoint proposal ablation: predeclaration

This is a new mechanism experiment prompted by the corrected positive/empty semantic audit. It does not revise prior parsing outputs or reinterpret B0 downstream as B2/B3 performance. Fixed original tune30 only; no confirmation execution is part of this entry.

For each existing predicted B0/B2/B3 pair (s,t), obtain the same four SAM3 text mask sets sT1,sT2,tT1,tT2. Unknown/same-class predictions yield empty proposals in both arms. No GT class is supplied to a model.

- Strict support retains the existing rule: each sT1 mask intersects union(tT2), excludes union(sT2) and union(tT1); repeat symmetrically for each tT2 mask.
- Positive-only support removes only the two negative endpoint exclusions: each sT1 mask intersects union(tT2), plus each tT2 mask intersects union(sT1).
- Nonempty filtering, source-before-target stable original slots, original DINO delta scorer, tie order and Top8 are fixed. No weights/threshold sweep, semantic gate, new candidate generator, or verifier.
- Primary comparison is each arm's full candidate union; fixed delta Top8 is secondary. Report full30, positive/empty, target/surface F1/P/R/counts and proposal coverage. B0 strict must reproduce archived R2_all and R3_dino_noverify@8 pixel counts.

Read-only metadata inventory before implementation found B0 68/68 required queries available; B2 38/52; B3 46/64. B0/B2/B3 have 17/13/16 cross-class predictions; the remaining 13/17/14 are empty under the original policy. Combined missing requests are 28, corresponding to both dates for these 14 task/class pairs:

- train_119_6to2: building; water
- train_1322_2to4: building
- train_147_1to5: non-vegetated ground
- train_1798_2to6: building
- train_2848_1to4: building; non-vegetated ground
- train_443_3to2: building
- train_1659_1to2: tree
- train_2677_2to1: building
- train_2822_1to2: building; tree
- train_555_4to2: building
- train_908_4to1: tree

Actual Slurm inventory writes immutable initial_inventory.json before inference and current inventory.json with per-case requirements. It checks exact text/query, pixel threshold .85, score threshold .5, resolution1008 and paired metadata/mask existence. GPU prepare uses one persistent existing Sam3Backend, generates only missing predicted-class queries, and writes exclusively to positive_endpoints_tune/cache. It does not load GT labels. CPU score refuses missing caches rather than shrinking denominators or treating unavailable masks as empty; it checks decoded shape and persists per-case strict/positive packed masks, stable slots, rankings and counts. Failed/restarted work reuses completed exact query caches and per-case metrics.

Operational split: tools/run_night_t3_positive_endpoints_prepare.sh requests one GPU,2CPU,32G,2h; score script requests CPU1,12G,3h and also supports inventory mode. No external API or VLM calls are made. Existing cached feature grids and original SAM caches are read-only. Source paths remain local provenance; no credentials are copied.

The real motivating case is train1808: positive endpoint truth intersection3479 was reduced to4 by sourceT2 exclusion. This predicts possible recall recovery, not net F1 benefit: removed exclusions can also admit unchanged support and substantially increase FP, especially on the19 empty tasks. B2's cross-date parsing information remains a separate intervention; the within-variant strict/positive-only contrast identifies the proposal-rule change. It is not a general semantic-understanding claim.

## Exploratory confirmation extension after the tune result

The completed tune pairing recovered3479 positive GT pixels for B2/B3 instead of4 under strict support, but all added TP came from the previously diagnosed train1808. That concentration is a reason to test generalization, not evidence of stable benefit. A separately authorized fixed confirmation run uses only B0 and B2. B2 was selected by positive-task label-pair agreement4/11 (the highest tested) and existing complete60 parsing; B3's roughly0.02 tune F1 difference is not pursued.

The module now accepts `--split confirm --variants B0 B2`; defaults remain tune and B0/B2/B3. New scripts `run_night_t3_positive_endpoints_confirm_prepare.sh` and `run_night_t3_positive_endpoints_confirm_score.sh` write exclusively to `positive_endpoints_confirm`, with independent settings, initial inventory, masks and summaries. They reuse existing B0/B2 parse records, fill only exact missing predicted-class SAM3 queries, retain strict/full-union and delta Top8 comparisons, and reproduce archived B0 confirmation strict counts. No tune outputs or settings are rewritten. Confirmation images have been exposed repeatedly: this is exploratory confirmation, never an unseen holdout or actual Agent evaluation.
