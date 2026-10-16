# CA_0721(13) SoM Proposal ablation analysis

## Protocol

- Code commit: `2f57a49` (`feat: ground verifier actions with region ids`)
- Output: `experiments/CA_0721(13)-bailian-proposal-ablation/`
- Jobs: Direct `44064`, Proposal `44065`, Hybrid `44066`; all completed on `gpu46`
- Samples: `test_20_15`, `test_78_13`, `test_85_16`
- Rollout is GT-free. Ground truth is loaded only after each arm completes for offline metrics.
- Proposal/Hybrid use numbered global region selection, local crop diagnosis, and runtime-owned distance-transform point geometry.

## Offline result

| Arm | test20 selected IoU | test78 selected IoU | test85 selected IoU | aggregate selected IoU |
|---|---:|---:|---:|---:|
| Direct | 0.84638554 | 0.75787116 | 0.30658070 | 0.69744116 |
| Proposal | 0.84638554 | 0.75787116 | 0.30658070 | 0.69744116 |
| Hybrid | 0.84638554 | 0.75787116 | **0.33716381** | **0.70886178** |

Hybrid improves test85 by `+0.03058311` IoU and aggregate IoU by `+0.01142062`.
This is one beneficial action on three smoke samples, not evidence of a stable expected gain.

## What happened

Proposal-only executed one deterministic negative point per sample. Every candidate reduced
offline IoU and was rolled back:

| Sample | Initial IoU | Candidate IoU | Delta |
|---|---:|---:|---:|
| test20 | 0.84638554 | 0.63723713 | -0.20914841 |
| test78 | 0.75787116 | 0.50503309 | -0.25283807 |
| test85 | 0.30658070 | 0.16696629 | -0.13961441 |

Hybrid also rejected the harmful test20 action, emitted no tool action on test78, and accepted
one useful test85 negative point. That point removed 408 false-positive change pixels without
changing TP/FN, increasing precision from `0.31759558` to `0.35053381` while recall stayed
`0.89837134`.

The geometry interface therefore worked as intended: Qwen selected `region_id`; runtime supplied
the exact proposal seed; Environment executed it; rollback protected all three harmful Proposal
candidates. But geometry correctness is not semantic correctness. The current `present` proposal
is a connected component of the change mask. On test20/test78 the selected component also
contained many true-positive pixels, so SimpleClick's negative composition removed a large mixed
component and destroyed recall. The candidate menu is still too coarse.

## Runtime issues exposed and repaired

After a rollback, staged verification reused the same accepted feedback and repeated the rejected
deterministic point. It now chooses a distinct, already-diagnosed safe region while excluding the
bounded rejection history; if none remains, it fails closed. A second state-machine bug kept
executing an authorized `finish` until `max_steps` because the identical-state path preserved
`stop=false`. An accepted `none/finish` feedback now sets `stop=true` on the identical finish
check. Focused Slurm regression job `44073` passed 49 tests.

## Conclusion and next experiment

The result supports the semantic-to-geometric split but does not yet show reliable mask
improvement. The immediate bottleneck moved from free coordinate generation to proposal/action
granularity and semantic selection:

1. split change components using T1/T2 object instances before offering negative actions;
2. reject a negative-point proposal when its editable component materially extends outside the
   audited proposal mask;
3. report proposal recall, selection accuracy, program point hit, tool success, and commit
   precision separately;
4. rerun more than three samples before drawing a performance conclusion.
