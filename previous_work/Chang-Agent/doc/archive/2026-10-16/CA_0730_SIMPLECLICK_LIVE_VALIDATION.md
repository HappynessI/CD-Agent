# CA_0730 SimpleClick live-trajectory validation

Date: 2026-07-30

## Question and scope

CA_0726 established under a frozen action schedule that the new negative-point
default can use the complete accepted SimpleClick session as model input while
restricting writes to the clicked live component inside the proposal ROI. Relative
to legacy session input plus legacy write scope, that fixed replay changed micro IoU
from `0.722826` to `0.719928`, improved Recall by `0.001784`, and reduced cumulative
authorization scope by `60.5%`.

CA_0730 asks a different question: does that already-selected mechanism remain safe
and useful when fresh SAM3 initialization, autonomous Qwen3.7 planning, candidate
review, rollback, the conjunctive point gate, and live proposal refresh all run
together? It is a three-sample live validation, not a repeated input/write ablation
and not evidence of LEVIR-CD generalization.

## Run identity

- completed run: `CA_0730(4)-simpleclick-full-session-component-write-live-3sample`
- arm: `qwen37-no-thinking`
- Slurm Job: `50213`, `gpu37`, `COMPLETED 0:0`, elapsed `00:36:07`
- source: commit `d849c6945396f9d0041a830562d6e4642ba96943`, dirty-worktree
  fingerprint `<hash omitted>`
- samples: `test_20_15`, `test_78_13`, `test_85_16`
- controller: `qwen3.7-plus-2026-05-26`, thinking disabled, staged Proposal v11.1
- initialization: fresh dual-view SAM3 for every sample; no cached masks
- point input/write policy: `full_session_mask` input and
  `clicked_component_within_proposal_roi` write scope
- selection: `conservative_best`, at most three actions per sample
- GT boundary: labels were opened only after `rollout_complete.json`

## Acceptance result

The predeclared contract was selected micro IoU at least `0.7155976055`, no write
outside the authorized clicked component, and live proposal refresh after every
accepted edit.

| Check | Result | Status |
| --- | ---: | --- |
| selected micro IoU | `0.70422933` (`-0.01136828` vs threshold) | fail |
| outside-authorized writes | `0` | pass |
| accepted edits with live proposal refresh | `5/5` | pass |

The run therefore **completed successfully as a process but failed the absolute IoU
acceptance contract**. Safety boundaries and refresh behavior passed.

## Metrics

| State | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| fresh live initialization | `0.69744116` | `0.72136906` | `0.95459950` | `0.82175592` |
| verifier-selected | `0.70422933` | `0.73489566` | `0.94406010` | `0.82644902` |
| within-run delta | `+0.00678817` | `+0.01352660` | `-0.01053940` | `+0.00469310` |
| CA_0723 historical reference | `0.72282586` | not compared | not compared | not compared |
| CA_0726 fixed-action new default | `0.719928` | `0.745452` | `0.954600` | `0.837161` |

The selected live result is `-0.01859653` below the CA_0723 historical reference
and about `-0.015699` below the CA_0726 fixed-action result. Those are useful
operational comparisons, not a controlled causal delta: CA_0730 started from a new
SAM3 initialization and allowed Qwen to choose a new trajectory.

## Per-sample trajectory

| Sample | Initial IoU | Selected IoU | Selected step | Accepted/rejected |
| --- | ---: | ---: | ---: | ---: |
| `test_20_15` | `0.84638554` | `0.81428825` | 1 | 1/2 |
| `test_78_13` | `0.75787116` | `0.77948648` | 3 | 3/0 |
| `test_85_16` | `0.30658070` | `0.31234428` | 1 | 1/2 |

There were nine point actions, five accepted candidates and four rejected candidates.
All three episodes ended `budget_exhausted`; convergence was `0/3`. There were no
fallback actions and no action-attempt errors.

The trajectories reveal two important behaviors:

- `test_78_13` accepted three clean false-positive removals, reducing FP by 421
  without changing TP. `test_85_16` step 1 similarly removed 83 FP without changing
  TP.
- `test_20_15` step 1 was accepted by the GT-free Verifier but removed 195 TP and
  only 17 FP, lowering offline IoU by `0.03209729`. The runtime cannot compare GT
  IoU online, so `conservative_best` means best among Verifier-authorized states, not
  an oracle guarantee that the initial mask is retained. This accepted error is the
  main trajectory-level failure to investigate.

`test_85_16` step 2 would have raised offline IoU to `0.31715731`, but the Verifier
rejected it as `candidate_effect_not_better`; the selected mask stayed at step 1.
Step 3 would have dropped IoU to `0.17013968`, but the hard gate rejected it and the
degraded candidate was never committed.

## Safety audit

All nine actions record:

- `model_input_scope_policy=full_session_mask`;
- `write_scope_policy=clicked_component_within_proposal_roi`;
- `outside_roi_pixels=0` and `nonlocal_changed_pixels=0`.

Across all attempted candidates, 2,487 pixels changed inside the authorized ROI and
zero changed outside. Only accepted candidates were committed: they changed 716
authorized pixels in total, again with zero outside. The combined proposal/absolute/
image gate hard-rejected three candidates and set
`verifier_skipped_by_hard_gate=true` for exactly those three. Every one of the five
accepted candidates persisted a `full_state_after_accepted_candidate` proposal
refresh and rebased the point session to the accepted live mask.

## Why fixed replay and live results differ

“Fixed action” in CA_0726 means the archived CA_0723 initial masks, pixel clicks,
per-view click histories, action order, and accept/rollback schedule were frozen.
Qwen, Bailian, fresh SAM3 initialization, and the new point gate were not involved.
That design isolated only SimpleClick model-input scope and write authorization.

CA_0730 instead regenerated SAM3 masks and let the Qwen Verifier decide regions and
candidate acceptance online. Its initial micro IoU was only `0.69744116`, already
`0.02538470` below the CA_0723 reference. The live loop improved that fresh starting
point, but not enough to reach the historical absolute threshold. Thus the fixed
replay validates the input/write mechanism, while CA_0730 exposes initialization and
controller-trajectory behavior.

## Startup diagnostics

Only Job `50213` is a completed experiment registry row. Earlier attempts remain as
failure diagnostics in their output directories:

| Attempt | Job | Result | Diagnosis |
| --- | ---: | --- | --- |
| `CA_0730(1)` | `50161` | `FAILED 1:0` | stale OmniOVCD runtime path; `No module named sam3` |
| `CA_0730(2)` | `50166` | `FAILED 1:0` | stale SegAgent runtime path; `No module named third_party` |
| `CA_0730(3)` | `50211` | `FAILED 1:0` | cold SimpleClick worker exceeded the old 300-second timeout |
| `CA_0730(4)` | `50213` | `COMPLETED 0:0` | corrected reference paths and 900-second cold-worker timeout |

Probe Job `50160` completed on `gpu37` with direct networking; the endpoint returned
HTTP 404, confirming reachability without a sample request.

## Conclusion

Keep the new SimpleClick input/write policy as mechanism-validated and safety-audited,
but do not treat CA_0730 as validation of the historical IoU contract. The next
debugging target is the GT-free candidate judgment that accepted `test_20_15` step 1,
followed by the fresh SAM3 initialization gap. Any follow-up comparison should freeze
the initialization when evaluating controller changes, or explicitly report the
initialization delta separately.
