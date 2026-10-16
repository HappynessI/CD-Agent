# CA_0726 SimpleClick input/write decoupling and point gate

Date: 2026-07-26

## Acceptance contract

The primary reference is the Qwen3.7 non-thinking arm of CA_0723(5), restricted
to `test_20_15`, `test_78_13`, and `test_85_16`. Its micro IoU is
`0.7228258641`. A new default is acceptable only when it improves a separately
reported safety or quality metric and satisfies both reported interpretations:

- no more than one IoU point lower: `IoU >= 0.7128258641`;
- no more than one percent relatively lower: `IoU >= 0.7155976055`.

The relative threshold is the stricter working acceptance line. Without a
measurable secondary improvement, the target remains no IoU regression.

## CA_0726(1) diagnosis

CA_0726(1) selected micro IoU `0.69744116`, below both limits. Offline GT
selection of the best already-generated candidate for each sample reaches only
`0.70693315`. This does not bound a new live trajectory after an accepted step,
but it proves that changing only the final selected step in the saved trajectory
is insufficient.

The pure proposal-relative `0.25` point gate rejected useful edits of 3, 177,
and 152 pixels because their small components changed by ratios `1.0`, `1.0`,
and `0.3725`. Across CA_0723 accepted edits and CA_0726 candidates, known useful
edits are at most 197 pixels (`0.3006%` of a 256x256 image), while the smallest
known harmful edit that should be pre-rejected is 408 pixels (`0.6226%`).

## Runtime change under test

Proposal authorization remains mandatory and fail-closed. For an authorized
negative point, Executor now separates:

- model input: complete accepted SimpleClick session by default, with
  `proposal_roi` and `clicked_component` retained as diagnostic modes;
- write authorization: only the live clicked component inside the proposal ROI,
  independent of model input.

After an accepted candidate, Environment replaces candidate-delta `d*` proposals
with a fresh full-state catalog before the next authorized action. This is required
for the Verifier's next `r*` plan to accumulate multiple accepted improvements.

The pre-Qwen point gate rejects only when all conditions hold:

```text
proposal_change_ratio > 0.25
AND changed_pixels > 256
AND changed_pixels / image_pixels > 0.004
```

The `256`-pixel and `0.004` thresholds are calibrated from 256x256 LEVIR
evidence and must be recalibrated for other image resolutions.

This sends small high-relative-ratio edits to candidate verification while still
pre-rejecting the observed 408/931/4016-pixel harmful candidates. The gate does
not decide whether a small edit is semantically correct.

## Fixed-action validation protocol

The first model-level validation reuses CA_0723(5)'s byte-identical initial masks,
pixel clicks, and accept/rollback schedule without Qwen, Bailian, a new SAM3
initialization, or the new gate. Three arms are required:

1. `legacy_session_legacy_write`;
2. `component_input_component_write`;
3. `legacy_session_component_write`.

The legacy arm must reproduce all archived raw and candidate hashes before the
comparison is considered valid. Ground truth is loaded only after all SimpleClick
calls and a replay-complete marker. The primary result uses the frozen source
acceptance schedule; per-step GT oracle results are diagnostic only.
All three arms also retain the same frozen per-view accepted click history, so
the treatments change only the model-input mask and authorized write scope.
Authorization-scope reduction is reported separately as a quantitative safety
improvement; Precision, Recall, and F1 remain the secondary quality metrics.

Completed output:

```text
outputs/CA_0726(2)-simpleclick-input-write-fixed-replay-3sample
```

Fixed replay Job `47087` completed on `gpu37` in `00:03:15` with exit `0:0`.
The legacy arm reproduced all 9 archived raw masks and all 9 candidate T1/T2/change
hash triples. All 18 SimpleClick calls completed before the GT barrier; negative
additions and writes outside each arm's authorization were both zero.

| Arm | IoU | Precision | Recall | F1 | IoU delta | authorized live pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy session + legacy write | 0.722826 | 0.749660 | 0.952816 | 0.839117 | 0 | 6330 |
| component input + component write | 0.714049 | 0.739150 | 0.954600 | 0.833172 | -0.008777 | 2499 |
| legacy session + component write | 0.719928 | 0.745452 | 0.954600 | 0.837161 | -0.002898 | 2499 |

Component-only input falls below the stricter relative-one-percent line
`0.715598` and is not eligible as the default under that contract. Decoupled
legacy-session input plus component-only write stays above the line, improves
Recall by `0.001784`, and reduces the cumulative live authorization scope by
`3831` pixels (`60.5%`). This is the fixed-action configuration carried into the
live validation.

## Live-run preflight

Probe Job `47090` on `gpu08` was cancelled by Slurm with real-time signal 53
before the batch script started and produced no stdout/stderr. The same direct
probe completed on `gpu37` as Job `47092` with exit `0:0`: required paths,
interpreters, GPU visibility, and matching irreversible proxy fingerprints were
confirmed; the configured workspace endpoint returned HTTP `404`, establishing
network reachability without sending a sample. The formal hosted run remains
unsubmitted until the external-data authorization required by the execution
layer is explicitly reconfirmed.

## Code validation

The final worktree passed two independent Slurm validations. Targeted/full jobs
`47076`/`47077` and verification rerun `47080`/`47081` all completed with
exit `0:0`; both full runs reported `202 tests passed`. The targeted suites cover
the live-catalog two-step integration, next-live completion facts, input/write
decoupling, and the combined-gate truth table. Logs and the completed test
manifest are in:

```text
outputs/CA_0726(3)-input-write-gate-tests
```
