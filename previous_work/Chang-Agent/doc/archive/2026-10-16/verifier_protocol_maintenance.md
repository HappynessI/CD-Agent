# Staged Verifier protocol maintenance

## Scope

The staged verifier is an opt-in alternative to the legacy rich single-pass
Qwen verifier.  It separates evidence inspection, diagnosis, action planning,
and candidate comparison while keeping all proposal geometry and editability
facts under Environment control.

The diagnosis prompt must remain model-capability neutral.  It must not encode
the old small-model shortcut that a white change region with a visible T1/T2
difference is normally correct.  Current diagnosis asks the MLLM to inspect
whole-region mask coverage, boundaries, and internal gaps; `mixed_error` is
valid when one proposal contains both supported and unsupported pixels.  The
runtime keeps geometry, schema, editability, and polarity checks, but does not
replace MLLM semantic judgment with a default `none` decision.

The protocol implementation is split across:

- `change_agent/verifier_protocol.py`: typed records and enums;
- `change_agent/adapters/staged_verifier.py`: state machine and cross-stage checks;
- `change_agent/adapters/stage_backends.py`: local Transformers and BaiLian
  OpenAI-compatible stage backends;
- `change_agent/adapters/bailian_adapter.py`: hosted Agent action generation;
- `tools/run_levir_change_agent.py`: CLI selection and run-manifest metadata.

## Verifier-input visualization

`tools/visualize_verifier_inputs.py` reconstructs the staged image payload from a
completed run without loading a model or contacting BaiLian. It writes the global
numbered proposal overview, per-region T1/T2 RGB crops, object/change-mask crops,
candidate added/removed delta crops, contact sheets, and JSON manifests. It uses
the run metadata for batch size and crop padding, and reconstructs saved point/box
tool composition for candidate-step presentation. Example:

```bash
/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python \
  tools/visualize_verifier_inputs.py \
  --run-dir outputs/CA_0722\(5\)-bailian-context-fix-3arm/proposal \
  --samples test_20_15 --initial-only \
  --output outputs/visualization
```

## Stage contract

The v11 initial-state path is:

1. Environment creates stable proposals, normalized geometry, authoritative
   white/black change-mask occupancy, and editable-seed facts.
2. Runtime exhaustively screens pending proposals in bounded global batches. The
   model cannot omit a difficult region: every selected batch is removed from the
   pending set until coverage is complete, and each exact screening reason is
   persisted with its regions.
3. One `audit` call per region jointly emits the copied mask state, physical
   T1/T2 target-class states, whole-component mask assessment, categorical
   evidence quality, and the complete checklist. Every checklist status must
   include an observable evidence string.
4. The local audit must resolve its persisted global hypothesis as
   `confirmed|refuted|uncertain`. Runtime validates this resolution together with
   copied-state, assessment/checklist, target-view, evidence-quality, and
   proposal-polarity consistency. It cannot silently discard an earlier
   over-segmentation or missing-extent observation.
5. `plan` selects a safe executable diagnosis with Environment-owned geometry,
   preferring the smallest component within the same error-priority class.
6. Executor builds a candidate. The attempted action, persisted initial
   diagnosis, and authoritative added/removed delta are returned through
   `candidate_evidence`; runtime derives benefit, harm, and comparison.
7. Only `comparison=better, accept=true` commits. A rejected candidate rolls
   back and replans; an accepted candidate rebuilds the catalog and performs a
   complete fresh v11 audit before another action or `finish`.

An accepted candidate rebuilds the proposal catalog and starts a fresh batched
audit because the previous region identities and mask facts may be stale after
an edit. Audit rounds do not consume Environment action steps; only executed
point/box edits do.

The candidate path uses only `candidate_evidence` with an action-scoped delta
record, previous/candidate object masks, and previous/candidate change masks.
Point-tool fragments with the same polarity are aggregated inside the
Environment-owned action scope, so all changed pixels are covered by one
semantic decision rather than many one-pixel calls. Qwen receives exact
added/removed masks and T1/T2 RGB crops with those delta pixels highlighted,
then labels physical target-class presence under the highlighted pixels; it does not reclassify a removed
black region as a current-state false positive/negative and does not author
transition flags. Runtime combines the local RGB judgment with the attempted
`positive_point|negative_point` and authoritative `delta_added|delta_removed`
polarity to derive intended improvement and introduced FP/FN harm. Evidence below
`--verifier-min-visual-confidence`, non-clear evidence, mixed/uncertain RGB states,
missing actions, box transitions, and unexpected polarity fail closed as
`comparison=uncertain`. Runtime derives `better` and commit only from sufficient
benefit without harm. After a
commit, the verifier independently rebuilds current-state proposals and diagnoses
remaining errors. Initial and post-commit `finish` additionally require complete
Environment audit coverage: every proposal must be selected, have sufficient
evidence, and be diagnosed `none`. An
identical candidate is handled programmatically as `unchanged` without a model
call.

The Environment commits a non-finish candidate only when all runtime gates pass
and the runtime-derived transition is `better`. Model-authored acceptance is not
part of the staged protocol.

`VerifierOutput` remains the compatibility envelope consumed by Environment,
but its roles are explicit: `comparison/accept` describe the transition, while
`error_type/error_region/suggested_action/stop` come from the independent
post-commit state assessment. `StageTrace.transition_assessment` retains the
runtime-derived effect flags, evidence-sufficiency gate, evidence summary, and
decision source for audit.

Rollback preserves only the prior accepted masks and accepted point-session
history. It must not reuse a rejected action instruction. Proposal and Hybrid
select a different cached Verifier diagnosis and update the persisted diagnosis
used by candidate verification. Direct calls
its full-context verifier in `replan` mode, with the rejected action, candidate
mask delta, candidate verdict, rejection reasons, and a bounded four-entry
rejection history. The current images/masks in that call are the accepted state;
additional masks are explicitly labelled as the rejected candidate. Runtime
assigns `comparison="uncertain"` and `accept=false`; Qwen must author a different
action or `finish`. If this replan is invalid, no action is authorized.

### Direct binary rubric

The current BaiLian operational path is `proposal_mode=direct`. The hosted model
can inspect complete T1/T2 RGB, predicted temporal masks, and the final change
mask without Proposal crops. Proposal and Hybrid remain controlled ablation arms.

Direct schema `direct_change_rubric_v3` forbids model-authored
`quality_score`, `progress_score`, `comparison`, and `accept`. Qwen returns exact
binary judgments with a short observable evidence string for each item:

- `evidence_sufficient`: hard gate for visual judgeability;
- `target_class_only`: auditable scope diagnostic. It must be true when Qwen
  correctly treats roads/vehicles/etc. as non-target false positives; it is not
  a stop gate, because a false mask can be correctly diagnosed while evidence
  remains actionable;
- `change_semantic_precision`: weight 3;
- `change_semantic_recall`: weight 3;
- `changed_object_extent`: weight 2;
- `change_boundary_alignment`: weight 1;
- `change_artifact_control`: weight 1.

Runtime computes quality as passed quality-weight divided by total quality weight.
Only `evidence_sufficient` is a hard gate and it does not add score. A failed
evidence gate authorizes no action. `error_type=none` is valid only when every
quality item and target-scope diagnostic pass; target-scope failure may still
carry an actionable FP/mixed diagnosis. For the
LEVIR `building` query, roads, parking areas, vehicles, vegetation, bare ground,
shadows, illumination, and registration differences are non-target context.
Predicted T1/T2 masks are fallible evidence rather than GT and are not required
to segment non-target or unchanged content merely to raise a score.

Candidate regional calls output the same typed physical RGB states as initial
evidence plus visual confidence and evidence quality. For a negative point, a
`delta_removed` region with clear unchanged target presence is a potential benefit,
while a real building/background transition is introduced false-negative harm.
For a positive point, clear real target transition supports benefit and unchanged
physical content is introduced false-positive harm. Runtime derives `better` from
benefit without harm, `worse` from harm without benefit, and otherwise fails closed.
Candidate `accept` is true only for runtime-derived `better`, after normal
Environment hard gates. Staged candidate transitions do not make a model
`decision` call.

Each model call has a minimal exact JSON schema.  The runtime rejects unknown
fields, unknown region IDs, non-enum values, string booleans, non-integer public
coordinates, points outside the proposal, negative clicks on black object-mask
seeds, and positive clicks on white seeds.  Invalid output never authorizes a
tool action or finish.

Runtime has no hard semantic mapping from a coarse T1/T2 state pair and a
white/black Proposal to `none`, `false_positive_change`, or
`false_negative`. A white component can contain both a real appearance change
and unsupported boundary/interior pixels. The runtime validates only structural
polarity (`false_positive_change` needs a white proposal and
`false_negative` needs a black proposal), then retains Qwen's semantic
`false_positive_change` or `mixed_error` result for planning.

`diagnosis` receives actual visual inputs, not only a serialized prior evidence
judgment. Candidate deltas deliberately do not run `candidate_diagnosis`. Every regional call starts with a global
overview whose active proposal is marked yellow, followed by exact regional
RGB/object-mask/change-mask crops. Hybrid includes T1/T2 object masks as marked
full-frame panels; Proposal keeps the smaller marked RGB/change overview.
Candidate calls additionally show previous accepted T1/T2 object-mask crops,
the previous accepted change-mask crop, exact added/removed delta-mask crops,
and original-color delta-only T1/T2 RGB crops. Black outside the delta is
presentation padding; a one-pixel cyan outer contour marks the delta without
overwriting its RGB values. Candidate records serialize separate
previous/candidate crop, delta-component, and seed occupancy facts.

### Atomic grounded regional audit and confidence policy

Staged v11 has no model-authored `visual_confidence`, diagnosis `confidence`, or
initial `quality_score`. Evidence uses only `clear|ambiguous|insufficient`.
Ambiguous/insufficient evidence fails closed as `uncertain_region` and prevents
candidate commit.

Every initial audit returns these statuses, each exactly
`pass|fail|not_applicable|uncertain` and paired with a non-empty visual evidence
string:

- `evidence_sufficient`;
- `target_class_only`;
- `white_pixels_supported`;
- `boundary_alignment`;
- `internal_holes_absent`;
- `changed_object_extent_complete`;
- `fragment_artifacts_absent`.

The same response must explicitly assess the whole audited component as
`correct|false_positive|false_negative|mixed|uncertain`. Runtime requires that
assessment to agree with the checklist-derived error. Failed target scope, white-pixel,
boundary, or fragment checks produce a false-positive diagnosis; failed hole
or extent checks produce a false-negative diagnosis; failures from both groups
produce `mixed_error`; insufficient/uncertain evidence produces
`uncertain_region`; only a fully passing applicable checklist produces `none`.
The per-region quality is the deterministic pass ratio over applicable quality
checks, and global initial quality is their mean. Model-authored numeric values
are accepted only by the legacy parser compatibility path used for archived
mock/replay data and are ignored for current runtime scoring and ordering.

The audit prompt states the `building` ontology explicitly: permanent roofed
structures are target; roads, parking, vehicles, trailers/mobile equipment,
vegetation, bare ground, shadows, illumination, and registration artifacts are
not. A normal non-scan audit receives a marked overview, exact local RGB/masks,
and an exact binary render of the audited connected component. A visible target
transition under part of a white component does not prove that the entire
component is correct. Empty-mask scan audits are the deliberate exception: the
Environment-owned tile box supplies only paired unmodified RGB crops because
the otherwise all-black masks and all-white tile geometry repeat presentation
facts rather than semantic evidence.

An evidence-grounding retry is field-local rather than a second semantic audit.
It is enabled only when replacing the rejected evidence string makes the first
response fully valid, including its region ID, T1/T2 states, seven checklist
statuses, polarity constraints, and runtime target resolution. The retry sees
only the authoritative-box T1/T2 RGB pixels; it receives no marked overview,
object/change mask, or component-geometry image. For provider minimum image
size, pixels are replicated with nearest-neighbor resize inside the exact box,
never imported from a neighboring region. The dedicated prompt is fixed and
positive: it does not echo the validation error, rejected evidence, mask facts,
presentation-color terms, proposal catalog, or Environment envelope. Runtime
deep-copies the validated anchor, takes only the new evidence string,
mechanically preserves every validated semantic field from the first response,
and runs the complete parser again. Once an anchor exists, every later retry
remains field-local and can never fall back to a full-context semantic rewrite;
if no safe anchor can be formed initially, the ordinary full-context repair
remains in force and the parser stays fail closed. Backend call history records
the selected `visual_input_scope`; the analyzer validates its allowed
stage/attempt combination and complete call accounting so pinned trajectories
can audit which runtime input contract was selected. The hosted wrapper carries
that metadata through private stage-call context and preserves the public
`complete_json(messages, *, prompt, call_kind)` override signature.

The dated v10 smoke entry point is:

```bash
NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0723_batched_rubric_v10.sh 2 test_85_16
```

It writes only below
`outputs/CA_0723(2)-bailian-proposal-batched-rubric-v10`, fixes the submitted
job to the node that passed the Bailian probe, uses one action step for smoke
validation, archives Slurm logs under the run's `proposal/logs/`, and sends a
QQ completion notification on success or failure. Unit and full-regression
tests use `tools/submit_ca0723_verifier_tests.sh` and write below the separate
`CA_0723(1)-verifier-batched-rubric-tests` directory.

### Point composition

A Proposal/Hybrid point plan carries its selected `region_id` across the runner to
Environment as trusted runtime metadata; it is not added to model-authored Action JSON.
Environment resolves that ID against the live `verifier_region_proposals`, requires the
point to equal the proposal's authoritative component seed, and supplies the proposal's
pixel box to Executor. Unknown, stale, or seed-mismatched authorization fails before a
segmentation worker runs. Proposal-grounded runner paths prefer the staged region ID;
a legacy Hybrid verifier may recover it only through one exact normalized-geometry match
in the live Environment catalog. Missing or ambiguous matches fail closed. Only the
explicitly selected Direct path may execute without proposal authorization.

Executor treats model input and authorized write scope as separate masks. The production
default sends the complete accepted SimpleClick session mask as `prev_mask`, preserving
the multi-object context used by CA_0723(5). The `proposal_roi` and
`clicked_component` inputs remain explicit diagnostic modes. Regardless of negative-point
input mode, the removal mask is the live white connected component containing the click
inside the adaptive Environment proposal ROI (with bounded seed padding and image
clipping). Absence from SimpleClick output therefore cannot delete a neighbouring
component. A positive point retains the proposal-local policy and merges only the clicked
prediction component inside the authorized ROI.
It never deletes an entire connected initial component merely because the click is
inside it, never adds pixels for a negative click, and never applies negative-click
changes outside the ROI. Tool audit records both raw-output mask pixels and the
composed removal count, selected region ID, authoritative proposal box, effective ROI,
full/model-input/session/write-scope pixel counts, input and write policies, and
local/nonlocal changed pixels. Legacy non-Proposal point callers retain the fixed-fraction
fallback scope.

Candidate-region generation treats both legacy and proposal-local point composition
modes as one authoritative action scope. All disconnected same-polarity fragments from
that single edit are aggregated into one full-coverage delta record; changing the
composition-mode audit label must be reflected in this whitelist, or fragmented edits
can incorrectly exceed the staged Verifier's total-region safety ceiling.

Environment also compares the point candidate's changed-pixel count with the selected
proposal's authoritative component area. The pre-Qwen hard gate now rejects only when
all three conditions hold: proposal ratio exceeds
`max_point_proposal_change_ratio=0.25`, changed pixels exceed
`max_point_proposal_change_pixels=256`, and full-image change exceeds
`max_point_image_change_ratio=0.004`. The small-edit exemption is necessary because a
3- or 177-pixel component can legitimately have ratio `1.0`. Every raw value, threshold,
subcondition, enable flag, and final gate result remains trajectory-auditable. The gate
is a catastrophic-edit precheck; smaller semantic harms still go to Qwen candidate
verification.

### Box authorization

Staged point and box plans both carry the selected `region_id` from runner to
Environment as trusted metadata outside Action JSON. Before SAM3 execution, Environment
looks up that ID in the current state's `verifier_region_proposals`, validates the live
pixel box, and requires exact equality with the parsed box action. Unknown/stale IDs or
geometry mismatch fail before the box worker runs. Trajectory evidence records the
region ID, authoritative box, equality result, and `environment_live_proposal` policy.
Direct/legacy actions without a staged region retain their explicit legacy path.

After an accepted point, Environment makes the exact composed target-view mask the
base of a new SimpleClick session and clears the replay history. This is deliberately
different from replaying clicks against the older base: connected-component merging
and ROI clipping mean the committed mask is generally not identical to SimpleClick's
raw full-frame prediction. Rejected points do not change the session. Accepted boxes
use the same rebase rule.

Candidate verification uses action-scoped `d*` delta proposals. After a candidate is
accepted, Environment immediately rebuilds the committed state's full `r*`/`s*`
proposal catalog before the next action. This matches the staged Verifier's accepted-
state replan namespace and prevents a valid next plan from being looked up against stale
delta IDs. Candidate-transition verification keeps the `d*` delta mask facts, while
accepted-state completion and replanning use the paired next-live mask facts/catalog.
The real two-action integration regression binds the replanned region ID and geometry to
the promoted live proposal before executing the second accepted action.

Direct candidate calls retain their binary candidate-effect contract, but now receive
full added/removed delta masks and exact delta T1/T2 RGB plus previous/candidate mask
crops. This prevents small valid edits from being presented only as nearly identical
full-frame masks.

## Backends

`LocalQwen3VLStageBackend` reuses already-loaded local Qwen model/processor
objects.  `BailianQwen3VLStageBackend` uses the OpenAI-compatible chat endpoint,
`response_format={"type":"json_object"}`, and base64 PNG inputs.  Both satisfy
the same `StageBackend.generate_stage` interface and return provider-independent
Python mappings.

Hosted structured calls explicitly set `enable_thinking=false`. This keeps
Qwen3.7-Plus synchronous JSON behavior comparable with the existing Qwen3-VL-Plus
protocol and prevents hidden reasoning tokens from consuming the bounded completion
budget before the required stage envelope is emitted.

The runner can instead pass `--bailian-enable-thinking` and an optional positive
`--bailian-thinking-budget`. These settings apply to the hosted staged Verifier
and are recorded in both run-level and trajectory metadata.
`tools/submit_ca0722_proposal_qwen37_thinking_v9.sh` uses a fixed budget of 256
for the CA_0722(10) non-thinking versus CA_0722(11) thinking comparison; it does
not change candidate evidence or runtime acceptance gates.

Local Transformers decoding does not provide a provider-side JSON constraint.
The staged backend therefore uses stage-aware extraction: it scans all complete
JSON objects and selects the envelope required by the current stage (`evidence`,
`diagnosis`, `plan`, or `decision`). It never accepts the first object merely
because it is valid JSON. The prompt places the output contract before a
delimited Environment-facts envelope, and a schema/semantic validation failure
is sent back to the same stage for a bounded repair attempt. Repair does not
relax the typed protocol; the response must still pass the exact stage parser.
Each staged call records a bounded raw response, parsed output, prompt hash,
latency, and validation error in `backend_calls` for debugging. Credentials
are not included in these records.

The local 2B GPU smoke run showed that diagnosis generation deterministically
omitted `confidence` on both the initial and repair attempt while preserving a
valid `error_type` and `target_view`. Diagnosis confidence is therefore an
optional ranking hint: an omitted value is normalized conservatively to `0.0`.
The action-bearing fields remain required and strictly validated; this
normalization cannot authorize an invalid target view, action, or geometry.

The next local GPU smoke run reached `plan` and exposed prompt-induced action
bias: a static `positive_point` example was copied even though the selected T2
seed was white. Plan templates are now derived from authoritative Environment
facts. A white seed produces a `negative_point` contract, a black seed produces
a `positive_point` contract, and the contract contains the exact component
seed. Runtime editability and geometry validation remain mandatory.

The following GPU smoke run reached candidate decision, where a static
`comparison=initial` example caused the local model to repeat an invalid label.
Decision templates are now mode-aware: initial uses `initial`; candidate uses
the valid non-initial `uncertain` example, while the parser still requires
candidate `accept=true` to require `comparison=better`, without converting an
explicit `accept=false` into acceptance.

The BaiLian path did not exhibit the copied-context parsing failure in the
smoke run because its request uses server-side `response_format=json_object`,
which returns one JSON message content. This constrains JSON syntax, not the
full application schema: the BaiLian run still exposed a semantic
`target_view` validation error and a later HTTP 400 candidate request, so it
uses the same stage-aware parser and repair interface.

Hosted vision providers also impose a minimum image side length.  Region
proposals are derived from connected components and can be smaller than that
limit even when the source LEVIR-CD images are 256x256.  Before a staged
BaiLian request, `_normalized_crop_box` expands only the provider-facing PIL
crop to at least 11x11 pixels, clamped to the source image.  Proposal geometry,
mask facts, and verifier semantics remain unchanged.

The hosted backend reads credentials only from an environment variable.  The
default is `DASHSCOPE_API_KEY`; the key is never included in trajectory metadata,
errors, or `last_call`.  Configure the endpoint through `--bailian-base-url` or
`DASHSCOPE_BASE_URL`.  A workspace-specific BaiLian base URL is preferred when
available.  Do not place API keys in command lines, JSON configs, CSV files, or
the repository.

`tools/run_with_bailian_csv.py` accepts the exported BaiLian workspace CSV only
as a local secret source: it loads the `apiKey` row into the child process and
sets `DASHSCOPE_BASE_URL` from the `openAiCompatible` row.  This is required for
workspace-scoped keys; the public DashScope endpoint is not a substitute.
Provider HTTP errors retain only a short, credential-redacted diagnostic.

For Slurm runs, probe the candidate node's workspace, model/tool paths, GPU,
proxy fingerprints, and workspace endpoint first, then submit with the same
node pinned via `--nodelist`.  `tools/submit_ca0720_bailian_fix.sh` requires the
probed node in `NODE` and requires `BAILIAN_NETWORK_MODE=direct|proxy`.  Use
`direct` only when the probe succeeds after unsetting proxy variables; use
`proxy` only when the same node has a working proxy listener and matching
fingerprints.  The helper archives Slurm stdout/stderr under the experiment
output's `logs/` directory before cleaning temporary `/tmp` copies.

The standard Direct-rubric entry point is
`tools/submit_ca0721_direct_rubric.sh`.  It accepts a run index followed by
sample names; `OUTPUT`, `NODE`, `BAILIAN_NETWORK_MODE`, `MAX_STEPS`, and
`SAMPLES_CSV` can be overridden without rebuilding a temporary script:

```bash
NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0721_direct_rubric.sh 11 \
  test_20_15 test_78_13 test_85_16
```

The helper performs `scontrol ping` before calling `sbatch`, pins the tested
node, and refuses an existing output directory.  A failure containing
`Error creating slurm stream socket` or `Slurmctld ... is DOWN` is a cluster
control-plane/connectivity failure; it is not a verifier or job-script error.

## Runner modes

The production LEVIR runner is Verifier-planned:

```text
--verifier qwen_zero_shot|qwen_staged|rule
--staged-verifier-backend local|bailian
--verifier-bailian-model qwen3.7-plus-2026-05-26
--proposal-mode direct|proposal|hybrid
```

The Verifier performs both semantic diagnosis and action planning. Environment
validates its normalized geometry and the Executor deterministically dispatches
SimpleClick or SAM3. There is no independent Agent decision in this runner.
`--agent-backend`, `--max-new-tokens`, `--action-retries`, and
`--bailian-model` remain compatibility arguments for archived launchers but are
not treatment variables when `--verifier-bailian-model` is explicit.

`direct` is the no-Proposal comparison arm: Qwen sees complete state and authors
action geometry; Proposals are not attached.
`proposal` uses regional crops and Environment-owned Proposal geometry. `hybrid`
uses complete state plus regional crops for diagnosis while retaining
Environment-owned Proposal geometry for execution and candidate verification.
All modes retain coordinate parsing, tool safety gates, rollback, and
offline-only GT evaluation.

Changing `--verifier-bailian-model` changes the controller's diagnosis and action
planning together. It is therefore a controller/Verifier ablation, not a
verifier-only ablation with a fixed independent Agent. The Proposal v9
Qwen3.7 comparison uses the fixed snapshot `qwen3.7-plus-2026-05-26` and sends
`enable_thinking=false` for deterministic structured stages. Job `44562` showed
that replacing both the initial planner and candidate verifier can stop before
candidate generation: its test85 selection text identified over-segmented bare
ground, while the structured diagnosis returned `none` at confidence `0.95`.
Do not interpret such a run as a direct candidate-evidence model comparison;
hold initial planning fixed when the experimental question is delta semantics.

## Required evaluation

Do not compare only aggregate IoU/F1.  For each model/protocol cell, report:

- stage schema-valid rate;
- diagnosis-valid and target-view-valid rates;
- executable-action rate;
- verifier-invalid and action-retry rates;
- number of tool actions and accepted/rejected candidates;
- initial versus selected per-sample IoU/precision/recall/F1;
- hosted request latency, token usage, and request ID when available.

Run the same fixed inputs, initial SAM3 artifacts, seed, proposal configuration,
and selection gates in every cell.  GT remains unavailable until rollout has
completed.

## Tests

For the current `wangyihan` deployment, runtime verification is performed only
as a short compute-node GPU smoke run. Do not launch additional CPU test runs.
Slurm stdout/stderr are temporary `/tmp` files and are removed by the job
wrapper; `outputs/` contains only experiment artifacts and structured protocol
diagnostics.

The protocol tests are CPU-only and do not call external services:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python \
  -m unittest tests.test_direct_verifier tests.test_staged_verifier \
  tests.test_bailian_adapter -v
```

`test_bailian_adapter` uses an injected fake HTTP opener and a temporary fake
environment key.  It verifies JSON mode, endpoint construction, usage/request
metadata, missing-key failure, and absence of credentials from audit records.

## 2026-07-23 batched-audit smoke record

The v10 smoke helper is `tools/submit_ca0723_batched_rubric_v10.sh`; its CPU
regression/targeted companion is `tools/submit_ca0723_verifier_tests.sh`.  Job
44826 ran on the previously probed `gpu46` node in direct Bailian mode with
`qwen3.7-plus-2026-05-26`, one action step, and at most three regions per
selection batch.  The output is
`outputs/CA_0723(2)-bailian-proposal-batched-rubric-v10`.

For `test_85_16`, the verifier audited all 19 connected-component proposals in
seven sequential batches (the trace records 19 selected region IDs and 19
judgments/diagnoses).  The runtime checklist removed model-authored numeric
confidence as intended, but the hosted model marked every checklist item for
all 19 regions as `pass` and authorized `finish` without an action.  Offline
GT evaluation was IoU `0.3066`, precision `0.3176`, recall `0.8984`, and F1
`0.4693` (2963 false-positive pixels, 156 false-negative pixels).  Thus the
experiment validates exhaustive coverage and deterministic runtime scoring,
but does not yet validate diagnosis accuracy: a discrete checklist alone does
not prevent systematic all-pass overconfidence when the visual evidence is
ambiguous.  Future ablations should add hard-negative calibration or an
independent rule-based contradiction gate before allowing `finish`.

## 2026-07-23 atomic-audit v11 follow-up

V11 replaces v10's lossy `select -> evidence -> diagnosis` chain with exhaustive
global screening followed by one grounded atomic regional audit. The exact
screening rationale is an input to that audit and must be explicitly confirmed,
refuted, or marked uncertain. It requires per-check evidence, validates the
model's whole-mask assessment against the checklist, and carries the chosen
diagnosis plus attempted action into candidate verification. Executor candidates
are still committed only by the existing runtime benefit-without-harm rule;
accepted edits are audited again from the new state and rejected edits roll back.

An invalid local audit after bounded repair no longer aborts all independent
regions. Runtime records that region as `uncertain_region` together with the exact
screening hypothesis and validation error; it still blocks `finish`, but another
consistent and executable FP/FN diagnosis may authorize an action. This preserves
fail-closed semantics without recreating v10's zero-action failure mode.

The first hosted attempt (`44855`--`44857`) was invalidated because all models
returned the categorical alias `evidence_quality=high`; strict parsing failed
before semantic auditing. V11 normalizes only that observed alias to `clear`, while
retaining all consistency and harm gates. Slurm jobs `44876` and `44877` passed the
updated focused protocol tests and complete 167-test regression suite. The
five-sample, three-arm hosted entry point is:

```bash
# Run this historical command only from the stable v11 tag.
NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0723_atomic_v11_5sample.sh 5
```

All arms keep Environment, Executor, inputs, and safety gates fixed while changing
the Verifier/controller model or thinking mode: Qwen3.7 thinking budget 256,
Qwen3.7 non-thinking, and Qwen3-VL-Plus non-thinking. Because this Verifier also
plans actions, this is not a pure verifier-only ablation with a fixed Agent. Each
arm uses the same five samples and up to three executed edits.

CA_0723(5) completed successfully in jobs `44878`, `44879`, and `44880`, with empty
stderr. The common initial aggregate IoU/F1 was `0.67629040/0.80688931`:

| Verifier | actions | accepted/rejected | selected IoU | selected F1 |
|---|---:|---:|---:|---:|
| Qwen3.7 non-thinking | 9 | 7/2 | 0.70009134 | 0.82359262 |
| Qwen3.7 thinking-256 | 9 | 6/3 | 0.69462518 | 0.81979802 |
| Qwen3-VL-Plus | 8 | 8/0 | 0.68465376 | 0.81281243 |

Qwen3.7 non-thinking is the recommended treatment. Its seven accepted candidates
all improved offline IoU, while its two rejected candidates were worse than the
retained state. On test85 it committed three successively re-verified edits and
improved IoU `0.30658070 -> 0.33580247`. Qwen3-VL-Plus also improved test85, but
misaccepted a harmful test20 deletion, showing why acceptance count must not replace
post-rollout candidate analysis. test50 remains structurally unactionable when the
Environment has no proposal from either predicted temporal mask.

## 2026-07-23 protocol v11.1 post-v11 audit hardening

The following changes apply only to the next run; they do not alter or relabel the
existing CA_0723(5) artifacts:

```bash
NODE=<probed-node> BAILIAN_NETWORK_MODE=<direct-or-proxy> \
  bash tools/submit_ca0723_atomic_v11_1_5sample.sh <run-index>
```

- `episode_summary.json` now separates `execution_status=completed` from
  `episode_outcome`. Outcomes include `converged`, `budget_exhausted`,
  `verifier_invalid`, and `no_executable_action`; the run manifest reports counts
  and convergence rate. An uncaught failure writes a redacted `failure.json`,
  marks the manifest `failed`, and leaves `.run_incomplete`.
- When both predicted temporal object masks are empty, Environment builds a
  deterministic 3x3 full-frame scan grid. The staged Verifier exhaustively screens
  and audits the tiles. A clear false negative is recovered with that tile's
  Environment-owned SAM3 box; the model never invents geometry. Box-candidate
  transition polarity is derived from the persisted false-negative diagnosis and
  remains subject to normal candidate harm and locality gates.
- Proposal rollback updates `_planned_diagnosis` to the alternate cached region
  before candidate verification. If no alternate exists, it clears the persisted
  diagnosis.
- BaiLian retries only transient 408/425/429/5xx and URL failures, with a default
  maximum of two retries and exponential backoff. Authentication/non-transient 4xx
  errors fail immediately. Provider diagnostics redact the complete parent key,
  `Bearer` values, and `sk-*` token-shaped values.
- Segmentation workers receive a least-privilege environment with API-key/token/
  secret/password variables removed. Successful worker streams remain discarded;
  bounded stdout/stderr are saved only when the worker fails or times out.
- The current launcher removes no-op legacy Agent options and v10 batching/min-area
  options. Parsers retain them for archived commands. The effective/compatibility
  split and rollback instructions are recorded in `doc/ablation_parameters.txt`.
- `tools/replay_verifier_challenge.py` is explicitly legacy because its whole-
  component negative deletion does not match the current ROI-clipped Executor. It
  requires an acknowledgement flag and is not a v11 validation path.

The action-scoped aggregation of disconnected same-polarity delta fragments remains
unchanged for cost reasons. It is a registered limitation, not evidence that every
component is independently harmless.
