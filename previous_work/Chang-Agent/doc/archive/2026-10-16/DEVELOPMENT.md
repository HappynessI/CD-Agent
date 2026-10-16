# Development log

## 2026-07-23 — Post-audit hardening v11.1

The production loop is now documented and implemented as
`Verifier planner -> Environment -> deterministic Executor`; the unused hosted
Agent initialization was removed. The new launcher is
`tools/submit_ca0723_atomic_v11_1_5sample.sh`. The historical v11 launcher points
users to tag `stable-ca0723-5-bailian-proposal-atomic-v11`, so new fallback and
status behavior cannot be confused with CA_0723(5).

V11.1 separates execution completion from episode convergence, isolates API keys
from segmentation workers, redacts provider errors, retries only transient hosted
failures, fixes persisted rollback diagnosis, and adds a deterministic 3x3 scan plus
Environment-owned SAM3 box when both temporal masks are empty. The legacy replay
tool remains available only with explicit acknowledgement because its whole-
component negative deletion does not match the current Executor.

## 2026-07-23 — Atomic grounded verifier v11

CA_0723(2) v10 exhaustively audited test85 but still produced zero actions: its
selection prose recognized over-segmentation and non-target content, then the
separate evidence/diagnosis calls lost that meaning and marked all 19 regions
all-pass. V11 removes that semantic handoff. Each region now has one target-aware
audit response containing copied mask state, RGB states, a whole-component mask
assessment, and a seven-item checklist with evidence for every judgment. Runtime
rejects assessment/checklist, mask-polarity, evidence-quality, and target-view
contradictions.

Planning now prefers the smallest safe executable diagnosis. Candidate review
receives the attempted action and persisted initial diagnosis together with exact
Environment-owned deltas. A better candidate commits and triggers a complete fresh
audit; all other candidates roll back and replan. The runner also supports a
separate `--verifier-bailian-model` and correctly validates/records an arbitrary
requested sample list. The later audit established that this Verifier also planned
actions; the arms were controller/Verifier comparisons rather than pure verifier
ablations with an independently fixed Agent.

Focused Slurm regression job `44850` and the 164-test full regression job `44851`
both completed successfully. The v11 stable tag contains the historical launcher
for three controller/Verifier arms over five samples while holding Environment,
Executor, initialization, seed, and safety gates fixed.

The first hosted attempt (`44855`--`44857`) exposed a categorical API mismatch:
all three arms returned `evidence_quality=high` although the prompt required
`clear|ambiguous|insufficient`, so every completed initial audit failed closed and
authorized no action. The run was invalidated and moved out of formal `outputs/`
to the workspace diagnostics directory.

V11 now normalizes only the observed categorical alias `high -> clear`; all other
unknown values remain invalid. More importantly, it restores exhaustive global
screening and persists each batch's exact suspected-error reason into every local
atomic audit. The local response must explicitly mark the hypothesis
`confirmed|refuted|uncertain`, and runtime requires that resolution to agree with
the whole-mask assessment and grounded checklist. Thus a useful global
over-segmentation observation cannot disappear into an unrelated all-pass response.
Slurm jobs `44868` and `44869` passed the updated focused suite and all 166 tests.

The next hosted pass confirmed that semantic continuity was restored: global
screening identified parking lots, bare ground, and vehicles, and local audits
explicitly confirmed those hypotheses. One response used `mixed` for a component
whose checklist had only precision failures. Rather than aborting the whole sample,
v11 now fails only that region closed as `uncertain_region`, preserves its hypothesis
and validation error, blocks `finish`, and continues auditing independent regions.
Candidate benefit/harm acceptance remains unchanged. Slurm jobs `44876` and `44877`
passed the focused suite and all 167 tests, including this continuation case.

CA_0723(5) then completed five samples in all three hosted arms on `gpu46`; all
stderr files were empty. The common initial aggregate was IoU `0.67629040` and F1
`0.80688931`. Qwen3.7 non-thinking (`44879`, 36m43s) was best: 9 actions, 7 accepted,
2 rejected, selected IoU `0.70009134`, and F1 `0.82359262`. Every accepted candidate
in this arm improved post-rollout IoU and both rejected test20 candidates were worse
than the retained state. test85 improved `0.30658070 -> 0.33580247` and FP fell
`2963 -> 2515`; test78 improved `0.75787116 -> 0.78081913`; test20 improved
`0.84638554 -> 0.84989316`.

Qwen3.7 thinking (`44878`, 59m14s) used 9 actions, accepted 6, rejected 3, and
reached aggregate IoU `0.69462518`; it correctly rejected all three harmful test20
candidates. Qwen3-VL-Plus (`44880`, 39m57s) accepted all 8 candidates and reached
IoU `0.68465376`, but it misaccepted a harmful test20 deletion
(`0.84638554 -> 0.78786202`). Qwen3.7 non-thinking is therefore the recommended
Verifier treatment; acceptance count alone is not a useful ranking signal.

Two limitations remain. test50 has no Environment proposal because both predicted
temporal masks are empty, so none of the three Verifiers can recover its 166 GT
false-negative pixels. On GT-empty test1, Qwen3-VL-Plus removed 375 of 626 false
positives, while the two Qwen3.7 treatments produced no selected action. The failed
CA_0723(3)/(4) attempts and logs were moved out of formal outputs into the workspace
diagnostics directory.

## 2026-07-22 — Proposal action-scoped candidate evidence

The completed CA_0722(5) run established a clean safety/utility split. Staged
schema v5 prevented every harmful Proposal/Hybrid commit and restored Hybrid
selected aggregate IoU from `0.63552202` to the common initial `0.69744116`,
but it rejected every Proposal/Hybrid candidate. In particular, a Proposal
negative-point candidate improved test85 from `0.30658070` to `0.33236925`
without being committed. Direct still accepted a harmful positive point and
fell to aggregate IoU `0.68976021`; Direct and Hybrid are therefore out of scope
for the next experiment.

Staged schema v6 treats one Environment-clipped tool edit as one evidence unit.
When Executor locality metadata is present, all same-polarity delta fragments
inside the authoritative action scope are aggregated without losing any delta
pixels. Candidate regional calls now show exact added/removed masks plus T1/T2
RGB crops with precisely those delta pixels highlighted; the prompt tells Qwen
to ignore unedited objects elsewhere in the rectangle. Generic/offline
candidate generation without action locality retains component-separated
records.

Runtime revalidates point editability against the accepted pre-action target
mask before candidate inspection. Initial and post-commit `finish` plans now
pass only when every Environment audit region was selected, received sufficient
visual evidence, and was diagnosed `none`; unselected regions can no longer be
silently converted into outer `accept=true, stop=true`.

Slurm job `44459` passed all 70 focused staged-verifier, regional-backend,
region-generation, Environment, Executor-locality, and runner-action tests.

The Proposal-only launcher now escapes Markdown backticks around its dynamic
`MODES` value. This prevents the shell from treating `proposal` as command
substitution while writing `experiment_manifest.md`; the issue affected only
manifest text, not the submitted arm command.

Proposal-only job `44461` completed successfully on `gpu46` in 2m31s. Schema v6
reduced each candidate transition from 4--18 disconnected records to one
action-scoped record, retained all delta pixels, and produced sufficient visual
evidence for all five candidates. All attempted deletions were judged to remove
real buildings and were rejected, so selected aggregate IoU stayed at the
initial `0.69744116`. The test85 rollout produced no candidate: its highest
confidence diagnosis selected an uneditable T1 seed even though another selected
region had an executable T2 diagnosis.

Schema v7 adds a programmatic executable-diagnosis fallback. Diagnoses remain
ranked by the existing safety priority and confidence, but the planner now skips
an item that cannot map to an Environment-authorized point and tries the next
independently diagnosed region. Candidate evidence and acceptance thresholds are
unchanged. Slurm job `44477` passed all 71 focused tests, including the new
invalid-top-diagnosis fallback regression.

Proposal-only job `44487` completed successfully on `gpu46` in 2m26s with empty
stderr, but v7 again selected no candidate and aggregate IoU remained
`0.69744116`. The v6/v7 test85 records were value-equivalent, while prompt hashes
differed because `EvidenceRecord.from_proposal()` iterated the `TARGET_VIEWS`
set: v6 serialized editable seed facts as T2/T1 and v7 as T1/T2. This process
hash-seed dependency also changed r1 from actionable T2 to `none`.

Schema v8 makes T1/T2 protocol order explicit. When a diagnosis names a target
whose seed cannot execute the required positive/negative point, runtime may
canonicalize it only if exactly one Environment target view satisfies that seed
precondition; otherwise it remains unauthorized. This lets the persistent
test85 r0/T1 diagnosis execute as the uniquely valid r0/T2 candidate without
weakening candidate acceptance. Slurm job `44514` passed all 73 focused tests;
separate processes with `PYTHONHASHSEED=1` and `98765` both serialized the target
view order as T1/T2. Job `44513` was an implementation-independent fixture
failure caused by clearing the smaller region while Proposal IDs are area-sorted;
the corrected region mapping passed.

Proposal-only job `44516` completed successfully on `gpu46` in 2m45s with empty
stderr. Deterministic target resolution worked: test85 attempted both r0/T2 and
r1/T2 instead of terminating without a candidate. The r1 candidate improved
offline IoU from `0.30658070` to `0.33236925` by removing 349 false-positive
pixels without removing any true-positive pixel. Runtime nevertheless rejected
it because candidate evidence classified the aggregated delta as T1 background
to T2 building with confidence `0.95`. Selected aggregate IoU therefore stayed
at `0.69744116`.

The remaining bottleneck is candidate RGB semantics, not evidence coverage or
action planning. A next iteration should expose the unchanged RGB values under
the exact delta directly (for example, delta-only original-color T1/T2 crops with
a contour marker) instead of relying on a strongly yellow-blended highlight in a
large action crop. Acceptance thresholds should remain unchanged until that
evidence presentation is tested.

Schema v9 implements this presentation change. Candidate regional calls now replace
the large RGB proposal rectangle and yellow fill with delta-only original-color T1/T2
renders: pixels outside the authoritative delta are black presentation padding, and a
cyan outer contour marks the delta without changing its RGB values. The runtime
confidence threshold, transition polarity, and harm gates are unchanged. The focused
regression covers RGB preservation, contour placement, and removal of the old
rectangular/yellow candidate inputs; submit the Proposal-only v9 launcher for the
hosted semantic check.

Proposal-only v9 job `44554` completed on `gpu46` in 2m45s with empty stderr.
The evidence presentation changed as intended, but selected aggregate IoU stayed
at `0.69744116`. test85's useful second deletion again removed 349 false-positive
pixels and no true positives, improving offline IoU to `0.33236925`, yet Qwen
rejected it as a real `T1 building -> T2 background` transition at confidence
`0.95`. This is the opposite temporal label from v8's `background -> building`
for the same useful delta, also at confidence `0.95`. The stable rejection with
unstable high-confidence semantics is evidence that the next ablation should
change the candidate-evidence vision model rather than weaken confidence or harm
gates.

The fixed-snapshot Qwen3.7-Plus comparison was submitted through
`tools/submit_ca0722_proposal_qwen37_v9.sh`. Slurm job `44562` completed on
`gpu46` in 1m28s with empty stderr and recorded
`qwen3.7-plus-2026-05-26` for both Agent and staged Verifier, with thinking
disabled. It produced zero candidates on all three samples and retained selected
aggregate IoU `0.69744116`. For test85 the selection stage correctly described
bare ground/dirt in r0 as over-segmented non-change, but the following evidence
and diagnosis calls returned `background -> building` and `error_type=none` at
confidence `0.95`. The contradiction prevented an executable action before the
v9 candidate evidence could be tested. test20/test78 also remained unchanged,
so no unsafe commit was introduced. This run does not establish that Qwen3.7 is
worse at delta semantics; it establishes that a full-model swap confounds the
candidate comparison with initial staged planning. The next clean comparison
must isolate the candidate-evidence backend or add a runtime consistency check
between selection rationale and structured diagnosis.

## 2026-07-22 — runtime candidate evidence and local negative edits

The CA_0722(4) audit separated state-cache stability from semantic candidate
quality. Proposal rejected the only material improvement because a removed
false-positive region was black and therefore failed the initial-state invariant
that `false_positive_change` requires white pixels. Hybrid avoided that parser
failure but accepted three removals described as fixing false negatives, including
large offline regressions. The staged evidence template also used a literal
`visual_confidence=0.0`, which every hosted regional call copied while downstream
diagnosis ignored that contradiction.

Staged schema v5 removes `candidate_diagnosis` and model-authored candidate
decisions from the active path. Qwen now reports only physical T1/T2 target-class
presence for each exact delta region. Runtime combines that evidence with the
attempted point action and authoritative `delta_added|delta_removed` polarity to
derive intended improvement and introduced FP/FN harm. Low-confidence, mixed,
uncertain, missing-action, unsupported-box, and unexpected-polarity transitions
are `uncertain` and cannot commit. Candidate records preserve separate previous
and candidate T1/T2 crop/component/seed mask facts. Direct candidates receive
explicit added/removed masks and exact delta RGB/change-mask crops so small edits
are no longer judged only at full-frame scale.

Negative SimpleClick execution no longer discards the worker prediction and
deletes the clicked initial connected component. It applies only subtraction
pixels supported by the SimpleClick output inside the deterministic point ROI;
outside-ROI pixels and additions are preserved. Trajectories retain raw-output
and composed-removal pixel counts. This makes the executable edit local enough
for the transition verifier to distinguish a useful correction from destructive
component deletion.

Slurm job `44403` passed all 81 focused staged-verifier, visual-backend,
executor-locality, region-fact, Direct-verifier, Environment, and runner-action
tests with exit code `0:0`.

## 2026-07-22 — separate candidate transition from accepted-state planning

The CA_0722(3) audit found that identical BaiLian candidate-decision prompts could
flip only `accept=true|false`, while `comparison=better` and the candidate masks
were unchanged. The staged verifier also combined candidate comparison feedback
with a post-commit remaining-error plan in one unlabelled decision, so `accept`,
`error_type`, and `suggested_action` could describe different states.

Staged Qwen decisions no longer author `comparison`, `accept`, or `stop`. Candidate
calls return one intended-improvement flag and three introduced-harm flags; runtime
derives `better` only for benefit without harm and commits only that result. After
commit, current-state proposals are rebuilt, remaining errors are diagnosed, the
accepted diagnosis cache is refreshed, and `stop` is derived only from an executable
`finish` plan. Initial calls provide quality text only; runtime owns readiness.

Regional visual input again starts with a global overview whose active proposal is
marked yellow. Proposal uses marked T1/T2/change context plus exact crops; Hybrid adds
T1/T2 object-mask panels to the marked overview. Candidate calls also include the
previous accepted change mask globally and in the exact crop. This retains global
grounding without the ten independent, unmarked images used by the first context fix.

The CA_0722 marked-transition submission entry now writes Slurm stdout/stderr to a
shared staging directory before archiving each arm under its own `logs/` directory;
it no longer depends on compute-local `/tmp` for management-node diagnosis. The
default request is 16 GiB per arm (over six times the 2.7 GiB peak observed in the
preceding three-arm rollout) and remains configurable through `MEMORY`, allowing the
three arms to run concurrently under the per-user memory QOS.

## 2026-07-22 — make Proposal and Hybrid visual context a real ablation

The staged regional backend previously validated `visual_context=proposal|hybrid` but
constructed the same image list for both values: a global SoM overview plus five local
crops. The tests also used payloads without a proposal catalog and therefore asserted
five crops for both modes while incorrectly naming the Hybrid case “full context”.

The selection stage remains shared and receives one global numbered SoM overview.
After selection, Proposal now receives exactly five local RGB/object-mask/change-mask
crops. Hybrid receives five independent full-frame T1/T2 RGB, T1/T2 object-mask, and
change-mask images followed by the same five crops. The staged schema is bumped to v3,
and run manifests record distinct decision-mode and proposal-semantics strings. The
first validation after this repair is a three-sample Direct/Proposal/Hybrid smoke
ablation; a larger rollout is deferred until this contract is verified end to end.

The first submission of this smoke run completed Direct but Proposal and Hybrid failed
inside the shared SimpleClick subprocess before producing trajectories. The compute
node's `segagent-env` was importing user-site NumPy 2.2.6 instead of its pinned NumPy
1.26.4; legacy `imgaug` then crashed on the removed `np.sctypes` attribute. The tool
adapter now sets `PYTHONNOUSERSITE=1` for both SimpleClick and SAM3 subprocesses. The
failed arms are invalid experiment artifacts and must be rerun with the same code
snapshot before comparing metrics.

## 2026-07-22 — CA_0721(13) SoM ablation and rollback-state repair

The first three-sample GT-free SoM rollout completed in
`experiments/CA_0721(13)-bailian-proposal-ablation/` (jobs `44064`–`44066`).
Hybrid selected aggregate IoU improved from `0.69744116` to `0.70886178`, entirely
from test85 (`0.30658070` to `0.33716381`). Proposal-only executed three harmful
candidates, but runtime rollback kept every selected output at the initial mask.

The run exposed two staged runtime bugs. Rollback reused the same deterministic point,
so `StagedQwenVerifier` now selects a distinct cached diagnosed region while excluding
failed actions. The identical-state finish path now converts an accepted `none/finish`
authorization into `stop=true`, preventing repeated finish steps. Slurm job `44073`
passed 49 focused tests. Full evidence and limitations are in
[`ca0721_13_som_ablation_analysis.md`](ca0721_13_som_ablation_analysis.md).

## 2026-07-22 — SoM global selection and programmatic geometry

The staged proposal path now follows the grounding protocol documented in
`GROUNDING_PAPERS_ANALYSIS.md`: Environment proposals are rendered as a
deterministic numbered T1/T2/change overview, and Qwen returns only a bounded
list of existing `region_id` values. Selected regions receive the existing
local crop evidence/diagnosis call with the overview attached for alignment.

The old model-authored `plan` stage is removed from the active protocol. For a
safe `false_positive_change` or `false_negative` diagnosis, the runtime creates
one point action at the proposal's distance-transform seed. `mixed_error` and
`uncertain_region` fail closed until a finer proposal/grid resolver exists;
they never fall back to a large guessed box. Proposal/hybrid rollout actions
now go directly from verifier output to Environment, eliminating the second
Qwen coordinate-generation step. Candidate audits still inspect every delta
proposal; only the initial/replan audits use global region selection.

The Direct path keeps its full-image limitation, but repair retries now carry
the previous invalid response and lock rubric pass values, `error_type`, and
candidate-effect flags. Repair may change only the invalid target/action/
geometry fields. This prevents a geometry repair from silently replacing an
actionable mixed/false-positive diagnosis with `none/finish`.

Regression coverage includes deterministic interior seeds, strict region-ID
selection, overview/crop image composition, fail-closed mixed plans, direct
repair semantic locking, and direct verifier action execution. Slurm job
`44060` passed all 47 focused tests.

## 2026-07-21 — Direct rubric actionability/replan repair

The final three-sample Direct-rubric run `CA_0721(12)-bailian-direct-rubric-v3`
showed that valid rubric JSON alone is insufficient: one negative click was
grounded on a black editable-mask seed and therefore made no change, while both
rollback replans repeated the rejected pixel action.  A repeated replan was
detected only after parsing, outside the retry loop, so the episode stopped
without a second model repair request.

`DirectQwenVerifier` now validates the deterministic point no-op before a
segmentation worker starts: negative points require a white current-mask seed.
Positive points remain legal on either current-mask state because SimpleClick
can expand a component from a white seed.  The same negative-point validator is
part of the Direct replan retry loop, along with the non-duplicate action invariant.
Schema-valid but non-executable or repeated replans now receive the backend's
existing repair prompt rather than immediately invalidating the episode.  These
are tool-contract checks only; semantic diagnosis, rubric judgments, and action
geometry remain authored by Qwen.  Regression coverage includes a repeated
replan followed by a repaired, distinct action.

The detailed evidence, observable verifier trace, Codex operational trace,
offline metrics, and remaining model-quality limitation are recorded in
[`ca0721_12_direct_rubric_trajectory_analysis.md`](ca0721_12_direct_rubric_trajectory_analysis.md).

## 2026-07-21 — Direct rollback replan

The Direct Proposal-ablation arm previously treated its verifier feedback as
its action source. After any candidate rollback, Environment correctly restored
the accepted masks but also restored that same action-bearing feedback, causing
Direct to retry an action now present in the rejected-action blacklist. Direct
therefore stopped after one candidate rather than performing the intended
multi-step loop.

Environment now exposes an optional rollback-replan hook. `DirectQwenVerifier`
receives the accepted state, rejected candidate masks, rejected action, mask
delta, candidate verdict, and rejection reason, then makes a new full-context
Qwen request. It also receives a bounded four-entry, GT-free rejection history
so a stateless provider call can avoid earlier failed actions. Replans must use
`comparison="uncertain"` and `accept=false`; they author a different executable
action or `finish`. Replan failure is fail-safe and authorizes no action.
Proposal and Hybrid continue using their independent Agent regeneration path.
Regression tests cover hard-gate rollback, accepted-state preservation, replan
audit, Direct replan payload, and prompt requirements.

## 2026-07-21 — staged Verifier semantic override removal and Proposal ablation

The original BaiLian run recorded repeated raw `false_positive_change` diagnoses
that were rejected by a runtime rule equating a white component with a clear
T1/T2 appearance difference to a wholly correct change region. This is invalid
for components containing a real change plus unsupported boundary/interior
pixels, and caused `test_85_16` to receive an erroneous initial `none`/finish
verdict. The override is removed; only polarity, schema, editability, and
geometry remain runtime-owned. Diagnosis stages now receive their stated RGB and
mask images rather than only prior serialized evidence. A valid initial
Verifier `stop=true` now ends the episode immediately; identical-state finish
preserves a previously authorized stop.

The runner now supports `--proposal-mode direct|proposal|hybrid`. Direct uses
full-state Qwen diagnosis and model-authored action geometry without Proposal
attachment. Proposal uses regional crops and Environment geometry. Hybrid uses
full state plus regional crops and Environment geometry. The three-arm BaiLian
experiment is launched into one `experiments/` parent with `direct/`,
`proposal/`, and `hybrid/` children.

The ablation then exposed two additional protocol/commit defects. Direct Qwen
used valid plain-language labels such as `missing_detection`, but the newly
introduced Direct parser rejected them because its error enum was narrower than
the prompt. The boundary now canonicalizes documented aliases to the shared
typed vocabulary and requires an explicit `accept` field. Separately, the
staged decision parser had required `accept=true` whenever
`comparison=better`; it now permits `better, accept=false`. Environment commit
now requires all safety gates, `comparison=better`, and `accept=true`, so a
model's explicit candidate rejection cannot be silently promoted to a commit.
The focused Slurm regression suite passed 43 tests. Final corrected results are
recorded in `experiments/CA_0721(8)-bailian-proposal-ablation/ablation_results.md`.

## 2026-07-21 — remove small-model diagnosis bias from staged prompt

The staged diagnosis prompt no longer presents `error_type=none` with
`confidence=0.0` as its semantic default and no longer says that a white change
region with a T1/T2 difference is normally correct.  It now asks the MLLM to
inspect full-region coverage, boundaries, and internal gaps; `mixed_error` is
explicitly used for proposals containing both supported and unsupported pixels.
Environment geometry, schema, editability, and polarity validation remain
unchanged.  Proposal generation remains enabled for this prompt-only ablation.

## 2026-07-21 — Bailian prompt + original Proposal Slurm test

After probing node paths, GPU visibility, proxy fingerprints, and the
workspace endpoint, job `42156` ran on `gpu43` with one GPU and three fixed
samples.  The inherited localhost proxy was unavailable on the compute node;
unsetting proxy variables made the endpoint reachable, so this run used the
same direct path as the prior successful Bailian case.  The result is
`outputs/CA_0721(5)-bailian-prompt-original-proposal-direct`.

Aggregate selected metrics were IoU `0.69744116`, precision `0.72136906`,
recall `0.95459950`, and F1 `0.82175592`.  No selected mask improved over its
own initial mask: `test_20_15` stayed at IoU `0.84638554`, `test_78_13` at
`0.75787116`, and `test_85_16` at `0.30658070`.  The run completed all three
samples successfully; Slurm logs are under the experiment's `logs/` directory.

The prompt cleanup changed the initial diagnosis behavior from the old
zero-confidence `none` outputs to high-confidence `none` outputs on these
samples, but did not make the model discover pixel-level false positives in
the initial proposals.  Candidate-stage diagnosis did identify several
`false_positive_change` regions, yet the corresponding candidates were not
accepted.  This is a prompt/model judgment limitation, not an Executor
failure.

## 2026-07-20 — opt-in temporal and instance-mask visualization

The LEVIR closed-loop runner now exposes `--visualize`. When enabled, trajectory
serialization exports the predicted T1/T2 binary masks and one binary PNG for every
connected-component T1/T2 instance used by OmniOVCD-style matching. Artifacts are
organized per sample and per trajectory step, including rejected candidates, under
`visualizations/<sample>/step_<index>/`. Each trajectory step records its relative
visualization directory. The default remains off, so non-visual runs keep their previous
artifact footprint.

## 2026-07-20 — first rich-Verifier GPU result and executable long-output protocol

Single-GPU job `41502` evaluated commit `26a79b7` in
`change_agent_levir_gpu_closed_loop_20260720_015048`. The runner completed successfully
in 171 seconds, but all three initial Verifier calls exhausted retries and safely stopped
at step 0. Aggregate IoU therefore remained the fixed initial `0.69744116`; this run is
diagnostic, not an accepted improvement.

The failure was localized before any Agent/tool action. With six rich region records per
call, one response hit the 1024-token ceiling after repeating a sentence and produced
incomplete JSON. The other responses covered six IDs but repeatedly used
`false_negative` for already-white change-mask components because they interpreted an
empty temporal object mask as the evaluated FN. One valid `true_change` label was also
discarded only because its advisory target/action fields were non-null.

The next protocol keeps Qwen's semantic authority and changes only executability and
definitions:

- Initial long-output batches contain two regions instead of six; all components and exact
  coverage are retained through additional batches. The output ceiling stays 1024.
- The prompt defines FP/FN against the final current change mask. A missing T1 or T2 object
  mask is explicitly not an FN by itself, and ordinary building appearance/disappearance is
  explicitly a true change when supported by RGB.
- `correct_unchanged` lets Qwen state that a black missing-proposal region correctly contains
  no real RGB change. Environment-provided `allowed_verdicts` exposes the white/black geometry
  constraint without choosing the semantic verdict.
- Local target/action inconsistencies no longer invalidate an otherwise useful diagnosis;
  Qwen's global synthesis owns the final error and correction. Impossible white-FN,
  black-FP, unknown-region, polarity, schema, and coverage contradictions remain invalid.
- Verifier generation is deterministic (`do_sample=false`) with repetition penalty `1.05`.
  These settings are included in run metadata and candidate cache identity.

This optimization does not restore any state-to-FP/FN or effect-to-better/worse runtime
mapping. Qwen still owns local semantics, quality/progress, global comparison, and correction.

Follow-up single-GPU job `41503` evaluated clean commit `99aca7a` in
`change_agent_levir_gpu_closed_loop_20260720_020152`. Reducing the batch to two eliminated
all token truncation: complete two-region responses were 797–992 characters. The remaining
failure was purely representational and deterministic: Qwen emitted `"T1"`/`"T2"` and the
string `"null"` instead of lowercase enums and JSON null. Every sample therefore stopped at
the initial fail-safe and aggregate IoU again remained `0.69744116`.

The parser now canonicalizes enum case, spaces/hyphens, and common null spellings before
validating against the same closed enum sets. Raw generations remain unchanged in evidence.
This normalization does not reinterpret a visual verdict or comparison; impossible geometry,
unknown labels/regions, incomplete coverage, invalid numeric ranges, and schema drift still
fail. Prompts also state the literal JSON-null requirement, and regression coverage includes
the exact field drift observed in job `41503`.

Single-GPU job `41504` then evaluated commit `7f8723e` in
`change_agent_levir_gpu_closed_loop_20260720_020605`. All initial schemas and global
syntheses were valid, proving the rich pipeline can execute end to end. `test_20_15`
generated two point candidates (one Qwen-judged unchanged, one rejected by hard locality
gates); `test_78_13` generated one hard-gated global candidate; `test_85_16` was incorrectly
finished at the initial state. No candidate was accepted and aggregate IoU stayed
`0.69744116`.

The semantic trace exposed contaminated visual evidence rather than a missing decision rule.
Qwen repeatedly described purple, pink, and cyan areas as if they were scene objects because
the old local panel blended mask colors into the T1/T2 RGB tiles. It labeled all 19
`test_85_16` proposals `true_change` and assigned quality `0.95`, even though that sample has
many false positives offline. Local true-change records also carried inappropriate point
suggestions, which anchored global synthesis toward large positive-point edits.

The v10 evidence protocol uses one component per rich call. Top tiles are clean T1/T2 RGB with
an external yellow correspondence ring that never overwrites audited pixels; bottom tiles are
explicitly identified as binary geometry and raw RGB difference, not scene colors. Candidate
calls also receive predicted T1/T2 masks. Global synthesis receives component area, box,
audit/delta polarity, verdict/effect, confidence, severity, and prose, but intentionally omits
local advisory target/actions so Qwen must independently plan the final correction. The prompt
asks Qwen itself to favor compact high-confidence errors and calibrate quality over the full
mask. There is still no programmatic semantic ranking or mixed rejection rule.

Job `41505` evaluated this clean-panel v10 at commit `6bf9360` and completed in 385 seconds.
It produced the first effective rich-Verifier result: aggregate IoU improved from
`0.69744116` to `0.69816521` with no sample regression. `test_20_15` improved from
`0.84638554` to `0.84746063`, and `test_85_16` improved from `0.30658070` to
`0.31478642`; `test_78_13` retained its initial mask. Qwen directly accepted both beneficial
candidates as `better`, while hard gates and Qwen `unchanged` rejected the others.

The result is useful but not yet semantically satisfactory. Every initial present component
(8, 8, and 19 regions) was still labeled `true_change` with near-constant 0.95 confidence.
Candidate regions likewise defaulted to the first allowed `added_true_change` label and sometimes
described the identical yellow correspondence ring as brightness evidence. This shows label-order
anchoring rather than grounded mixed/error reasoning, even though the final accepted edits happened
to improve offline IoU. Aggregate performance also remains below the preserved compact baseline
`0.70117909`.

The v11 protocol removes colored outlines entirely: top tiles are unannotated clean RGB at identical
crop coordinates, and the separate binary geometry locates the evaluated pixels. Before emitting
its rich verdict/effect, Qwen must record `t1_state` and `t2_state` as building, background, mixed,
or uncertain. These states and the final semantic verdict are all Qwen outputs and are passed to
global synthesis. Runtime validates their schemas but deliberately does not map state pairs to
FP/FN or better/worse, preserving the Verifier's core responsibility.

Job `41506` evaluated v11 at commit `2b2c7e5`. It stopped all three samples safely at the
first local region and retained aggregate IoU `0.69744116`. The new state chain worked:
Qwen consistently observed `background/background` and explained that no real change was
visible. It nevertheless called the already-white component `correct_unchanged` on the first
attempt and `false_negative` on retry. Thus the visual observation improved, but FP/FN terms
remained inverted relative to the final change mask.

The v12 local schema requires Qwen to echo the authoritative mask state
(`white_predicted_change` or `black_predicted_unchanged`) before its RGB states and verdict.
The prompt gives an explicit four-case semantic table. Local terminology conflicts are retained
with a `geometry_consistency` flag and passed to global Qwen for a second semantic review rather
than invalidating the entire audit. The final synthesis still cannot target an already-white
region as FN or an already-black region as FP. This is structural validation, not a programmatic
replacement verdict.

For better spatial grounding without artificial colors, the panel now contains six tiles: clean
T1/T2, exact binary geometry, T1/T2 focus views that preserve every inside RGB pixel while
dimming only outside context, and raw difference. Candidate output similarly echoes authoritative
added/removed polarity before its RGB states and effect. Runtime verifies only that these copied
geometry fields match the Environment; Qwen still produces every semantic label, score,
comparison, and correction.

Job `41507` evaluated v12 at commit `78aa4fb` and again stopped safely at step 0 with
aggregate IoU `0.69744116`, but it confirmed the semantic fix. On all three samples Qwen's
first response copied the mask state as the shorthand `"white"`, emitted
`background/background`, selected `false_positive`, and explained that the white change was
unsupported by RGB. The only rejection was that the schema expected the expanded literal
`white_predicted_change`; retry then regressed to the old FN wording.

The parser now canonicalizes only this authoritative geometry shorthand (`white`/`black` and
their predicted-change variants) before checking it against the proposal. This is analogous to
the existing T1/T2 case and JSON-null normalization: raw output is retained, and the Qwen
`false_positive` verdict is neither created nor changed by runtime code.

Job `41508` evaluated commit `25ddec4`. The white shorthand was accepted, but the added
focus tiles caused a new evidence artifact: Qwen emitted `dark/dark` as RGB states for later
regions because pixels outside the component had been deliberately dimmed. Since `dark` cannot
be safely canonicalized to building or background, `test_20_15` stopped at region 1 and
`test_78_13` at region 4; all samples retained the initial aggregate IoU `0.69744116`.

The v13 panel removes focus tiles and every other RGB annotation. It contains only byte-identical
T1/T2 crops, a separate binary geometry tile, and raw difference. The prompt explicitly states
that no RGB pixel is outlined, recolored, masked, brightened, or darkened. The mask-state and
T1/T2 reasoning fields remain unchanged, so this removes an artificial cue without weakening
Qwen's diagnostic role.

Job `41509` evaluated v13 at commit `331d8ea` and completed every long regional diagnosis:
35/35 components produced valid rich JSON without truncation. Qwen correctly diagnosed most
present regions as false positives from `background/background` RGB evidence (6/8 in
`test_20_15`, 7/8 in `test_78_13`, and 18/19 in `test_85_16`). This is a substantial semantic
improvement over the former all-`true_change` behavior. No candidate was accepted, so aggregate
IoU stayed at the fixed initial `0.69744116`.

The remaining failure was correction planning rather than local recognition. Global Qwen selected
`test_85_16/r11` for a positive correction even though its own local diagnosis called r11 the
only supported true change and described the other regions as false positives. On the other two
samples it selected a negative click on an object-mask side whose exact component seed was black,
producing a no-op before later hard-gated attempts. The padded crop also leaves very small
components visually under-resolved.

The v14 evidence protocol keeps Qwen as the decision-maker and addresses those failures without
deriving a semantic label in code. Every proposal now reports exact component pixel occupancy and
seed occupancy in the editable T1/T2 masks. Global Qwen uses these facts to choose the target side;
runtime rejects only a guaranteed no-op negative click. Local summaries are ordered from smaller
to larger components to reduce largest/first-region anchoring, while Qwen still chooses which
error matters. Each panel adds byte-identical tight T1/T2 crops alongside padded context, binary
geometry, and raw difference. Finally, global synthesis must be coherent with Qwen's own local
output: it cannot request correction of a region that the local pass explicitly and
geometry-consistently marked correct. This forces a Qwen retry rather than replacing its verdict.

## 2026-07-20 — preserve the first effective baseline and restore a semantic Verifier

The first closed-loop result that met the acceptance criteria is preserved as the
annotated Git tag `closed-loop-effective-v1`. The tag and `main` were pushed before
this redesign; both resolved to `a39d3510ca63f9dbbdd0f93bccef13cd9942b1c4` at that
point. Its accepted GPU artifact remains
`change_agent_levir_gpu_closed_loop_20260719_151344` (job 41410): aggregate IoU
`0.70117909`, exact audit coverage, and no per-sample regression. This gives the compact,
programmatic Verifier a reproducible rollback/reference point.

The next design deliberately restores Qwen as the semantic error-diagnosis and correction
core while retaining the 1024-token ceiling:

- Initial batched calls now output rich region records: FP/FN/true-change/mixed/uncertain,
  target view, proposed correction, confidence, severity, and diagnostic prose.
- Candidate batched calls explain each exact added/removed delta, including beneficial and
  harmful subparts of a mixed component, rather than reducing it to two temporal-state words.
- A separate global Qwen call directly emits quality and progress scores, candidate
  better/worse/unchanged/uncertain, the principal remaining error, exact region ID, next action,
  and a multi-sentence rationale. Mixed or conflicting evidence is not automatically rejected;
  Qwen must weigh it and make the global decision.
- The removed runtime path no longer maps elementary state/effect labels to semantic verdicts or
  better/worse. Runtime authority is limited to schema/range checks, complete coverage, valid
  region/polarity constraints, exact Environment geometry, identical-state handling, caching,
  rollback, and locality/area hard gates.
- Rich output remains bounded by batching all regions and then requesting one synthesis. Candidate
  cache identity includes the new schema, 1024-token setting, threshold, masks, action, facts,
  regions, query, images, and model identity, so old compact decisions cannot be reused.

CPU regression coverage was rewritten around this boundary: Qwen-owned scores and finish,
FP/FN geometry constraints, mixed-but-better acceptance, harmful rejection, full multi-batch
coverage, synthesis region validation, invalid-output fail-safe, identical/cached decisions,
prompt contents, and the 1024-token generation argument. A GPU result is not claimed for the
new rich protocol until a separate closed-loop run is requested and completed.

## 2026-07-19 — accepted three-sample GPU validation

GPU job `41410` (`change_agent_levir_gpu_closed_loop_20260719_151344`) validated commit
`c87b707` on one GPU and completed successfully in 115 seconds. The run meets the
closed-loop acceptance criteria:

- Aggregate conservative-selected IoU is `0.70117909`, above the fixed initial baseline
  `0.69744116` by `0.00373793`; F1 improves from `0.82175592` to `0.82434482`.
- Every sample's selected IoU is at least its initial IoU. `test_20_15` remains
  `0.84638554`, `test_85_16` remains `0.30658070`, and `test_78_13` improves from
  `0.75787116` to `0.76467070`.
- `test_78_13` accepts a 135-pixel pure false-positive removal. TP/FN remain
  `11506/430`, FP falls from `3246` to `3111`, and the candidate finishes normally.
- Initial audit coverage is exactly `1.0` for all samples. Their 8, 8, and 19 components
  are processed in 2, 2, and 4 batches. Every candidate that reaches semantic
  verification also records exact delta coverage `1.0`.
- The size-aware guard rejects the harmful `test_20_15` 352-pixel deletion: it represents
  6.53% of the previous change mask and mask-context disagrees with clean RGB, so its
  final effect is `uncertain`. The global `test_85_16` candidate is rejected earlier by
  area, locality, and target-mask-change gates. Both samples retain their initial masks.

This is the first post-migration run to accept a genuinely beneficial saved candidate,
improve aggregate IoU, preserve every per-sample baseline, and maintain exact initial
and candidate audit coverage simultaneously. The result is accepted; further changes
should use this output and commit as the regression baseline.

## 2026-07-19 — full initial batching and size-aware semantic consensus

GPU job `41407` (`change_agent_levir_gpu_closed_loop_20260719_145637`) completed on one
GPU in 140 seconds with exact component anchors, but exposed an unsafe interaction
between proposal ordering and fallible RGB semantics. In `test_85_16`, Qwen labeled the
largest 684-pixel component unchanged in both temporal images. The exact negative point
therefore removed all 684 pixels, candidate verification called the removal beneficial,
and selected IoU fell from `0.30658070` to `0.16696629`; aggregate IoU fell from the
baseline `0.69744116` to `0.67360342`. The edit was genuinely local and fully audited,
so locality and coverage gates could not detect the semantic error.

Post-rollout GT analysis, performed only after completion, showed that 636/684 removed
pixels were true change. It also showed the complementary opportunity hidden by the
old largest-six initial cap: `test_78_13` contains 127-, 135-, 177-, and 254-pixel pure
false-positive components, and `test_85_16` contains multiple 83–173-pixel pure false
positives. These safer components never reached the initial Qwen call.

- Initial proposals now retain every connected component and use the configured six as
  a per-call batch size, matching the already full-coverage candidate protocol. Exact
  initial coverage is therefore `1.0`; one invalid batch invalidates the full diagnosis.
- When several judgments yield actionable errors, programmatic selection chooses the
  smallest component rather than the largest. Exact Environment seeds remain mandatory.
- Small candidate edits remain decidable from the clean-RGB temporal facts even when the
  advisory mask-context label disagrees, preserving the prior 135-pixel beneficial case.
  If any component or the complete delta exceeds 5% of the previous change-mask area,
  however, mask-context and RGB effects must agree; otherwise its final effect becomes
  `uncertain` and is rejected.
- Schema/cache and configuration metadata are bumped to record full initial batching and
  the 0.05 unilateral-evidence threshold. The output ceiling remains 1024 tokens.

Regression coverage includes two-batch initial full coverage, smallest-component action
selection, rejection of a large cross-evidence disagreement, and acceptance of a small
beneficial edit despite the same advisory disagreement.

## 2026-07-19 — exact component anchors and outlined RGB correspondence

GPU job `41396` (`change_agent_levir_gpu_closed_loop_20260719_143908`) completed on one
GPU in 147 seconds and confirmed exact full candidate-delta coverage: `test_85_16` split
13 components into five batches and audited all 96 pixels. All initial calls were valid
and the locality gate continued to reject a catastrophic global SimpleClick candidate.
Aggregate selected IoU/F1 nevertheless remained at the baseline
`0.69744116`/`0.82175592` because all first-step point coordinates landed in padded crop
background rather than on the audited component. The 96-pixel candidate improved
`test_85_16` offline IoU from `0.30658070` to `0.31390233`, but its exact components were
visually mixed (48 GT-positive and 48 false-positive pixels) and were conservatively
rejected rather than weakening the all-components-beneficial rule.

- Every initial and candidate proposal now records both pixel and normalized
  connected-component seeds. Point feedback uses the exact seed as a degenerate
  `error_region`, and the Agent point example contains that coordinate with an explicit
  instruction to copy it exactly. Box actions retain the padded proposal box.
- Initial and candidate RGB panels draw a one-pixel yellow ring immediately outside the
  exact component in both temporal crops. The audited pixels retain their original RGB;
  the amplified difference remains computed from raw, unannotated T1/T2 crops.
- The prompt directs Qwen to classify the pixels inside that ring. This removes the
  padded-box spatial ambiguity seen in job `41396` without exposing predicted masks or
  abstract FP/FN/action semantics to the decisive visual call.
- The Verifier schema/cache version is bumped so no pre-outline judgment can be reused.

Regression coverage checks exact seed normalization and injection, preservation of RGB
inside the component, the external yellow ring, and raw-difference isolation.

## 2026-07-19 — full delta batching and programmatic initial semantics

GPU job `41377` (`change_agent_levir_gpu_closed_loop_20260719_135752`) completed on one
GPU in 95 seconds, but none of its candidates reached the new RGB temporal-state call.
`test_20_15` improved offline IoU from `0.84638554` to `0.84762580`, yet one of nine
delta pixels fell outside the three-proposal global cap. `test_85_16` similarly had five
uncovered pixels, while `test_78_13` stopped at initial verification after twice labeling
ordinary current-change proposals as false negatives.

- Candidate proposal construction now returns every added/removed connected component.
  The configured value `3` is `max_regions_per_batch`; the Verifier performs as many
  batches as required and still enforces exact total `coverage_ratio=1.0`.
- Each batch independently records advisory mask-context attempts and decisive clean-RGB
  temporal-state attempts. Advisory failure never blocks the corresponding RGB batch;
  one invalid decisive batch invalidates the complete candidate.
- Initial Qwen output is also reduced to elementary T1/T2 RGB states. Predicted masks,
  current-change presence, FP/FN terms, target views, and action semantics are hidden
  from the model call. Runtime geometry derives initial verdict, target view, and action.
- Existing change plus different decisive RGB states derives `true_change`; existing
  change plus equal states derives FP or matching uncertainty; missing change plus
  different states derives FN; missing change plus equal states derives no error.
- Rejected normalized action JSON is persisted in trajectory history. Duplicate-action
  retry prompts quote it as forbidden and require a different action type or geometry,
  while the Environment hard rejection remains unchanged.
- The schema/cache version is bumped, run/config metadata names the per-batch limit, and
  the 1024-token ceiling remains unchanged.

Regression coverage includes four mixed-polarity delta components across two batches,
full pixel coverage, initial state-to-verdict/action derivation, rejection of legacy
abstract initial labels, and persistent duplicate-action constraints.

## 2026-07-19 — RGB temporal facts replace exact dual-label consensus

The closed loop `change_agent_levir_gpu_closed_loop_20260719_205825` preserved the
initial aggregate IoU (`0.69744116`) but exposed two over-conservative filters. Initial
`true_change` answers with an irrelevant T1/T2 target exhausted retries, and a genuinely
beneficial 135-pixel false-positive removal in `test_78_13` was rejected because both
visual branches repeated the same wrong abstract `removed_true_change` label.

- The decisive candidate call no longer asks Qwen for an abstract action-effect label.
  For each exact delta component it emits only elementary T1/T2 RGB states:
  `building`, `background`, `mixed`, or `uncertain`. Action type, delta polarity, and
  added/removed statistics are hidden from this call to avoid semantic anchoring.
- Runtime code combines those states with Environment-owned delta polarity. Equal,
  decisive states mean no temporal change; different decisive states mean a temporal
  change. This deterministically maps additions/removals to the existing beneficial or
  harmful effect labels and then to `better/worse`.
- The mask-context effect call remains as audit evidence, but it is not a veto. Its
  disagreement or JSON failure is recorded without suppressing a decisive beneficial
  RGB result. Only an invalid or mixed/uncertain RGB response remains conservative.
- Non-actionable initial `true_change`/`uncertain` target views are canonicalized to
  `null` with `schema_warnings`; the semantic judgment is not discarded for a cosmetic
  field that has no downstream action meaning.
- The cache schema and run manifests now identify `rgb_temporal_state` evidence, so old
  exact-consensus decisions cannot be silently reused.

Regression coverage includes conflicting mask/RGB conclusions, malformed advisory
mask JSON, RGB uncertainty, target-view canonicalization, and deterministic derivation
of beneficial and harmful candidate effects.

## 2026-07-19 — dual-visual candidate consensus and invalid-initial fail-safe

The valid GPU regression `change_agent_levir_gpu_closed_loop_20260719_203719`
confirmed that programmatic comparison cannot correct a wrong local effect label:
Qwen called 189 newly added false-positive pixels `added_true_change` in
`test_20_15`, while it called a beneficial 135-pixel false-positive removal
`removed_true_change` in `test_78_13`.

- Candidate effects now require exact agreement between two model calls with deliberately
  different visual evidence. `mask_context` retains previous/candidate masks and temporal
  overlays; `rgb_counterfactual` hides predicted masks and shows clean T1/T2 crops, the
  exact component delta, and raw RGB difference.
- Per-region disagreement is converted to `uncertain`; the programmatic comparison gate
  therefore rejects it. Both calls, retry histories, per-mode labels, agreement flags,
  and consensus labels are saved in Verifier evidence.
- The cache schema is bumped and includes the consensus modes, preventing reuse of older
  single-view decisions.
- An invalid initial Verifier result now stops the production runner before Agent action
  generation. This prevents an exploratory edit from being committed against an
  unaudited baseline; the initial mask remains selected with stop reason
  `initial_verifier_invalid`.

## 2026-07-19 — component-safe delta audit and replay identity

- Candidate deltas are no longer collapsed by polarity. Up to three connected components
  receive separate panels and exact `delta_pixels`; explicit covered/uncovered counts and
  `coverage_ratio` force conservative rejection if the panel budget cannot cover all edits.
- Candidate labels are now unambiguous visual facts: `added_true_change`,
  `added_false_change`, `removed_false_positive`, `removed_true_change`, `mixed`, and
  `uncertain`. Mixed labels or a combination of beneficial and harmful components derives
  `uncertain`; only uniformly beneficial components derive `better`.
- Locality, area, and component-count hard gates run before Qwen, so candidates already
  known unsafe consume no Verifier generation. The output ceiling remains 1024 tokens.
- Decision keys now include original images, previous/candidate masks, pixel action, query,
  schema version, model identity, generation settings, proposals, and exact region facts.
  Cache evidence records `decision_key`, `decision_step`, `cache_hit`, and
  `reused_from_step`.
- Every trajectory step stores T1/T2/change-mask SHA256 values. Replay follows the original
  accepted-state chain, uses the recorded matching configuration, and fails before Qwen if
  any reconstructed candidate hash differs from the online trajectory. Ground truth is
  opened only after all Verifier calls for a sample.
- The first GPU regression on this revision showed all three samples immediately finishing
  after only six `true_change` labels; `test_85_16` had just 54.8% initial audit coverage.
  Initial coverage is now explicit, uncovered pixels deterministically localize an
  `uncertain_region` and forbid finish, and non-actionable judgments strictly require
  `target_view=null` as declared by the compact schema.

## 2026-07-19 — compact delta-effect verifier and duplicate-candidate safety

The `change_agent_levir_gpu_closed_loop_20260719_173613` audit exposed three
structural failures: a white `candidate_added` component could be called a false
negative, identical candidates received inconsistent pairwise labels, and six verbose
region feedback sentences could exceed the Qwen3-VL-2B output budget.

- Initial region output is now compact: each fixed `rN` maps to
  `[verdict,target_view]`. Per-region prose is no longer requested. An ordinary white
  component cannot be labeled `false_negative`; that label requires a
  `temporal_difference_missing` proposal.
- Candidate verification no longer repeats the initial six-region analysis or asks Qwen
  to emit `better/worse`. The current component-safe protocol is documented above.
- The runtime derives comparison deterministically. Any harmful effect makes the
  candidate worse, any uncertainty remains uncertain, and every inspected effect must
  be beneficial before the candidate can be better. A delta with uninspected pixels is
  invalid and therefore rejected.
- Candidate decisions are cached by a SHA256 over previous masks, candidate masks, and
  action. The Environment separately fingerprints actions and refuses an exact action
  that was already rejected on the same live state.
- Saved-candidate replay now reconstructs local point/box composition rather than using
  a worker's raw full-image mask as the temporal candidate mask.
- Verifier output budget defaults to 1024 tokens. The compact schema remains mandatory;
  the larger ceiling provides headroom for occasional Qwen formatting drift and retries.
  Trajectory and run-manifest source
  metadata now includes `git_worktree_sha256`, covering tracked diffs and untracked file
  contents without persisting the diff itself.

CPU regressions cover compact initial labels, impossible white false negatives,
programmatic beneficial/harmful comparisons, candidate cache hits, delta polarity and
coverage, repeated-action rejection, and worktree fingerprinting. No GPU rollout was
started as part of this implementation; current behavior still requires a saved-candidate
replay followed by the fixed three-sample closed loop.

## 2026-07-19 — replay challenge, safe initial finish, and pairwise delta crops

- Added `tools/replay_verifier_challenge.py`, which replays the saved tool candidates
  from the previous three-sample run through the GT-free Verifier. GT is opened only
  after each Verifier response to assign offline `better/worse/unchanged` labels.
  The report is atomically committed, so a failed model/configuration run removes its
  temporary directory and leaves no failed result under `outputs/`.
- The Environment and Agent now allow a direct initial `finish` only when the initial
  Verifier is valid, reports `error_type=none`, and sets `comparison=initial, stop=true`.
  Invalid or actionable initial feedback still requires a segmentation tool action.
- Pairwise local panels now show previous-vs-candidate change pixels explicitly:
  previous red, candidate green, and delta blue, alongside the T1/T2 and semantic-mask
  views. This makes the actual candidate edit visible at the same crop scale used for
  regional diagnosis.
- Added regressions for safe initial finish, initial Agent prompt authorization, local
  delta visibility, atomic failure cleanup, and declared comparison tolerances.

## 2026-07-19 — region-grounded diagnosis and categorical pairwise gate

- Replaced Qwen's joint absolute `quality_score` plus continuous `progress_score`
  prediction with two focused stages. The regional stage classifies fixed proposals;
  the candidate stage emits only `better`, `worse`, `unchanged`, or `uncertain`.
  Qwen runtime outputs now leave both numeric score fields null.
- Added Environment-owned Verifier proposals from change-mask connected components,
  T1/T2 object-mask XOR components, and candidate added/removed delta components.
  Proposals are padded, deduplicated, capped at six, serialized with their sources and
  exact pixel counts, and converted to normalized boxes by the runtime.
- Preserved the five labeled full-image inputs and added a 384x384 local four-panel view
  per proposal. The panel explicitly enlarges the RGB crops, binary change mask, and
  color-coded temporal masks, so small white components remain visible.
- Region responses must cover every Environment `region_id` exactly once with
  `true_change`, `false_positive`, `false_negative`, or `uncertain`. Qwen no longer
  generates localization coordinates; actionable `error_region` is the selected
  Environment proposal box.
- Added authoritative mask facts and a semantic consistency gate: when the runtime
  counts any white change pixels, a model response claiming the mask is empty is invalid
  and retried. The proposal builder retains even a single white pixel despite the normal
  minimum-area filter.
- Environment candidate commit now requires categorical `comparison=better` while
  retaining verifier-validity, locality, area, topology, rollback, and full trajectory
  gates. Pairwise trajectories use the latest accepted state as best instead of
  manufacturing an absolute scalar rank.
- Added regressions for proposal construction/source merging, one-pixel preservation,
  full-plus-local visual inputs, empty-mask contradiction retries, fixed-box diagnosis,
  pairwise-only output, identical-state validation, and rejection of worse candidates.

Validation:

```text
Python byte compilation: passed
Unit tests: 68 passed
git diff --check: passed
```

## 2026-07-19 — restore the official SAM3 CUDA autocast contract

- The first migrated A800 smoke job (`40907`) reached fresh SAM3 initialization but
  failed in the fused ViT MLP with a BF16-activation/FP32-weight dtype mismatch.
- The isolated segmentation worker had omitted the CUDA autocast context used by the
  official SAM3 inference examples. SAM3 text initialization and box inference now
  run inside a scoped autocast context: BF16 on supported GPUs and FP16 otherwise.
  CPU execution keeps a no-op context.
- Added regression coverage for CPU, CUDA BF16, and CUDA FP16-fallback selection.
  Slurm released the failed job's GPU; its diagnostic output was removed after the
  final successful rollout, as requested during output cleanup.
- The follow-up job (`40909`) passed the fused MLP and exposed a separate normal
  zero-detection edge case: SAM3 can return empty semantic, instance, and mask arrays.
  The adapter now maps an explicit empty detector result to an all-zero mask and
  confidence map while still raising when the state contains no mask output keys.
  Added a regression for this distinction. Slurm released the failed job's GPU; its
  diagnostic output was removed after the final successful rollout.
- Job `40910` then completed inference and failed only while persisting a BF16
  diagnostic tensor, because PyTorch cannot directly expose BF16 storage to NumPy.
  The adapter now promotes BF16 to FP32 strictly at the diagnostic serialization
  boundary, with a real BF16 tensor regression. Its GPU was released, and the failed
  diagnostic output was removed after the final successful rollout.
- The final one-GPU job (`40911`) completed all three fixed LEVIR-CD samples on one
  A800 in 114 seconds and released its allocation. Aggregate conservative-selected
  IoU/F1 were `0.69744116`/`0.82175592`. The run confirmed that field-free point
  actions execute, retry exhaustion no longer invokes a synthetic box, and rejected
  candidates roll back without contaminating the selected prediction.
- Remaining audit findings are model-policy issues rather than runtime failures:
  ten Agent responses omitted required point/box coordinates, all six Verifier
  localizations covered the whole image, and a post-rollout-improving candidate for
  `test_85_16` (IoU `0.30658070` to `0.38875878`) was rejected because the GT-free
  Verifier score stayed at zero. See
  `outputs/change_agent_levir_gpu_smoke_20260719_030624/CLOSED_LOOP_AUDIT.md`.

## 2026-07-18 — expose both temporal object masks to the Verifier

- Expanded every Qwen Verifier request from three to five labeled visual inputs:
  `T1 original image`, `T2 original image`, `Predicted T1 object mask`,
  `Predicted T2 object mask`, and `Current change mask`.
- The diagnostic prompt explicitly states that T1/T2 masks are predictions rather than
  GT and that the change mask is reconstructed from those masks and OmniOVCD matching.
- Added a regression asserting the exact input order and labels, so future prompt changes
  cannot silently make the model infer image/mask roles from position alone.

## 2026-07-18 — no synthetic box fallback and rejected-candidate rollback

- Removed the runner-generated SAM3 box action that previously executed after Qwen
  action retries were exhausted. Exhaustion now records all rejected raw outputs,
  sets `episode_stop_reason=action_retry_exhaustion_without_state_change`, and safely
  exports the current/history-best state even when no tool action ran.
- Added an Environment candidate-commit boundary. Tool candidates are rejected when
  the Verifier is invalid, the score does not improve beyond `selection_epsilon`, or
  the absolute change-mask area jump exceeds `max_selection_area_delta`.
- Rejected candidate masks, Verifier outputs, tool evidence, area ratios, and rejection
  reasons remain in the trajectory, but the live Environment state and valid feedback
  roll back to the previous accepted version. The step index still advances so repeated
  failures cannot create an infinite loop.
- History-best selection excludes rejected candidates; raw `verifier_best` remains an
  audit artifact. Added regressions for retry exhaustion without fallback execution,
  invalid-Verifier rollback, and excessive-area rollback.

## 2026-07-18 — staged Verifier diagnostics and derived actions

- Split Qwen Verifier handling into a diagnostic stage and a dedicated localization
  request. Missing `error_region` is retained as a semantic diagnosis instead of
  being discarded immediately; one localization request is issued before declaring
  the Verifier invalid.
- Removed model authority over `accept`, `stop`, and `suggested_action`. The runtime
  derives them from `error_type`, `error_region`, and the quality threshold. Invalid
  feedback is explicitly marked `verifier_valid=false`, has
  `localization_valid=false`, `suggested_action=null`, and `stop=false`.
- Invalid feedback retains the previous valid diagnostic text and never tells the
  Agent to finish. Environment termination now uses the derived `stop` field.
- Added regressions for contradictory model fields, successful second-stage
  localization, failed localization with retained feedback, and invalid no-finish
  fallback behavior.

## 2026-07-18 — system-owned Agent coordinate protocol

- Removed `coordinate_frame` from the Qwen action schema and the runner's safety
  fallback. The prompt still defines normalized `[0,1000]` XY/XYXY, while the model
  now emits only action fields and coordinates.
- Made the retired field optional at the parser boundary. Correct legacy declarations
  remain accepted for old artifacts/clients, conflicting values remain invalid, and
  missing declarations no longer consume retries or trigger safety fallbacks.
- Added point, box, prompt, Environment, and trajectory regressions for field-free
  actions. This directly covers the five valid point actions rejected in
  `change_agent_levir_fresh_qwen_20260718_133335` solely for omitting the field.

## 2026-07-18 — normalized protocol, fresh SAM3, and zero-shot Verifier

- Unified every Agent/Verifier-facing region under normalized `[0,1000]` XY/XYXY;
  pixel coordinates now exist only after `ActionParser` inside the Environment.
- Required `coordinate_frame=normalized_1000_xy` for every point/box Agent action;
  missing frames are rejected, repeated all-`<=255` values produce an audit warning,
  and no automatic pixel correction is attempted.
- Added configurable `verifier_best`, `conservative_best`, and `initial` selection;
  every run exports initial, verifier-best, last, and selected masks.
- Added explicit coordinate-space metadata and regression coverage for the 256-pixel
  coordinate mismatch observed in the first three-sample rollout.
- Removed the cached-mask option from the LEVIR runner. Each sample now starts a fresh
  dual-view SAM3 text-prompt worker that loads SAM3 once, processes T1/T2, and saves
  both masks plus all confidence/presence/object-score arrays and worker parameters.
- Added a structured, GT-free Qwen3-VL zero-shot Verifier that shares the already
  loaded Agent model/processor, preserves raw verifier responses, and infers
  `target_view` visually instead of alternating labels.
- Removed the synthetic `index % 2` target-view labels, target-view training array,
  classifier head, action head, and their losses. Training schema/checkpoints are now
  version 3 and retain only quality/error-map/error-type supervision.
- SAM3 initialization persistence is limited to selected diagnostics (mask/logit,
  confidence/presence/object scores, and selected FPN feature arrays), not every
  Transformer activation or attention map.
- Fixed fresh SAM3 mask construction after the first live attempt exposed that
  `semantic_mask_logits` are already probabilities. The adapter now mirrors
  OmniOVCD's semantic/instance/object/presence fusion and `prob_thd=0.4` threshold,
  instead of treating every positive probability as foreground.
- Enforced the first-tool requirement at the Environment boundary after a live retry
  showed that prompt-only guidance allowed three consecutive `finish` actions. Agent
  retries now include the exact validation error, including missing coordinate frames.
- Added an explicitly logged, mask-bounded SAM3 box safety fallback after all action
  retries are rejected, so a zero-shot episode can finish without treating `finish` as
  an executable segmentation decision.
- Made the Qwen zero-shot Verifier return an explicit `uncertain_region` safe fallback
  after malformed structured outputs exhaust retries; raw output, validation errors,
  and `fallback=true` remain in verifier evidence instead of aborting the rollout.
- Kept `RuleBasedVerifier` only as the explicit `--verifier rule` ablation; the real
  runner defaults to `qwen_zero_shot`.
- Renamed offline report fields to `verifier_selected_step` and
  `verifier_selected`, avoiding an implication that the selection is GT-oracle best.

Validation:

```text
Python byte compilation: passed
Unit tests: 33 passed
Runner/worker CLI parsing: passed
git diff --check: passed
```

## 2026-07-17 — overlap-presence matching and full LEVIR runner

- Changed the default matching mode from one-to-one greedy to OmniOVCD-compatible,
  directional `overlap_presence` with threshold `0.25`.
- Kept `greedy_one_to_one` as an explicit ablation and added candidate-pair,
  directional-coverage, and split/merge-ambiguity evidence.
- Separated `t12_min_instance_area` and `cd_min_instance_area`; both default to zero.
- Added isolated real-model subprocess adapters and a segmentation worker for
  SimpleClick point actions and SAM3 box actions.
- Added `tools/run_levir_change_agent.py` for the three fixed LEVIR-CD samples. It
  saves per-step masks/trajectories, verifier feedback, tool reports, history-best
  predictions, and performs GT evaluation only after a rollout-complete marker.
- Expanded the matching regression suite to cover split/merge, additions,
  disappearances, unrelated instances, directional coverage, and area filters.

Validation before the full GPU run:

```text
Python byte compilation: passed
Unit tests: 19 passed
git diff --check: passed
```

## 2026-07-17 — v0–v3 skeleton

- Added validated public/hidden state protocols and a strict JSON Action parser.
- Added GT-free reset/step, point/box dispatch, candidate-state rebuilding, Verifier
  feedback, finish handling, trajectory persistence, and history-best selection.
- Added pure NumPy instance extraction/matching as a deterministic adapter fallback.
- Added the modern Qwen3-VL-2B multimodal Adapter with explicit T1/T2/mask labels.
- Added SimpleClick and OmniOVCD/SAM3 adapter boundaries for isolated environments.
- Added a rule Verifier baseline, frozen-feature trainable head, and offline GT mask
  perturbation labels.
- Added CPU unit tests and a two-round synthetic smoke loop.

Validation performed with Python 3.10.20 / NumPy 1.26.4:

```text
12 unit tests passed
two-round trajectory smoke passed
SegAgent external prev_mask forwarding smoke passed
```

The model-level checks below are now validated with local weights. They are kept in
separate processes because SAM3 and SimpleClick use incompatible dependency stacks.

## 2026-07-17 — training/rollout/server smoke preparation

- Added a deterministic `train_verifier.py --smoke` path. It exercises the trainable
  head, quality/error-map/classification losses, optimizer update, checkpoint writing,
  and memory telemetry.
- Added `rollout_smoke.py`, which uses the real `OmniOVCDAdapter` boundary and mock
  tool callbacks to exercise two feedback-loop steps without GT.
- Added `SAM3ProcessorAdapter` for concrete `set_image`, `set_text_prompt`, and
  `add_geometric_prompt` integration, with a fake-processor regression test.
- Added `qwen3vl_smoke.py` and `download_qwen3vl.py`; the latter keeps model files
  under `models/`, outside the Git repository.
- Added `collect_runtime_manifest.py` and `configs/runtime_cpu.json` for reproducible
  local/server runs.

Smoke results on the current host:

```text
Verifier: 2 epochs, loss 5.3406 -> 5.3378, RSS ~630 MB, CUDA unavailable
Adapter rollout: 2 steps, RSS ~38 MB, CUDA unavailable
```

The current host reports no NVIDIA driver (`nvidia-smi` cannot communicate with the
driver), so real GPU VRAM and CUDA model generation remain pending. Qwen3-VL download
is being performed from the official Hugging Face endpoint in tmux; its completion
status and artifact path are recorded in `logs/qwen3vl_download_official.log`.

The local Qwen3-VL checkpoint subsequently downloaded successfully to
`/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct` (4,255,140,312-byte
`model.safetensors`). The real CPU generation smoke passed:

```text
load_seconds: 5.115
RSS: 37.87 MB -> 981.57 MB (delta 943.70 MB)
raw JSON: {"target_view":"t1","action":"positive_point","coordinate":[500,100]}
parsed pixel coordinate: [32, 6]
transformers: 4.57.3
```

The smoke process reported a transient CUDA probe inconsistency (one process saw 8
devices, while an independent probe and `nvidia-smi` could not initialize NVML). Since
the run used `device_map=cpu` and allocated/reserved 0 MB VRAM, no GPU memory claim is
made until the server driver is healthy.

The Qwen3-VL-to-Environment rollout smoke also passed. The generated raw response was
accepted by the ActionParser, executed as an Environment step, and followed by a
Verifier-accepted `finish`; the trajectory was saved under
`/tmp/change_agent_qwen_rollout_smoke2/trajectory.json`. A single-GPU probe with
`CUDA_VISIBLE_DEVICES=0` currently reports no usable CUDA device, so no GPU was
allocated. When the driver is repaired, use one visible GPU and `device_map=auto` for
the same script; do not expose all GPUs for this smoke.

Real-weight adapter and Verifier checks:

```text
SAM3 text + geometric prompt (CPU): 50.949 s, peak RSS ~7,493 MB,
  models/sam3/sam3.pt, text masks 0, box masks 1
SimpleClick ViT-L external-mask point (CPU): 10.807 s, peak RSS ~3,124 MB,
  models/SimpleClick/cocolvis_vit_large.pth, output 256x256
LEVIR-CD verifier sample builder: 2 paired samples, 10 feature channels
LEVIR-CD verifier head: 1 epoch, loss 5.6379 -> 5.6904, peak RSS ~864 MB
```

The SAM3 and SimpleClick commands intentionally run independently; loading both
large checkpoints together is outside the bounded smoke-test memory budget. The
Qwen-to-Environment smoke remains the supported full Agent rollout, while the two
segmentation adapters are validated at their real model boundaries.

GPU validation (one visible L20, `CUDA_VISIBLE_DEVICES=7`):

```text
Qwen3-VL smoke: CUDA allocated ~4,067 MB, peak ~4,147 MB, load 6.893 s
Qwen3-VL → Environment rollout: CUDA allocated ~4,067 MB, peak ~4,147 MB,
  1 step, 14.188 s
SAM3 text + box smoke: CUDA allocated ~3,525 MB, peak ~3,909 MB, 13.540 s
SimpleClick point smoke: CUDA allocated ~1,249 MB, reserved ~1,450 MB,
  peak ~1,390 MB, 7.167 s
```

The earlier false “CUDA unavailable” result came from running Python inside the
restricted sandbox, which does not expose `/dev/nvidia*`; `nvidia-smi` and the same
commands in the device-enabled context see eight NVIDIA L20 GPUs with driver
595.58.03.
