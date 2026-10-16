# Proposal ablation

Run the three BaiLian Qwen3-VL-Plus arms with the same samples, initialization,
seed, tools, matching, safety gates, and offline evaluator. GT remains closed
until every rollout finishes.

- `direct`: current BaiLian operational mode. Full T1/T2/object-mask/change-mask
  context; Qwen emits binary rubric judgments, short evidence, diagnosis, action,
  and normalized geometry. No Proposal is attached to Environment state.
- `proposal`: selection receives one numbered global SoM overview. Regional calls
  receive an RGB/change overview with the active region marked yellow plus exact
  RGB, T1/T2 object-mask, and change-mask crops.
- `hybrid`: regional calls use the same active mark, with full T1/T2 object masks
  included in the marked overview, plus the same exact crops. Environment Proposal
  seed/box grounds execution and delta verification in both modes.

Staged candidate verification asks Qwen only for physical T1/T2 target presence in
each exact delta region. It does not reuse initial-state error diagnosis and never
asks Qwen for comparison, benefit/harm flags, `accept`, or `stop`. Runtime combines
clear, sufficiently confident RGB evidence with point-action direction and delta
polarity, derives comparison/commit, then performs an independent accepted-state
assessment for remaining error, next action, and termination.

`tools/submit_ca0721_proposal_ablation.sh` creates one parent directory outside
`outputs/`:

```text
experiments/CA_0721(<run>)-bailian-proposal-ablation/
  direct/
  proposal/
  hybrid/
```

For the repaired visual-context contract, submit the three-arm smoke run with
`tools/submit_ca0722_context_fix_ablation.sh`. It writes a self-contained parent under
`outputs/`, pins all arms to the already-probed node, archives Slurm logs below each
arm, and sends one completion notification per arm.

CA_0722(5) retired Direct and Hybrid from the immediate optimization loop:
Direct still accepted a harmful candidate, while Hybrid matched Proposal exactly
after v5 failed closed. The next experiment is Proposal-only. Schema v6 aggregates
the full point-action delta inside the authoritative ROI, shows exact delta masks
and highlighted RGB, and requires complete region coverage before `finish`.
Submit it with `tools/submit_ca0722_proposal_scoped_v6.sh`.

CA_0722(6) confirmed that action-scoped aggregation works: every candidate edit
became one complete evidence unit and all five candidate assessments had
`evidence_sufficient=true`. No candidate committed and aggregate IoU remained
`0.69744116`. The useful test85 path was never attempted because the highest
confidence diagnosis named an uneditable T1 seed and the initial planner did not
fall through to the valid r1/T2 diagnosis. Schema v7 preserves every v6 safety
gate but ranks diagnosed errors until it finds the first Environment-executable
action. Submit it with `tools/submit_ca0722_proposal_executable_v7.sh`.

CA_0722(7) also selected no candidate and retained aggregate IoU `0.69744116`.
The fallback was not exercised because r1 changed from the v6 actionable T2
diagnosis to `none`. The semantic evidence and proposal order were identical,
but prompt hashes differed: `editable_seed_white` was built by iterating the
unordered `TARGET_VIEWS` set and serialized as T2/T1 in v6 versus T1/T2 in v7.
Schema v8 fixes the order to T1/T2 and resolves an invalid target view only when
Environment facts identify exactly one view with the required editable seed.
Ambiguous or impossible targets still fail closed. Submit it with
`tools/submit_ca0722_proposal_deterministic_v8.sh`.

CA_0722(8) verified the deterministic action path but did not improve selected
aggregate IoU (`0.69744116`). test85 now attempted two T2 deletions. The second
candidate improved offline IoU from `0.30658070` to `0.33236925`, removing 349
false-positive pixels and no true positives, but candidate evidence labeled the
action-scoped delta as background-to-building at confidence `0.95` and rejected
it. The next Proposal experiment should preserve original RGB values in a
delta-only view and use contours rather than a dominant fill highlight; do not
lower the runtime evidence or harm gates to force acceptance.

Schema v9 implements that evidence-only change. Candidate regional calls replace the
large rectangular RGB crop and yellow fill with delta-only T1/T2 RGB crops: original
colors are retained only on authoritative delta pixels, black is presentation padding,
and a one-pixel cyan outer contour marks the delta without overwriting its colors.
The candidate evidence threshold, polarity checks, and harm gates are unchanged.
Submit it with `tools/submit_ca0722_proposal_delta_only_v9.sh`.

CA_0722(9) completed successfully in Slurm job `44554` (2m45s, empty stderr),
but selected aggregate IoU remained `0.69744116`. test85 again attempted two T2
deletions; the second removed 349 false-positive pixels without losing any true
positive and reached offline IoU `0.33236925`, but was rejected. With the v9
delta-only input Qwen labeled it `T1 building -> T2 background` at confidence
`0.95`; v8 had labeled the same useful delta in the opposite temporal direction,
also at `0.95`. The unchanged rejection under opposite high-confidence labels
indicates that the remaining bottleneck is fine-grained visual semantic capability,
not action planning, delta coverage, or the runtime safety threshold. The next
controlled experiment should switch the candidate-evidence vision model while
keeping the v9 representation and all runtime gates fixed.

The next controlled run uses the fixed BaiLian snapshot
`qwen3.7-plus-2026-05-26` with thinking disabled, while retaining the v9 evidence
representation, samples, thresholds, and safety gates. The launcher records the
model explicitly in the parent manifest:
`tools/submit_ca0722_proposal_qwen37_v9.sh`.

CA_0722(10) completed successfully in Slurm job `44562` on `gpu46` (1m28s,
empty stderr), but it did not reach candidate-delta verification. On test85,
Qwen3.7's selection reason explicitly identified bare ground/dirt inside r0 as
over-segmented non-change, then its evidence and diagnosis stages labeled the
same region `T1 background -> T2 building` and `error_type=none`, both at
confidence `0.95`. With no authorized action, every sample produced zero
candidates and selected aggregate IoU remained `0.69744116`. test20 and test78
also executed no action, so the conservative safety behavior did not regress.
This full-model replacement is therefore inconclusive about Qwen3.7's v9
candidate-delta semantics: the model failed earlier through a cross-stage
selection/diagnosis contradiction. A future model comparison should either
enforce consistency between those stages or hold the v9 initial planner fixed
and replace only the candidate-evidence model.

CA_0722(11) is the direct thinking-mode follow-up to CA_0722(10). It keeps the
fixed Qwen3.7 snapshot, Proposal v9 evidence, prompts, samples, seed, thresholds,
and safety gates, while changing hosted calls to `enable_thinking=true` with a
fixed `thinking_budget=256`. Submit it with
`tools/submit_ca0722_proposal_qwen37_thinking_v9.sh`. The run manifest records
both values so the treatment cannot be confused with the preceding non-thinking
run.

Each child contains its own `logs/`, trajectories, feedback, masks,
predictions, and `per_sample_metrics.json`. Compare initial-error localization,
small-change recall, invalid/unsafe tool actions, accepted-candidate IoU/F1,
audited-region coverage, token usage, endpoint failures, and latency. Do not
rank arms from aggregate IoU alone.

Direct schema `direct_change_rubric_v3` canonicalizes common model aliases such
as `missing_detection` to `false_negative`, while retaining JSON and geometry
validation. Qwen cannot author quality, progress, comparison, or acceptance.
Runtime weights change precision/recall/extent/boundary/artifact booleans, applies
the evidence-sufficiency hard gate, retains target-scope as an audit diagnostic,
then derives candidate comparison from one
benefit and three harm flags. Proposal and Hybrid retain their staged decision
contract. In every arm, Environment commits only `better, accept=true`
candidates after its own safety gates.

Negative points use the SimpleClick output as an ROI-clipped subtraction and no
longer delete the entire clicked initial connected component. Direct candidates
receive explicit full and local delta views; Proposal/Hybrid candidate records retain
separate previous/candidate object-mask and change-mask facts.

After any rollback, the accepted masks and their exact rebased point-session masks
remain authoritative; the rejected candidate is retained only for audit and replan
context. Direct does not reuse the rejected `suggested_action`: it makes a new
full-context Qwen call with the rejected action, mask delta, and rejection
reason, plus a bounded four-entry history of recent rejected actions. The
replan contract uses `candidate_effect=null`; runtime assigns
`comparison="uncertain"` and `accept=false`, while Qwen supplies a different
executable action or a `finish`.

For the post-fix Direct-only three-sample run, first probe a chosen GPU node,
then pin the run to that same node:

```bash
NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0721_direct_replan.sh 9
```

The run writes to `outputs/CA_0721(9)-bailian-direct-rollback-replan/`, with
Slurm stdout/stderr archived below its `logs/` directory.

## CA_0723 atomic v11 five-sample controller/Verifier ablation

V10 demonstrated that exhaustive region coverage alone is insufficient: a useful
selection rationale can be lost across independent evidence and diagnosis calls.
V11 uses one grounded target-aware audit per Environment region and carries the
selected diagnosis and attempted action into candidate verification. Runtime keeps
exclusive authority over geometry, transition polarity, acceptance, rollback, and
post-commit re-audit.

The corrected variant also preserves the global screening reason inside every
atomic audit and requires an explicit confirmed/refuted/uncertain resolution. It
normalizes the observed hosted alias `high` to categorical `clear`, but unknown
qualities and every semantic contradiction still fail closed. The initial
`CA_0723(3)` hosted attempt was invalidated before semantic audit and moved to the
workspace diagnostics directory. A contradictory local audit now fails only that
region closed and cannot authorize finish, while independent valid diagnoses remain
actionable. Use run index 5 for the corrected experiment.

The historical `tools/submit_ca0723_atomic_v11_5sample.sh` is frozen behind the
`stable-ca0723-5-bailian-proposal-atomic-v11` tag. It held Environment, Executor,
inputs, and safety gates fixed and submitted three Verifier-planner treatments.
The Verifier generates the diagnosis and action plan, so this is not a pure
verifier-only ablation with an independent fixed Agent:

- `qwen37-thinking`: `qwen3.7-plus-2026-05-26`, thinking budget 256;
- `qwen37-no-thinking`: the same snapshot without thinking;
- `qwen3vl-plus`: `qwen3-vl-plus` without thinking.

All arms use `test_20_15`, `test_78_13`, `test_85_16`, `test_1_1`, and
`test_50_8`, seed 42, Proposal mode, identical fresh SAM3 initialization, and at
most three action steps. Compare diagnosis/action rate, candidate acceptance,
post-commit re-audit behavior, and per-sample initial-to-selected metrics—not only
aggregate IoU.

CA_0723(5) completed on `gpu46` in jobs `44878`--`44880`. Qwen3.7 non-thinking
was best, moving five-sample aggregate IoU/F1 from the common
`0.67629040/0.80688931` to `0.70009134/0.82359262`. It executed nine actions,
accepted seven, rejected two, and selected every accepted improvement correctly in
offline evaluation. test85 reached IoU `0.33580247` after three accepted edits,
compared with `0.30658070` initially. Qwen3.7 thinking reached `0.69462518`; Qwen3-VL-Plus
reached `0.68465376` and misaccepted a harmful test20 edit. Use the non-thinking
Qwen3.7 arm as the primary v11 result. None of the five episodes ended with
`verifier_authorized_finish`; this result demonstrates useful budgeted edits, not
semantic convergence.

Protocol v11.1 uses `tools/submit_ca0723_atomic_v11_1_5sample.sh`. It is the
next-run hardening and adds a 3x3 full-frame scan only when both temporal object
masks are empty. A suspected black scan tile is corrected with its
Environment-owned SAM3 box and re-verified like any other candidate. It also adds
bounded transient-network retry, explicit execution/outcome status, credential
isolation, and the Proposal rollback diagnosis-state fix. These changes are not
part of the historical CA_0723(5) metrics.
