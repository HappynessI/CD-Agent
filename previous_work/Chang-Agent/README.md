# Change-Agent

文档入口见 [doc/README.md](doc/README.md)。2026-10-16 以前的开发说明、协议维护和实验计划已集中归档到 `doc/archive/2026-10-16/`，按历史资料阅读；后续新文档直接维护在 `doc/`。

2026-10-16 目录整理：SAM3、SimpleClick 与训练依赖共用工作区 `Chang-Agent/third_party` 中的 OmniOVCD、SegAgent、LLaMA-Factory；环境统一位于工作区 `conda_envs`。活动脚本及环境安装引用已同步更新，历史实验记录保留原路径。

2026-10-16 收拢：本目录是唯一开发目录，使用原本地 Qwen 分支的较新代码。[项目总结](doc/archive/2026-10-16/summary.md)与现有维护报告保留；旧实验输出已清理，`outputs` 留空供新实验使用。旧主目录和旧运行快照分别保存于本地 Git 分支 `archive/direct-verifier-working-copy-20261016`、`archive/baseline-0801-working-copy-20261016`。原 `Chang-Agent-local-qwen3vl` 和 `run_worktree_archive` 已移除。历史报告里的旧工作目录、实验路径及验收记录仅作历史依据；依赖旧输出的重放或验收需要先重新生成产物。

历史 Qwen 工具验证记录完整保留在 `diagnostics/legacy/` 下的六个 `qwen3vl_*` 目录，原始日志、结果和中间文件均保留；旧流程脚本已更新到该路径。历史回执内的原路径保持原样。新的项目诊断放在本项目 `diagnostics/` 下。工作区整理和模型下载的辅助记录已按用户要求删除。

Change-Agent is a GT-free iterative change-detection research framework built around
the existing `SegAgent` and `OmniOVCD` repositories. The runtime loop keeps T1/T2
semantic masks, instances, matching, and model evidence inside the Environment. The
current production runner is Verifier-planned: the Verifier receives T1, T2,
predicted T1/T2 object masks, the current change mask, and history, then selects a
diagnosis and action. The Environment validates that plan and the Executor applies a
deterministic SimpleClick or SAM3 edit. There is no separate Agent policy in this
path. These masks are model predictions, not GT.

The implementation follows [`../CHANGE_AGENT_DEVELOPMENT_SPEC.md`](../CHANGE_AGENT_DEVELOPMENT_SPEC.md).

本轮完整开发记录见：[DEVELOPMENT_WORK_REPORT.md](doc/archive/2026-10-16/DEVELOPMENT_WORK_REPORT.md)。
分步 Verifier 与商业模型接入维护说明见：
[`doc/archive/2026-10-16/verifier_protocol_maintenance.md`](doc/archive/2026-10-16/verifier_protocol_maintenance.md)。

## Current milestone

The current code implements the v0–v3 research skeleton:

- strict `State`, `Action`, and `VerifierOutput` protocols;
- JSON extraction plus view/action/coordinate/box validation;
- a Qwen3-VL-2B Adapter using `AutoProcessor`, `apply_chat_template`, and
  `Qwen3VLForConditionalGeneration`;
- a single public `[0,1000]` coordinate protocol for Verifier plans and pixel XY
  coordinates only inside the Environment;
- an inference-only Environment with no GT argument or label state;
- SimpleClick point and SAM3 box boundaries;
- per-step instance extraction, default OmniOVCD overlap-presence matching, optional
  one-to-one greedy ablation, and change-mask reconstruction;
- a region-grounded Qwen3-VL zero-shot Verifier that produces rich local diagnoses and
  directly synthesizes quality, progress, candidate comparison, and correction while sharing
  the Agent model weights, plus a
  transparent rule Verifier ablation and a legacy trainable
  frozen-feature Verifier head for offline quality/error-map/error-type experiments;
- offline GT perturbations for Verifier supervision;
- feedback-driven iteration, finish rejection, complete trajectory artifacts, and
  history-best state selection.

The LEVIR runner in `tools/run_levir_change_agent.py` performs fresh
SAM3 text-prompt initialization for both temporal views on every sample; cached masks
are not accepted by the entry point. It saves the initial masks, confidence maps,
presence/object scores, prompt/configuration, and worker report before running
Verifier-planned actions. Worker stdout/stderr are retained only on failure.
Qwen3-VL zero-shot verification is the default and the rule Verifier remains an
explicit `--verifier rule` ablation.

An opt-in `--verifier qwen_staged` path uses a v11.1 atomic, target-aware regional
audit. Each region call must jointly report the copied mask state, T1/T2 physical
state, whole-component mask assessment, and a complete checklist with observable
evidence for every item. Exhaustive global screening runs first, and its exact
suspected-error reason is persisted into each selected region; the atomic audit must
explicitly confirm, refute, or mark that hypothesis uncertain. Runtime rejects
contradictions inside that record, so a model cannot identify over-segmentation in
one stage and silently turn the same region into an unrelated all-pass diagnosis.
Candidate comparison uses only clear local RGB presence evidence plus runtime-owned
action/delta polarity. Point-tool delta fragments are aggregated into one
Environment-owned action-ROI record and shown as exact added/removed masks plus
original-color delta-only RGB crops with a cyan contour; low-confidence or contradictory evidence fails closed
rather than reusing initial-state black/white diagnosis rules. A staged `finish`
requires every Environment audit region to have sufficient evidence and a `none`
diagnosis.
The planner prefers a safe executable diagnosis with the smallest component first.
If both temporal object masks are empty, Proposal mode uses a deterministic full-frame
grid scan. The Verifier may identify a black tile as a false negative; recovery uses
the Environment-owned tile box with SAM3, never model-authored geometry.
After Executor produces a candidate, the attempted action, persisted initial
diagnosis, and exact mask delta are returned to the Verifier. Only a runtime-derived
`better` transition is committed; rejected candidates roll back, while accepted
candidates trigger a complete fresh regional audit before the next action or finish.
The same staged interface supports local Transformers weights and BaiLian
`qwen3-vl-plus`. Legacy `--agent-backend`, `--max-new-tokens`, and
`--action-retries` arguments remain parseable for archived launchers but do not
configure the production action path.
API credentials are read only from `DASHSCOPE_API_KEY` (or the variable named by
`--bailian-api-key-env`), are removed from segmentation-worker environments, and are
never stored in run artifacts. Hosted 429/5xx and transient URL failures use bounded
exponential retry; authentication and other non-transient 4xx failures do not retry.

`--proposal-mode direct|proposal|hybrid` provides a controlled Proposal ablation.
Direct sends full state to Qwen and accepts model-authored action geometry. Proposal
deterministically audits every Environment region with a marked overview, local RGB
and mask crops, and the exact audited component geometry. Hybrid adds full-frame mask
state while keeping the same exhaustive audit and Environment-owned execution geometry.
`--verifier-bailian-model` selects the hosted controller/Verifier model. Changing it
changes diagnosis and action planning together, so this is a controller ablation, not
a verifier-only comparison with an independently fixed Agent.
The current optimization path uses Proposal with Environment-owned geometry.
Direct and Hybrid remain available for historical ablations but are not part of
the next BaiLian experiment after CA_0722(5).
See [`doc/archive/2026-10-16/proposal_ablation.md`](doc/archive/2026-10-16/proposal_ablation.md).

The completed five-sample CA_0723(5) v11 run demonstrated useful closed-loop edits. The
recommended Qwen3.7 non-thinking Verifier executed nine actions, accepted seven,
rejected two, and improved aggregate IoU from `0.67629040` to `0.70009134`; test85
improved from `0.30658070` to `0.33580247`. Every accepted candidate improved
offline IoU and both rejected candidates were worse than the retained mask. This is
not a convergence claim: none of the five episodes ended with
`verifier_authorized_finish`. New artifacts separate `execution_status` from
`episode_outcome` and report convergence explicitly.
The post-audit changes are protocol v11.1 and must be evaluated as a new run.

### Visualize staged Verifier inputs

To prepare presentation images from a completed staged Proposal run, use
`tools/visualize_verifier_inputs.py`. The script makes no model/API call. It exports
the numbered global proposal overview, each region's T1/T2 RGB crop and binary mask
crops, candidate added/removed delta crops, contact sheets, and a JSON manifest:

```bash
/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python \
  tools/visualize_verifier_inputs.py \
  --run-dir outputs/CA_0722\(5\)-bailian-context-fix-3arm/proposal \
  --samples test_20_15 \
  --initial-only \
  --output outputs/visualization
```

`step_000_initial/00_global_overview.png` is the selection-stage input. Each
`step_000_initial/rN/` directory contains local evidence inputs for region `rN`;
`contact_sheet.png` is a compact presentation view. Omit `--initial-only` to also
reconstruct candidate-step delta inputs from saved tool artifacts.

The offline training schema deliberately has no `target_view` target. Earlier smoke
data alternated T1/T2 by sample index, which was not a real label and must not be used
for training. A future trained target-view policy requires real supervision or a
separately validated latent/tool-ranking objective.

## Coordinate boundary

- Agent JSON coordinates and Verifier `error_region` are normalized XY/XYXY integers
  in `[0,1000]`. The runtime owns this protocol; Agent actions contain only the action
  fields and coordinates and do not repeat coordinate configuration metadata.
- `coordinate_frame=normalized_1000_xy` remains accepted only for compatibility with
  older action payloads. It is no longer required or requested from Qwen; a conflicting
  legacy value is rejected rather than allowed to override the system protocol.
- `ActionParser` is the only public-to-internal conversion boundary.
- Parsed actions, Environment state, and SimpleClick use original-image pixel XY.
- Trajectories preserve the raw normalized payload, parsed pixel action, and a warning
  after two consecutive actions whose public values are all `<=255`; no automatic
  pixel-coordinate correction is performed.
- SAM3 geometric prompts use normalized center/size values created by the Executor.

## Verifier feedback boundary

- The Environment still owns geometry: it enumerates every auditable initial component or
  candidate added/removed component, gives it a stable rN/dN identifier, measures exact
  pixel coverage, and batches panels without dropping the remaining components.
- Qwen is again the semantic Verifier rather than an elementary-state classifier. Its initial
  regional pass diagnoses true_change, correct_unchanged, false_positive, false_negative,
  mixed, or uncertain against the final change mask; it also returns the target view,
  proposed correction, confidence, severity, and
  one-to-three sentences explaining the local visual evidence. Each rich diagnosis also records
  authoritative white/black change-mask state and T1/T2 building/background/mixed/uncertain
  states as an explicit grounding chain; runtime checks the copied geometry but does not derive
  the verdict from the RGB states.
- Candidate regional passes diagnose the actual action delta as added_true_change,
  added_false_change, removed_false_positive, removed_true_change, mixed, or
  uncertain, again with a corrective proposal and detailed feedback. A mixed component must
  describe both its beneficial and harmful portions instead of being automatically rejected.
- A separate Qwen global-synthesis call sees the images, current/previous masks, authoritative
  geometry facts, action, and every local diagnosis. Qwen directly returns quality_score,
  progress_score, better/worse/unchanged/uncertain, the main remaining error, an exact
  region ID, and the next correction. Thus mixed, uncertainty, or simultaneous beneficial and
  harmful evidence are weighed by the Verifier rather than collapsed by a program rule.
- Direct is a separate full-context contract. Qwen returns seven binary rubric judgments
  with short observable evidence, an error type, and action geometry. It never returns
  `quality_score`, `progress_score`, `comparison`, or `accept`. Runtime computes weighted
  quality from semantic precision/recall, changed-object extent, boundary alignment, and
  artifact control. Evidence sufficiency is a hard gate; target-class-only reasoning is
  retained as an auditable diagnostic and does not block an actionable FP/mixed diagnosis.
  For the building query, roads, parking areas, vehicles, vegetation, bare ground, shadows,
  illumination, and registration differences are explicitly non-target context.
- A Direct candidate additionally reports whether the intended error improved and whether
  it introduced a false positive, false negative, or boundary/artifact regression. Runtime
  derives comparison and acceptance from those booleans. Benefit without harm is `better`;
  harm without benefit is `worse`; neither is `unchanged`; simultaneous benefit and harm is
  `uncertain`.
- Runtime checks are deliberately limited to protocol and safety invariants: exact JSON schema,
  enums/ranges, complete region coverage, valid region IDs, added/removed polarity, the fact that
  an already-white component cannot be a false negative, exact region-to-coordinate conversion,
  local/global Qwen outputs cannot select the same region as both correct and erroneous, a
  negative click cannot target a black seed in its editable mask, identical-state handling,
  SHA256 decision caching, rollback, and locality/area hard gates.
  Runtime code does not infer semantic better/worse from effect labels.
- The Verifier generation ceiling is configurable; CA_0723(5) used 1536 tokens. Rich
  diagnosis uses one exact local
  region per call followed by one global synthesis. The local panel contains both padded and
  tight unannotated clean T1/T2 RGB, separate binary geometry, and raw difference. Exact
  per-component and click-seed occupancy in the editable T1/T2 masks is supplied separately, so
  Qwen can diagnose the scene and plan an executable correction without artificial RGB cues.
  Verifier generation is deterministic
  with a small repetition penalty; these settings are part of the decision-cache identity.
- An initial state can finish only when no remaining error is reported and its quality score meets
  the configured threshold. In Direct, this score and acceptance are runtime-derived from the
  rubric; in Proposal/Hybrid, the staged model decision remains authoritative. A candidate may
  still contain a localized remaining error, in which case the accepted state continues with the
  next correction. Invalid output never authorizes an action or stop.
- `tools/replay_verifier_challenge.py` is retired for v11 validation: its negative
  point deletes a whole connected component and does not match the current ROI-clipped
  Executor. It now requires an explicit legacy acknowledgement.

The runner supports `verifier_best`, `conservative_best`, and `initial` selection
policies. All attempted candidate masks are retained, and initial, verifier-best,
last-attempted, and selected prediction masks are exported. A tool candidate is accepted
only when the Verifier is valid, its categorical comparison is `better`, it sets
`accept=true`, and its
absolute mask-area jump stays within the configured limit. Rejected candidates
remain auditable in the trajectory, while the next Verifier plan resumes from the
previous accepted state. If the Verifier authorizes no executable action, the episode
stops without inventing a fallback action.
If the initial Verifier exhausts its own retries and remains invalid, the production
runner stops before requesting any tool action and exports the unchanged initial
state.
Candidate decisions are cached by the SHA256 of previous masks, candidate masks, and
action. An exact action rejected on an unchanged live state cannot execute again.
Rejected normalized action JSON is retained in history and injected into retries as a
forbidden action; the next response must change its tool type or geometry and select a
different unresolved region.

The runner serializes the already-validated Verifier plan into the public Action JSON
contract. Point plans use the exact Environment anchor, box plans use the exact
Environment box, and finish contains no geometry.

## Local editing and safety gates

- A positive point merges only the tool-predicted connected component containing the
  click. Proposal/Hybrid point execution resolves the Verifier-selected `region_id`
  against the live Environment proposal. A missing or stale staged region ID fails
  closed before tool execution. For a negative point, the default SimpleClick
  `prev_mask` is the complete accepted session mask so the model retains segmentation
  context, while the removal boundary is independently restricted to the live white
  component containing the authorized seed inside the proposal ROI. Diagnostic replay
  may select `proposal_roi` or `clicked_component` input without broadening that write
  boundary. The region ID and ROI remain runtime metadata rather than model-authored
  Action JSON.
- A SAM3 box result replaces pixels only inside the requested pixel XYXY box; pixels
  outside that ROI are copied from the previous target-view mask. Staged box actions
  carry their region ID separately from Action JSON, and Environment requires the parsed
  box to exactly match that live proposal before SAM3 runs.
- Every tool result records target-mask XOR statistics: action ROI, changed and
  outside-ROI pixels, outside-ROI ratio, target-mask change ratio, largest changed
  component, and before/after component counts.
- Candidate acceptance rejects non-`better` runtime-derived decisions, excessive outside-ROI
  changes, excessive target-mask changes, excessive component-count jumps, and excessive
  change-mask area jumps. The proposal-point precheck rejects only when all three limits
  are exceeded: proposal-relative change (`0.25`), absolute change (`256` pixels), and
  full-image change (`0.004`). This preserves small-component edits for Qwen candidate
  verification while retaining a GT-free catastrophic-edit guard. Thresholds and every
  subdecision are trajectory fields.
- Actionable `error_region` always comes from the bounded Environment proposal set.
  Unknown/duplicate region IDs, missing proposal judgments, false-positive judgments
  with no white pixels, and mask-emptiness contradictions are rejected and retried.
- Successful segmentation-worker stdout/stderr are discarded. A failed or timed-out
  worker retains bounded `stdout.log`/`stderr.log` beside its structured artifacts;
  API keys are stripped from the worker environment before launch.

### CA_0730(4) SimpleClick live-trajectory validation

The current Proposal default has already passed a controlled fixed-action replay:
negative points keep the complete accepted SimpleClick session as model input while
writes are restricted to the clicked live component inside the authorized proposal
ROI. On the frozen three-sample action schedule this reduced cumulative authorized
pixels by `60.5%`, changed micro IoU from `0.722826` to `0.719928`, and changed Recall
from `0.952816` to `0.954600`.

CA_0730(4) is the separate live-trajectory validation of that existing default. It
does not repeat the fixed-action mechanism comparison and does not claim causal or
LEVIR-CD generalization improvement. It tests whether autonomous Qwen3.7 non-thinking
planning, candidate acceptance, the conjunctive point gate, and accepted-state live
proposal refresh together preserve the three-sample acceptance contract:

- selected micro IoU must be at least `0.7155976055`: **failed**, observed
  `0.70422933` (`-0.01136828`);
- writes outside the authorized clicked component must remain zero: **passed**;
- accepted edits must replan against the refreshed live proposal catalog: **passed**
  for all `5/5` accepted edits.

Job `50213` completed on `gpu37` in `00:36:07` with exit `0:0`. All nine point
actions used the complete accepted session mask as SimpleClick input. Five candidates
were accepted, four were rejected, three of the rejections were hard-gated before
Verifier review, and no fallback or action-attempt error occurred. All episodes
exhausted the three-action budget and none converged.

The fresh SAM3 initialization had micro IoU `0.69744116`; the selected result therefore
improved its own live starting point by `0.00678817`. This does not contradict the
failed absolute contract: the archived CA_0723 reference (`0.72282586`) and CA_0726
fixed replay (`0.719928`) used byte-fixed historical initialization and/or frozen
actions, whereas CA_0730 performed fresh initialization and autonomous planning.
See [`doc/archive/2026-10-16/CA_0730_SIMPLECLICK_LIVE_VALIDATION.md`](doc/archive/2026-10-16/CA_0730_SIMPLECLICK_LIVE_VALIDATION.md)
for the per-sample trajectories, safety audit, and startup diagnostics.

To reproduce it, first complete a new fixed-node Bailian probe, then submit the single
three-sample arm with:

```bash
NODE=<probed-node> BAILIAN_NETWORK_MODE=<direct-or-proxy> \
  bash tools/submit_ca0730_simpleclick_live_3sample.sh
```

The completed output is
`outputs/CA_0730(4)-simpleclick-full-session-component-write-live-3sample`.

## Runtime audit

Each sample writes `episode_summary.json` with its stop reason, loop count,
accepted/rejected candidates, tool count, action-attempt errors, elapsed time, and
selected steps. `execution_status=completed` means the process completed;
`episode_outcome` separately reports `converged`, `budget_exhausted`,
`verifier_invalid`, or `no_executable_action`. The run manifest includes outcome counts
and convergence rate. An uncaught runtime failure atomically updates the manifest to
`failed`, writes a redacted `failure.json`, and leaves `.run_incomplete` in place.
Every invalid action attempt records its loop index and validation error.
Trajectory metadata resolves Git from
the repository path and records commit, dirty state, and a SHA256 fingerprint of tracked
diffs plus untracked file contents. The run manifest records the same source fingerprint,
Python, platform, parent-process seeds, model metadata checksums, and deterministic policy.
Isolated segmentation workers are launched through `seeded_segmentation_worker.py`,
which seeds Python, NumPy, and PyTorch before model construction and appends that seed
record to each worker report.

## Local smoke commands

The isolated `omniovcd-env` can prepare the Qwen3-VL dependency set without changing
the legacy SegAgent environment:

```bash
/Data/wyh/CD-SegAgent/omniovcd-env/bin/pip install \
  'accelerate>=0.26' 'qwen-vl-utils>=0.0.8'
HF_ENDPOINT=https://huggingface.co \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python tools/download_qwen3vl.py
```

On a host without a working NVIDIA driver, load and generate on CPU:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/qwen3vl_smoke.py --device-map cpu
```

The smoke report records load time, process RSS, CUDA availability, raw JSON, and the
parsed Action. On a CUDA server use `--device-map auto`; the report then includes
PyTorch allocated/reserved VRAM.

Verifier and adapter-loop smoke tests do not need model weights:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --smoke
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py --agent qwen3vl \
  --model-path /Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct \
  --device-map cpu
```

`configs/runtime_cpu.json` documents the no-GPU fallback. Use
`tools/collect_runtime_manifest.py` at the start of real runs to capture the commit,
software, model path, split, seed, and CUDA state.

On this host, GPU access requires an execution context with `/dev/nvidia*` mounted.
The validated low-occupancy profile is one L20 only:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py --agent qwen3vl --device-map auto \
  --model-path /Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct
```

Use [`configs/runtime_gpu_l20.json`](configs/runtime_gpu_l20.json) as the corresponding
runtime manifest. A shell launched only inside the restricted filesystem sandbox can
still report `cuda.is_available() == false`; verify with `nvidia-smi` in the device-
enabled execution context.

Real local-weight adapter checks are isolated by dependency stack:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/sam3_smoke.py --device cpu --resolution 1008
PYTHONPATH=/Data/wyh/CD-SegAgent/change_agent:/Data/wyh/CD-SegAgent/SegAgent:/Data/wyh/CD-SegAgent/SegAgent/third_party/SimpleClick \
  /Data/wyh/CD-SegAgent/segagent-env/bin/python \
  tools/simpleclick_smoke.py --device cpu
```

Prepare real offline Verifier samples and run a bounded training smoke:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/build_verifier_samples.py \
  --dataset-root /Data/wyh/CD-SegAgent/OmniOVCD/dataset/LEVIR-CD/test_256 \
  --max-samples 2 --output outputs/verifier_levir_smoke.npz
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --samples outputs/verifier_levir_smoke.npz \
  --epochs 1 --batch-size 1 --output outputs/verifier_levir_smoke.pt
```

## Environment isolation

Do not upgrade the legacy SegAgent environment (`transformers==4.31.0`). Install the
new Agent branch in a separate environment:

```bash
python -m pip install -e '.[qwen3vl]'
```

Install training dependencies separately when training the Verifier:

```bash
python -m pip install -e '.[train]'
```

OmniOVCD/SAM3 and SimpleClick should remain in their existing isolated environments.
Their adapters accept injected callbacks/wrappers so incompatible CUDA and
Transformers stacks do not need to share one process.

## Run tests and smoke loop

The test suite only requires NumPy and Pillow:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python tools/smoke_loop.py --output outputs/smoke
```

The smoke output includes `trajectory.json` and one `.npy` change mask per step. The
LEVIR runner also accepts `--visualize`; when enabled it writes each step's binary
`t1_mask.png`/`t2_mask.png` and every connected-component instance used by matching
under `visualizations/<sample>/step_<index>/{t1,t2}_instances/`. Run metadata records
the Git commit, Python/platform, seed, and dataset split. Production runs should
additionally pass config/checkpoint/GPU/software metadata through
`ChangeAgentEnvironment(run_metadata=...)`.

## Adapter contract

`OmniOVCDAdapter` receives two injected callbacks:

1. `initialize_masks(t1_image, t2_image, query)` returns T1/T2 masks plus optional
   no-GT model evidence.
2. `segment_box_callback(image, normalized_cxcywh, query)` executes SAM3 geometric
   prompting.

`SAM3ProcessorAdapter` provides those callbacks directly from OmniOVCD's public
`SAM3ImageProcessor` API. It uses text prompts to initialize both temporal masks and
`add_geometric_prompt` for normalized box actions, while keeping processor state and
features hidden from the Agent.

`SimpleClickAdapter` calls the underlying SimpleClick predictor directly so the point
session's target-view mask is actually supplied as `prev_mask`. Because the Executor
commits only a locally composed subset of the raw prediction, every accepted point or
box starts a new session from the exact committed target-view mask. Rejected candidates
leave the session unchanged. Every tool result is checked for shape and locality before
it can update live state.

The upstream OmniOVCD checkout contains three CPU-compatibility fixes used by the
adapter smoke: device-aware SAM3 decoder coordinate caches, CPU-safe positional
encoding precomputation, and a non-pinned CPU geometry scale path. These are committed
in that checkout only; model weights remain outside Git.

## Safety boundary

Runtime `ChangeAgentEnvironment` only accepts `inference_only=True`; `reset` has no GT
parameter. GT-derived perturbations and labels live in `perturbations.py`, which is an
offline training utility and is never imported by the Environment.
