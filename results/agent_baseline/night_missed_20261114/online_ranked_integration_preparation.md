# Frozen ranked online integration — preparation only

No online promotion or run is authorized by this preparation. The frozen six tasks have zero intersection with the night 30-task pilot. A separate six-task manifest still must be prepared with the frozen gates and evidence ranking, without GT-driven selection. No such data preparation or online session was launched here.

The opt-in runner `--ranked-manifest` now reaches the actual loop entrypoint `--missed-ranked-manifest`. Only that branch instantiates FrozenRankedCandidates. The normal provider, global ask, Gate6, answer handling and match_exemplar are unchanged. A ranked record exposes only original evidence Top10 masks and source provenance, subtracts the current answer, and never fills empty slots from rank 11 onward. The Agent receives proposal handles; accepting and submitting remains its decision. Offline GT and old_answer fields are not copied into proposal records.

For the ranked experiment both arms explicitly use MissedVerifier four. This controls verifier prompt/rendering, request parameters and retry behavior; the legacy no-ranked baseline branch remains unchanged. The sole experimental tool difference is availability of find_missed. Both arms share task initialization, three seed offsets, planner model qwen3.7-flash, verifier model qwen3.7-flash and the common session tool budget. The baseline has no find_missed Top10 pool, so this is a tool-availability comparison, not an equal-candidate-count ranking ablation.

Fresh-response evidence: runner configs include group/seed/arm/stem/attempt; loop output adds task/split/stem. MissedVerifier is instantiated at that session directory's missed_verifier subdirectory. Each call creates a new Planner and invokes complete; it never reads an existing response or offline old_answer. Repeated calls in a session receive incremented request numbers. Successful completed sessions can be skipped on resume, but their answers are not injected into another seed or arm. Ordinary within-session proposal verification reuse does not constitute cross-seed reuse. Shared provider-side token-prefix caching may affect billed tokens, not replay stored experiment answers.

Validation pending Slurm: `PYTHONPATH=code python -B -m unittest agent_baseline.tests.test_frozen_ranked_candidates agent_baseline.tests.test_find_missed`. Tests cover actual Toolbox dispatch, no answer mutation, source-only output, fixed Top10 order/no refill, runner arm flags and the real loop --help parser. Management-node checks were limited to source review and git diff --check.

Remaining before an authorized run: integrate/review patch, execute Slurm tests, decide promotion from completed ranking confirmation, independently prepare all six frozen task records and initial masks, validate matching frozen gates/settings, then prepare serial 36-session runner jobs. No pilot manifest substitution is allowed.


条件性缓存盘点：原6任务仍为SECOND train1884/train368各新建/拆除，LEVIR train293/train78。`pilot/online_selection`三组清单与初始PNG/RGB存在；历史source根`exp6_staging/batch99_find_missed_pre/artifacts`内base6/6、五来源候选30/30、四张独立图双期DINO8/8均存在。此为路径存在性检查，未解包生成新版候选。

若确认支持在线，最小缺口是独立六任务推理manifest入口：按冻结gate与V0设置，复用原S5 union/组件、gate_features、全来源dedup与evidence排序，输出provider兼容settings/manifest/masks；无需旧pilot答案模板或GT。现有night_missed_rank.prepare依赖旧pilot模板，不能直接套用，也不能把pilot获益图换入六任务。是否实施取决于已声明主对照的实际确认结果；当前没有新Agent会话。
