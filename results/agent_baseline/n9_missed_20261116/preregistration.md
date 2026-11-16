# N9 find_missed 分任务标定预注册（2026-11-16）

用户已授权执行N9，覆盖任务书旧“只写文档”。本线预算初分实际API¥20，总两线¥30；必要新候选真实请求qwen3.7-flash，串行、有限重试，按每次usage累计计量，不以预估费跳过必要方法。主Agent统一exp-loop调度。先完成资产QC，T3第一阶段audit完成后才开始FM调参。

## 数据、划分与曝光

仅使用既有授权开发任务。SECOND继承`night_t3_20261114/audit.json`原query dev60的30/30图像划分，两个方向一致；WHU官方dev60按seed20261116对排序图像名shuffle成30/30，三个方向一致，与报告128互斥；LEVIR从既有`dev2/names.txt`开发200按seed20261116抽取60，再按同种子排序shuffle为30/30。不读取测试集，不新建任务定义，不按GT或效果选图。每行60任务，共360任务、180物理图像。

在任何N9评分前，将每数据集confirm图像排序后用seed20261117打乱，首10图冻结为在线名单；同数据集多方向用相同10图。缺数据如实报告，不跨划分补足。资产脚本先写`frozen_protocol.json`，再校验GT和缓存。N8旧5/5划分不覆盖，本轮SECOND继承原query30/30，可能将旧pilot的confirm移入tune；全部开发池此前已有曝光，本轮不是未触及的独立holdout。

现行WHU任务及官方修订已获用户授权。保留其任务instruction/direction/scope，不自动改写历史`building_rebuild_excluded`范围。官方v3 C0=官方变化，C1=官方变化∩明确0↔1，其他官方正像素为两项C1共用ignore；C0不忽略，官方负像素始终有效。ignore只在离线评分作用，不参与候选、触发或VLM输入。复用起点是固定预测经既有f37_corr组件规则核验（保留非no），不是Agent终答；报告明确起点定义。

## 候选与profile

仅从任务direction映射any/build/demolish。允许数据集名称定位资产和分行评估，不允许在线据此切换配置。完整原S1–S5缓存重建，S5使用既有DINO020定义；统一欧氏半径3圆盘膨胀扣除当前答案，8连通组件面积至少25像素。保留每层候选TP/FP/召回损失作离线诊断。

复用N8双期7通道概率（building通道5）和DINO特征；缺失时在GPU节点真实生成。any使用abs(p2-p1)，build使用p2-p1，demolish使用p1-p2，候选区域内取均值。排序为max(语义差,0)×max(DINO差,0)×面积^alpha，同分按稳定候选index。

| profile | 来源组合 | tau_sem | 候选DINO下限 | alpha | Verifier | K |
|---|---|---|---|---|---|---|
| any | 全五源；S1/S2/S5 | .2,.4 | .4,.6 | 0 | V2 | 1,3 |
| build | 全五源 | .05,.2 | .2 | 0,.5 | V0,V3,V4 | 5,10 |
| demolish | S1/S2/S5 | 0,.05,.2 | .2 | 0 | V0,V4 | 10,15 |

网格共16/24/12配置。any来源组合若产生相同候选及输入可去重请求，但评分保留两配置的等价记录。强门限any用于保护高起点；build的宽/原N8门限与弱面积权重测试召回排序；demolish包含零门限检验全局语义阈值造成的损失。

**DINO是逐候选过滤，不是另一个任务残差面积触发。** 过滤后至少存在1个已扣除膨胀答案、面积≥25的净候选才送审。没有额外任务面积门限，不继承N8跨任务选出的1024。候选生成率与VLM调用率独立报告；高DINO并不节约已发生的特征或候选生成成本。

## 核验、选择和确认

V0保留既有四图与问法；V2外部亮度0.3+当前答案蓝色alpha0.4+旧问法/图例；V3原图+保守区域二分类；V4=V2视图+V3问法/图例。复用已验证renderer和文案，固定模型、请求参数、同候选seed；记录四幅实际编码图、完整请求与响应。新候选/新图真实请求。仅在实际四图编码、完整文本、模型、seed与全部参数精确一致时允许复用旧答，并记录来源；仅mask相同不够。首实现以全新核验为默认，避免不可靠缓存等价判断。

先完成各profile所有预声明tune配置；选择目标为该profile两条数据集tune行各自pooled F1净增量的等权均值，**绝不混合跨集像素计算pooled**。并列优先更少实际候选核验、再更小K、再预声明网格顺序。保存两个数据集行全部P/R/F1、增量和成本，三个profile各自冻结，恰好同值也不是共享选择。允许冻结非正最优以完整确认，不能据confirm再次选参。

确认独立逐行列LEVIR any、WHU any/build/demolish、SECOND build/demolish的样本量、起点/补漏P/R/F1、净delta、新增TP/FP、胜负平、空GT FP、触发率、候选数和费用；各行内pooled与图像等权macro均报告，图像级bootstrap和leave-one-image-out判断集中性。不存在跨集pooled总分。

每条confirm行Δpooled F1>0即独立晋级对应预冻结10图×2臂×3seeds真实在线；其他数据集同profile退化不能拖停或抵消该行。在线两臂相同任务/初始设置/seed/模型，仅补漏工具是否开放不同；不把固定工具流程写成Agent收益。逐行记录晋级理由，实际完成后才计为完成。离线零或负收益行保留失败诊断，不继续confirm拟合。

## 产物与首阶段复现

输出主仓库`results/agent_baseline/n9_missed_20261116/`。新代码位于独立n9-missed工作区，旧N7/N8产物不覆盖。首阶段`run_n9_missed_assets.sbatch`：gre/gpu42、1CPU、4GB、40分钟，运行4项协议测试及N8物理变换回归，再资产QC。无API。缺特征才运行`run_n9_missed_fill.sbatch`（单GPU、2CPU、24GB、4小时，纯本地模型可由主Agent选择节点）；后续候选重建`run_n9_missed_prepare.sbatch`（1CPU、16GB、4小时）。解释器固定`/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python`，PYTHONPATH包含工作区code及既有code_w/ovcd_v0。不在管理节点执行测试、数组检查或实验。

阶段接口：`n9_missed_pipeline prepare --split tune|confirm`生成各配置所需请求并集；`verify`串行真实请求并逐attempt归档；`n9_missed_score --split tune`独立冻结三个profile，`--split confirm`只评冻结方案并输出逐行晋级与图像bootstrap/LOO。统一入口`run_n9_missed_verify.sbatch tune|confirm`先跑7项N9协议/配对/跨行拒绝测试，再prepare、verify、score；申请gre/gpu42 1CPU12GB8小时，BAILIAN_CSV由主Agent现有环境注入。只有T3第一阶段audit完成且资产/重建就绪后才由主Agent提交tune，confirm严格在tune冻结之后。当前复用旧隔离工作区`cd-agent-worktrees/n7-missed`的新n9文件，避免NFS新worktree检出阻塞，全部旧代码不修改。

## 在线入口（只准备，按确认逐行晋级）

`n9_missed_online prepare`只读取逐行`offline_decision.json`中的正向行，写该行冻结10图、相同M_init PNG与任务清单；不新选图。`run_n9_missed_online.sbatch LINE SEED`运行一行一seed的20个串行会话，三seed共60；主Agent最多同时运行两作业。gre/gpu42、单GPU、4CPU、48GB、8小时，同allocation清代理后使用CSV workspace GET `/models`，必须HTTP200再启动同节点worker和真实会话；只记录host/status/proxy布尔，不输出凭据或模型列表。

独立`n9_missed_agent_entry`只在本进程挂接N9 Toolbox，不修改默认Agent入口或默认工具文件。baseline不暴露find_missed，两臂普通ask、其他工具、系统Prompt及起点完全相同。新增工具由Agent自主调用，内部按当时current mask重做欧氏dilate3、25像素组件、冻结profile过滤/排序及K核验，返回所有送审proposal及逐候选yes/no理由；不修改current，不自动并入答案。Agent决定选择、合并、提交或不使用；不强制调用一次。

影像路径仅用于绑定缓存，profile仅由direction映射。源与特征可缓存，在线净候选及实际视图每次重新生成，绝不并入离线旧预测或复用离线旧答。普通ask现有的同会话精确outline缓存可保存工具核验结果；改变/合并后的轮廓不沿用该回答。内部每次API共享当前会话RunGuard，token/费用及attempt进入Recorder相同账本；请求、失败、mask快照独立保留以核对。实际在线预算为FM当前分配额减去已发生离线费用，各任务作业共用在线批次账本；主Agent可在总30元内统一重分。

`run_n9_missed_online_tests.sbatch`为无API契约测试（mock网络），覆盖当前答案变化后重新扣除、baseline不暴露工具、profile方向约束、proposal不修改current、真实Planner记账路径和Recorder费用归档。该测试不等于已完成真实在线实验。

在线来源覆盖限制：复用的完整S1–S5来源缓存在原始生成时已针对冻结M_init处理。在线每次确实根据当前答案重新膨胀扣除、拆净组件、过滤、排名和真实核验，但如果Agent删除原M_init的像素，旧来源池不保证重新提出最初被排除的区域。这不是在线重新推理所有来源模型，不据此宣称完整重找被删目标；不复用或自动并入旧答案。

断点契约：N9 fill按每张影像/每类特征检查最终NPZ，已有文件不重算，新增概率/特征先写`.partial.npz`再原子替换，保留累计完成清单。prepare沿用N8逐任务重建：NPZ完整写出后才写任务JSON完成标志，N9将JSON标志改为原子写；重启跳过已有完整任务记录并最终汇总manifest。无需因缺缓存跳过方法；完成资产QC后先补齐，再重建。旧N8源码不修改。新增在线契约测试通过真实loop_json schema入口与Toolbox.execute验证分发和metadata，并模拟首候选付费成功、第二候选失败，核对partial token/费用和原始attempt仍进入Recorder路径。测试尚待Slurm执行，不将代码审查当实测通过。
