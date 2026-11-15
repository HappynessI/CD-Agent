# N8 T3 seed1016：四任务、两臂完整轨迹审阅

只读已结束的 baseline_s1016/n8_s1016 八条轨迹及冻结快照 [seed1016_summary.json](online/seed1016_summary.json)。未读取活跃seed2027目录，未运行图像计算、模型、作业或修改参数。以下是日志与模型理由的机制审阅，不是新增人工GT或对模型叙述的视觉事实背书。

seed1016全部20会话均由模型成功submit。整体F1从3.79833到33.79057，但空目标FP由6617到114806，空正确从1/3到0/3；目标类F1反而从6.57065到1.52424。N8的53517个TP中，52835来自368单例，其余九例合计682TP；baseline除368外还有2949TP。因此不能用总F1大涨概括为普遍检索或Agent收益。

表内TP/FP/FN均使用既定排除示例E口径。原steps/final recorder的eval包含E，不能直接引用为T3外部检索指标。例如2848 baseline提交E后原recorder显示8377TP/F1=1，但排除E后是正确空预测；2828 N8提交E后原recorder有613TP，正式口径却为0TP、43900FN。

| 任务 | Baseline TP/FP/FN | N8 TP/FP/FN | 主要机制 |
|---|---:|---:|---|
| 368 | 714 / 4735 / 69863 | 52835 / 4923 / 17742 | N8轨迹放弃小候选集，另调植被分割与实例消失匹配；大召回不能归给Top5直接覆盖 |
| 2848（空） | 0 / 0 / 0 | 0 / 80437 / 0 | N8返回空后Agent另做语义差，泛变化核验未守住示例转移 |
| 2828 | 1107 / 3055 / 42793 | 0 / 0 / 43900 | N8的候选被核验接受，但Agent误解示例为屋顶翻修，主动只交示例 |
| 1884（空） | 0 / 4030 / 0 | 0 / 29950 / 0 | N8稠密池带来更大误检支持，Agent保留全部并把泛变化yes当同转移yes |

## 368：大召回是工具策略转向后的结果

证据：[baseline步骤](online/baseline_s1016/T3.1/val/train_368_1to2/steps.jsonl)、[N8步骤](online/n8_s1016/T3.1/val/train_368_1to2/steps.jsonl)、[N8候选记录](online/n8_s1016/T3.1/val/train_368_1to2/n8_retrieval/call_000/inference.json)。

Baseline第3步取得8个候选，union7771像素；第4步inspect误用set，得到bad_args，随后第5步转mask。第6—11步看组件和示例，第12步删除组件1/3/4共2322像素，只保留5449像素的组件2。Agent认为被删大组件是反方向生长，保留者是植被清除。第14步首次submit触发核验，唯一组件答yes；第15步原样确认提交。最终714TP/4735FP，说明“核验yes”并不保证示例类的像素精度。

N8第2步取得5个候选、union4694像素；内部E概率预测building→non-vegetated ground，走96候选的实例支路。该概率是provider内部观测，不是提供给planner的真实类别。第3步同样inspect(set)错误，第4步转mask后仅2连通组件。第6—8步Agent认为一个组件转移相反，并质疑高相似度的意义。第9—10步遂另调segment_text('vegetation', score_thr=.3)，T1返回12实例/60511像素，T2返回8实例/12609像素；第11步match_instances(overlap_thr=.1)得9个消失实例，第12步转为mask#8，57758像素。第13—21步逐组件查看，其中组件7面积52735。第22步union示例得到mask#9（59179像素），不再使用Top5的mask#2构成最终答案。

第23步ask()因没有current失败；第24步把已是Mask的mask#9传to_mask再次bad_handle；第25步改为ask(regions=[mask#9])恢复，10组件全yes。第26—27步提交mask#9。排除E后52835TP/4923FP来自这套重新生成的实例消失候选。核验理由中也含建筑/道路等泛变化，因此不能称核验严格验证了相同转移。准确结论是：此轨迹的自主扩大搜索找到了一个高覆盖大区域；改变provider可能改变了后续决策，但没有受控反事实证明改进全由provider造成。

## 2848：正确的空候选没有约束住后续自主搜索

证据：[baseline步骤](online/baseline_s1016/T3.1/val/train_2848_1to4/steps.jsonl)、[N8步骤](online/n8_s1016/T3.1/val/train_2848_1to4/steps.jsonl)、[N8候选记录](online/n8_s1016/T3.1/val/train_2848_1to4/n8_retrieval/call_000/inference.json)。

Baseline第1步返回8候选/24510像素。第2步inspect(set)及第7步无overlay的带ids view均bad_args，之后恢复。第13—14步select后得到6544像素子集；第18步也调用T2 green vegetation语义分割（160252像素），但没有把它直接提交。第20—24步继续检查；第25—26步认为其余候选是不同或反方向变化，只提交8377像素示例E。正式口径为正确空集。

N8第3步retrieve返回n=0/area=0：E预测端点为non-vegetated ground→同类（.4582/.4340），稠密支路为空。第4步Agent没有把这当作无需外推的证据，反而猜测示例是否未正确定义。第5—6步用相同box分别segment_prompt；第7—8步改用全图segment_semantic('green vegetation')，T2为160252像素、T1为94745像素。第9步T2减T1得到mask#6：81005像素、59组件，正式GT外全部FP。

第12步ask检查59组件，56yes、3no。第13步planner明确认识到有建筑、反向绿→褐等不匹配变化，却反复引用共同system中的“a change of land-cover category ... with no category restriction”，最终把same transition解释为泛指T1→T2时间变化。第14步只删除no的组件27/6/12，共568像素；第15—16步提交余下80437像素/56组件。全程没有工具调用错误，因此不能归因接口失败。它是**空候选之后的策略扩张、任务解释漂移与泛变化核验**共同造成的空FP，不是provider直接返回了80437像素错误候选。

## 2828：有候选且核验通过，仍被Agent主动全拒

证据：[baseline步骤](online/baseline_s1016/T3.1/val/train_2828_1to5/steps.jsonl)、[N8步骤](online/n8_s1016/T3.1/val/train_2828_1to5/steps.jsonl)。

Baseline第2步8候选union4162像素；第4步试图给props添加组件标签报bad_args，第5步转mask并设current。第6—10步查看9组件，第11步ask全yes，第12—13步全部提交，正式1107TP/3055FP。部分yes理由把反光、屋顶亮斑等也解释成类别变化，不能把这些全当正确检测。

N8第2步5候选union3201像素，第5步转成4组件的mask#2。第3步inspect(set)、第4步props标签view、第13步缺overlay的标签view共3次bad_args，第15步to_mask(mask#0)另有bad_handle；均未造成最终运行失败。第11步对4组件ask全yes，但第12—18步Agent反复重新解释示例，认为其是“已有屋顶由深色变浅色”，而候选是新建、植被变铺装等不同转移。第19步主动提交mask#0；首次提交报告明确提示4个yes候选、3201像素没有进入当前答案，Agent第20步仍坚持只交示例。

正式口径因此全空，损失全部43900剩余GT。此例说明Agent可不服从核验，不能把漏检简单归因于Verifier误拒；核验没有拒绝任何候选。直接证据是示例语义理解与最终选择造成拒绝；候选本身的覆盖问题另行存在，不能由这条轨迹算出新provider的独立净效应。

## 1884：候选误检放大，Agent未执行同转移筛选

证据：[baseline步骤](online/baseline_s1016/T3.1/val/train_1884_4to3/steps.jsonl)、[N8步骤](online/n8_s1016/T3.1/val/train_1884_4to3/steps.jsonl)、[N8候选记录](online/n8_s1016/T3.1/val/train_1884_4to3/n8_retrieval/call_000/inference.json)。

Baseline第2步8候选union7023像素；第3步inspect(set)错误后第4步转mask得到6组件。第6—11步查看，第12步ask六个全yes；Agent第13—14步仍按水体相关转移删除组件1/2/6，去掉2993像素，而不是盲从yes。第15步加入914像素示例，第17步补核验新轮廓，第18—19步提交。正式空FP4030，仍非正确空集，但自主筛选减少了误检。

N8内部E预测裸地→树（.3179/.3727），走稠密支路，重构19候选；第1步Top5 union29950像素，比baseline支持范围明显扩大。第2步inspect(set)错误，第3步转mask后得到5组件；第5—10步逐区域查看。第11步把全部候选加E，得到30864像素mask#3；第12步ask五个全yes，理由分别涉及裸地/稀疏植被→树、工业结构→植被、暗湿表面→较亮地面。

Agent此前识别示例为水体被填充/植被覆盖，也注意到候选多是裸地→植被；但第13—15步仍把泛变化yes概括为与示例一致，没有像baseline删掉不同转移组件。最终直接保留全部29950候选像素，正式均FP。这里既有provider预测E端点与稠密支持造成的误检扩张，也有Agent语义放宽、核验未验证同转移的问题。不能只说“核验过严”，也不能只说“Agent导致”而忽略更大的初始错误支持。

## 跨案例边界与后续解释

四例八会话各调用retrieve一次并成功返回，故不是“工具未使用”的子样本；但使用后选择差异巨大：368重新生成候选、2848空后扩张、2828拒绝全部候选、1884保留全部候选。普通inspect/view/to_mask参数错误在多例出现，均有具体恢复路径；大误检2848反而无工具错误，不应把科学失败统一归到schema。

共同system的any-change口径和user的same-transition要求存在解释张力，已在两臂input.json中核实；尤其2848的planner直接引用前者为放宽选择辩护。在线ask回答的是泛land-cover change，不是离线N8证据卡“候选与E同转移”核验。这一共用Harness限制会影响两臂，不能说只有N8注入了错误类别，也不能把二者核验效果混为一谈。此处仅记录证据，不在运行中改提示或规则。

2848与1884合计空FP从4030到110387，覆盖本seed绝大部分空FP恶化；368提供绝大部分TP增益。当前结果是高度集中的召回收益与明显空集失守并存，尚不支持“整体更可靠”。后续seed仍按冻结方案完成，结论必须保留主gate/K3离线失败、Top5为预声明对照的探索在线，以及10图已有confirm/tune曝光，不能因本seed总F1上升宣布主系统成功。
