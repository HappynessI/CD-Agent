# G3 / G4 多任务回放

只读取已有轨迹。所有图片内嵌，长边不超过 512 像素。打开 index.html。

选择规则：三种子合并最终逐会话 F1 中位数；排除非 model 候选（全兜底除外）；最近者并列依成功工具类别数降序、图名、运行名。中位数按三种子全部有最终 F1 的会话计算，兜底仍计入分布。类别按成功且未跳过的调用统计，分类来自 make_demo.CATEGORIES。
浮点距离差不超过 1e-12 视为并列。缺最终 F1 的会话不补零；缺一个种子直接报错。selection.json 保存计数、种子和选择依据。

| 页面 | 任务 / 数据 | 运行 / 图名 | 最终 F1 | 中位数 | 字节 |
|---|---|---|---|---|---|
| [levir_c0](levir_c0.html) | T1.1xC0 / LEVIR | json_qwen37flash_g3_s2 / val_21 | 82.38% | 82.32% | 2800626 |
| [levir_t51](levir_t51.html) | T5.1 / LEVIR | json_qwen37flash_g3_s1 / val_2 | 83.82% | 83.92% | 1764475 |
| [second_c1](second_c1.html) | T1.1xC1 / SECOND | second_g3_appeared_s1 / train_1537 | 65.49% | 65.60% | 2466150 |
| [second_c2](second_c2.html) | T1.1xC2 / SECOND | second_g3box_any_s2 / train_276 | 49.81% | 49.93% | 1091303 |
| [second_c4](second_c4.html) | T1.1xC4 / SECOND | second_g3box_appeared_s1 / train_1799 | 67.46% | 67.43% | 1652343 |
| [levir_t52](levir_t52.html) | T5.2 / LEVIR | json_qwen37flash_g3_t52_k3_s2 / val_51 | 86.21% | 86.21% | 1621516 |
| [levir_t53](levir_t53.html) | T5.3 / LEVIR | json_qwen37flash_g3_t53v2_fb_s2 / val_25 | 82.53% | 82.48% | 1440830 |
| [second_t13](second_t13.html) | T1.3 / SECOND | json_qwen37flash_g4_t13_any_s3 / train_2286 | 38.28% | 38.25% | 1696388 |
| [second_t31](second_t31.html) | T3.1 / SECOND | json_qwen37flash_g3_t31_any_s3 / train_2468_4to2 | 10.55% | 10.49% | 1677381 |

这些是典型会话，不代表最好或最差。旧 G1 在 ../demo_g1/index.html。

复现：指定外部运行根目录，不把服务器路径写入代码。当前目录执行，先设 PYTHONPATH=code、TMPDIR=.runtime/tmp 和单线程变量。

```bash
$PY -m agent_baseline.make_g3_pages --run-root "$RUN_ROOT" --out-dir results/demo_g3
$PY -m agent_baseline.make_tools_page --out-dir results/tools
```

输入第 4 格按归档展示给定掩码、反馈、区域或示例。缺记录写未记录。点/方位叠图只使用公开几何，不展示隐藏误差区域。
桌面两侧 sticky，手机使用单栏并取消固定。父级 layout/aside/body 不设置 overflow:hidden。浅深主题沿用 G1 变量。
所有回放保留失败调用、自动报告、返回掩码/候选；G4 提醒与求交只依轨迹字段标记。T1.3 本会话如无 region，不会伪造求交步骤。

baselines.html 解释固定路线、规则和 Agent。入口顶部提供三段导航。每个回放的对照面板在流程图和时间线之间。
示例图沿用 LEVIR 中位数候选，并要求规则至少删过一个组件。选择记录在 baseline_example.json。
LEVIR 规则掩码按已有 f37_corr 答案删除 no 的组件恢复，仅为离线展示，不调用模型。未缓存的固定匹配中间步骤明确标未缓存。
C2/C4 的答案编号来自裁剪前组件。T5.2 是固定响应 C；T5.3 展示两个固定响应，验证器挑选不是删除规则。T3.1 对照按报告口径剔除示例，页头保留记录器原始 F1。
