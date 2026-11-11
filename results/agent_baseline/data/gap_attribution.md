# 补漏归因

问题：Agent 补到了哪些固定路线没有的真变化，来自哪次工具调用？

三个种子的像素数和组件数合并；F1 为各种子整组像素汇总后 F1 的均值。区间为配对图级 bootstrap：128 图有放回抽样，三个种子共享抽样索引，种子 1016，5000 次，95% 百分位区间。

补漏 = A & ~F，扣除 F 后再取 8 连通，无最小面积。同一原组件与 F 重叠时仅保留剩余像素，可能拆为多个补漏组件。TP/FP 是像素口径，不能当作完整新发现建筑数。

“去掉补漏”预测是 A & F。Δ = 实际 F1 − 去掉补漏 F1，单位百分点；这是固定最终答案上的消融估计，不是重跑 Agent 或工具因果效应。

|组|会话|补漏组件|补漏 TP|补漏 FP|补漏 TP / 全部 TP|实际 F1|去掉补漏 F1|Δ [95% CI]|未知像素|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
|second_appeared_g2|384|3460|1201270|560190|21.6%|72.30|64.03|+8.27 [+3.71, +14.28]|0.00%|
|second_disappeared_g2|384|1416|282109|254530|28.0%|50.29|41.90|+8.39 [+1.65, +16.87]|0.00%|
|second_appeared_g3|384|3368|1075231|482996|19.8%|71.31|63.69|+7.62 [+3.29, +13.14]|0.00%|
|second_disappeared_g3|384|1626|311397|429706|30.4%|48.45|40.98|+7.47 [-1.64, +18.45]|0.00%|
|whu_any_g3|384|2026|480083|374731|11.9%|64.27|60.74|+3.53 [-0.67, +9.14]|0.00%|
|whu_appeared_g3|384|1168|458725|384198|11.7%|72.86|69.76|+3.10 [-2.15, +10.05]|0.00%|
|whu_disappeared_g3|384|389|64844|107449|24.1%|25.86|21.42|+4.44 [-0.62, +13.05]|0.00%|

## 工具与参数

组件数为该档涉及的补漏组件数。同一组件跨档会重复计数；同档多个调用只计一次。TP/FP 每像素只归一个来源，可相加。完整调用与每组件分摊见 JSON 的 session_details。

来源限定在最终句柄的祖先。subtract 沿 a，match_instances 沿同一时相实例；多个有效祖先包含同一像素时选最早来源调用。morph 新造像素归 morph。未知祖先证据不被后续调用覆盖。这里归的是像素产生来源，不代表后续选择、匹配、验证没有作用。

本版本 add_region 只复制 other 中已存在的区域，因此像素来源继续追到 other；没有单列来源不表示该操作未被使用。refine_region 的内部 prompt 新增像素才归 refine_region。

### second_appeared_g2

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t2, score_thr=0.3, tile=512|2342|544487|282067|45.3%|65.9%|
|segment_prompt time=t2, mode=box, box_size=>10%|537|533241|80709|44.4%|86.9%|
|segment_prompt time=t2, mode=box, box_size=1–10%|544|74517|173704|6.2%|30.0%|
|segment_text query_class=其他, query=warehouse, time=t2, score_thr=0.5, tile=512|1|27603|0|2.3%|100.0%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.1, tile=512|4|14246|11308|1.2%|55.7%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.5, tile=512|1|6489|8996|0.5%|41.9%|
|morph op=close, radius=5|3|559|329|0.0%|63.0%|
|segment_prompt time=t2, mode=points|1|110|30|0.0%|78.6%|
|refine_region time=t2, mode=box, box_size=1–10%|1|18|2483|0.0%|0.7%|
|segment_prompt time=t2, mode=box, box_size=<1%|29|0|563|0.0%|0.0%|
|refine_region time=t2, mode=box, box_size=<1%|1|0|1|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `second_g2_appeared_s3/T1.1xC1/val/train_943`：图 `train_943`，补漏 TP 98006，FP 2。
  - 组件 1，框 [145, 128, 512, 512]，TP/FP=98006/2；第 10 步第 1 调用 segment_prompt，参数 {"time": "t2", "box": [180, 130, 512, 490]}，TP/FP=98006/2。
- `second_g2_appeared_s1/T1.1xC1/val/train_943`：图 `train_943`，补漏 TP 96448，FP 7249。
  - 组件 1，框 [126, 88, 512, 512]，TP/FP=96448/7249；第 1 步第 1 调用 segment_prompt，参数 {"time": "t2", "box": [150, 80, 512, 450]}，TP/FP=96166/2；第 8 步第 1 调用 segment_prompt，参数 {"time": "t2", "box": [150, 80, 350, 250]}，TP/FP=282/7247。

### second_disappeared_g2

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t1, score_thr=0.3, tile=512|941|148475|131656|52.6%|53.0%|
|segment_prompt time=t1, mode=box, box_size=>10%|209|92299|29473|32.7%|75.8%|
|segment_prompt time=t1, mode=box, box_size=1–10%|120|25435|65113|9.0%|28.1%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.2, tile=512|73|15041|16106|5.3%|48.3%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.1, tile=512|59|513|1920|0.2%|21.1%|
|segment_prompt time=t1, mode=box, box_size=<1%|10|346|4020|0.1%|7.9%|
|segment_prompt time=t2, mode=box, box_size=1–10%|4|0|4594|0.0%|0.0%|
|segment_prompt time=t2, mode=box, box_size=<1%|1|0|1648|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `second_g2_disappeared_s2/T1.1xC1/val/train_519`：图 `train_519`，补漏 TP 70358，FP 17601。
  - 组件 2，框 [112, 4, 414, 437]，TP/FP=70195/17529；第 11 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [100, 30, 400, 440]}，TP/FP=70195/17529。
  - 组件 1，框 [343, 3, 350, 24]，TP/FP=77/18；第 11 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [100, 30, 400, 440]}，TP/FP=77/18。
  - 组件 12，框 [410, 306, 415, 312]，TP/FP=21/0；第 11 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [100, 30, 400, 440]}，TP/FP=21/0。
- `second_g2_disappeared_s1/T1.1xC1/val/train_519`：图 `train_519`，补漏 TP 24523，FP 675。
  - 组件 24，框 [359, 330, 428, 365]，TP/FP=1490/0；第 1 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.3}，TP/FP=1490/0。
  - 组件 31，框 [393, 400, 452, 435]，TP/FP=1316/0；第 1 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.3}，TP/FP=1316/0。
  - 组件 11，框 [294, 243, 337, 274]，TP/FP=1141/0；第 1 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.3}，TP/FP=1141/0。

### second_appeared_g3

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t2, score_thr=0.3, tile=512|2252|543905|272576|50.6%|66.6%|
|segment_prompt time=t2, mode=points|37|249365|9594|23.2%|96.3%|
|segment_prompt time=t2, mode=box, box_size=1–10%|652|126736|161087|11.8%|44.0%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.2, tile=512|6|39641|11746|3.7%|77.1%|
|morph op=close, radius=10|125|34084|0|3.2%|100.0%|
|segment_text query_class=其他, query=blue roof, time=t2, score_thr=0.5, tile=512|1|27351|0|2.5%|100.0%|
|segment_text query_class=其他, query=new building, time=t2, score_thr=0.3, tile=512|5|24042|0|2.2%|100.0%|
|morph op=close, radius=5|192|9106|0|0.8%|100.0%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.1, tile=512|2|6857|1294|0.6%|84.1%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.5, tile=512|57|6658|26223|0.6%|20.2%|
|segment_text query_class=其他, query=solar panel, time=t2, score_thr=0.3, tile=512|1|6576|47|0.6%|99.3%|
|morph op=fill_holes, radius=None|23|904|0|0.1%|100.0%|
|segment_prompt time=t2, mode=box, box_size=<1%|17|6|403|0.0%|1.5%|
|morph op=close, radius=3|1|0|26|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `second_g3_appeared_s3/T1.1xC1/val/train_943`：图 `train_943`，补漏 TP 100862，FP 36。
  - 组件 1，框 [122, 127, 512, 512]，TP/FP=100862/36；第 16 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[300, 250]], "point_labels": [1]}，TP/FP=100862/36。
- `second_g3_appeared_s1/T1.1xC1/val/train_943`：图 `train_943`，补漏 TP 100747，FP 28。
  - 组件 1，框 [123, 127, 512, 512]，TP/FP=100747/28；第 21 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[350, 250]], "point_labels": [1]}，TP/FP=100747/28。

### second_disappeared_g3

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t1, score_thr=0.3, tile=512|1189|164628|203391|52.9%|44.7%|
|segment_prompt time=t2, mode=points|5|77495|5755|24.9%|93.1%|
|segment_text query_class=其他, query=bare soil, time=t2, score_thr=0.5, tile=512|59|47713|3555|15.3%|93.1%|
|segment_prompt time=t1, mode=box, box_size=1–10%|144|9584|80717|3.1%|10.6%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.1, tile=512|10|9574|10838|3.1%|46.9%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.4, tile=512|6|1106|4020|0.4%|21.6%|
|segment_prompt time=t1, mode=box, box_size=<1%|40|1083|4266|0.3%|20.2%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.5, tile=512|173|214|117164|0.1%|0.2%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `second_g3_disappeared_s3/T1.1xC1/val/train_519`：图 `train_519`，补漏 TP 77495，FP 5755。
  - 组件 1，框 [129, 21, 454, 469]，TP/FP=77494/5742；第 9 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[200, 200], [300, 300], [400, 400]], "point_labels": [1, 1, 1]}，TP/FP=77494/5742。
  - 组件 2，框 [416, 250, 417, 251]，TP/FP=1/0；第 9 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[200, 200], [300, 300], [400, 400]], "point_labels": [1, 1, 1]}，TP/FP=1/0。
  - 组件 3，框 [448, 355, 450, 359]，TP/FP=0/7；第 9 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[200, 200], [300, 300], [400, 400]], "point_labels": [1, 1, 1]}，TP/FP=0/7。
- `second_g3_disappeared_s1/T1.1xC1/val/train_519`：图 `train_519`，补漏 TP 47713，FP 3555。
  - 组件 26，框 [158, 129, 449, 465]，TP/FP=46624/3531；第 6 步第 1 调用 segment_text，参数 {"time": "t2", "text": "bare soil"}，TP/FP=46624/3531。
  - 组件 6，框 [346, 64, 378, 115]，TP/FP=397/0；第 6 步第 1 调用 segment_text，参数 {"time": "t2", "text": "bare soil"}，TP/FP=397/0。
  - 组件 44，框 [138, 188, 150, 218]，TP/FP=122/0；第 6 步第 1 调用 segment_text，参数 {"time": "t2", "text": "bare soil"}，TP/FP=122/0。

### whu_any_g3

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t2, score_thr=0.3, tile=512|1177|219490|181787|45.7%|54.7%|
|segment_prompt time=t2, mode=points|3|119728|1707|24.9%|98.6%|
|segment_text query_class=其他, query=white roof, time=t2, score_thr=0.2, tile=512|1|55670|0|11.6%|100.0%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.3, tile=512|161|40437|102095|8.4%|28.4%|
|segment_prompt time=t1, mode=box, box_size=1–10%|195|21194|13844|4.4%|60.5%|
|segment_prompt time=t2, mode=box, box_size=1–10%|243|10685|42990|2.2%|19.9%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.1, tile=512|181|8027|1714|1.7%|82.4%|
|morph op=close, radius=5|17|3420|5967|0.7%|36.4%|
|segment_prompt time=t2, mode=box, box_size=<1%|16|1353|2123|0.3%|38.9%|
|segment_text query_class=building, query=building, time=t1, score_thr=0.1, tile=512|13|38|17966|0.0%|0.2%|
|segment_prompt time=t1, mode=points|18|37|38|0.0%|49.3%|
|morph op=close, radius=3|1|4|0|0.0%|100.0%|
|segment_prompt time=t1, mode=box, box_size=<1%|31|0|4500|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `whu_g3_any_s3/T1.1xC0/val/train_1071`：图 `train_1071`，补漏 TP 60250，FP 422。
  - 组件 10，框 [0, 50, 243, 512]，TP/FP=60250/375；第 8 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [0, 20, 130, 120]}，TP/FP=624/98；第 10 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[100, 200]], "point_labels": [1]}，TP/FP=59626/277。
  - 组件 1，框 [23, 15, 24, 16]，TP/FP=0/1；第 8 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [0, 20, 130, 120]}，TP/FP=0/1。
  - 组件 2，框 [4, 24, 5, 25]，TP/FP=0/1；第 8 步第 1 调用 segment_prompt，参数 {"time": "t1", "box": [0, 20, 130, 120]}，TP/FP=0/1。
- `whu_g3_any_s2/T1.1xC0/val/train_1071`：图 `train_1071`，补漏 TP 60085，FP 8865。
  - 组件 6，框 [0, 56, 243, 512]，TP/FP=60085/8574；第 7 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.1}，TP/FP=2/8281；第 17 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[100, 300]], "point_labels": [1]}，TP/FP=60083/293。
  - 组件 1，框 [0, 8, 9, 30]，TP/FP=0/126；第 7 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.1}，TP/FP=0/126。
  - 组件 2，框 [0, 31, 1, 33]，TP/FP=0/2；第 7 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.1}，TP/FP=0/2。

### whu_appeared_g3

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_prompt time=t2, mode=points|7|190025|7939|41.4%|96.0%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.3, tile=512|651|178746|272949|39.0%|39.6%|
|refine_region time=t2, mode=box, box_size=>10%|1|36889|1114|8.0%|97.1%|
|segment_prompt time=t2, mode=box, box_size=1–10%|347|33188|61898|7.2%|34.9%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.1, tile=512|9|11469|12211|2.5%|48.4%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.5, tile=512|84|6410|2423|1.4%|72.6%|
|segment_text query_class=building, query=building, time=t2, score_thr=0.4, tile=512|14|1245|8317|0.3%|13.0%|
|segment_prompt time=t2, mode=box, box_size=<1%|35|747|6331|0.2%|10.6%|
|morph op=close, radius=3|7|6|2|0.0%|75.0%|
|segment_prompt time=t1, mode=box, box_size=1–10%|5|0|4701|0.0%|0.0%|
|morph op=close, radius=5|1|0|3135|0.0%|0.0%|
|segment_prompt time=t1, mode=points|9|0|3157|0.0%|0.0%|
|refine_region time=t2, mode=box, box_size=1–10%|1|0|21|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `whu_g3_appeared_s3/T1.1xC1/val/train_1071`：图 `train_1071`，补漏 TP 60148，FP 277。
  - 组件 1，框 [0, 56, 243, 512]，TP/FP=60148/277；第 11 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[100, 200]], "point_labels": [1]}，TP/FP=60148/277。
- `whu_g3_appeared_s1/T1.1xC1/val/train_1071`：图 `train_1071`，补漏 TP 60046，FP 368。
  - 组件 1，框 [0, 57, 243, 512]，TP/FP=60046/368；第 10 步第 1 调用 segment_prompt，参数 {"time": "t2", "points": [[150, 250], [200, 300], [100, 200]], "point_labels": [1, 1, 1]}，TP/FP=60046/368。

### whu_disappeared_g3

|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|
|---|---:|---:|---:|---:|---:|
|segment_text query_class=building, query=building, time=t1, score_thr=0.3, tile=512|137|39216|60511|60.5%|39.3%|
|segment_prompt time=t1, mode=box, box_size=1–10%|231|13101|28729|20.2%|31.3%|
|segment_prompt time=t1, mode=points|2|12527|4253|19.3%|74.7%|
|segment_prompt time=t1, mode=box, box_size=<1%|12|0|6468|0.0%|0.0%|
|morph op=close, radius=4|1|0|337|0.0%|0.0%|
|morph op=close, radius=5|2|0|2|0.0%|0.0%|
|segment_text query_class=其他, query=building shed warehouse, time=t1, score_thr=0.2, tile=512|8|0|7149|0.0%|0.0%|

核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。

- `whu_g3_disappeared_s2/T1.1xC1/val/train_183`：图 `train_183`，补漏 TP 20553，FP 1425。
  - 组件 1，框 [0, 161, 149, 412]，TP/FP=20553/1425；第 1 步第 1 调用 segment_text，参数 {"time": "t1", "text": "building", "score_thr": 0.3}，TP/FP=20553/1425。
- `whu_g3_disappeared_s3/T1.1xC1/val/train_183`：图 `train_183`，补漏 TP 12507，FP 0。
  - 组件 1，框 [10, 180, 131, 384]，TP/FP=12507/0；第 7 步第 1 调用 segment_prompt，参数 {"time": "t1", "points": [[50, 300]], "point_labels": [1]}，TP/FP=12507/0。

## 结论

- SECOND 四组中，building 文本门限 0.3 的来源合计补回 TP 1401495、带入 FP 889690，精确率 61.2%。固定路线门限为 0.5；放宽门限确实提供了补漏来源，也需要筛除 FP。
- SECOND G3 点提示合计补回 TP 326860、FP 15349，精确率 95.5%。这是实际最终答案保留的像素来源；不能据此把所有点提示调用都视为有效。
- WHU 各方向与配置的描述性合计：segment_text 提供最多补漏 TP（560748，占 55.9%），同时带入 FP 667122，精确率 45.7%。不同组不是独立图样本，不据此做显著性检验。
- 主要带入 FP 的参数档（按 FP 量取前三）：whu_appeared_g3 的 segment_text query_class=building, query=building, time=t2, score_thr=0.3, tile=512，TP/FP=178746/272949，精确率 39.6%；second_disappeared_g3 的 segment_text query_class=building, query=building, time=t1, score_thr=0.3, tile=512，TP/FP=164628/203391，精确率 44.7%；second_appeared_g2 的 segment_prompt time=t2, mode=box, box_size=1–10%，TP/FP=74517/173704，精确率 30.0%。
- 补漏 F1 差的区间跨零：second_disappeared_g3、whu_any_g3、whu_appeared_g3、whu_disappeared_g3。这些组仍有真实补回的 TP，但整体 F1 收益尚未确认。

未知来源共 0 像素。归因是祖先内最早像素来源；相同 building 查询也可能通过不同匹配、选择和组合把固定路线丢掉的像素带回，不能把这类收益都理解成换查询。

## 核对与限制

固定路线均核对 settings.json、生成代码，并逐图重算 TP/FP/FN 对照 counts.json。SECOND 路径另对照 second/metrics_spec.json。所有会话的最终掩码计数对照 final.json；未跳过错误兜底或空答案。

多祖先不同来源的歧义数、三个种子的分开结果、每图每组件分摊均写入 JSON。未运行模型、API、Slurm、GPU；未做可选 LEVIR 参照。未重建已执行工具，也未用验证器回答替代真值。

运行时间 1337.9 秒；读取 OSError 重试 2 次。所有轨迹路径相对 R。
