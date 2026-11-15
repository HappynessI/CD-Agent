"""Round-1 content of results/ovcd_v0_rounds.json (texts written by Claude; numbers from RUN/eval)."""
import json
from pathlib import Path

from common import REPO, RUN

F = '../figures/ovcd_v0/round_1/'


def table(r=1, imc=None):
    lab = {'M0': 'M0', 'A': 'M0+A', 'B': 'M0+A+B', 'C': 'M0+A+B+C'}
    cols = ['k', '状态', 'F1^c', 'IoU^c', 'P', 'R', '整目标漏检率', '整目标误检率', '逐图平均 F1', '平均调用', 'MLLM 秒/图']
    rows = []
    for k in (10, 20, 50):
        d = json.loads((RUN / 'eval' / f'r{r}_k{k}_test.json').read_text())
        for st, v in d['states'].items():
            if st == 'M0' and k != 10:
                continue
            rows.append([k, lab[st]] + [f"{v[x]*100:.2f}" for x in ('f1', 'iou', 'precision', 'recall')] +
                        [f"{v['miss_rate']*100:.1f}", f"{v['false_alarm_rate']*100:.1f}", f"{v['mean_image_f1']*100:.2f}",
                         f"{v['mean_calls']:.1f}", f"{v['mean_mllm_seconds']:.2f}"])
    rows.append(['—', 'DynamicEarth M-C-I 原文', '53.6', '36.6'] + ['—'] * 7)
    rows.append(['—', 'DynamicEarth I-M-C 原文' + (f'（本次复现 {imc[0]:.2f} / {imc[1]:.2f}）' if imc else '（复现待补）'), '69.7', '53.5'] + ['—'] * 7)
    return dict(columns=cols, rows=rows)


def f1s(name, cfg='r1_k20'):
    d = json.loads((RUN / 'eval' / f'{cfg}_test.json').read_text())['per_image']
    return {st: round(d[st][name], 4) for st in ('M0', 'A', 'B', 'C')}


ROUND = dict(
    id=1, title='第 1 轮', stage='第 1 轮 · M4 完成（已评测试集）',
    config=[['数据集', 'LEVIR-CD（验证 64 对 / 测试 128 对，1024²）'], ['类别', 'building'],
            ['初始化', 'DynamicEarth M-C-I（SAM ViT-H + DINOv2 + SegEarth-OV），复现 53.76 / 36.76'],
            ['MLLM', 'Qwen3.5-4B 本地，yes/no 首词元概率 + 生成类型与理由'],
            ['分割工具', 'SAM ViT-H（框提示，CPU 解码缓存嵌入）'],
            ['阶段 A', 'M0 连通域按面积从大到小，P(yes) < τ_A 删除'],
            ['阶段 B', 'T7 = M-C-I 近阈值 SAM 提议（每图前 20）；P(yes) ≥ τ_B 时 T5 画框 → SAM 两期并集'],
            ['阶段 C', 'SAM 外接框重分割，IoU(新,旧) ≥ 门限才替换'],
            ['预算', 'k = 10 / 20 / 50；A 最多用 75%'],
            ['GPU', '约 1.7 GPU 小时（M-C-I 0.38，SAM 0.08，Qwen 0.9，其余冒烟）']],
    changed='首轮。',
    sections=dict(
        M0=dict(
            did='用 DynamicEarth 官方 M-C-I 代码与参数在 LEVIR 1024² 全图上推理，**不改任何决策**，另外导出每个 SAM 提议（每图约 600 个）的 DINOv2 变化分数、两期 building 概率和是否进 M0。3 张验证图上官方流程与导出流程逐像素一致；每图约 7.2 GPU 秒。',
            result='测试集 F1^c **53.76**、IoU^c **36.76**（原文 53.6 / 36.6，差 0.16），P 45.75、R 65.16。验证集 F1^c 57.82。I-M-C（APE-DINO）的环境本轮没跑通；第 2 轮补上：测试集 69.67 / 53.46（原文 69.7 / 53.5）。',
            issues='- 官方 SegEarth-OV 的上采样器保留 Dropout，输出带随机性；我们按图固定种子。\n- 网络慢：权重改从 ModelScope 下载并按 HF 的 sha256 核对；CLIP ViT-B/16 用 timm 发布的同一份 OpenAI 权重（已核对 fp16 无损）。\n- I-M-C 需要 16 GB 权重和 detectron2/detrex 编译，环境第一次因 detrex 的 omegaconf 版本冲突失败。',
            next='用这份 M0 和逐提议分数做 M1 错误分析；I-M-C 环境修好后补对比行。',
            figs=[dict(src=F + 'm0_repro.png', caption='M-C-I 复现与原文对比（I-M-C 只有原文数字）')]),
        M1=dict(
            did='在验证集（选择）和测试集（描述）的 M0 上统计整目标漏检/误检率、误检像素构成、漏检像素的 ①②③⑤ 格（沿用 miss_share 定义，基准为 M0）、理想删除与理想补入的上限；在验证集上比较阶段 A 的六种排序（理想核实下的删除收益曲线）和 T7 候选覆盖率。',
            result='- 整目标漏检率 验证 43.1% / 测试 50.5%；整目标误检率 38.2% / 39.3%。\n- 误检像素占全部错误 75% / 69%，但其中约 70% 是**命中目标外多出的像素**，整块误检只占约 30%。\n- 漏检像素 69% 是 ② 孤立无信号。\n- 理想上限（验证）：A 删除最优集合 65.43（+7.6），B 补回整目标漏检 69.85（+12.0）。\n- 排序：按面积从大到小最好（每图查 15 个：62.97）；小连通域更常是误检（AUC 0.88），但删掉它们挽回的像素少。\n- T7：前 20 个候选只命中 21% 的整目标漏检，前 50 个 34%。',
            issues='误检大部分是边界外溢，A 删整块只能处理三成；漏检主要是 M-C-I 根本没出提议的目标，T7 的近阈值提议只覆盖五分之一。',
            next='A 用面积降序；按误检像素占比把 75% 预算给 A。',
            figs=[dict(src=F + 'm1_composition.png', caption='误检与漏检像素的构成（验证、测试）'),
                  dict(src=F + 'm1_queue.png', caption='左：阶段 A 各排序的理想删除曲线；右：T7 候选覆盖（验证集）')]),
        M2=dict(
            did='每个 M0 连通域：T2 近景（外接框 4 倍、384²、红色轮廓）→ T3 Qwen3.5-4B 核实（系统提示放口径卡，回答是否、伪变化类型、一句理由，读 yes/no 概率）→ P(yes) < τ_A 则 T4 删除。τ_A 在验证集上按 k 分别选。',
            result='- MLLM 判别力：P(yes) 区分该删/该留的 AUC 0.85，**按面积加权只有 0.69**。\n- 验证集 τ_A 曲线几乎平坦（0–0.7 之间 57.6–58.0）；选中 0.2 / 0.1 / 0.5。\n- 验证集 A 相对 M0：+0.10 / +0.10 / +0.21；**测试集 +0.02 / 0.00 / −0.19**。\n- 逐图平均 F1 在测试集上 +1.6–1.9，主要来自空真值图上的误检被删光（单图 F1 从 0 跳到 1）。',
            issues='按面积降序先核实的是大块、精度中等的连通域，Qwen 看到里面有建筑就答“是”；判别力恰好在像素权重大的地方最弱。k=50 测试集删除的像素中真阳性占 29.7%，高于 F1/2，累计 F1 因而下降（例 test_27 uid 19，精度 58%，被判 no_change 删除）。',
            next='Codex 用缓存精确核算：A 改按 M-C-I 变化分数升序并重选 τ_A，验证集 A 可到 58.88（k=20，现为 57.92）。列为第 3 轮改动。',
            figs=[dict(src=F + 'm2_tau_a.png', caption='验证集：删除阈值 τ_A 与 F1^c'),
                  dict(src=F + 'm2_m3_pyes.png', caption='P(yes) 的分布：左为阶段 A，右为阶段 B')]),
        M3=dict(
            did='T7 取 M-C-I 未进 M0、接近阈值的 SAM 提议，按软接受分排序、掩码 NMS 后每图前 20 个。每个候选（未被当前掩码覆盖时）：T2 → T3（1 次调用）→ P(yes) ≥ τ_B 则 T5：Qwen 在近景里画框（1 次调用）→ SAM 用框在 T1、T2 各分割一次取并集加入掩码；框不可用时退回候选掩码。',
            result='- B 的判别力强：候选中 20% 是真漏检，P(yes) 区分它们的 AUC **0.94**。\n- 画框 3,839 次全部可解析（失败率 0%），但对真漏检候选，框 + SAM 并集与真值的 IoU 平均 **0.13**，低于候选掩码本身的 **0.22**；只有 28% 的情况框更好。\n- τ_B < 0.8 时补入都让验证集变差；选中 0.95。验证集 B 相对 A +0.26 / +0.23 / +0.01；**测试集 +0.07 / −3.03 / −3.15**。',
            issues='**反常排查（与 Codex 一起）**：测试集的骤降主要来自 test_96 的一次调用。候选外接框 180×286，4 倍近景就是整图；Qwen 回答 [0,0,1000,1000]，SAM 用整图框新增 972,791 像素，其中 972,774 是误检，占 B 新增误检的 81%。只撤销这一步，B 仍比 A 低 0.10（其余补入像素精度 24%）。验证集里也有 4 次整框，但都在小近景上。B 的“判断”强、“生成掩码”弱。',
            next='第 2 轮去掉 T5：核实通过后直接加入候选掩码（验证集最终 F1：k=20 +0.72，k=50 +1.97）。',
            figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：加入阈值 τ_B 与 F1^c（A 已固定）'),
                  dict(src=F + 'm3_box.png', caption='真漏检候选：T5 框 + SAM 并集 vs 候选掩码本身（近景内与真值的 IoU）')]),
        M4=dict(
            did='对 B 之后的每个连通域，用外接框提示 SAM（T1、T2 各一次，取与原域 IoU 高者），IoU(新,旧) ≥ 门限才替换；所有替换一次性合并，与访问顺序无关。门限在验证集上选，含“不修”和“总是替换”。',
            result='- 验证集：总是替换会掉 0.5；门限 0.7–0.8 时 +0.2。选中 0.8 / 0.8 / 0.7。\n- 测试集 C 相对 B：+0.17 / +0.41 / +0.49。\n- 代价：CPU 上每图约 11 秒。',
            issues='新旧 IoU 门限只限制形状变化、不判断对错：test_96 上 C 接受了与错误大区域 IoU 0.89 的替换。测试集 k=20 的 C 让像素指标上升，整目标误检率却从 40.0% 升到 41.8%（连通域被拆分或合并）。',
            next='本轮不改 C；之后考虑只修 A 核实为“是”的连通域。',
            figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的替换门限')]),
    ),
    summary_table=table(1, (69.67, 53.46)),
    summary_note='测试集，每轮只评这一次。k=10 的 M0 行与 k=20、50 相同，只列一次。MLLM 秒/图是 16 条一批的缓存时间按批大小分摊；C 在 CPU 上每图约 11 秒；M-C-I 初始化约 7.2 GPU 秒/图未计入。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=20）'),
                  dict(src=F + 'test_budget.png', caption='测试集预算曲线（每个 k 在验证集上独立选参）'),
                  dict(src=F + 'test_composition.png', caption='测试集误检与漏检像素（k=20）')],
    examples=[dict(cfg='r1_k20', split='test', name='test_15.png',
                   why='三个阶段都起作用的例子：A 删除 2 个误检（含阴影），B 补入 3 处，C 替换 7 个连通域的边界；本图 F1 0.363 → 0.451。', f1=f1s('test_15.png')),
              dict(cfg='r1_k20', split='test', name='test_96.png',
                   why='效果变差的例子：B 中一个候选的 4 倍近景就是整图，Qwen 画出整幅框，SAM 用它分出几乎整张图，新增约 97 万误检像素；C 又接受了与这块错误区域相似的替换。本图 F1 0.880 → 0。', f1=f1s('test_96.png'))],
    discussion='**Codex（batch58 代码审查）**：在看测试集前发现并修复了 C 的替换顺序问题、缺缓存被静默跳过、理想上限只删一轮、画框解析不严格、B 诊断定义不一致、缺少耗时统计。\n\n**Codex（batch59 第 1 轮讨论）**：独立核实了 test_96 的原因并修正数字口径；指出 k=50 的 A 在测试集变差是真实误删；逐图平均与累计 F1 方向相反来自空真值图的 0↔1 跳变。\n\n**第 2 轮改什么**：双方一致——只去掉 T5，B 直接用候选掩码。依据只来自验证集：框后掩码 IoU 0.13 < 候选 0.22；无 T5 的完整验证集网格最终 F1 58.54 / 59.09 / 60.23（+0.14 / +0.72 / +1.97），固定原 τ_B 也有提升。\n\n**后续候选**（Codex 排序，Claude 同意）：第 3 轮 A 改为 score_asc 排序（缓存核算验证集 A 58.88）；第 4 轮 B 队列 20 → 50。暂不改口径卡：Codex 指出“轮廓内有无建筑变化”与“整块是否值得保留”是两个问题，改提示前要先分开它们。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text()) if p.exists() else dict(rounds=[])
    data['rounds'] = [r for r in data['rounds'] if r['id'] != 1] + [ROUND]
    data['rounds'].sort(key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
