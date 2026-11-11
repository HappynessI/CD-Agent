#!/usr/bin/env python3
"""G3/G4 replay site in the format of the G1 replay pages (results/agent_baseline/demo_g1).

Pages: index.html (entry), baselines.html (Agent, fixed route, fixed rule), one replay page per task.
Reads saved traces, saved fixed-route and rule masks, and saved verifier answers. Runs no model.
CSS comes from the G1 pages (site_assets/hub.css) plus site_assets/site.css.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.dont_write_bytecode = True
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_baseline.make_demo import (JS, chart, esc, jpeg, mask_path, metric_f1, open_image, pct, proposal_thumbnail,
                                      read_json, read_mask, read_steps, read_text, report_text, resolve_asset, retry_read,
                                      text_block, after_stop_count)
from agent_baseline.make_baselines_page import load_baselines, measures, scored_image, image_answers, select_example
from agent_baseline.make_g3_pages import audit_html
from agent_baseline.site_svg import agent_figure, route_figure, rule_figure

ASSETS = Path(__file__).resolve().parent / 'site_assets'
CSS = (ASSETS / 'hub.css').read_text(encoding='utf-8') + (ASSETS / 'site.css').read_text(encoding='utf-8')
SWITCH_TOOLS = {'segment_everything', 'segment_semantic', 'extract_features', 'compare_features', 'classify', 'match_exemplar'}
CATEGORIES = {
    'view': '查看', 'inspect': '查看', 'segment_text': '分割', 'segment_prompt': '分割', 'segment_everything': '分割',
    'segment_semantic': '分割', 'match_instances': '比较', 'classify': '比较', 'match_exemplar': '比较',
    'extract_features': '特征', 'compare_features': '特征', 'ask': '验证', 'select': '运算', 'to_mask': '运算',
    'combine': '运算', 'morph': '运算', 'python': '运算', 'remove_region': '修改', 'add_region': '修改',
    'refine_region': '修改', 'submit': '结束',
}

# page key -> task code, task text, dataset, the fourth input chip, the input arrow label, route family.
TASKS = {
    'levir_c0': ('T1.1 × C0', '建筑变化 · 从零生成', 'LEVIR-CD', None, '两期影像', 'imc'),
    'levir_t51': ('T5.1', '建筑变化 · 修正给定掩码', 'LEVIR-CD', '给定掩码 mask#0', '影像和掩码', 't51'),
    'second_c1': ('T1.1 × C1', '新建建筑 · 带方向', 'SECOND', None, '两期影像', 'direction'),
    'second_c2': ('T1.1 × C2', '建筑变化 · 限定区域', 'SECOND', '区域 region#0', '影像和区域', 'region'),
    'second_c4': ('T1.1 × C4', '新建建筑 · 限定区域', 'SECOND', '区域 region#0', '影像和区域', 'region'),
    'levir_t52': ('T5.2', '建筑变化 · 点反馈修正', 'LEVIR-CD', '给定掩码 mask#0', '影像和掩码', 't52'),
    'levir_t53': ('T5.3', '建筑变化 · 语言反馈修正', 'LEVIR-CD', '给定掩码 mask#0', '影像和掩码', 't53'),
    'second_t13': ('T1.3', '全部地类变化 · 不限类别', 'SECOND', None, '两期影像', 't13'),
    'second_t31': ('T3.1', '同类转换 · 按图内示例', 'SECOND', '示例掩码 mask#0', '影像和示例', 't31'),
}
INSTRUCTION_CHIP = {'levir_t52': '指令（含点反馈）', 'levir_t53': '指令（含语言反馈）'}

HARNESS_NOTES = {
    'G2': 'G2：补上答案到不了规划器的漏洞；预算用完时补报告。',
    'G3': 'G3：框提示的框超过整图 10% 时报错；报告区域外像素和方向错误；反馈任务只验证相关组件。',
    'G4': 'G4：最后 3 步提醒当前答案；提交时与 region#0 求交；虚线框工具附用途说明。',
}
OFF_NOTE = '不启用：plan、python、skill、L2 仔细验证。'

IMC_STEPS = [
    ('分割两期建筑', ['segment_text(T1)', 'segment_text(T2)'], ['文字 "building"，分块 512。', '实例分数 ≥ 0.5，像素 ≥ 0.85。']),
    ('匹配两期实例', ['match_instances'], ['重叠 ≥ 0.25 视为同一实例。', '输出：出现、消失。']),
]
ASK_RULE = dict(decide=['答 no：删除这个组件。', '答 yes：保留这个组件。', '答案无法解析：保留。', '规则只删除，不补漏。'])

# Route families: the button label, the figure, and the reported scores (README_ai sections).
ROUTES = {
    'imc': dict(label='T1.1 × C0（LEVIR、WHU）', title='固定路线 I-M-C(SAM3)：LEVIR、WHU 的 T1.1 × C0',
                inputs=['T1 较早影像', 'T2 较晚影像'], steps=IMC_STEPS + [('合并', ['to_mask'], ['出现 ∪ 消失。'])],
                rule=ASK_RULE,
                scores=['LEVIR 验证 64 张：固定路线 83.15，固定规则 86.35，Agent G3 85.76（第 10.5、10.10 节）。',
                        'WHU 128 张：固定路线 56.59，固定规则 62.75，Agent G3 64.27（第 10.11 节）。']),
    't51': dict(label='T5.1', title='T5.1：给定掩码就是固定路线掩码',
                inputs=['T1 较早影像', 'T2 较晚影像', '给定掩码 mask#0'],
                steps=[('原样提交', ['submit'], ['给定掩码来自 I-M-C(SAM3)。', '固定路线不修改给定掩码。'])],
                rule=ASK_RULE, scores=['LEVIR 验证 64 张：给定掩码 83.15，固定规则 86.35，Agent G3 86.23（第 10.5、10.10 节）。']),
    'direction': dict(label='T1.1 × C1 方向', title='固定路线：SECOND、WHU 的 T1.1 × C1（只取一个方向）',
                      inputs=['T1 较早影像', 'T2 较晚影像'], steps=IMC_STEPS + [('取一个方向', ['to_mask'], ['新建：只取出现。', '拆除：只取消失。'])],
                      rule=dict(ASK_RULE, question='问法带方向：原址重建不算新建。'),
                      scores=['SECOND 新建 128 张：固定路线 57.04，固定规则 63.74，Agent G3 71.31（第 10.6、10.10 节）。',
                              'WHU 新建 128 张：固定路线 62.25，固定规则 70.26，Agent G3 72.86（第 10.11 节）。']),
    'region': dict(label='T1.1 × C2 / C4 区域', title='固定路线：SECOND 的 T1.1 × C2、C4（限定区域）',
                   inputs=['T1 较早影像', 'T2 较晚影像', '区域 region#0'],
                   steps=IMC_STEPS + [('合并或取方向', ['to_mask'], ['C2：出现 ∪ 消失。', 'C4：只取出现。']),
                                      ('裁到区域', [], ['与 region#0 逐像素求交。'])],
                   rule=dict(ASK_RULE, decide=['答 no：删除这个组件。', '答 yes：保留这个组件。', '先在整图上删除，', '再与 region#0 求交。']),
                   scores=['C2：固定路线 54.67，固定规则 60.69，Agent G3 62.77（第 10.7、10.10 节）。',
                           'C4：固定路线 62.63，固定规则 67.99，Agent G3 71.94（第 10.7、10.10 节）。']),
    't52': dict(label='T5.2 点反馈', title='固定响应 C：T5.2 点反馈（没有固定规则）',
                inputs=['T1、T2 影像', '给定掩码 mask#0', '点反馈'],
                steps=[('误检点', ['remove_region'], ['删除点所在的给定组件。']),
                       ('漏检点', ['segment_prompt(T1)', 'segment_prompt(T2)'], ['SAM ViT-H 点提示。', '每期取质量最高的候选。', '取面积较小的一个。']),
                       ('合并', ['add_region'], ['把候选并入给定掩码。'])],
                scores=['LEVIR 验证 64 张，k=3：给定掩码 83.15，固定响应 C 84.73，Agent G3 84.59（第 10.8、10.10 节）。',
                        '这个任务没有固定规则。对照是固定响应。']),
    't53': dict(label='T5.3 语言反馈', title='固定响应 1：T5.3 语言反馈（没有固定规则）',
                inputs=['T1、T2 影像', '给定掩码 mask#0', '语言反馈'],
                steps=[('定位', [], ['按 3×3 方位格定位。', '漏检候选：两期 SAM3 实例，', '给定掩码覆盖 < 50%。', '误检候选：给定组件。']),
                       ('选择', [], ['固定响应 1：格内面积最大。', '固定响应 2：验证器 p_yes', '（漏检取最高，误检取最低）。']),
                       ('执行', ['add_region', 'remove_region'], ['每条反馈执行一次。'])],
                scores=['LEVIR 验证 64 张：给定掩码 83.15，固定响应 1 82.73，固定响应 2 83.22，Agent G3 82.89（第 10.10 节）。',
                        '这个任务没有固定规则。固定响应 2 用验证器挑选，不是删除规则。']),
    't13': dict(label='T1.3 不限类别', title='固定路线：T1.3 不限类别（提议 + 特征比较）',
                inputs=['T1 较早影像', 'T2 较晚影像'],
                steps=[('提议', ['segment_everything(T1)', 'segment_everything(T2)'], ['SAM ViT-H 自动掩码。']),
                       ('比较两期特征', ['compare_features'], ['DINOv2 区域特征。', '1 − 余弦 ≥ 0.22 为候选。']),
                       ('合并', ['combine'], ['候选取并集。'])],
                rule=dict(ASK_RULE, question='问法：describe（在开发集 60 张上选）。'),
                scores=['SECOND 128 张：固定路线 60.88，固定规则 61.11，Agent G3 46.53，G4 47.71（第 10.10、10.11 节）。']),
    't12': dict(label='T1.2 多类别', title='固定路线：T1.2 建筑 + 低矮植被',
                inputs=['T1 较早影像', 'T2 较晚影像'],
                steps=[('建筑', ['segment_text', 'match_instances'], ['I-M-C(SAM3) 建筑变化。']),
                       ('低矮植被', ['segment_semantic(T1)', 'segment_semantic(T2)'], ['SegEarth-OV 两期语义。', '两期异或。']),
                       ('合并', ['combine', 'morph'], ['两类取并集。', '删除 < 64 像素的组件。'])],
                rule=ASK_RULE, scores=['SECOND 128 张：固定路线 46.26，固定规则 47.50，Agent G3 41.22，G4 38.95（第 10.10、10.11 节）。']),
    'c1t': dict(label='C1 地类转换', title='固定路线：T1.1 × C1 地类转换（低矮植被或非植被地表 → 建筑）',
                inputs=['T1 较早影像', 'T2 较晚影像'],
                steps=IMC_STEPS[:1] + [('新建组件', ['match_instances'], ['只取出现。']),
                                       ('前期地类', ['segment_semantic(T1)'], ['组件内规定地类', '覆盖 ≥ 25% 时保留。'])],
                rule=dict(ASK_RULE, question='问法：只接受规定的前期地类。'),
                scores=['SECOND 128 张：固定路线 62.82，固定规则 62.70，Agent G3 64.84（第 10.10 节）。']),
    't31': dict(label='T3.1 图内示例', title='固定路线：T3.1 按图内示例找同类转换（没有固定规则）',
                inputs=['T1 较早影像', 'T2 较晚影像', '示例掩码 mask#0'],
                steps=[('提议', ['segment_everything(T1)', 'segment_everything(T2)'], ['SAM ViT-H 自动掩码。']),
                       ('变化向量', ['extract_features'], ['DINOv2：T2 特征 − T1 特征。']),
                       ('与示例比较', ['match_exemplar'], ['余弦 ≥ 0.6 的提议取并集。', '去掉示例本身的像素。'])],
                scores=['SECOND 128 个任务：固定路线 19.39，Agent G3 20.34（第 10.10 节）。', '这个任务没有固定规则。']),
    't3x': dict(label='T3.2–T3.4 外部示例', title='固定路线：T3.2、T3.3、T3.4 用另一对图的示例（没有固定规则）',
                inputs=['T1 较早影像', 'T2 较晚影像', '外部示例图块'],
                steps=[('提议', ['segment_everything'], ['SAM ViT-H 自动掩码。']),
                       ('与示例比较', ['extract_features'], ['T3.2：T2 外观，门限 0.55。', 'T3.3：两期变化向量，门限 0.15。', 'T3.4：正例 − 负例，门限 0.20。']),
                       ('合并', ['combine'], ['超过门限的提议取并集。'])],
                scores=['SECOND：T3.2 14.44，T3.3 15.43，T3.4 12.84（固定路线，第 10.11 节）。Agent 没有全量运行。']),
    't54': dict(label='T5.4 旧地图', title='固定路线：T5.4 用新影像更新旧地图',
                inputs=['2012 旧地图 map#0', '2016 影像'],
                steps=[('分割新影像建筑', ['segment_text(T2)'], ['SAM3，与 I-M-C 相同的门限。']),
                       ('与旧地图比较', [], ['与旧地图逐像素异或。', '不删除小组件。'])],
                rule=dict(ASK_RULE, question='验证器对 82% 的组件答 no。'),
                scores=['WHU 128 张：固定路线 59.40，固定规则 30.97（第 10.11 节）。Agent 没有全量运行。']),
    't41': dict(label='T4.1 面积', title='固定路线：T4.1 只要大建筑的变化',
                inputs=['T1 较早影像', 'T2 较晚影像', '指令中的面积门限'],
                steps=IMC_STEPS + [('合并', ['to_mask'], ['出现 ∪ 消失。']), ('面积筛选', ['select'], ['保留 > 260 m² 的组件', '（> 1040 像素）。'])],
                rule=dict(ASK_RULE, decide=['答 no：删除这个组件。', '答 yes：保留这个组件。', '再按同样的面积筛选。']),
                scores=['LEVIR 验证 64 张：固定 + 面积 77.13，固定规则 + 面积 78.48（第 10.11 节）。Agent 全量在运行。']),
}


# Per-task controls (README_ai sections 10.5–10.13). Each row: task, data, fixed route, rule, other controls,
# where the settings were chosen, report-set F1 (fixed / rule / other), section.
IMC_TEXT = 'I-M-C(SAM3)：两期 segment_text("building")，分块 512，实例分数 ≥ 0.5，像素 ≥ 0.85 → match_instances（重叠 ≥ 0.25）→ 出现 ∪ 消失'
CONTROLS = [
    ('T1.1 × C0', 'LEVIR 验证 64 张', IMC_TEXT,
     '固定路线的组件 → flash 验证器（L1：4 张裁剪图、跨时相对应问法）→ 删除答 no', 'M-C-I 固定路线（参照）',
     '像素门限 0.85 在验证集 64 张上选（第 10.3 节：有选择偏差）；L2 验证器在 dev2 上比较，没有上线', '83.15 / 86.35', '10.5'),
    ('T1.1 × C0', 'SECOND 128 张', IMC_TEXT + '（不分方向）', '没有（C0 只做了固定路线）', '—',
     '沿用 LEVIR 的设置，零样本，不调', '51.83 / — ', '10.6'),
    ('T1.1 × C0', 'WHU 128 张', IMC_TEXT, '同 LEVIR 的验证器和问法，删除答 no', '对官方变化标签另报一行：固定 64.93、规则 76.35',
     '沿用 LEVIR 的设置，零样本；WHU 开发 60 张没有用来调参', '56.59 / 62.75', '10.11'),
    ('T1.1 × C1', 'SECOND、WHU 128 张', 'I-M-C(SAM3)，只取一个方向：新建 = 出现，拆除 = 消失',
     '带方向的问法 + SECOND 范围卡「范围卡加一句定义」：原址重建不算新建也不算拆除；删除答 no',
     '不分方向的固定路线（参照，显示不管方向会掉多少分）',
     '范围卡写法在 SECOND 开发 60 张上从 5 种写法中选（看了 128 张试跑的问题后改，第 10.6 节写明）',
     'SECOND 新建 57.04 / 63.74，拆除 32.84 / 37.96；WHU 新建 62.25 / 70.26，拆除 21.71 / 22.17', '10.6、10.11'),
    ('T5.1', 'LEVIR 验证 64 张', '给定掩码 = 不分方向的固定路线（I-M-C(SAM3)，像素 0.85），原样提交', '同 T1.1 × C0 的规则', '—',
     '同 T1.1 × C0', '83.15 / 86.35', '10.5'),
    ('T5.1 × C1', 'SECOND 128 张', '给定掩码 = 不分方向的固定路线；对照：给定掩码原样、给定掩码只留对应方向', '带方向的规则（同 T1.1 × C1）', '—',
     '同 T1.1 × C1', '新建：原样 49.03、只留方向 57.04 / 规则 63.74；拆除：原样 13.11、只留方向 32.84 / 规则 37.96', '10.6'),
    ('T1.1 × C1 转换', 'SECOND 128 张', '新建组件 ∩ T1 低矮植被或非植被地表（SegEarth-OV）；组件被覆盖 ≥ 25% 时保留整个组件，不设最小面积',
     '同一个验证器，问法只接受规定的前期地类；删除答 no', '—', 'SECOND 开发 60 张上选（component0.25_min0）', '62.82 / 62.70', '10.10'),
    ('T1.1 × C2 / C4', 'SECOND 128 张', 'I-M-C(SAM3)（C4 只取出现）→ 与框 region#0 逐像素求交',
     '在整图组件上删除答 no，再与框求交', '事后 ∩ 框：Agent 的 C0 / C1 结果直接与框求交',
     '框按种子 1016 冻结；没有新门限', 'C2：54.67 / 60.69，事后 66.71；C4：62.63 / 67.99，事后 76.12', '10.7、10.10'),
    ('T5.2 点反馈', 'LEVIR 验证 64 张', '固定响应 C（主对照）：误检点删除点所在的给定组件；漏检点在 T1、T2 各做一次 SAM ViT-H 点提示，取面积小的候选并入',
     '没有', '固定响应 A：点落在 T2 的 SAM3 建筑里用 T2，否则用 T1；固定响应 B：取三个候选里面积最小的',
     'B、C 是看了 A 的失败后加的，没有调门限（第 10.8 节写明）', 'k=1 / k=3：C 84.07 / 84.73；A 67.00 / 56.47；B 73.83 / 69.36', '10.8'),
    ('T5.3 语言反馈', 'LEVIR 验证 64 张', '固定响应 1：按 3×3 方位格定位，选格子内面积最大的候选', '没有',
     '固定响应 2：格子内用验证器 p_yes 挑选（漏检取最高，误检取最低）。它用验证器，但不是删除规则', '沿用 LEVIR 设置；10 张试跑后改了误检句', '给定 83.15；固定响应 1 82.73；固定响应 2 83.22', '10.10'),
    ('T1.3 不限类别', 'SECOND 128 张', 'SAM ViT-H 两期提议 → DINOv2 区域特征 1 − 余弦 ≥ 0.22 → 候选取并集',
     'describe 问法，删除答 no', 'AnyChange 46.80；删除 p_yes < 0.9 的变体 61.42',
     '门限在 SECOND 开发 60 张上选；问法在同 60 张的 1,649 个候选上从 3 种中选', '60.88 / 61.11', '10.10'),
    ('T1.2 多类别', 'SECOND 128 张', '建筑 = I-M-C(SAM3) 的变化；低矮植被 = SegEarth-OV 两期分割后逐类异或；两类取并集，删除 < 64 像素的组件',
     '同一个验证器，删除答 no', '—', 'SECOND 开发 60 张上选（类别和 sam3_min64）', '46.26 / 47.50', '10.10'),
    ('T3.1 图内示例', 'SECOND 128 个任务', 'SAM 提议 → DINOv2 两期变化向量 → 与示例的余弦 ≥ 0.6 的提议取并集；预测和真值都去掉示例像素',
     '没有', '—', 'SECOND 开发 60 张上选 0.6', '19.39（703 个任务全集 19.80）', '10.10'),
    ('T3.2 / T3.3 / T3.4', 'SECOND 1,823 个任务', '同 T3.1 的提议和特征：T3.2 比 T2 外观（门限 0.55），T3.3 比两期变化向量（0.15），T3.4 正例减负例（0.20）',
     '没有', '—', '开发 912 个任务上选门限，并列取小', '14.44 / 15.43 / 12.84', '10.11'),
    ('T4.1 面积', 'LEVIR 验证 64 张', '固定路线 + 面积筛选：只保留 > 260 m²（> 1,040 像素）的组件', '规则 + 同样的面积筛选', '—',
     '260 m² 用 dev2 的组件面积分布选（约 1/3 组件达标）', '77.13 / 78.48', '10.11'),
    ('T5.4 旧地图', 'WHU 128 张', '2016 影像 SAM3 建筑 → 与 2012 旧地图逐像素异或，不删小组件',
     '旧问法「地图与影像是否不同」：验证器对 82% 的组件答 no', '新问法「只看 T2，轮廓内现在是不是建筑」，方向由 harness 结合旧地图判断（开发 60 张：规则 67.13，固定 66.22）',
     'min_area=0 在 WHU 开发 60 张上选；问法在同 60 张上比较', '59.40 / 30.97（旧问法）', '10.11、10.13'),
]


def controls_table():
    rows = ''.join(f'<tr><td class="task">{esc(t)}<span>{esc(d)}</span></td><td>{esc(f)}</td><td>{esc(r)}</td><td>{esc(o)}</td><td>{esc(w)}</td><td class="num">{esc(sc)}</td><td>{esc(sec)}</td></tr>'
                   for t, d, f, r, o, w, sc, sec in CONTROLS)
    return ('<section class="panel example" id="controls"><h2>各任务的对照是什么</h2>'
            '<ul><li>规则只删除，不补漏。</li><li>不同任务的对照是人按任务定义分别写的。Agent 在所有任务上用同一套代码。</li>'
            '<li>「固定规则」在不同任务里不是同一个东西：验证问题、候选来源和后处理都随任务变化。下表逐项写明。</li>'
            '<li>分数是报告集上的 F1ᶜ（固定 / 规则），来源是 README_ai 的对应章节。</li></ul>'
            '<div class="tablewrap"><table class="hubtable controls"><thead><tr><th>任务</th><th>固定路线</th><th>规则</th><th>其他对照</th><th>设置在哪选</th><th>F1（固定 / 规则）</th><th>章节</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></section>')


def figure(uri, label, extra=''):
    return (f'<figure class="{extra}"><button type="button" class="zoom" aria-label="放大：{esc(label)}">'
            f'<img src="{uri}" alt="{esc(label)}" loading="lazy"></button><figcaption>{esc(label)}</figcaption></figure>')


def page(title, body, script=''):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body>{body}{script}</body></html>')


class Store:
    """Values that need the server data (images, masks, traces), cached per page.

    Data step (mode 'build'): every get() computes its value and the store is saved as JSON.
    Page step (mode 'read'): get() only reads the saved value, so the HTML is assembled in seconds
    without the server data. Text, CSS and figures live in the page step and need no data step.
    """
    def __init__(self, path, mode):
        self.path, self.mode = Path(path), mode
        self.data = json.loads(self.path.read_text(encoding='utf-8')) if mode == 'read' else {}

    def get(self, key, compute):
        if self.mode == 'read':
            if key not in self.data:
                raise KeyError(f'{self.path.name} has no cached value {key!r}; run the data step (--stage data) again.')
            return self.data[key]
        if key not in self.data:
            self.data[key] = compute()
        return self.data[key]

    def save(self):
        if self.mode == 'build':
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding='utf-8')


def harness_of(info):
    backends = info.get('backends') or {}
    return 'G5' if backends.get('gate5') else 'G4' if backends.get('gate4') else 'G3' if backends.get('gate3') else 'G2' if backends.get('gate2') else 'G1'


def harness_notes(version):
    order = ['G2', 'G3', 'G4']
    notes = [HARNESS_NOTES[v] for v in order[:order.index(version) + 1]] if version in order else []
    return notes + [OFF_NOTE]


def gate_caption(version, stops=None):
    rules = ['规划器第一次调用 submit 时，提交门暂停提交，并返回报告。这个暂停不表示拒绝。',
             '规划器再次调用 submit 时，只有掩码含未验证组件，提交门才暂停提交。',
             '每次暂停时，验证器验证两类组件：掩码中的未验证组件；已从给定掩码删除且没有答案的组件。',
             '答案只作为证据。规划器决定是否删除组件。']
    changes = {
        'G2': ['预算用完时，当前答案有未验证组件：harness 补一次报告。',
               '分批调用 ask 时，重复调用保护不再拦截。'],
        'G3': ['框提示的框超过整图 10% 时，segment_prompt 返回错误。',
               '有区域或方向约束时，报告列出区域外像素和方向错误。harness 不裁剪掩码。',
               '反馈任务：自动验证只覆盖反馈附近和改动过的组件。'],
        'G4': ['最后 3 步，当前答案不是最新掩码时，状态行提醒规划器。harness 不替换掩码。',
               '有 region#0 时，提交的掩码与区域求交。求交前后的掩码都写入轨迹。',
               '虚线框工具的说明附一句用途。'],
    }
    order = ['G2', 'G3', 'G4']
    rows = ''.join(f'<li><strong>{v}</strong>：{esc(c)}</li>' for v in order[:order.index(version) + 1] if version in order for c in changes[v])
    after = '这些情况下，验证器在会话结束后验证还没有答案的组件。答案只写入验证记录。'
    if stops is not None:
        after += f'这个会话的结束后验证次数：{stops}。工具名后的 ×N 是这个会话的调用次数。灰色工具在这个会话中没有调用。'
    return ('<figcaption class="syscap"><strong>验证器只接收裁剪图，只返回答案。只有规划器可以调用工具箱。</strong><div class="capgrid">'
            '<div><h3>提交门的规则</h3><ol>' + ''.join(f'<li>{esc(r)}</li>' for r in rules) + '</ol></div>'
            '<div><h3>规划器收不到答案的 3 种情况</h3><ul><li>规划器在第 30 步直接调用 submit。</li><li>规划器连续 3 步没有调用工具。</li>'
            '<li>再次提交的掩码没有未验证组件，但规划器删除了未验证的给定组件。</li></ul><p>' + esc(after) + '</p></div>'
            + (f'<div><h3>{esc(version)} 相对 G1 的改动</h3><ul>{rows}</ul></div>' if rows else '') + '</div></figcaption>')


TERMS = [
    ('会话', 'Agent 处理一张图的全过程。'), ('步', '规划器的一次回复，以及这次回复发出的工具调用。'),
    ('当前掩码', '会话中最新的候选结果。'), ('最终掩码', '会话结束时提交的掩码。'), ('组件', '掩码中的一个 8 连通区域。'),
    ('给定掩码', 'T5 任务的输入掩码。这个掩码来自固定路线。'), ('区域 region#0', 'C2、C4 任务给定的框。答案只能在框内。'),
    ('示例掩码', 'T3.1 给定的一处变化。Agent 找同类的变化。'), ('规划器', '决定调用哪些工具、保留哪些组件、何时提交的模型。'),
    ('验证器', '判断一个组件是否是真实变化的模型。它独立于规划器。'), ('harness', '运行规划器和工具的程序。G1–G4 是它的版本。'),
    ('提交门 Gate', 'harness 中的一个结构。规划器调用 submit 时，提交门判断是否暂停提交。'),
    ('固定路线', '程序按固定顺序和固定参数调用工具。不用规划器，不用验证器。'),
    ('固定规则', '在固定路线的组件上运行验证器，删除答 no 的组件。不用规划器。'),
    ('固定响应', '反馈任务（T5.2、T5.3）的固定路线。程序按反馈的位置直接修改给定掩码，不用规划器。'),
    ('虚线框工具', '默认关闭的执行工具，用命令行开关 --tools-t13 打开。T1.2、T1.3、T3 使用。'),
    ('F1ᶜ', '累计全部图的 TP、FP、FN 后计算的 F1。单张图的 F1 写作 F1。'), ('真值', '数据集提供的变化标注。真值只用于事后评分。'),
]


def terms_block(open_details=True):
    rows = ''.join(f'<div><dt>{esc(a)}</dt><dd>{esc(b)}</dd></div>' for a, b in TERMS)
    return f'<details class="terms"{" open" if open_details else ""}><summary>术语</summary><dl class="termgrid">{rows}</dl></details>'


# ---------------------------------------------------------------- comparison on one image

def comparison_scores(data, agent, truth):
    """F1, P, R of fixed route, fixed rule and Agent on one image, with the task's own exclusions."""
    ignore = data.get('ignore')
    if ignore is not None:
        truth, agent = truth & ~ignore, agent & ~ignore
    out = {}
    fixed, rule = data.get('fixed'), data.get('rule') if data.get('rule') is not None else data.get('response')
    for key, pred in (('fixed', fixed), ('rule', rule), ('agent', agent)):
        if pred is not None:
            out[key] = measures(pred & ~ignore if ignore is not None else pred, truth)
    return out, truth, ignore


def baseline_names(key):
    if key == 'levir_t52':
        return '固定响应 C', '固定规则'
    if key == 'levir_t53':
        return '固定响应 1', '固定响应 2'
    return '固定路线', '固定规则'


def comparison_data(record, data, agent, truth, t2, encode):
    """Data step: the 3 pictures and scores of one image."""
    scores_, truth_used, ignore = comparison_scores(data, agent, truth)
    fixed, rule = data.get('fixed'), data.get('rule') if data.get('rule') is not None else data.get('response')
    deleted = fixed & ~data['rule'] if fixed is not None and data.get('rule') is not None else None
    pictures = {}
    for pred, sk in ((fixed, 'fixed'), (rule, 'rule'), (agent, 'agent')):
        if pred is not None:
            pictures[sk] = encode(scored_image(pred & ~ignore if ignore is not None else pred, truth_used, t2, deleted if sk == 'rule' else None))
    return dict(scores=scores_, pictures=pictures, deleted=deleted is not None, ignore=ignore is not None,
                same=bool(fixed is not None and rule is not None and np.array_equal(fixed, rule)),
                answers=[dict(id=a['id'] if isinstance(a['id'], str) else int(a['id']), area=int(a['area']), answer=a.get('answer', '未记录'), reason=a.get('reason', '')) for a in data.get('answers') or []])


def comparison_panel(record, c):
    """Page step: the HTML of the three-method comparison."""
    key = record['page']
    fixed_name, rule_name = baseline_names(key)
    family = TASKS[key][5]
    cols = []
    for name, sk in ((fixed_name, 'fixed'), (rule_name, 'rule'), ('Agent（最终掩码）', 'agent')):
        if sk not in c['pictures']:
            reason = '这个任务没有固定规则。' if sk == 'rule' else '没有找到这个对照的掩码。'
            cols.append(f'<div><h3>{esc(name)}</h3><div class="empty">{reason}</div></div>')
            continue
        m = c['scores'][sk]
        cols.append(f'<div><h3>{esc(name)}</h3>{figure(c["pictures"][sk], name + "：与真值对比")}'
                    f'<p class="score"><strong>F1 {m["f1"] * 100:.2f}</strong> · P {m["precision"] * 100:.2f} · R {m["recall"] * 100:.2f}</p></div>')
    body = (f'<section class="panel compare"><h2>同一张图上的三种做法</h2>'
            f'<p class="route">{esc(ROUTES[family]["title"])}。<a href="baselines.html#{family}">查看这条路线的结构</a>。</p>'
            '<div class="trio">' + ''.join(cols) + '</div>'
            '<div class="legend"><span><i class="tp"></i>TP 命中</span><span><i class="fp"></i>FP 误检</span><span><i class="fn"></i>FN 漏检</span>'
            + ('<span>黄色虚线：固定规则删除的组件</span>' if c['deleted'] else '') + '</div>')
    if c['same']:
        body += f'<p class="muted">这张图上，{esc(rule_name)}的掩码与{esc(fixed_name)}的掩码相同。</p>'
    if c['ignore']:
        body += ('<p class="muted">这 3 个分数按 T3.1 的评分方法计算：预测和真值都去掉示例本身的像素。页头的 F1 是记录器的值，没有去掉示例。'
                 'G3 的 384 个 T3.1 会话中，209 个会话按这个方法计算的 F1 是 0。</p>')
    if c['answers']:
        rows = ''.join(f'<tr><td>{esc(a["id"])}</td><td>{esc(a["area"])}</td><td>{esc(a["answer"])}</td><td>{esc(a["reason"])}</td></tr>' for a in c['answers'])
        body += (f'<details><summary>固定规则的验证器答案（{len(c["answers"])} 个组件，点击展开）</summary>'
                 '<p class="muted">固定规则的请求在会话之外单独发送。同一个组件的答案可以与会话中的答案不同。</p>'
                 f'<div class="tablewrap"><table><thead><tr><th>组件</th><th>面积（像素）</th><th>答案</th><th>理由</th></tr></thead><tbody>{rows}</tbody></table></div></details>')
    return body + '</section>'


# ---------------------------------------------------------------- replay page

def input_cell(directory, info, key, sources, encode):
    """The fourth cell of the input grid."""
    initial = info.get('initial_handle')
    t2 = sources[1].copy()
    if key in ('second_c2', 'second_c4'):
        region = directory / 'masks' / 'region_0.png'
        if region.exists():
            yy, xx = np.nonzero(read_mask(region))
            if len(xx):
                ImageDraw.Draw(t2).rectangle((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())), outline='#ffd400', width=max(3, t2.width // 120))
            return [encode(t2), 'region', None]
    if key in ('levir_t52', 'levir_t53'):
        from agent_baseline.feedback import cell_mask
        canvas = t2
        if initial and mask_path(directory, initial).exists():
            small = Image.fromarray(read_mask(mask_path(directory, initial)).astype('uint8') * 255)
            edge = np.asarray(small.filter(ImageFilter.MaxFilter(3))) != np.asarray(small.filter(ImageFilter.MinFilter(3)))
            a = np.array(canvas); a[edge] = [255, 60, 60]; canvas = Image.fromarray(a)
        draw = ImageDraw.Draw(canvas)
        for entry in (info.get('task_meta') or {}).get('feedback') or []:
            if 'point' in entry:
                x, y = entry['point']; r = max(6, canvas.width // 80)
                draw.ellipse((x - r, y - r, x + r, y + r), fill='#ffd400', outline='black', width=2)
            elif 'where' in entry:
                yy, xx = np.nonzero(cell_mask(entry['where'], (canvas.height, canvas.width)))
                if len(xx):
                    draw.rectangle((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())), outline='#ffd400', width=max(3, canvas.width // 150))
        return [encode(canvas), 'feedback', None]
    if key == 'second_t31':
        handle = (info.get('initial_view') or {}).get('overlay') or initial
        if handle and mask_path(directory, handle).exists():
            return [encode(open_image(mask_path(directory, handle))), 'exemplar', handle]
    if initial and mask_path(directory, initial).exists():
        return [encode(open_image(mask_path(directory, initial))), 'given', initial]
    return [None, 'none', None]


def input_figure(key, cell):
    uri, kind, handle = cell
    if uri is None:
        return f'<figure><div class="empty">没有给定掩码</div><figcaption>{esc(TASKS[key][1])}</figcaption></figure>'
    caption = {'region': 'T2 和区域 region#0（黄框）', 'exemplar': f'示例掩码 {handle}', 'given': f'给定掩码 {handle}',
               'feedback': 'T2、给定掩码（红线）和反馈' + ('点' if key == 'levir_t52' else '方位格') + '（黄色）'}[kind]
    return figure(uri, caption)


def answers_html(call):
    try:
        results = json.loads(call.get('result_text') or '').get('results') or []
    except (ValueError, AttributeError):
        return ''
    rows = [f'<li><span class="{"down" if r.get("answer") == "no" else "up" if r.get("answer") == "yes" else "muted"}">{esc(r.get("answer") or "未解析")}</span> '
            f'组件 {esc(r.get("id"))}{"" if r.get("p_yes") is None else " · p_yes " + format(r["p_yes"], ".2f")}：{esc(r.get("reason") or "")}</li>'
            for r in results if isinstance(r, dict)]
    return f'<details><summary>验证器答案（{len(rows)} 条，点击展开）</summary><ul class="answers">{"".join(rows)}</ul></details>' if rows else ''


def result_block(value, threshold=420):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) > threshold:
        return f'<details><summary>返回内容（{len(text)} 字符，点击展开）</summary><pre>{esc(text)}</pre></details>'
    return f'<pre>{esc(text)}</pre>'


def ledger_html(entries):
    if entries is None:
        return ''
    rows, against = [], 0
    for entry in sorted(entries, key=lambda e: -e.get('area', 0)):
        records = entry.get('records', [])
        latest, kept = entry.get('latest'), entry.get('kept')
        conflict = (latest == 'no' and kept) or (latest == 'yes' and not kept)
        against += conflict
        last = records[-1] if records else {}
        score = last.get('p_yes')
        rows.append(f'<tr{" class=against" if conflict else ""}><td>{esc(entry.get("area"))}</td><td>{esc(entry.get("bbox"))}</td>'
                    f'<td>{len(records)}</td><td>{esc(latest or "没有有效答案")}</td><td>{esc(last.get("type") or "")}</td>'
                    f'<td>{"" if score is None else f"{score:.2f}"}</td><td>{"保留" if kept else "删除"}</td><td>{esc(last.get("reason") or "")}</td></tr>')
    return (f'<details class="ledger"><summary>验证记录：验证过的组件 {len(entries)} 个。最终决定与最后答案不同的组件 {against} 个（彩色背景标出）。</summary>'
            '<p class="muted">这张表列出验证过的组件，包含中间掩码的组件和已删除的组件。行数不等于最终掩码的组件数。p_yes 是模型输出的数值，没有校准。</p>'
            '<div class="tablewrap"><table><thead><tr><th>面积</th><th>外接框</th><th>答案数</th><th>最后答案</th><th>类型</th><th>p_yes</th><th>最终决定</th><th>最后理由</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div></details>')


def render_replay(root, record, store, quality=62):
    from functools import lru_cache
    key = record['page']
    code, text, dataset, chip4, arrow_label, _ = TASKS[key]
    directory = Path(root or '.') / record['run'] / record['task'] / 'val' / record['image']  # only read in the data step
    S = store.get
    info = S('info', lambda: read_json(directory / 'input.json'))
    final = S('final', lambda: read_json(directory / 'final.json'))
    rows = S('rows', lambda: read_steps(directory))
    entries = S('verification', lambda: read_json(directory / 'verification.json') if (directory / 'verification.json').exists() else None)
    version = harness_of(info)
    encode = lambda im, limit=512: jpeg(im, min(limit, 512), quality)
    name = record['image']

    @lru_cache(None)
    def gt():
        return read_mask(resolve_asset(directory, info['gt_mask_path'])) if info.get('gt_mask_path') else read_mask(
            Path(info['source_images'][0]).parent.parent / 'label' / (name + '.png'))

    @lru_cache(None)
    def sources():
        return [open_image(resolve_asset(directory, p)) for p in info['source_images']]

    @lru_cache(None)
    def t2():
        im = sources()[1].copy(); im.thumbnail((512, 512), Image.Resampling.LANCZOS)
        return im

    panels = {}

    def mask_images(handle):
        mask = read_mask(mask_path(directory, handle))
        small = Image.fromarray(mask.astype('uint8') * 255).resize(t2().size, Image.Resampling.NEAREST)
        gt_small = np.asarray(Image.fromarray(gt()).resize(t2().size, Image.Resampling.NEAREST))
        pred = np.asarray(small) > 127
        edge = np.asarray(small.filter(ImageFilter.MaxFilter(3))) != np.asarray(small.filter(ImageFilter.MinFilter(3)))
        overlay = np.array(t2()); overlay[edge] = [239, 80, 80]
        compare = (np.array(t2()).astype(float) * .22).astype('uint8')
        compare[pred & gt_small] = [44, 206, 120]; compare[pred & ~gt_small] = [239, 80, 80]; compare[~pred & gt_small] = [69, 141, 255]
        return [encode(Image.fromarray(mask.astype('uint8') * 255)), encode(Image.fromarray(overlay)), encode(Image.fromarray(compare))]

    def mask_assets(handle):
        thumb, overlay, compare = S(f'mask:{handle}', lambda: mask_images(handle))
        panels[handle] = (figure(overlay, 'T2 和当前掩码（红色轮廓）') + figure(compare, '与真值对比（模型在运行时看不到）'))
        return thumb

    final_metric = final.get('eval') or {}
    final_f1 = metric_f1(final_metric)
    calls = [c for r in rows for c in r.get('calls', [])]
    tp, fp, fn = (final_metric.get(k) for k in ('tp', 'fp', 'fn'))
    prec = tp / (tp + fp) if tp is not None and tp + fp else None
    rec = tp / (tp + fn) if tp is not None and tp + fn else None
    stats = [('最终 F1', pct(final_f1)), ('精确率 P', pct(prec)), ('召回率 R', pct(rec)), ('harness', version),
             ('步数', str(final.get('planning_rounds', sum(r['step'] > 0 for r in rows)))), ('工具调用', str(final.get('tool_calls', len(calls)))),
             ('耗时', f'{final["elapsed_seconds"]:.1f} 秒' if final.get('elapsed_seconds') is not None else '没有记录'),
             ('费用', f'¥{final["cost_cny"]:.4f}' if final.get('cost_cny') is not None else '没有记录')]
    title = f'{code} · {text}'
    head = (f'<header><div class="headtop"><div><div class="eyebrow"><a href="index.html">回放列表</a> · {esc(dataset)} 验证集 · {esc(name)}</div><h1>{esc(title)}</h1></div>'
            '<label class="js-only">外观 <select id="theme"><option value="auto">跟随系统</option><option value="light">浅色</option><option value="dark">深色</option></select></label></div>')
    head += '<div class="warning">这个会话的最终 F1 接近这个任务全部会话的中位数。它不是挑选出的最好结果。</div>'
    head += '<div class="metrics">' + ''.join(f'<div class="metric"><strong>{esc(v)}</strong><span>{k}</span></div>' for k, v in stats) + '</div>'
    if code.startswith('T5'):
        head += f'<p>给定掩码 F1 {pct(metric_f1(final.get("initial_eval")))} → 最终 F1 {pct(final_f1)}。</p>'
    head += f'<details open><summary>指令原文</summary><pre>{esc(info.get("instruction", "没有记录"))}</pre></details></header>'

    counts = Counter(c['tool'] for c in calls)
    enabled = {s['function']['name'] for s in info.get('schemas', [])}
    switch = sorted(enabled & SWITCH_TOOLS)
    chips = ['T1 较早影像', 'T2 较晚影像', INSTRUCTION_CHIP.get(key, '指令（一句话任务）')] + ([chip4] if chip4 else []) + ['系统提示和工具定义']
    answers = sum(len(e.get('records', [])) for e in entries) if entries else 0
    svg = agent_figure(version, chips, arrow_label, counts, answers, sum(c.get('status') == 'report' for c in calls),
                       final.get('termination_reason', '没有记录'), switch, harness_notes(version))
    system = (f'<section class="panel system"><h2>系统流程（{version}）</h2><figure><div class="sysscroll">{svg}</div>'
              + gate_caption(version, S('stops', lambda: after_stop_count(directory))) + '</figure>' + terms_block(False) + '</section>')

    final_handle = final.get('final_handle')
    c = S('compare', lambda: comparison_data(record, load_baselines(root, record, directory),
                                             read_mask(resolve_asset(directory, final['mask_path'])), gt(), t2(), encode))
    compare, scores_ = comparison_panel(record, c), c['scores']

    left = '<aside><section class="inputs panel sticky"><h2>输入和真值</h2><div class="pair">'
    left += ''.join(figure(S(f'source:{k}', lambda k=k: encode(sources()[k])), label) for k, label in enumerate(['T1 · 较早影像', 'T2 · 较晚影像']))
    left += figure(S('gt', lambda: encode(Image.fromarray(gt().astype('uint8') * 255))), '真值（模型在运行时看不到）')
    left += input_figure(key, S('input4', lambda: input_cell(directory, info, key, sources(), encode)))
    left += '</div><p class="muted">页面图像是缩小版。点击图像可以放大。真值和 F1 只用于事后评分。</p></section></aside>'

    timeline = ['<section aria-label="步骤时间线"><h2>Agent 做了什么</h2><p class="muted">页面按记录顺序回放。点击步骤卡片可以切换步骤。按 ← 或 → 键也可以切换。第 0 步是初始化。</p>'
                '<noscript><p>JavaScript 没有启用。时间线和最终结果仍然可以阅读。</p></noscript>']
    states = []
    for i, row in enumerate(rows):
        step, handle = row['step'], row.get('current_mask')
        value = metric_f1(row.get('eval')) if handle else None
        label = '第 0 步 · 初始化' if step == 0 else f'第 {step} 步'
        states.append(dict(mask=handle, label=label, metric=f'当前 F1 {pct(value)}' if handle else '没有当前掩码'))
        odd = step > 0 and not row.get('calls')
        body = (f'<article class="step" data-index="{i}" data-step="{step}" id="step-{step}"><div class="stephead"><h3>{label}'
                + (' <span class="badge">异常</span>' if odd else '') + f'</h3><button class="js-only" type="button" data-select="{i}">查看此步</button></div>')
        thought = (row.get('planner') or {}).get('thought')
        body += f'<div class="thought">{esc(thought)}</div>' if thought else '<p class="muted">这一步没有模型输出。</p>'
        if row.get('report'):
            kind = '当前答案提醒（G4）' if 'Your current answer is' in row['report'] else '报告'
            body += f'<div class="reportbox">{kind}：{esc(row["report"])}</div>'
        if step == 0:
            views = S('views0', lambda: [encode(open_image(p)) for p in (directory / 'images' / f'view_0_{t}.jpg' for t in ('t1', 't2')) if p.exists()])
            if len(views) == 2:
                body += '<details><summary>规划器收到的初始图像（缩小版）</summary><div class="pair">' + ''.join(
                    figure(uri, f'初始 T{k + 1}') for k, uri in enumerate(views)) + '</div></details>'
        for call in row.get('calls', []):
            tool = call['tool']
            failed = not call.get('ok', False)
            held = call.get('status') == 'report'
            body += f'<div class="call{" error" if failed else ""}"><span class="badge">{"提交门" if held else CATEGORIES.get(tool, "其他")}</span><span class="toolname">{esc(tool)}</span>'
            if held:
                body += '<span class="muted"> · 提交门暂停了这次提交，并返回报告。这不表示拒绝。</span>'
            if failed:
                body += f'<span class="errorlabel"> · 调用失败 {esc(call.get("error_code") or "")}</span>'
            if call.get('status') == 'skipped_due_to_submit':
                body += '<span class="muted"> · 提交后跳过</span>'
            body += f'<details><summary>调用参数</summary><pre>{esc(json.dumps(call.get("args", call.get("raw_arguments", {})), ensure_ascii=False, indent=2))}</pre>'
            if tool == 'ask' and not call.get('args'):
                body += '<p class="muted">不传参数时，ask 验证当前掩码中的全部未验证组件。验证顺序是面积从大到小。每次调用最多验证 64 个组件。</p>'
            body += '</details>'
            report = report_text(call)
            if report:
                body += f'<div class="reportbox">{esc(report)}</div>'
            else:
                if tool == 'ask':
                    body += answers_html(call)
                body += '<div class="muted">返回内容</div>' + result_block(call.get('result_text', '没有记录'))
            if call.get('result_images'):
                body += '<div class="pair">' + ''.join(figure(S(f'image:{p}', lambda p=p: encode(open_image(resolve_asset(directory, p)))), f'规划器收到的 T{j + 1}（缩小版）')
                                                       for j, p in enumerate(call['result_images'][:2])) + '</div>'
            provenance = call.get('provenance') or {}
            if provenance.get('pre_region_handle'):
                body += f'<div class="reportbox">提交时与区域求交（G4）：删除区域外 {esc(provenance.get("region_clipped_px", "没有记录"))} 像素。</div>'
            for created in dict.fromkeys(call.get('handles_created') or []):
                if created.startswith(('inst#', 'props#')) and tool in SWITCH_TOOLS | {'segment_everything'}:
                    uri = S(f'props:{created}', lambda created=created: (lambda im: encode(im) if im is not None else None)(proposal_thumbnail(directory, created)))
                    if uri is not None:
                        body += figure(uri, f'返回的候选（{created}，全部候选的并集）', 'thumb')
            before, after = call.get('current_before'), call.get('current_after')
            if before != after:
                if after:
                    delta = (call.get('eval') or {}).get('delta_f1')
                    change = '<span class="muted">这是第一个当前掩码，没有前值可以比较</span>' if delta is None else \
                        f'<span class="{"up" if delta > 0 else "down" if delta < 0 else "muted"}">ΔF1 {delta * 100:+.2f} 个百分点</span>'
                    body += f'<p>当前掩码：{esc(before or "无")} → {esc(after)} · F1 {pct(metric_f1(call.get("eval")))}<br>{change}</p>'
                    body += figure(mask_assets(after), '这次更新后的当前掩码', 'thumb')
                else:
                    body += '<p class="muted">当前掩码被清空。</p>'
            body += '</div>'
        if odd:
            body += '<p class="warning">异常：这一步没有调用任何工具。原因未确认。</p>'
        if handle:
            mask_assets(handle)
            body += f'<p class="muted">这一步结束时的当前掩码：{esc(handle)} · F1 {pct(value)}</p>'
        else:
            body += '<p class="muted">没有当前掩码。工具的中间结果见上方返回内容。</p>'
        timeline.append(body + '</article>')
    if final_handle:
        mask_assets(final_handle)
    if not states or states[-1]['mask'] != final_handle:
        states.append(dict(mask=final_handle, label='最终提交', metric=f'最终 F1 {pct(final_f1)}'))
    curve = chart(states, rows).replace('逐步 F1 曲线，点可点击或用回车选择', '每一步的 F1 曲线。点击数据点或按回车键可以选择步骤。').replace('>步数<', '>步<')
    timeline.append('</section>')

    right = ('<aside><div class="results sticky panel"><h2>掩码与结果</h2><div class="controls js-only"><button id="prev" type="button" aria-label="上一步">←</button>'
             f'<button id="next" type="button" aria-label="下一步">→</button><button type="button" data-select="{len(states) - 1}">最终结果</button></div>'
             f'<p id="current-label">{esc(states[-1]["label"])}</p><p id="current-metric">{esc(states[-1]["metric"])}</p>'
             '<div data-mask-panel="none"' + (' hidden' if final_handle else '') + '><div class="empty">没有当前掩码</div></div>')
    for handle, panel in panels.items():
        right += f'<div data-mask-panel="{esc(handle)}"' + ('' if handle == final_handle else ' hidden') + '>' + panel + '</div>'
    right += ('<div class="legend"><span><i class="tp"></i>TP 命中</span><span><i class="fp"></i>FP 误检</span><span><i class="fn"></i>FN 漏检</span></div>'
              + curve + '<p class="muted">曲线只显示已记录的当前掩码 F1。曲线中断的位置没有 F1 记录，不表示 0 分。指标按原分辨率计算，不使用缩小图。</p></div></aside>')
    errors = Counter(c['tool'] for c in calls if not c.get('ok', False))
    footer = ('<footer><h2>最终提交</h2>'
              f'<p>提交来源：{esc(final.get("submitted_by", "没有记录"))} · 结束原因：{esc(final.get("termination_reason", "没有记录"))}</p>'
              f'<pre>{esc(final.get("summary", "没有记录"))}</pre>')
    if 'pre_region_handle' in final:
        footer += f'<p>提交时与区域求交（G4）：{esc(final["pre_region_handle"])} → {esc(final_handle)}，删除区域外 {esc(final.get("region_clipped_px", "没有记录"))} 像素。</p>'
    footer += ledger_html(entries)
    footer += (f'<p>变化组件数：{esc(final.get("change_component_count", "没有记录"))} · 变化面积：{esc(final.get("change_area_px", "没有记录"))} 像素。'
               '<strong>组件按 8 连通计算。组件数不等于目标个数。</strong></p>')
    footer += '<p>工具调用次数（包含失败和跳过的调用）：' + esc('；'.join(f'{k} × {v}（失败或跳过 {errors[k]} 次）' for k, v in sorted(counts.items()))) + '</p>'
    footer += '<p>记录器计算 F1。Agent 在运行时看不到 F1。真值不参与规划。真值和预测都为空时，这张图的 F1 记为 1。页面图像是 JPEG 缩小版。面积和指标按原分辨率记录。</p></footer>'
    state_json = json.dumps(states, ensure_ascii=False).replace('<', '\\u003c')
    body = (head + system + compare + '<main class="layout">' + left + ''.join(timeline) + right + '</main>' + footer +
            '<dialog id="zoom-dialog" aria-label="放大图像"><button id="close-zoom" class="close" type="button">关闭（Esc）</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>')
    script = f'<script id="states" type="application/json">{state_json}</script><script>{JS}</script>'
    return page(f'{title} · {name}', body, script), dict(version=version, scores=scores_)


# ---------------------------------------------------------------- entry page

def render_index(records):
    rows = []
    for r in records:
        code, text, dataset = TASKS[r['page']][:3]
        sc = r['scores']
        values = {k: sc[k]['f1'] * 100 for k in ('agent', 'fixed', 'rule') if k in sc}
        best = max(values.values()) if values else None
        fixed_name, rule_name = baseline_names(r['page'])

        def cell(k, label):
            if k not in values:
                return '<td class="num muted">—</td>'
            note = '' if label in ('固定路线', '固定规则', '') else f' <span class="muted">{esc(label)}</span>'
            return f'<td class="num{" best" if values[k] == best else ""}">{values[k]:.2f}{note}</td>'
        rows.append(f'<tr><td><a href="{r["file"]}">{esc(r["image"])}</a></td><td class="task">{esc(code)}<span>{esc(text)}</span></td><td>{esc(dataset)}</td>'
                    f'<td class="hv">{esc(r["version"])}</td>{cell("agent", "")}{cell("fixed", fixed_name)}{cell("rule", rule_name)}</tr>')
    cards = [('baselines.html', '三种做法', '固定路线、固定规则和 Agent 用同一套工具。差别只在谁做决定。', '按任务查看固定路线和固定规则的结构，以及一张图的分步示例。'),
             ('errors.html', '典型错误', '固定路线和 Agent 的错误按类统计：LEVIR、SECOND、WHU。', '每类附损失最大的样例。'),
             ('../tools/index.html', '核心工具', '23 个原子工具：功能、输入、输出、后端模型、开关。', '其中 14 个默认可用，9 个要用命令行开关打开。'),
             ('../demo_g1/index.html', 'G1 回放', 'LEVIR-CD 建筑变化，G1 配置，4 个回放。', '包含提示词全文。')]
    entry = '<nav class="entry" aria-label="子页面">' + ''.join(
        f'<a class="card" href="{h}"><h2>{esc(t)}</h2><p>{esc(a)}</p><p>{esc(b)}</p><p class="go">打开 →</p></a>' for h, t, a, b in cards) + '</nav>'
    head = ('<header><h1>多任务回放（G3 / G4）</h1><p>规划器和验证器都使用 qwen3.7-flash。每个任务展示 1 个会话。这个会话的最终 F1 接近这个任务全部会话的中位数。</p>'
            '<p>表中的 F1 是这一张图上的分数。每个任务在全部图上的分数见「三种做法」。</p></header>')
    svg = agent_figure('G3', ['T1 较早影像', 'T2 较晚影像', '指令（一句话任务）', '给定掩码、区域或示例', '系统提示和工具定义'],
                       '其他输入', switch_tools=SWITCH_TOOLS, harness_notes=harness_notes('G4'),
                       toolbox_note='14 个默认工具和 6 个要开关的工具。工具只生成带新编号的结果。')
    system = (f'<section class="panel system"><h2>系统流程（G3、G4）</h2><figure><div class="sysscroll">{svg}</div>{gate_caption("G4")}</figure></section>')
    table = ('<main class="hub"><section class="panel"><h2>9 个任务</h2><div class="tablewrap"><table class="hubtable"><thead><tr><th>页面</th><th>任务</th><th>数据</th><th>harness</th>'
             '<th>Agent F1</th><th>固定路线 F1</th><th>固定规则 F1</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
             '<ul><li>粗体是这张图上 3 种做法中最高的 F1。</li><li>T5.2 的对照是固定响应 C。T5.3 的两个对照是固定响应 1 和固定响应 2。T5.2、T5.3、T3.1 没有固定规则。</li>'
             '<li>T3.1 的分数去掉了示例本身的像素。按这个方法计算，G3 的 384 个 T3.1 会话中有 209 个 F1 是 0，中位数是 0。</li>'
             '<li>C4 和 T5.3 的这张图上，两个对照的掩码相同，所以分数相同。</li><li>一张图的分数不代表整个任务。</li></ul></section>'
             '<section class="panel terms">' + terms_block().replace('<details class="terms" open><summary>术语</summary>', '<h2>术语</h2>').replace('</dl></details>', '</dl>') + '</section></main>')
    return page('多任务回放（G3 / G4）', head + entry + system + table)


# ---------------------------------------------------------------- three methods page

def cached_steps(root, image, encode):
    """The fixed route on one LEVIR image, step by step, from the saved SAM3 instances.

    Step 2 (match_instances) was not saved by the fixed route. It is pure arithmetic on the saved instances
    (instances.match, overlap 0.25), so it is recomputed here and checked against the saved fixed-route mask.
    """
    import io
    from agent_baseline import instances
    path = Path(root) / 'sam3_soft/t512s256/val' / (image + '.npz')
    fixed_path = Path(root) / 'init/sam3_imc_p85/val' / (image + '.png')
    if not path.exists():
        return None

    def load():
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
            shape = tuple(map(int, z['shape'])); sets = []
            for time in (0, 1):
                rows = []
                scores, origins, offsets = z[f'scores{time}'], z[f'origins{time}'], z[f'offs{time}']
                pixels, values = z[f'idx{time}'], z[f'val{time}']
                for i in np.flatnonzero(scores >= .5):
                    if origins[i][0] % 512 or origins[i][1] % 512:
                        continue
                    idx = np.sort(pixels[offsets[i]:offsets[i + 1]][values[offsets[i]:offsets[i + 1]] >= round(.85 * 255)]).astype(np.int64)
                    if len(idx):
                        ys, xs = idx // shape[1], idx % shape[1]
                        rows.append(dict(idx=idx, area=int(idx.size), bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
                sets.append(rows)
            return shape, sets
    shape, (t1, t2) = retry_read(load)
    appeared, disappeared, _ = instances.match(t1, t2, 0.25)
    m = lambda rows: instances.to_mask(rows, shape)
    union = m(appeared) | m(disappeared)
    saved = read_mask(fixed_path) if fixed_path.exists() else None
    same = saved is not None and np.array_equal(union, saved)
    pic = lambda mask: encode(Image.fromarray(mask.astype('uint8') * 255))
    return dict(images=[pic(m(t1)), pic(m(t2)), pic(m(appeared)), pic(m(disappeared)), pic(union)],
                counts=[len(t1), len(t2), len(appeared), len(disappeared)], same=bool(same), saved=saved is not None)


def steps_html(d):
    if d is None:
        return '<p>这张图的 SAM3 实例没有保存。</p>'
    n1, n2, na, nd = d['counts']
    captions = [f'1. segment_text(T1)：{n1} 个实例', f'1. segment_text(T2)：{n2} 个实例', f'2. match_instances：出现 {na} 个',
                f'2. match_instances：消失 {nd} 个', '3. to_mask：出现 ∪ 消失 = 固定路线掩码']
    note = ('第 2 步的结果在生成页面时用保存的实例重新计算（纯运算，重叠门限 0.25）。'
            + ('合并后的掩码与保存的固定路线掩码逐像素相同。' if d['same'] else
               '合并后的掩码与保存的固定路线掩码不完全相同。' if d['saved'] else '没有找到保存的固定路线掩码，无法比对。'))
    return '<div class="steps-figs">' + ''.join(figure(u, c) for u, c in zip(d['images'], captions)) + f'</div><p class="muted">{note}</p>'


def numbered_components(data, t2, encode):
    labels = data.get('labels')
    if labels is None:
        return None
    canvas = t2.copy(); w, h = canvas.size
    answers = {r['id']: r.get('answer') for r in data['answers']}
    raw = Image.fromarray((labels > 0).astype('uint8') * 255).resize(canvas.size, Image.Resampling.NEAREST)
    edge = np.asarray(raw.filter(ImageFilter.MaxFilter(3))) != np.asarray(raw.filter(ImageFilter.MinFilter(3)))
    arr = np.array(canvas); arr[edge] = [60, 220, 140]
    removed = np.isin(labels, [k for k, v in answers.items() if v == 'no'])
    small = Image.fromarray(removed.astype('uint8') * 255).resize(canvas.size, Image.Resampling.NEAREST)
    no_edge = np.asarray(small.filter(ImageFilter.MaxFilter(3))) != np.asarray(small.filter(ImageFilter.MinFilter(3)))
    arr[no_edge] = [255, 70, 70]
    canvas = Image.fromarray(arr); draw = ImageDraw.Draw(canvas)
    for ident in sorted(answers):
        yy, xx = np.nonzero(labels == ident)
        if len(xx):
            draw.text((int(xx.mean() * w / labels.shape[1]), int(yy.mean() * h / labels.shape[0])), str(ident), fill='white', stroke_width=1, stroke_fill='black', anchor='mm')
    return [encode(canvas), len(answers), sum(v == 'no' for v in answers.values())]


def render_baselines(root, records, example, example_file, store):
    encode = lambda im, limit=512: jpeg(im, min(limit, 512), 70)
    groups = [('agent', 'Agent', None), ('fixed', '固定路线', list(ROUTES)), ('rule', '固定规则', None)]
    agent_svg = agent_figure('G3', ['T1 较早影像', 'T2 较晚影像', '指令（一句话任务）', '给定掩码、区域或示例', '系统提示和工具定义'],
                             '其他输入', switch_tools=SWITCH_TOOLS, harness_notes=harness_notes('G4'),
                             toolbox_note='14 个默认工具和 6 个要开关的工具。工具只生成带新编号的结果。')
    panels = [f'<div class="route-panel" data-group="agent" id="agent"><div class="routebox"><div class="sysscroll">{agent_svg}</div>'
              '<p>规划器决定调用哪些工具、问哪些组件、是否采纳答案、何时提交。规划器可以补漏，也可以修改组件。</p></div></div>']
    for key, route in ROUTES.items():
        fixed_svg = route_figure(route['title'], route['inputs'], route['steps'], '固定响应掩码' if key in ('t52', 't53') else '固定路线掩码')
        note = ('<p>固定响应是反馈任务的固定路线：程序按反馈的位置直接修改给定掩码，不用规划器。</p>' if key in ('t52', 't53') else '')
        panels.append(f'<div class="route-panel" data-group="fixed" data-route="{key}" id="{key}"><div class="routebox"><div class="sysscroll">{fixed_svg}</div>'
                      + note + ''.join(f'<p class="scores">{esc(line)}</p>' for line in route['scores']) + '</div></div>')
    rows = ''.join(f'<tr><td>{esc(r["label"])}</td><td>{esc(r["rule"].get("question", "与 Agent 相同的问法。"))}</td></tr>' for r in ROUTES.values() if r.get('rule'))
    without = '、'.join(r['label'] for r in ROUTES.values() if not r.get('rule'))
    panels.append(f'<div class="route-panel" data-group="rule" id="rule"><div class="routebox"><div class="sysscroll">{rule_figure()}</div>'
                  '<p>所有任务的固定规则都是这一个结构。不同任务只有两处不同：输入的固定路线掩码，以及验证问题。</p>'
                  f'<div class="tablewrap"><table class="hubtable"><thead><tr><th>任务</th><th>验证问题</th></tr></thead><tbody>{rows}</tbody></table></div>'
                  f'<p class="scores">没有固定规则的任务：{esc(without)}。各任务的分数写在「固定路线」中对应任务的下方。</p></div></div>')
    tabs = '<div class="tabs" role="group" aria-label="做法">' + ''.join(
        f'<button type="button" data-tab="{g}" aria-pressed="{"true" if g == "agent" else "false"}">{esc(label)}</button>' for g, label, _ in groups) + '</div>'
    subtabs = ''.join(f'<div class="subtabs js-only" data-for="{g}" role="group" aria-label="任务" hidden>' + ''.join(
        f'<button type="button" data-route="{k}" aria-pressed="false">{esc(ROUTES[k]["label"])}</button>' for k in keys) + '</div>' for g, _, keys in groups if keys)
    structure = ('<section class="panel example" id="structure"><h2>3 种做法的结构</h2>'
                 '<p>3 种做法使用同一套工具。固定规则和 Agent 使用同一个验证器。差别只在谁做决定。</p>'
                 '<ul><li><strong>固定路线</strong>：程序按固定顺序和固定参数调用工具。不用规划器，不用验证器。不同任务的固定路线不同。</li>'
                 '<li><strong>固定规则</strong>：程序把固定路线的掩码拆成组件，全部组件批量送给验证器，删除答 no 的组件。不用规划器。不补漏。所有任务使用同一个结构。</li>'
                 '<li><strong>Agent</strong>：规划器决定每一步。验证器的答案只作为证据。</li></ul>'
                 '<p>选择「固定路线」后，再选择任务。</p>'
                 + tabs + subtabs + ''.join(panels) + '</section>')

    # One LEVIR image, step by step.
    d = Path(root or '.') / example['run'] / example['task'] / 'val' / example['image']
    from functools import lru_cache

    @lru_cache(None)
    def loaded():
        info, final = read_json(d / 'input.json'), read_json(d / 'final.json')
        t2 = open_image(resolve_asset(d, info['source_images'][1])); t2.thumbnail((512, 512))
        return (load_baselines(root, example, d), read_mask(resolve_asset(d, final['mask_path'])),
                read_mask(Path(info['source_images'][0]).parent.parent / 'label' / (example['image'] + '.png')), t2)
    c = store.get('compare', lambda: comparison_data(dict(example, page='levir_c0'), *loaded(), encode))
    compare = comparison_panel(dict(example, page='levir_c0'), c)
    comp = store.get('components', lambda: numbered_components(loaded()[0], loaded()[3], encode))
    steps = store.get('steps', lambda: cached_steps(root, example['image'], encode))
    compare = compare.replace('<h2>同一张图上的三种做法</h2>', f'<h2>结果：{esc(example["image"])} 上的 3 种做法</h2>')
    rows = ''.join(f'<tr><td>{esc(a["id"])}</td><td>{esc(a["area"])}</td><td>{esc(a["answer"])}</td><td>{esc(a["reason"])}</td></tr>' for a in c['answers'])
    comp_html = '<div class="empty">没有找到组件</div>' if comp is None else figure(
        comp[0], f'固定路线的 {comp[1]} 个组件。红色：验证器答 no，固定规则删除（{comp[2]} 个）。绿色：保留。')
    walk = (f'<section class="panel example"><h2>一张图的分步示例：LEVIR-CD {esc(example["image"])}</h2>'
            f'<p>这张图来自 LEVIR-CD 验证集。Agent 在这张图上的 F1 接近 T1.1 × C0 全部会话的中位数。<a href="{esc(example_file)}">打开 Agent 在这张图上的回放</a>。</p>'
            '<h3>固定路线</h3>' + steps_html(steps) +
            '<h3>固定规则</h3><div class="pairs">' + comp_html +
            f'<details><summary>验证器答案（{len(c["answers"])} 个组件，点击展开）</summary><div class="tablewrap"><table><thead><tr><th>组件</th><th>面积（像素）</th><th>答案</th><th>理由</th></tr></thead><tbody>{rows}</tbody></table></div></details></div></section>')
    head = ('<header><div class="eyebrow"><a href="index.html">回放列表</a></div><h1>3 种做法：固定路线、固定规则、Agent</h1>'
            '<p>这个页面说明 3 种做法的结构，并在一张图上比较它们的结果。</p></header>')
    limits = ('<section class="panel example"><h2>读分数时的限制</h2><ul>'
              '<li>固定路线和固定规则的参数也在验证集上看过。这些分数有选择偏差（第 10.3、10.5、10.6 节）。</li>'
              '<li>一张图的分数不代表整个任务。全部图上的分数写在每条路线的结构图下方。</li>'
              '<li>Agent 的全部图分数是 3 次运行的合并 F1ᶜ。置信区间见 README_ai.md 中对应的章节。</li></ul></section>')
    script = '''<script>(()=>{document.documentElement.classList.add('has-js');
const panels=[...document.querySelectorAll('.route-panel')],tabs=[...document.querySelectorAll('[data-tab]')],subs=[...document.querySelectorAll('.subtabs')];
const first={fixed:'imc',rule:'imc'};let group='agent',route={fixed:'imc',rule:'imc'};
function show(){tabs.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tab===group)));
subs.forEach(s=>{s.hidden=s.dataset.for!==group;s.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.route===route[group])))});
panels.forEach(p=>p.hidden=!(p.dataset.group===group&&(group!=='fixed'||p.dataset.route===route[group])));}
tabs.forEach(b=>b.onclick=()=>{group=b.dataset.tab;show()});
subs.forEach(s=>s.querySelectorAll('button').forEach(b=>b.onclick=()=>{route[s.dataset.for]=b.dataset.route;show()}));
const h=location.hash.slice(1);if(h==='rule'){group='rule';}else if(h&&document.querySelector('.route-panel[data-group=fixed][data-route="'+h+'"]')){group='fixed';route.fixed=h;}
show();
const dialog=document.getElementById('zoom-dialog');document.querySelectorAll('.zoom').forEach(b=>b.addEventListener('click',()=>{const i=b.querySelector('img');document.getElementById('zoom-image').src=i.src;document.getElementById('zoom-caption').textContent=i.alt;dialog.showModal();}));
document.getElementById('close-zoom').onclick=()=>dialog.close();dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});})();</script>'''
    dialog = '<dialog id="zoom-dialog" aria-label="放大图像"><button id="close-zoom" class="close" type="button">关闭（Esc）</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>'
    return page('3 种做法', head + structure + controls_table() + walk + compare + limits + dialog, script)


# ---------------------------------------------------------------- typical errors page

ERROR_SETS = [('LEVIR_C0', 'LEVIR C0'), ('SECOND_C0', 'SECOND C0'), ('SECOND_新建', 'SECOND 新建'), ('SECOND_拆除', 'SECOND 拆除'),
              ('WHU_C0', 'WHU C0'), ('WHU_新建', 'WHU 新建'), ('WHU_拆除', 'WHU 拆除'), ('SECOND_T1.2', 'T1.2 主路线'), ('SECOND_T1.3', 'T1.3 主路线')]
PIXEL_CLASSES = ['粘连误检', '整目标误检', '方向错', '原址重建', '整目标漏检', '部分漏检或边界', '小目标漏检', '大目标漏检']
TRACE_CLASSES = ['兜底提交', '大框失败', '验证器误删 TP', '规划器补进 FP', '主路线选错']


def errors_data(root):
    """Data step: numbers from analysis/error_taxonomy.py and the largest-loss example of each class (JPEG data URIs)."""
    base = Path(root) / 'error_taxonomy'
    summary = read_json(base / 'error_taxonomy.json')
    out = dict(r4=summary.get('r4_comparison'), sets={})
    for key, _ in ERROR_SETS:
        out['sets'][key] = {}
        for method, rec in (summary['summary'].get(key) or {}).items():
            cats = {}
            for name, c in rec.get('categories', {}).items():
                ex = (c.get('examples') or [None])[0]
                pics = None
                if ex:
                    pics = {k: jpeg(open_image(base / ex[k]), 320, 70) for k in ('t2', 'gt', 'overlay') if ex.get(k)}
                    pics.update(image=ex['image'], session=ex.get('session'))
                cats[name] = dict(share=c.get('pixel_share'), denominator=c.get('denominator'), images=c.get('images'),
                                  sessions=c.get('sessions'), repair=c.get('f1_repair_pp'), example=pics)
            out['sets'][key][method] = dict(f1=rec.get('pooled_f1'), images=rec.get('images'), sessions=rec.get('sessions'), categories=cats)
    return out


def render_errors(d):
    tabs = '<div class="tabs" role="group" aria-label="数据集">' + ''.join(
        f'<button type="button" data-set="{k}" aria-pressed="{"true" if i == 0 else "false"}">{esc(label)}</button>' for i, (k, label) in enumerate(ERROR_SETS)) + '</div>'
    panels = []
    for i, (key, label) in enumerate(ERROR_SETS):
        methods = d['sets'].get(key) or {}
        cols = []
        for method, rec in methods.items():
            name = '固定路线' if method == '固定路线' else method.replace('Agent_', 'Agent（') + '）'
            rows, figs = [], []
            for cname in PIXEL_CLASSES + TRACE_CLASSES:
                c = rec['categories'].get(cname)
                if not c:
                    continue
                share = '—' if c['share'] is None else f'{c["share"] * 100:.1f}%'
                den = {'FP': '占 FP', 'FN': '占 FN'}.get(c['denominator'], '占 FP+FN')
                repair = '' if c['repair'] is None else f'{c["repair"]:+.2f}'
                rows.append(f'<tr><td>{esc(cname)}</td><td class="num">{share} <span class="muted">{den}</span></td><td class="num">{esc(c["images"])}</td>'
                            f'<td class="num">{repair}</td></tr>')
                ex = c.get('example')
                if ex and ex.get('overlay'):
                    cap = f'{cname}：{ex["image"]}' + (f'（{ex["session"].split("/")[0]}）' if ex.get('session') else '')
                    figs.append(f'<div class="errfig"><div class="pair">{figure(ex["overlay"], cap + " · 预测与真值")}{figure(ex["gt"], "真值")}</div></div>')
            cols.append(f'<div><h3>{esc(name)} · F1 {rec["f1"]:.2f}</h3><p class="muted">{esc(rec["images"])} 张图，{esc(rec["sessions"])} 个会话</p>'
                        '<div class="tablewrap"><table class="hubtable"><thead><tr><th>类</th><th>像素占比</th><th>涉及图</th><th>理想修复 ΔF1</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div><details><summary>每类损失最大的样例（{len(figs)} 个，点击展开）</summary>{"".join(figs)}</details></div>')
        panels.append(f'<div class="err-panel" data-set="{key}"{"" if i == 0 else " hidden"}><h3>{esc(label)}</h3><div class="trio duo">{"".join(cols)}</div></div>')
    r4 = d.get('r4') or {}
    attach = []
    for key, label in ERROR_SETS[:7]:
        c = ((d['sets'].get(key) or {}).get('固定路线') or {}).get('categories', {}).get('粘连误检')
        if c:
            attach.append(f'{label} {c["share"] * 100:.1f}%')
    head = ('<header><div class="eyebrow"><a href="index.html">回放列表</a></div><h1>典型错误</h1>'
            '<p>离线统计，不调用模型。Agent 合并 3 个种子；固定路线算一次。每个数据集分两栏：固定路线和 Agent。</p></header>')
    intro = ('<section class="panel example"><h2>读这一页之前</h2><ul>'
             f'<li><strong>粘连误检</strong>（FP 所在的连通域里也有 TP）占固定路线 FP 的比例：{esc("；".join(attach))}。'
             f'旧文档写的 LEVIR「54%」是像素门限 0.5 时的整数说法；按门限 0.5 复算是 {r4.get("older_share", 0) * 100:.1f}%，现用门限 0.85 是 {r4.get("current_share", 0) * 100:.1f}%。</li>'
             '<li>像素类：粘连误检和整目标误检互斥，合起来是全部 FP；整目标漏检和部分漏检互斥，合起来是全部 FN。方向错、原址重建、小目标漏检、大目标漏检是附加标签。</li>'
             '<li>轨迹类只对 Agent：兜底提交、大框失败、验证器误删 TP、规划器补进 FP、主路线选错（只 T1.2、T1.3）。</li>'
             '<li>理想修复 ΔF1：只把这一类错误像素改对后的 F1 增量。各类有重叠，不能相加，也不是改 harness 后能拿到的分数。</li>'
             '<li>SECOND C0 的 Agent 只有 G2 的运行；其他建筑任务用 G3。</li></ul></section>')
    concl = ('<section class="panel example"><h2>结论</h2><div class="capgrid"><div><h3>Agent 的典型错误（用来改 harness）</h3><ul>'
             '<li>SECOND 新建、SECOND C0：理想修复最大的是「验证器误删 TP」（2.60、5.33 分）。</li>'
             '<li>LEVIR、SECOND 拆除、WHU 新建：最大的是「规划器补进 FP」（1.47、5.09、2.64 分）。</li>'
             '<li>WHU C0、WHU 拆除：最大的是「兜底提交」（2.98、3.18 分），会话数少但单个损失大。</li>'
             '<li>T1.2、T1.3：「主路线选错」仍是最大来源。G4 把 T1.2 涉及的会话从 293 降到 90，T1.3 从 184 降到 115。</li></ul></div>'
             '<div><h3>固定路线的典型错误（Agent 有机会拿分的地方）</h3><ul>'
             '<li>SECOND、WHU：整目标漏检占 FN 的 46%–87%，大目标漏检占 FN 的 10%–41%。补漏是主要机会。</li>'
             '<li>WHU C0、新建：粘连误检占 FP 的 79%–80%，误检大多和真变化连在一起（边界或相连的伪变化）。</li>'
             '<li>SECOND：整目标误检占 FP 的 79%–93%，验证器删除这些组件是规则胜过固定路线的原因。</li></ul></div></div></section>')
    script = """<script>(()=>{document.documentElement.classList.add('has-js');const b=[...document.querySelectorAll('[data-set]')].filter(x=>x.tagName==='BUTTON'),p=[...document.querySelectorAll('.err-panel')];
b.forEach(x=>x.onclick=()=>{b.forEach(y=>y.setAttribute('aria-pressed',String(y===x)));p.forEach(q=>q.hidden=q.dataset.set!==x.dataset.set)});
const dialog=document.getElementById('zoom-dialog');document.querySelectorAll('.zoom').forEach(z=>z.addEventListener('click',()=>{const i=z.querySelector('img');document.getElementById('zoom-image').src=i.src;document.getElementById('zoom-caption').textContent=i.alt;dialog.showModal();}));
document.getElementById('close-zoom').onclick=()=>dialog.close();})();</script>"""
    dialog = '<dialog id="zoom-dialog" aria-label="放大图像"><button id="close-zoom" class="close" type="button">关闭（Esc）</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>'
    body = head + intro + '<section class="panel example"><h2>按数据集、按方法</h2>' + tabs + ''.join(panels) + '</section>' + concl + dialog
    return page('典型错误', body, script)


def build(root, out, selections, cache, stage='all'):
    """stage 'data': compute and cache (compute node). 'page': HTML from the cache only (seconds). 'all': both."""
    cache, out = Path(cache), Path(out)
    stages = ['data', 'page'] if stage == 'all' else [stage]
    for current in stages:
        mode = 'build' if current == 'data' else 'read'
        if current == 'data':
            from agent_baseline.pick_demo import select_all
            records = read_json(selections) if selections else select_all(root)
            example = select_example(root, records)
            (cache).mkdir(parents=True, exist_ok=True)
        else:
            meta = read_json(cache / 'records.json')
            records, example = meta['records'], meta['example']
            out.mkdir(parents=True, exist_ok=True)
        for record in records:
            record['file'] = record['page'] + '.html'
            store = Store(cache / f'{record["page"]}.json', mode)
            document, info = render_replay(root, record, store)
            store.save()
            record.update(version=info['version'], scores=info['scores'])
            if current == 'page':
                size = audit_html(document)['bytes']
                if size > 3_000_000:
                    raise ValueError(record['file'] + ' 超过 3 MB')
                (out / record['file']).write_text(document, encoding='utf-8')
                print(f'{record["file"]}: {record["image"]} {info["version"]} ' + ' '.join(f'{k}={v["f1"] * 100:.2f}' for k, v in info['scores'].items()) + f' {size} bytes', flush=True)
        example_file = next((r['file'] for r in records if r['image'] == example['image'] and r['run'] == example['run']), 'baseline_example.html')
        if example_file == 'baseline_example.html':
            store = Store(cache / 'baseline_example.json', mode)
            document, _ = render_replay(root, dict(example, page='levir_c0'), store)
            store.save()
            if current == 'page':
                (out / example_file).write_text(document, encoding='utf-8')
        store = Store(cache / 'baselines.json', mode)
        baselines = render_baselines(root, records, example, example_file, store)
        store.save()
        store = Store(cache / 'errors.json', mode)
        errors = render_errors(store.get('errors', lambda: errors_data(root)))
        store.save()
        if current == 'data':
            (cache / 'records.json').write_text(json.dumps(dict(records=records, example=example), ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'数据已缓存：{cache}')
            continue
        for name, document in (('index.html', render_index(records)), ('baselines.html', baselines), ('errors.html', errors)):
            audit_html(document)
            (out / name).write_text(document, encoding='utf-8')
        keep = ['page', 'task', 'dataset', 'run', 'image', 'f1', 'median', 'version']
        (out / 'selection.json').write_text(json.dumps([dict({k: r.get(k) for k in keep}, scores={k: v['f1'] for k, v in r['scores'].items()}) for r in records],
                                                       ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'总量 {sum(p.stat().st_size for p in out.glob("*") if p.is_file())} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, help='Run directory with the traces (data step only).')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True, help='Directory of the cached page data.')
    parser.add_argument('--stage', choices=['all', 'data', 'page'], default='all',
                        help='data: compute node, reads the traces. page: HTML from the cache only; seconds, no server data.')
    parser.add_argument('--selections', type=Path, help='Saved pick_demo --typical output; without it, the selection is recomputed.')
    args = parser.parse_args()
    if args.stage != 'page' and args.run_root is None:
        parser.error('--run-root is required for the data step')
    build(args.run_root, args.out_dir, args.selections, args.cache, args.stage)


if __name__ == '__main__':
    main()
