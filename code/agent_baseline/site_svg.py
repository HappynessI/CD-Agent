"""The Agent system figure of the G1 replay pages (results/agent_baseline/demo_g1), as a function.

With the G1 defaults it reproduces the hand-edited G1 figure byte for byte (tests/test_site_svg.py).
Later harness versions reuse the same layout: a different fourth input chip, an extra tool group
for the switch-enabled tools, and one line per harness change at the bottom.
"""
from html import escape


def esc(value):
    return escape(str(value), quote=True)


# (category, tools); the column x positions and widths are those of the G1 figure.
GROUPS = [
    ('感知工具', '读取信息，不修改当前掩码', [('查看', ['view', 'inspect']), ('分割', ['segment_text', 'segment_prompt']), ('比较', ['match_instances'])]),
    ('执行工具', '筛选实例，或生成新掩码', [('运算', ['select', 'to_mask', 'combine', 'morph']), ('修改', ['remove_region', 'add_region', 'refine_region'])]),
    ('事件触发工具', '启动验证器或提交门', [('验证', ['ask']), ('结束', ['submit'])]),
]
COLUMNS = [(372, 140), (518, 116), (640, 104), (750, 66)]
VERIFIER_TOOLS = {'ask', 'submit'}
# Tools offered only with a command-line switch (T1.2/T1.3/T3); shown as a fourth group when a session has them.
SWITCH_GROUP = ('开关工具', '--tools-t13 打开。SAM ViT-H、DINOv2、SegEarth-OV', [
    ('分割', ['segment_everything', 'segment_semantic']),
    ('特征', ['extract_features', 'compare_features']),
    ('比较', ['classify', 'match_exemplar']),
])
SWITCH_COLUMNS = [(372, 160), (538, 150)]


class Svg:
    def __init__(self):
        self.bits = []

    def rect(self, x, y, w, h, cls='box'):
        self.bits.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>')

    def text(self, x, y, value, cls='', anchor='start'):
        self.bits.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(value)}</text>')

    def line(self, points, kind, dashed=False):
        path = ' '.join(f'{x},{y}' for x, y in points)
        self.bits.append(f'<polyline class="edge edge-{kind}{" dash" if dashed else ""}" points="{path}" marker-end="url(#arrow-{kind})"/>')


def agent_figure(version='G1', inputs=None, input_label='影像和掩码', counts=None, answers=None, holds=None, ending=None,
                 switch_tools=(), harness_notes=None, toolbox_note=None):
    """counts=None draws the entry-page figure; a dict draws one session (tools with 0 calls are dimmed).

    inputs: the chip texts of the input box (default: the five G1 chips).
    switch_tools: switch-enabled tools offered in this session; they get a fourth tool group.
    harness_notes: lines at the bottom of the frame (default: the G1 line).
    """
    session = counts is not None
    inputs = inputs or ['T1 较早影像', 'T2 较晚影像', '指令（一句话任务）', '给定掩码（仅 T5.1）', '系统提示和工具定义']
    notes = harness_notes if harness_notes is not None else ['G1 不启用：plan、python、skill、L2 仔细验证。L1 是快速验证。']
    extra = 30 * len(SWITCH_GROUP[2]) + 28 + 6 if switch_tools else 0   # height of the fourth tool group
    box_bottom = 826 + extra                     # bottom of the toolbox
    note_top = box_bottom + 20
    frame_h = note_top + 22 * (len(notes) - 1) + 10 - 16
    height = frame_h + 32
    s = Svg()
    s.bits.append(f'<svg class="sys" viewBox="0 0 1540 {height}" role="img" aria-label="{version} 系统流程图。输入进入规划器和工具箱。规划器调用工具箱。'
                  '验证器与规划器分离，只接收裁剪图，只返回答案。提交门处理 submit。检查通过后，系统输出最终掩码。真值只用于事后评分。"><defs>')
    for name in ('main', 'plan', 'ver', 'mute'):
        s.bits.append(f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      f'<path class="head-{name}" d="M0 0 10 5 0 10z"/></marker>')
    s.bits.append('</defs>')
    s.rect(280, 16, 970, frame_h, 'frame')
    s.text(300, 46, 'Agent 系统：规划器与验证器分离', 'title')

    # Input: the chips spread evenly between y=130 and y=386.
    s.rect(20, 84, 180, 376)
    s.text(36, 112, '输入', 'head')
    step = 256 / (len(inputs) - 1)
    for n, item in enumerate(inputs):
        y = round(130 + step * n)
        s.rect(32, y, 156, 42, 'chip')
        s.text(110, y + 26, item, 'chiptext', 'middle')
    s.line([(200, 209), (300, 209)], 'main')
    s.text(240, 199, '第一条消息', 'label', 'middle')
    s.line([(200, 430), (300, 430)], 'main')
    s.text(240, 420, input_label, 'label', 'middle')

    # Planner, with the toolbox directly below it.
    s.rect(300, 84, 540, 250, 'box plan')
    s.text(320, 114, '规划器 Planner', 'head')
    s.text(820, 114, 'qwen3.7-flash', 'mono', 'end')
    for y, line in zip((152, 186, 220, 254, 288), ('输入：系统提示、指令、工具返回内容、状态行、最近的图像。',
                                                   '每步写一句理由，并发出 1–8 个工具调用。', '每个会话最多 30 步。',
                                                   '决定保留或删除哪些组件，决定何时提交。', '读取验证器答案后，可以采纳答案，也可以不采纳。')):
        s.text(320, y, line, 'body')
    s.line([(440, 334), (440, 400)], 'plan')
    s.text(432, 372, '工具调用（JSON）', 'label', 'end')
    s.line([(660, 400), (660, 334)], 'plan')
    s.text(668, 372, '返回内容、图像、状态行', 'label')

    total = 14 + len(switch_tools)
    s.rect(300, 400, 540, box_bottom - 400)
    s.bits[-1] = s.bits[-1].replace('class="box"', 'class="box plan"')
    s.text(320, 428, '工具箱 Toolbox', 'head')
    s.text(820, 428, toolbox_note or f'{total} 个工具。工具只生成带新编号的结果，不修改已有结果。', 'label', 'end')
    ends = {}

    def tool(x, width, y, name, kind):
        count = (counts or {}).get(name, 0)
        s.rect(x, y, width, 24, f'tool tool-{kind}' + (' off' if session and not count else ''))
        s.text(x + 7, y + 17, name, 'mono')
        if count:
            s.text(x + width - 6, y + 17, f'×{count}', 'mono', 'end')
        ends[name] = x + width

    y = 440
    for title, note, rows in GROUPS:
        height_g = 30 * len(rows) + 28
        s.rect(316, y, 508, height_g, 'group')
        s.text(328, y + 18, title, 'small strong')
        s.text(428, y + 18, note, 'label')
        for r, (cat, names) in enumerate(rows):
            ty = y + 26 + 30 * r
            s.text(328, ty + 17, cat, 'cat')
            for (x, width), name in zip(COLUMNS, names):
                tool(x, width, ty, name, 'ver' if name in VERIFIER_TOOLS else 'plan')
        y += height_g + 6
    if switch_tools:
        title, note, rows = SWITCH_GROUP
        s.rect(316, y, 508, 30 * len(rows) + 28, 'group')
        s.text(328, y + 18, title, 'small strong')
        s.text(428, y + 18, note, 'label')
        for r, (cat, names) in enumerate(rows):
            ty = y + 26 + 30 * r
            s.text(328, ty + 17, cat, 'cat')
            for (x, width), name in zip(SWITCH_COLUMNS, names):
                if name in switch_tools:
                    tool(x, width, ty, name, 'plan')
        y += 30 * len(rows) + 28 + 6
    s.text(320, y + 17, 'segment_text 使用 SAM3。segment_prompt 使用 SAM ViT-H。', 'label')
    s.rect(316, y + 30, 508, 30, 'chip')
    s.text(328, y + 50, '工作状态：结果编号（mask#k、inst#k）、当前掩码、验证记录', 'label')

    # Verifier.
    s.rect(960, 84, 270, 250, 'box ver')
    s.text(980, 114, '验证器 Verifier', 'head')
    s.text(1214, 114, 'flash · L1', 'mono', 'end')
    for y2, line in zip((142, 165, 188, 211, 234, 257, 280), ('每个组件发送一次独立请求。', '输入：4 张裁剪图。', 'T1、T2 各一张近景和一张远景。',
                                                              '红线标出被验证的组件。', '不接收规划器的理由和对话历史。', '没有工具。', '输出：yes 或 no、变化类型、理由。')):
        s.text(980, y2, line, 'body')
    if session:
        s.text(980, 322, f'这个会话的答案数：{answers}', 'small strong')
    s.line([(960, 150), (840, 150)], 'ver')
    s.text(900, 140, '答案和报告', 'label', 'middle')
    s.text(900, 170, '不自动删除组件', 'label strong', 'middle')
    s.line([(ends['ask'], 696), (900, 696), (900, 270), (960, 270)], 'ver')
    s.text(908, 400, 'ask', 'label mono')
    s.text(908, 420, '规划器按需请求验证', 'label')

    # Submission gate.
    s.rect(960, 600, 270, 224, 'box ver')
    s.text(980, 630, '提交门 Gate', 'head')
    s.text(1214, 630, 'harness', 'mono', 'end')
    for y2, line in zip((662, 690, 718), ('收到 submit 后判断两个条件：', '条件 1：这是第一次 submit。', '条件 2：掩码含未验证组件。')):
        s.text(980, y2, line, 'body')
    s.text(980, 750, '不读取真值，不修改掩码。', 'small muted')
    if session:
        s.text(980, 782, f'这个会话暂停提交 {holds} 次', 'small strong')
    s.line([(ends['submit'], 726), (960, 726)], 'ver')
    s.line([(1150, 600), (1150, 334)], 'ver')
    s.text(1142, 450, '任一条件成立时：', 'label', 'end')
    s.text(1142, 470, '自动验证并返回报告', 'label', 'end')
    s.line([(1230, 660), (1330, 660)], 'ver')
    s.text(1290, 650, '都不成立', 'label', 'middle')
    s.text(1290, 680, '提交', 'label', 'middle')

    # Output, and the scoring the model never sees.
    s.rect(1330, 600, 190, 112)
    s.text(1346, 628, '输出', 'head')
    s.text(1346, 654, '最终二值变化掩码', 'body')
    s.text(1346, 676, '决定说明 summary', 'body')
    if session:
        s.text(1346, 700, f'结束方式：{ending}', 'small strong')
    s.line([(1425, 712), (1425, 742)], 'mute', True)
    s.text(1433, 732, '掩码', 'label')
    s.rect(1330, 742, 190, 82, 'box dashbox')
    s.text(1346, 768, '事后评分', 'head')
    s.text(1346, 792, '只有这里使用真值。', 'body')
    s.text(1346, 814, '模型看不到分数。', 'body')
    s.line([(780, 84), (780, 66), (1425, 66), (1425, 600)], 'mute', True)
    s.text(1415, 92, '不经过提交门：', 'label', 'end')
    s.text(1415, 112, '连续 3 步没有工具调用', 'label', 'end')
    s.text(1415, 132, '或第 30 步结束', 'label', 'end')
    for n, line in enumerate(notes):
        s.text(300, note_top + 22 * n, line, 'label')
    s.bits.append('</svg>')
    return ''.join(s.bits)


VERIFIER_LINES = ('每个组件发送一次独立请求。', '输入：4 张裁剪图。', '红线标出被验证的组件。', '与 Agent 使用同一个验证器和问法。',
                  '输出：yes 或 no、变化类型、理由。')


def route_figure(title, inputs, steps, output='固定路线掩码', rule=None, label=None):
    """A fixed route (row 1) and, when rule is given, the deletion rule on its components (row 2).

    steps: [(head, tools, lines)]; tools are drawn as tool chips, lines as body text.
    rule: None, or dict(question=..., decide=[...], output=...).
    Classes are the G1 figure classes, so the page CSS styles both figures alike.
    """
    s = Svg()
    box_w, gap, x0 = 250, 44, 232
    line_h = 22

    def block_height(tools, lines):
        return 46 + 30 * len(tools) + line_h * len(lines) + 10

    row1_h = max([block_height(t, l) for _, t, l in steps] + [44 + 46 * len(inputs), 130])
    row2_h = 0
    if rule:
        row2_h = max(block_height([], VERIFIER_LINES + (rule.get('question', ''),)), block_height([], rule['decide']), 150)
    top1, top2 = 70, 70 + row1_h + 70
    height = top2 + (row2_h + 30 if rule else 0) + 20
    width = x0 + len(steps) * (box_w + gap) + 220
    width = max(width, 1100)
    s.bits.append(f'<svg class="sys route" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><defs>')
    for name in ('main', 'plan', 'ver', 'mute'):
        s.bits.append(f'<marker id="r-arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      f'<path class="head-{name}" d="M0 0 10 5 0 10z"/></marker>')
    s.bits.append('</defs>')

    def edge(points, kind, dashed=False):
        path = ' '.join(f'{x},{y}' for x, y in points)
        s.bits.append(f'<polyline class="edge edge-{kind}{" dash" if dashed else ""}" points="{path}" marker-end="url(#r-arrow-{kind})"/>')

    s.text(20, 40, title, 'title')
    s.rect(20, top1, 180, row1_h)
    s.text(36, top1 + 28, '输入', 'head')
    for n, item in enumerate(inputs):
        s.rect(32, top1 + 44 + 46 * n, 156, 36, 'chip')
        s.text(110, top1 + 67 + 46 * n, item, 'chiptext', 'middle')
    x = x0
    edge([(200, top1 + row1_h // 2), (x, top1 + row1_h // 2)], 'main')
    for n, (head, tools, lines) in enumerate(steps):
        s.rect(x, top1, box_w, row1_h, 'box plan')
        s.text(x + 14, top1 + 26, f'{n + 1}. {head}', 'head')
        y = top1 + 40
        for tool in tools:
            s.rect(x + 14, y, box_w - 28, 24, 'tool tool-plan')
            s.text(x + 21, y + 17, tool, 'mono')
            y += 30
        for line in lines:
            s.text(x + 14, y + 16, line, 'body')
            y += line_h
        x += box_w
        edge([(x, top1 + row1_h // 2), (x + gap, top1 + row1_h // 2)], 'main')
        x += gap
    out_x = x
    s.rect(out_x, top1, 190, 110)
    s.text(out_x + 16, top1 + 28, '输出', 'head')
    s.text(out_x + 16, top1 + 56, output, 'body')
    s.text(out_x + 16, top1 + 80, '不用规划器。', 'small muted')
    s.text(out_x + 16, top1 + 100, '不用验证器。', 'small muted')
    if rule:
        # Row 2: components -> the same verifier -> program decision -> output.
        mid = top2 + row2_h // 2
        edge([(out_x + 95, top1 + 110), (out_x + 95, top1 + row1_h + 30), (110, top1 + row1_h + 30), (110, top2)], 'main')
        s.text(130, top1 + row1_h + 22, '固定路线掩码', 'label')
        s.rect(20, top2, 180, row2_h)
        s.text(36, top2 + 28, '拆成组件', 'head')
        for k, line in enumerate(('8 连通组件。', '每个组件', '问一次验证器。')):
            s.text(36, top2 + 56 + 22 * k, line, 'body')
        edge([(200, mid), (x0, mid)], 'main')
        s.rect(x0, top2, box_w + 90, row2_h, 'box ver')
        s.text(x0 + 14, top2 + 26, '验证器 Verifier', 'head')
        s.text(x0 + box_w + 76, top2 + 26, 'flash · L1', 'mono', 'end')
        y = top2 + 40
        for line in VERIFIER_LINES + ((rule['question'],) if rule.get('question') else ()):
            s.text(x0 + 14, y + 16, line, 'body')
            y += line_h
        x = x0 + box_w + 90
        edge([(x, mid), (x + gap, mid)], 'ver')
        x += gap
        s.rect(x, top2, box_w + 40, row2_h, 'box')
        s.text(x + 14, top2 + 26, '程序判断', 'head')
        y = top2 + 40
        for line in rule['decide']:
            s.text(x + 14, y + 16, line, 'body')
            y += line_h
        x += box_w + 40
        edge([(x, mid), (x + gap, mid)], 'main')
        x += gap
        s.rect(x, top2, 190, 96)
        s.text(x + 16, top2 + 28, '输出', 'head')
        s.text(x + 16, top2 + 56, rule.get('output', '固定规则掩码'), 'body')
        s.text(x + 16, top2 + 80, '不用规划器。不补漏。', 'small muted')
    if label:
        s.text(20, height - 8, label, 'label')
    s.bits.append('</svg>')
    return ''.join(s.bits)
