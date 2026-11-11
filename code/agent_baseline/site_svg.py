"""The Agent system figure of the G1 replay pages (results/agent_baseline/demo_g1), as a function.

With the G1 defaults it reproduces the hand-edited G1 figure byte for byte (tests/test_site_svg.py).
Later harness versions reuse the same layout: a different fourth input chip, an extra tool group
for the switch-enabled tools, and one line per harness change at the bottom.
"""
from html import escape


def esc(value):
    return escape(str(value), quote=True)


# (category, subtitle, rows of (subgroup, tools)); the column x positions and widths are those of the G1 figure.
# 读取 tools only look; everything that runs a model or an operation, including segmentation, matching and
# features, is 执行. 事件触发 starts the verifier or the submission gate.
GROUPS = [
    ('读取工具', '读取影像或已有结果，不执行操作', [('查看', ['view', 'inspect'])]),
    ('执行工具', '执行操作，产生新的实例、掩码或特征',
     [('分割', ['segment_text', 'segment_prompt']), ('比较', ['match_instances']),
      ('运算', ['select', 'to_mask', 'combine', 'morph']), ('修改', ['remove_region', 'add_region', 'refine_region'])]),
    ('事件触发工具', '启动验证器或提交门', [('验证', ['ask']), ('结束', ['submit'])]),
]
COLUMNS = [(372, 140), (518, 116), (640, 104), (750, 66)]
VERIFIER_TOOLS = {'ask', 'submit'}
# Tools offered only with the --tools-t13 switch (T1.2, T1.3, T3). They join the 执行 group in extra rows and
# get a dashed outline.
EXEC_ROWS_WITH_SWITCH = [('分割', ['segment_text', 'segment_prompt']), ('', ['segment_everything', 'segment_semantic']),
                         ('比较', ['match_instances', 'match_exemplar', 'classify']),
                         ('特征', ['extract_features', 'compare_features']),
                         ('运算', ['select', 'to_mask', 'combine', 'morph']),
                         ('修改', ['remove_region', 'add_region', 'refine_region'])]
SWITCH_TOOLS = {'segment_everything', 'segment_semantic', 'match_exemplar', 'classify', 'extract_features', 'compare_features'}
WIDE_COLUMNS = [(372, 140), (518, 130), (654, 90)]


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
    groups = GROUPS
    if switch_tools:
        rows = [(cat, [n for n in names if n not in SWITCH_TOOLS or n in switch_tools]) for cat, names in EXEC_ROWS_WITH_SWITCH]
        groups = [GROUPS[0], (GROUPS[1][0], GROUPS[1][1], [r for r in rows if r[1]])] + GROUPS[2:]
    extra = sum(30 * len(rows) + 34 for _, _, rows in groups) - 312 + (20 if switch_tools else 0)
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
    ends, ys = {}, {}

    def tool(x, width, y, name, kind):
        count = (counts or {}).get(name, 0)
        s.rect(x, y, width, 24, f'tool tool-{kind}' + (' sw' if name in SWITCH_TOOLS else '') + (' off' if session and not count else ''))
        ys[name] = y
        s.text(x + 7, y + 17, name, 'mono')
        if count:
            s.text(x + width - 6, y + 17, f'×{count}', 'mono', 'end')
        ends[name] = x + width

    y = 440
    for title, note, rows in groups:
        height_g = 30 * len(rows) + 28
        s.rect(316, y, 508, height_g, 'group')
        s.text(328, y + 18, title, 'small strong')
        s.text(428, y + 18, note, 'label')
        for r, (cat, names) in enumerate(rows):
            ty = y + 26 + 30 * r
            s.text(328, ty + 17, cat, 'cat')
            columns = WIDE_COLUMNS if switch_tools and title == '执行工具' and cat in ('', '特征') else COLUMNS
            if cat == '比较' and len(names) == 3:
                columns = [(372, 140), (518, 116), (640, 104)]
            for (x, width), name in zip(columns, names):
                tool(x, width, ty, name, 'ver' if name in VERIFIER_TOOLS else 'plan')
        y += height_g + 6
    s.text(320, y + 17, 'segment_text 使用 SAM3。segment_prompt 使用 SAM ViT-H。', 'label')
    if switch_tools:
        y += 20
        s.text(320, y + 17, '虚线框：默认关闭，用 --tools-t13 打开（SAM ViT-H、DINOv2、SegEarth-OV）。', 'label')
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
    s.line([(ends['ask'], ys['ask'] + 12), (900, ys['ask'] + 12), (900, 270), (960, 270)], 'ver')
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
    s.line([(ends['submit'], ys['submit'] + 12), (960, ys['submit'] + 12)], 'ver')
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


def _figure_start(s, title, width, height, label):
    s.bits.append(f'<svg class="sys route" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label or title)}"><defs>')
    for name in ('main', 'plan', 'ver', 'mute'):
        s.bits.append(f'<marker id="r-arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      f'<path class="head-{name}" d="M0 0 10 5 0 10z"/></marker>')
    s.bits.append('</defs>')
    s.text(20, 40, title, 'title')


def _edge(s, points, kind, dashed=False):
    path = ' '.join(f'{x},{y}' for x, y in points)
    s.bits.append(f'<polyline class="edge edge-{kind}{" dash" if dashed else ""}" points="{path}" marker-end="url(#r-arrow-{kind})"/>')


def route_figure(title, inputs, steps, output='固定路线掩码', label=None):
    """One fixed route: input -> numbered steps (tool chips and text) -> output. No planner, no verifier."""
    s = Svg()
    box_w, gap, x0, line_h = 250, 44, 232, 22
    block = lambda tools, lines: 46 + 30 * len(tools) + line_h * len(lines) + 10
    row_h = max([block(t, l) for _, t, l in steps] + [44 + 46 * len(inputs), 130])
    top = 70
    mid = top + row_h // 2
    width = max(x0 + len(steps) * (box_w + gap) + 210, 1100)
    _figure_start(s, title, width, top + row_h + 30, label)
    s.rect(20, top, 180, row_h)
    s.text(36, top + 28, '输入', 'head')
    for n, item in enumerate(inputs):
        s.rect(32, top + 44 + 46 * n, 156, 36, 'chip')
        s.text(110, top + 67 + 46 * n, item, 'chiptext', 'middle')
    x = x0
    _edge(s, [(200, mid), (x, mid)], 'main')
    for n, (head, tools, lines) in enumerate(steps):
        s.rect(x, top, box_w, row_h, 'box plan')
        s.text(x + 14, top + 26, f'{n + 1}. {head}', 'head')
        y = top + 40
        for tool in tools:
            s.rect(x + 14, y, box_w - 28, 24, 'tool tool-plan')
            s.text(x + 21, y + 17, tool, 'mono')
            y += 30
        for line in lines:
            s.text(x + 14, y + 16, line, 'body')
            y += line_h
        x += box_w
        _edge(s, [(x, mid), (x + gap, mid)], 'main')
        x += gap
    oy = mid - 55
    s.rect(x, oy, 190, 110)
    s.text(x + 16, oy + 28, '输出', 'head')
    s.text(x + 16, oy + 56, output, 'body')
    s.text(x + 16, oy + 80, '不用规划器。', 'small muted')
    s.text(x + 16, oy + 100, '不用验证器。', 'small muted')
    s.bits.append('</svg>')
    return ''.join(s.bits)


def rule_figure():
    """The fixed rule, one structure for every task: the fixed-route mask -> components -> all of them to
    the verifier in one batch (one independent request per component) -> program decision -> output."""
    s = Svg()
    width, height, top, h = 1540, 400, 70, 290
    mid = top + h // 2
    _figure_start(s, '固定规则：在固定路线掩码上批量验证组件，删除答 no 的组件', width, height, None)
    # Input: the fixed-route mask of the task.
    s.rect(20, top, 200, h)
    s.text(36, top + 28, '输入', 'head')
    for n, item in enumerate(['固定路线掩码', 'T1、T2 影像（裁剪用）', '任务的验证问题']):
        s.rect(32, top + 50 + 50 * n, 176, 38, 'chip')
        s.text(120, top + 74 + 50 * n, item, 'chiptext', 'middle')
    _edge(s, [(220, mid), (264, mid)], 'main')
    # Split into components.
    s.rect(264, top, 220, h, 'box plan')
    s.text(280, top + 28, '1. 拆成组件', 'head')
    for k, line in enumerate(('8 连通组件。', '每个组件一个编号。', '组件 1、2 … N。')):
        s.text(280, top + 60 + 24 * k, line, 'body')
    _edge(s, [(484, mid), (528, mid)], 'main')
    # The batch: a stack of component cards.
    s.rect(528, top, 250, h, 'box plan')
    s.text(544, top + 28, '2. 批量送出', 'head')
    for k in range(4):
        y = top + 52 + 34 * k
        s.rect(552 + 6 * k, y, 180, 28, 'tool tool-plan')
        s.text(562 + 6 * k, y + 19, f'组件 {k + 1}' if k < 3 else '组件 N', 'mono')
    s.text(544, top + 220, '全部 N 个组件一起送出。', 'body')
    s.text(544, top + 244, '每个组件一次独立请求，', 'body')
    s.text(544, top + 268, '请求并发执行。', 'body')
    for k in range(3):
        _edge(s, [(778, mid - 30 + 30 * k), (822, mid - 30 + 30 * k)], 'ver')
    # The verifier: the same one the Agent uses.
    s.rect(822, top, 290, h, 'box ver')
    s.text(838, top + 28, '3. 验证器 Verifier', 'head')
    s.text(1098, top + 28, 'flash · L1', 'mono', 'end')
    for k, line in enumerate(('与 Agent 使用同一个验证器和问法。', '输入：一个组件的 4 张裁剪图。', 'T1、T2 各一张近景和一张远景。',
                              '红线标出这个组件。', '输出：yes 或 no、变化类型、理由。')):
        s.text(838, top + 60 + 24 * k, line, 'body')
    _edge(s, [(1112, mid), (1156, mid)], 'ver')
    # Program decision and output.
    s.rect(1156, top, 190, h, 'box')
    s.text(1172, top + 28, '4. 程序判断', 'head')
    for k, line in enumerate(('答 no：删除组件。', '答 yes：保留组件。', '无法解析：保留组件。', '只删除，不补漏。')):
        s.text(1172, top + 60 + 24 * k, line, 'body')
    _edge(s, [(1346, mid), (1370, mid)], 'main')
    s.rect(1370, mid - 55, 150, 110)
    s.text(1386, mid - 27, '输出', 'head')
    s.text(1386, mid + 1, '固定规则掩码', 'body')
    s.text(1386, mid + 25, '不用规划器。', 'small muted')
    s.text(20, height - 12, '规划器不参与。程序不看图，只按验证器的答案删除组件。', 'label')
    s.bits.append('</svg>')
    return ''.join(s.bits)
