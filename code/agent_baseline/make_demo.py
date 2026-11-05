#!/usr/bin/env python3
"""Render recorder traces as a self-contained, offline Chinese replay page."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import html
import io
import json
from pathlib import Path
import sys

# Works both in this staging directory and after installation beside recorder.py.
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent.parent
if not (CODE / 'agent_baseline').is_dir():
    CODE = Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code')
sys.path.insert(0, str(CODE))
import agent_baseline  # Makes the existing script-style common module importable.
from common import load_gt, load_mask_png, scores
import numpy as np
from PIL import Image, ImageFilter

CATEGORIES = {
    'view': '看', 'inspect': '看', 'segment_text': '分割', 'segment_prompt': '分割',
    'match_instances': '比较', 'ask': '核验', 'select': '运算', 'to_mask': '运算',
    'combine': '运算', 'morph': '运算', 'remove_region': '修改', 'add_region': '修改',
    'refine_region': '修改', 'submit': '结束',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_steps(directory):
    path = Path(directory) / 'steps.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return '未记录' if value is None else f'{value * 100:.2f}%'


def metric_f1(metric):
    return (metric or {}).get('f1')


def resolve_asset(directory, value):
    path = Path(value)
    # Prefer the archive when a session has been copied to a new location.
    if path.is_absolute() and path.parent.name in ('images', 'masks'):
        local = directory / path.parent.name / path.name
        if local.exists():
            return local
    return path if path.is_absolute() else directory / path


def mask_path(directory, handle):
    return directory / 'masks' / f'mask_{handle.split("#")[1]}.png'


def jpeg(image, limit=512, quality=80):
    image = image.copy().convert('RGB')
    image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality, subsampling=0)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def figure(uri, label, extra=''):
    return (f'<figure class="{extra}"><button type="button" class="zoom" aria-label="放大：{esc(label)}">'
            f'<img src="{uri}" alt="{esc(label)}" loading="lazy"></button>'
            f'<figcaption>{esc(label)}</figcaption></figure>')


def text_block(value, title=None, threshold=420):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    body = f'<pre>{esc(text)}</pre>'
    if title is not None or len(text) > threshold:
        label = title or f'返回原文（{len(text)} 字符，展开）'
        return f'<details><summary>{esc(label)}</summary>{body}</details>'
    return body


def delta_label(metric):
    value = (metric or {}).get('delta_f1')
    if value is None:
        return '<span class="muted">首次候选，无前值可比</span>'
    cls = 'up' if value > 0 else 'down' if value < 0 else 'muted'
    return f'<span class="{cls}">ΔF1 {value * 100:+.2f} 个百分点</span>'


def summary_text(summary, stem, final_f1):
    if summary is None:
        return '<p class="muted">未提供同配置整集汇总；整集 F1ᶜ、逐图平均 F1 和相对排名未知。</p>'
    count = summary.get('images')
    count_label = str(count) if count is not None else '数量未记录的'
    line = (f'同配置 {count_label} 张：整集 F1ᶜ {pct(summary.get("f1"))} · '
            f'逐图平均 F1 {pct(summary.get("mean_image_f1"))}')
    per = summary.get('per_image_f1', {})
    if per and final_f1 is not None:
        rank = 1 + sum(v > final_f1 for v in per.values())
        line += f' · 本图第 {rank}/{len(per)}（F1 从高到低，同分并列）'
    else:
        line += ' · 本图排名未知（缺逐图指标）'
    if final_f1 is not None and summary.get('f1') is not None:
        line += f' · 较整集 F1ᶜ {(final_f1 - summary["f1"]) * 100:+.2f} 个百分点'
    if count != 64:
        line += '。此汇总未确认覆盖完整 64 张。'
    if per and stem + '.png' not in per and stem not in per:
        line += ' 本图不在汇总列表中，排名仅为插入位置。'
    return f'<p>{esc(line)}</p>'


def fixed_text(fixed):
    if fixed is None:
        return ''
    # Native fixed_routes JSON, or a mapping of route name -> native metrics.
    rows = [(fixed.get('name', '固定流程'), fixed)] if 'f1' in fixed else list(fixed.items())
    entries = [f'{name}：F1ᶜ {pct(row.get("f1"))}' for name, row in rows if isinstance(row, dict) and 'f1' in row]
    if not entries:
        raise ValueError('--fixed 需要 fixed_routes 输出，或 {对照名称: {f1: 0..1}}')
    return '<p>不用 Agent 的对照 · ' + esc('；'.join(entries)) + '</p>'


TOOL_ROWS = [('看', ['view', 'inspect']), ('分割', ['segment_text', 'segment_prompt']), ('比较', ['match_instances']),
             ('运算', ['select', 'to_mask', 'combine', 'morph']), ('修改', ['remove_region', 'add_region', 'refine_region']),
             ('核验', ['ask']), ('结束', ['submit'])]
TOOL_NOTE = {'segment_text': ' · SAM3', 'segment_prompt': ' · SAM ViT-H'}


def system_figure(task=None, counts=None, stops=0, reports=None, answers=None, ending=None):
    """The G1 system as one left-to-right flow; with a session's numbers when they are given."""
    session = counts is not None
    bits = ['<svg class="sys" viewBox="0 0 1336 700" role="img" aria-label="G1 系统流程：输入进入规划器，工具箱在规划器正下方；'
            '验证器与规划器分离，只收裁图、只回答案；提交门拦截 submit，通过后输出掩码；真值只用于事后评分。"><defs>']
    for name in ('main', 'plan', 'ver', 'gate', 'mute'):
        bits.append(f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                    f'<path class="head-{name}" d="M0 0 10 5 0 10z"/></marker>')
    bits.append('</defs>')

    def box(x, y, w, h, cls='box'):
        bits.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>')

    def text(x, y, value, cls='', anchor='start'):
        bits.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(value)}</text>')

    def link(points, kind, dashed=False):
        path = ' '.join(f'{x},{y}' for x, y in points)
        bits.append(f'<polyline class="edge edge-{kind}{" dash" if dashed else ""}" points="{path}" marker-end="url(#arrow-{kind})"/>')

    # The system boundary first, so everything else draws over it.
    box(226, 16, 884, 652, 'frame')
    text(250, 45, 'Agentic system：规划与验证解耦', 'title')

    # Input.
    box(16, 78, 170, 270)
    text(32, 103, '输入', 'head')
    given = '给定掩码（仅 T5.1）' if task is None else '给定掩码 mask#0' if task == 'T5.1' else 'T1.1 无给定掩码'
    items = ['T1 较早影像', 'T2 较晚影像', '指令（一句话任务）', given, '系统提示和工具定义']
    for n, item in enumerate(items):
        y = 116 + 44 * n
        off = task is not None and task != 'T5.1' and n == 3
        box(28, y, 146, 36, 'chip off' if off else 'chip')
        text(101, y + 23, item, 'small', 'middle')
    link([(186, 145), (250, 145)], 'main')
    text(218, 136, '首条消息', 'label', 'middle')
    link([(186, 322), (250, 322)], 'main')
    text(218, 313, '两期影像' if task and task != 'T5.1' else '影像、掩码', 'label', 'middle')

    # Planner, with the toolbox directly below it.
    box(250, 78, 470, 134, 'box plan')
    text(270, 106, '规划器 Planner', 'head')
    text(700, 106, 'qwen3.7-flash', 'mono', 'end')
    for y, line in zip((130, 152, 174, 196), ('读对话：系统提示、指令、工具返回、状态行、最近的图',
                                              '每轮写一句理由，发 1–8 个工具调用，最多 30 次回复',
                                              '决定留哪些组件、删哪些、何时提交',
                                              '读到验证器的答案后，可以采纳，也可以反对')):
        text(270, y, line, 'body')
    link([(400, 212), (400, 282)], 'plan')
    text(392, 251, '工具调用（JSON）', 'label', 'end')
    link([(560, 282), (560, 212)], 'plan')
    text(568, 251, '结果、图、句柄、状态行', 'label')

    box(250, 282, 470, 328, 'box plan')
    text(270, 308, '工具箱 Toolbox', 'head')
    text(700, 308, '14 个工具；工具不改旧句柄，只出新句柄', 'label', 'end')
    ends = {}
    for row, (group, tools) in enumerate(TOOL_ROWS):
        y = 322 + 34 * row
        text(270, y + 18, group, 'small muted')
        x = 314
        for tool in tools:
            count = (counts or {}).get(tool, 0)
            label = tool + TOOL_NOTE.get(tool, '') + (f' ×{count}' if count else '')
            width = round(len(label) * 6.9 + 14)
            kind = 'ver' if tool == 'ask' else 'gate' if tool == 'submit' else 'plan'
            box(x, y, width, 26, f'tool tool-{kind}' + (' off' if session and not count else ''))
            text(x + 7, y + 17, label, 'mono')
            ends[tool] = x + width
            x += width + 8
    box(266, 562, 438, 36, 'chip')
    text(278, 585, '工作状态：句柄 mask#k、inst#k · 当前掩码 · 核验账本（轮廓 → 全部回答）', 'label')

    # Verifier, on the far side of the decoupling boundary.
    box(850, 78, 240, 222, 'box ver')
    text(870, 104, '独立验证器 Verifier', 'head')
    text(870, 124, 'qwen3.7-flash · L1', 'mono')
    for y, line in zip((148, 170, 190, 212, 234, 256), ('每个组件一次全新请求', '只看 4 张裁图：T1、T2 的', '近景和远景，红线圈出组件',
                                                        '不知道规划器的理由', '没有工具，没有对话历史', '回答 yes/no、类型、一句理由')):
        text(870, y, line, 'body')
    if session:
        text(870, 284, f'本次回答 {answers} 次', 'small strong')
    bits.append('<line class="boundary" x1="838" y1="70" x2="838" y2="318"/>')
    text(846, 322, '解耦边界：只传裁图和答案', 'label strong')
    link([(850, 130), (720, 130)], 'ver')
    text(785, 122, '答案、提交前报告', 'label', 'middle')
    text(785, 148, '不自动删除', 'label strong', 'middle')
    link([(ends['ask'], 505), (790, 505), (790, 230), (850, 230)], 'ver')
    text(782, 330, 'ask', 'label mono', 'end')
    text(782, 346, '按需核验', 'label', 'end')

    # Submission gate.
    box(850, 450, 240, 160, 'box gate')
    text(870, 476, '提交门 Gate', 'head')
    text(1074, 476, 'harness', 'mono', 'end')
    for y, line in zip((500, 522, 544), ('拦下 submit，判断两件事：', '· 这是第一次提交？', '· 掩码里有未核验组件？')):
        text(870, y, line, 'body')
    text(870, 570, '不读真值，不改掩码', 'small muted')
    if session:
        text(870, 594, f'本次拦截 {reports} 次', 'small strong')
    link([(ends['submit'], 539), (850, 539)], 'gate')
    text(785, 531, 'submit', 'label mono', 'middle')
    link([(1040, 450), (1040, 300)], 'ver')
    text(1032, 385, '是：自动核验并报告', 'label', 'end')
    link([(1090, 500), (1148, 500)], 'gate')
    text(1119, 491, '否：结束', 'label', 'middle')

    # Output and the scoring that the model never sees.
    box(1148, 410, 172, 130, 'box')
    text(1164, 435, '输出', 'head')
    text(1164, 460, '最终二值变化掩码', 'body')
    text(1164, 482, '决定说明 summary', 'body')
    if session:
        text(1164, 512, f'结束方式：{ending}', 'small strong')
    link([(1234, 540), (1234, 590)], 'mute', True)
    text(1242, 570, '掩码', 'label')
    box(1148, 590, 172, 94, 'box dashbox')
    text(1164, 615, '事后评分', 'head')
    text(1164, 638, '真值只在这里出现', 'body')
    text(1164, 660, '模型运行时看不到', 'body')

    # Fallback that bypasses the gate.
    link([(650, 78), (650, 62), (1234, 62), (1234, 410)], 'mute', True)
    text(660, 54, '没调 submit 或回复用完：用当前掩码兜底', 'label')
    text(250, 650, 'G1 未开启：plan、python、skill、L2 核验。', 'label')
    bits.append('</svg>')
    notes = ['第一次 submit 一定被拦下并得到报告，这不是拒绝；之后只有提交的掩码含未核验组件才再次拦下。',
             '每次拦下时，验证器核验这个掩码的未核验组件，以及已被去掉、还没有答案的给定组件。答案只作证据，由规划器决定删不删。',
             '三种情况下答案到不了规划器：第 30 次回复直接提交；模型停下不再调工具；再次提交的掩码没有未核验组件，但又去掉了没问过的给定组件。这些组件在结束后补问，答案只进账本。']
    if session:
        notes.append(f'这次运行结束后补问 {stops} 次。' + ('这些答案没有参与决定。' if stops else ''))
    caption = '<figcaption><strong>规划和验证是两个独立模型，跨过边界的只有裁图和答案；工具箱只听规划器调用。</strong>' + ''.join(f'<br>{esc(n)}' for n in notes) + '</figcaption>'
    return '<section class="panel system"><h2>系统流程（G1）</h2><figure><div class="sysscroll">' + ''.join(bits) + '</div>' + caption + '</figure></section>'


def ledger_section(directory):
    """All verifier answers of the session: what was asked, what it said, what stayed."""
    path = directory / 'verification.json'
    if not path.exists():
        return ''
    entries = read_json(path)
    rows = []
    against = 0
    for entry in sorted(entries, key=lambda e: -e.get('area', 0)):
        records = entry.get('records', [])
        latest = entry.get('latest')
        kept = entry.get('kept')
        conflict = (latest == 'no' and kept) or (latest == 'yes' and not kept)
        against += conflict
        last = records[-1] if records else {}
        score = last.get('p_yes')
        rows.append(f'<tr{" class=against" if conflict else ""}><td>{esc(entry.get("area"))}</td><td>{esc(entry.get("bbox"))}</td>'
                    f'<td>{len(records)}</td><td>{esc(latest or "无有效答案")}</td><td>{esc(last.get("type") or "")}</td>'
                    f'<td>{"" if score is None else f"{score:.2f}"}</td><td>{"留" if kept else "去"}</td><td>{esc(last.get("reason") or "")}</td></tr>')
    summary = f'验证器账本：{len(entries)} 个问过的轮廓，{against} 个最终决定和最后答案不同（标色）'
    return (f'<details class="ledger"><summary>{esc(summary)}</summary><p class="muted">单位是问过的轮廓，含临时候选和已删区域，不等于最终组件数。'
            'p_yes 是模型输出的读数，没有校准。</p><div class="tablewrap"><table><thead><tr><th>面积</th><th>框</th><th>回答次数</th><th>最后答案</th><th>类型</th><th>p_yes</th><th>最终</th><th>最后理由</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div></details>')


def after_stop_count(directory):
    path = directory / 'messages.jsonl'
    if not path.exists():
        return 0
    with path.open(encoding='utf-8') as handle:
        return sum('"final-stage"' in line for line in handle)


def report_text(call):
    try:
        return json.loads(call.get('result_text') or '').get('report')
    except (ValueError, AttributeError):
        return None


def answers_list(call):
    """One line per verifier answer, instead of a long JSON block."""
    try:
        results = json.loads(call.get('result_text') or '').get('results') or []
    except (ValueError, AttributeError):
        return ''
    rows = [f'<li><span class="{"down" if r.get("answer") == "no" else "up" if r.get("answer") == "yes" else "muted"}">{esc(r.get("answer") or "未解析")}</span> '
            f'组件 {esc(r.get("id"))}{"" if r.get("p_yes") is None else " · p_yes " + format(r["p_yes"], ".2f")}：{esc(r.get("reason") or "")}</li>' for r in results if isinstance(r, dict)]
    return f'<ul class="answers">{"".join(rows)}</ul>' if rows else ''


CSS = r'''
:root{color-scheme:light;--bg:#f4f5f7;--panel:#fff;--text:#172530;--muted:#596975;--line:#cbd3da;--accent:#185b91;--tint:#edf4fa;--up:#187040;--down:#ad3440;--warn:#855418;--warnbg:#fff4df;--tp:#2cce78;--fp:#ef5050;--fn:#458dff;--backdrop:#0009}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#111920;--panel:#19232c;--text:#e5edf4;--muted:#a8b7c3;--line:#3b4b58;--accent:#8ac4f4;--tint:#243544;--up:#79d79a;--down:#ff9b9f;--warn:#f1c77d;--warnbg:#382f21}}
:root[data-theme=dark]{color-scheme:dark;--bg:#111920;--panel:#19232c;--text:#e5edf4;--muted:#a8b7c3;--line:#3b4b58;--accent:#8ac4f4;--tint:#243544;--up:#79d79a;--down:#ff9b9f;--warn:#f1c77d;--warnbg:#382f21}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;overflow-wrap:anywhere}
header,footer{padding:18px 24px;background:var(--panel)}header{border-bottom:1px solid var(--line)}h1{font-size:23px;line-height:1.3;margin:5px 0 10px}h2{font-size:17px;margin:0 0 10px}h3{font-size:15px;margin:0}p{margin:7px 0}.eyebrow,.muted,figcaption{color:var(--muted)}.warning{font-weight:700;color:var(--warn);background:var(--warnbg);padding:8px 12px;border-left:3px solid var(--warn)}
.headtop{display:flex;justify-content:space-between;gap:12px;align-items:start}.metrics{display:flex;flex-wrap:wrap;gap:8px 25px;margin:12px 0}.metric strong{display:block;font-size:21px;line-height:1.25;font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(180px,.8fr) minmax(320px,1.55fr) minmax(240px,1fr);gap:16px;padding:18px 24px;max-width:1800px;margin:auto}.layout>*{min-width:0}.panel,.step{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px}.step{margin-bottom:12px;scroll-margin-top:16px}.step.active{border-color:var(--accent);border-left:4px solid var(--accent);padding-left:9px}.stephead{display:flex;align-items:center;justify-content:space-between;gap:10px}.thought{border-left:2px solid var(--line);padding-left:10px;white-space:pre-wrap;margin:12px 0}.call{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}.call.error{border-left:3px solid var(--down);padding-left:10px}.errorlabel,.down{color:var(--down)}.up{color:var(--up)}.badge{background:var(--tint);color:var(--accent);padding:2px 6px;border-radius:3px;margin-right:6px}.toolname{font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;margin:8px 0}details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}
figure{margin:10px 0;min-width:0}figcaption{font-size:12px;margin-top:4px}img{display:block;width:100%;height:auto;max-width:100%;border-radius:3px}button,select{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:5px 9px;cursor:pointer;max-width:100%}button:hover{border-color:var(--accent)}button:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.zoom{display:block;padding:0;border:0;width:100%;cursor:zoom-in}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.thumb{max-width:180px}.sticky{position:sticky;top:12px}.sticky figure img{max-height:27vh;object-fit:contain;background:var(--bg)}.legend{display:flex;gap:12px;font-size:12px;flex-wrap:wrap}.legend i{display:inline-block;width:9px;height:9px;margin-right:4px}.legend .tp{background:var(--tp)}.legend .fp{background:var(--fp)}.legend .fn{background:var(--fn)}.chart{width:100%;display:block;color:var(--muted)}.chart text{fill:currentColor;font-size:10px}.chart .axis{stroke:var(--line)}.chart .trace{fill:none;stroke:var(--accent);stroke-width:2}.chart .point{fill:var(--panel);stroke:var(--accent);stroke-width:2;cursor:pointer}.chart .point.active{fill:var(--accent);stroke-width:3}.chart .cursor{stroke:var(--muted);stroke-dasharray:3 3}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.empty{padding:25px 10px;text-align:center;color:var(--muted);border:1px dashed var(--line);margin:12px 0}.js-only{display:none}.has-js .js-only{display:initial}[hidden]{display:none!important}footer{border-top:1px solid var(--line)}footer pre{max-width:1200px}dialog{border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);max-width:95vw;max-height:95vh;padding:14px}dialog::backdrop{background:var(--backdrop)}dialog img{max-width:85vw;max-height:80vh;object-fit:contain;width:auto}dialog .close{float:right;margin-bottom:10px}
@media(max-width:900px){.layout{grid-template-columns:1fr;padding:16px}.sticky{position:static}.sticky figure img{max-height:none}.inputs .pair{grid-template-columns:1fr 1fr}header,footer{padding:16px}.headtop{flex-wrap:wrap}}
.system{margin:18px 24px 0;max-width:1752px}.sysscroll{overflow-x:auto}.sys{display:block;width:100%;min-width:1040px;height:auto;color:var(--text)}.sys text{fill:currentColor;font-size:13px;paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round}.sys .title{font-size:16px;font-weight:700}.sys .head{font-size:15px;font-weight:700}.sys .body{font-size:13px}.sys .small{font-size:12.5px}.sys .label{font-size:12px;fill:var(--muted)}.sys .muted{fill:var(--muted)}.sys .strong{font-weight:700;fill:var(--text)}.sys .mono{font:11.5px ui-monospace,SFMono-Regular,Consolas,monospace}.sys rect{fill:var(--panel);stroke:var(--line);stroke-width:1.2}.sys .frame{fill:none;stroke-width:1.5}.sys .plan{stroke:var(--accent);stroke-width:2}.sys .ver{stroke:var(--up);stroke-width:2}.sys .gate{stroke:var(--warn);stroke-width:2}.sys .dashbox{stroke-dasharray:5 4}.sys .chip{fill:var(--tint);stroke:none}.sys .tool{fill:var(--tint);stroke:none}.sys .tool-ver{stroke:var(--up);stroke-width:1.5}.sys .tool-gate{stroke:var(--warn);stroke-width:1.5}.sys .off{opacity:.45}.sys .edge{fill:none;stroke-width:1.6}.sys .edge-main{stroke:var(--text)}.sys .edge-plan{stroke:var(--accent)}.sys .edge-ver{stroke:var(--up)}.sys .edge-gate{stroke:var(--warn)}.sys .edge-mute{stroke:var(--muted)}.sys .dash{stroke-dasharray:6 5}.sys .boundary{stroke:var(--up);stroke-width:1.4;stroke-dasharray:3 4}.sys .head-main{fill:var(--text)}.sys .head-plan{fill:var(--accent)}.sys .head-ver{fill:var(--up)}.sys .head-gate{fill:var(--warn)}.sys .head-mute{fill:var(--muted)}.system figure{margin:0}.system figcaption{font-size:13px;margin-top:8px;color:var(--text)}.tablewrap{overflow-x:auto}.ledger table{border-collapse:collapse;width:100%;font-size:12px}.ledger th,.ledger td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top;white-space:nowrap}.ledger td:last-child{white-space:normal;min-width:320px}.ledger .against{background:var(--warnbg)}.answers{margin:8px 0;padding-left:18px;font-size:13px}.reportbox{white-space:pre-wrap;background:var(--warnbg);border-left:3px solid var(--warn);padding:8px 10px;margin:8px 0;font-size:13px}.hub{max-width:1752px;margin:auto;padding:18px 24px}.hubtable{border-collapse:collapse;width:100%}.hubtable th,.hubtable td{border-bottom:1px solid var(--line);padding:8px;text-align:left}a{color:var(--accent)}
@media(max-width:900px){.system{margin:12px 16px 0}.hub{padding:16px}}
@media(prefers-reduced-motion:no-preference){.step{transition:border-color .15s}}
'''

JS = r'''
(() => {
  document.documentElement.classList.add('has-js');
  const states = JSON.parse(document.getElementById('states').textContent);
  const cards = [...document.querySelectorAll('.step')];
  const points = [...document.querySelectorAll('.point')];
  const panels = [...document.querySelectorAll('[data-mask-panel]')];
  let current = states.length - 1;
  function choose(index, scroll=false) {
    current = Math.max(0, Math.min(index, states.length - 1));
    const state = states[current];
    cards.forEach(c => c.classList.toggle('active', Number(c.dataset.index) === current));
    points.forEach(p => { const active = Number(p.dataset.index) === current;
      p.classList.toggle('active', active); p.setAttribute('aria-pressed', String(active)); });
    panels.forEach(p => p.hidden = p.dataset.maskPanel !== (state.mask || 'none'));
    document.getElementById('current-label').textContent = state.label;
    document.getElementById('current-metric').textContent = state.metric;
    document.getElementById('prev').disabled = current === 0;
    document.getElementById('next').disabled = current === states.length - 1;
    const cursor = document.getElementById('chart-cursor');
    cursor.setAttribute('x1', state.x); cursor.setAttribute('x2', state.x);
    if (scroll) cards.find(c => Number(c.dataset.index) === current)?.scrollIntoView({block:'nearest'});
  }
  cards.forEach(c => c.addEventListener('click', e => {
    if (!e.target.closest('details, .zoom, a, input, select')) choose(Number(c.dataset.index));
  }));
  document.querySelectorAll('[data-select]').forEach(b => b.addEventListener('click', () => choose(Number(b.dataset.select), true)));
  points.forEach(p => {
    p.addEventListener('click', () => choose(Number(p.dataset.index), true));
    p.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault(); choose(Number(p.dataset.index), true);
    }});
  });
  document.getElementById('prev').onclick = () => choose(current - 1, true);
  document.getElementById('next').onclick = () => choose(current + 1, true);
  document.addEventListener('keydown', e => {
    if (document.getElementById('zoom-dialog').open || e.target.closest('input,textarea,select')) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault(); choose(current + (e.key === 'ArrowRight' ? 1 : -1), true);
    }
  });
  const dialog = document.getElementById('zoom-dialog');
  document.querySelectorAll('.zoom').forEach(b => b.addEventListener('click', () => {
    const image = b.querySelector('img');
    document.getElementById('zoom-image').src = image.src;
    document.getElementById('zoom-image').alt = image.alt;
    document.getElementById('zoom-caption').textContent = image.alt;
    dialog.showModal();
  }));
  document.getElementById('close-zoom').onclick = () => dialog.close();
  dialog.addEventListener('click', e => {if (e.target === dialog) dialog.close();});
  document.getElementById('theme').onchange = e => {
    if (e.target.value === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = e.target.value;
  };
  document.getElementById('theme').value = document.documentElement.dataset.theme || 'auto';
  choose(current);
})();
'''


def chart(states, rows):
    maximum = max([r['step'] for r in rows] + [1])
    x = lambda step: 34 + 274 * step / maximum
    bits = ['<svg class="chart" viewBox="0 0 330 144" role="group" aria-label="逐步 F1 曲线，点可点击或用回车选择">',
            '<text x="5" y="12">F1</text>']
    for y, label in [(22, '100%'), (68, '50%'), (114, '0%')]:
        bits.append(f'<line class="axis" x1="34" x2="308" y1="{y}" y2="{y}"/><text x="1" y="{y + 4}">{label}</text>')
    segment = []
    for i, row in enumerate(rows):
        value = metric_f1(row.get('eval')) if row.get('current_mask') else None
        states[i]['x'] = x(row['step'])
        if value is None:
            if segment:
                bits.append('<polyline class="trace" points="' + ' '.join(segment) + '"/>')
            segment = []
        else:
            segment.append(f'{x(row["step"]):.2f},{114 - 92 * value:.2f}')
    if segment:
        bits.append('<polyline class="trace" points="' + ' '.join(segment) + '"/>')
    for i, row in enumerate(rows):
        value = metric_f1(row.get('eval')) if row.get('current_mask') else None
        if value is not None:
            label = f'第 {row["step"]} 步，F1 {pct(value)}'
            bits.append(f'<circle class="point" data-index="{i}" tabindex="0" role="button" aria-label="{label}" cx="{x(row["step"]):.2f}" cy="{114 - 92 * value:.2f}" r="4"><title>{label}</title></circle>')
    if len(states) > len(rows):
        states[-1]['x'] = x(maximum)
    for step in sorted({0, maximum // 2, maximum}):
        bits.append(f'<text text-anchor="middle" x="{x(step):.2f}" y="132">{step}</text>')
    bits += ['<text x="285" y="143">步数</text>', '<line id="chart-cursor" class="cursor" x1="308" x2="308" y1="17" y2="117"/>', '</svg>']
    return ''.join(bits)


def render(run_dir, split, config_summary=None, fixed=None, note=''):
    directory = Path(run_dir).resolve()
    info, final = read_json(directory / 'input.json'), read_json(directory / 'final.json')
    rows = read_steps(directory)
    task = final.get('task_id', info.get('task_id', ''))
    name = directory.name
    gt = load_gt(split, name + '.png')
    sources = [Image.open(resolve_asset(directory, p)).convert('RGB') for p in info['source_images']]
    t2 = sources[1].copy()
    t2.thumbnail((512, 512), Image.Resampling.LANCZOS)
    gt_small = np.asarray(Image.fromarray(gt).resize(t2.size, Image.Resampling.NEAREST))
    masks, thumbs, panels = {}, {}, {}

    def mask_assets(handle):
        if handle not in masks:
            mask = load_mask_png(mask_path(directory, handle))
            masks[handle] = mask
            thumbs[handle] = jpeg(Image.fromarray(mask.astype('uint8') * 255))
            small = Image.fromarray(mask.astype('uint8') * 255).resize(t2.size, Image.Resampling.NEAREST)
            pred = np.asarray(small) > 127
            # A two-sided contour remains visible at the presentation resolution.
            edge = np.asarray(small.filter(ImageFilter.MaxFilter(3))) != np.asarray(small.filter(ImageFilter.MinFilter(3)))
            overlay = np.array(t2)
            overlay[edge] = [239, 80, 80]
            compare = (np.array(t2).astype(float) * .22).astype('uint8')
            compare[pred & gt_small] = [44, 206, 120]
            compare[pred & ~gt_small] = [239, 80, 80]
            compare[~pred & gt_small] = [69, 141, 255]
            panels[handle] = (figure(jpeg(Image.fromarray(overlay)), 'T2 + 当前掩码红色轮廓') +
                              figure(jpeg(Image.fromarray(compare)), '与真值对比 · 运行时模型不可见'))
        return thumbs[handle]

    title = 'T5.1 修正已有掩码' if task == 'T5.1' else 'T1.1 从零生成'
    final_metric = final.get('eval') or {}
    pr = scores(final_metric) if all(k in final_metric for k in ('tp', 'fp', 'fn')) else {}
    final_f1 = metric_f1(final_metric)
    initial_f1 = metric_f1(final.get('initial_eval'))
    rounds = sum(r['step'] > 0 for r in rows)
    calls = [c for r in rows for c in r.get('calls', [])]
    stats = [('最终 F1', pct(final_f1)), ('IoU', pct(final_metric.get('iou'))),
             ('精确率 P', pct(final.get('precision', pr.get('precision')))),
             ('召回率 R', pct(final.get('recall', pr.get('recall')))),
             ('步数', str(final.get('planning_rounds', rounds))),
             ('工具调用', str(final.get('tool_calls', len(calls)))),
             ('耗时', f'{final["elapsed_seconds"]:.1f} 秒' if final.get('elapsed_seconds') is not None else '未记录'),
             ('费用', f'¥{final["cost_cny"]:.4f}' if final.get('cost_cny') is not None else '未记录')]
    head = f'<header><div class="headtop"><div><div class="eyebrow"><a href="index.html">回放列表</a> · 变化检测 · Agent 轨迹展示 / {esc(split)} / {esc(name)}</div><h1>{title}</h1></div><label class="js-only">外观 <select id="theme"><option value="auto">跟随系统</option><option value="light">浅色</option><option value="dark">深色</option></select></label></div>'
    head += '<div class="warning">这是挑出来的样例，不代表平均水平</div>'
    head += '<div class="metrics">' + ''.join(f'<div class="metric"><strong>{esc(v)}</strong><span>{k}</span></div>' for k, v in stats) + '</div>'
    if task == 'T5.1':
        head += f'<p><strong>初始 F1 {pct(initial_f1)} → 最终 F1 {pct(final_f1)}</strong></p>'
    head += summary_text(config_summary, name, final_f1) + fixed_text(fixed)
    head += f'<details open><summary>指令原文</summary>{text_block(info.get("instruction", "未记录"), threshold=100000)}</details>'
    if note:
        head += f'<p>展示备注：{esc(note)}</p>'
    head += '</header>'

    left = '<section class="inputs panel"><h2>输入和参照</h2><div class="pair">'
    left += ''.join(figure(jpeg(im), label) for im, label in zip(sources, ['T1 · 较早影像', 'T2 · 较晚影像'])) + '</div>'
    left += figure(jpeg(Image.fromarray(gt.astype('uint8') * 255)), '真值 · 运行时模型不可见')
    initial_handle = info.get('initial_handle')
    if task == 'T5.1':
        if initial_handle and mask_path(directory, initial_handle).exists():
            left += figure(mask_assets(initial_handle), '初始掩码')
        elif info.get('init_mask_path') and resolve_asset(directory, info['init_mask_path']).exists():
            left += figure(jpeg(Image.open(resolve_asset(directory, info['init_mask_path']))), '初始掩码')
        else:
            left += '<p class="muted">初始掩码未归档。</p>'
    left += '<p class="muted">图像均为缩小展示；点击图像可放大。真值及所有 F1 仅供事后评估。</p></section>'

    timeline = ['<section aria-label="步骤时间线"><h2>Agent 做了什么</h2><p class="muted">按记录顺序回放。点卡片切步；← / → 切换。第 0 步是初始化。</p><noscript><p>JavaScript 未启用：时间线和最终结果仍可阅读。</p></noscript>']
    states = []
    for i, row in enumerate(rows):
        step, handle = row['step'], row.get('current_mask')
        value = metric_f1(row.get('eval')) if handle else None
        label = '第 0 步 · 初始化' if step == 0 else f'第 {step} 步'
        states.append(dict(mask=handle, label=label, metric=f'当前 F1 {pct(value)}' if handle else '尚未产生掩码'))
        body = f'<article class="step" data-index="{i}" data-step="{step}" id="step-{step}"><div class="stephead"><h3>{label}</h3><button class="js-only" type="button" data-select="{i}">查看此步</button></div>'
        thought = (row.get('planner') or {}).get('thought')
        body += f'<div class="thought">{esc(thought)}</div>' if thought else '<p class="muted">初始化（无模型发言）</p>' if step == 0 else '<p class="muted">本步未记录模型发言。</p>'
        if step == 0:
            initial_views = [directory / 'images' / f'view_0_{time}.jpg' for time in ('t1', 't2')]
            if all(p.exists() for p in initial_views):
                body += '<details><summary>模型收到的初始图像（归档缩小版）</summary><div class="pair">' + ''.join(figure(jpeg(Image.open(p), 640), f'初始 T{k + 1}') for k, p in enumerate(initial_views)) + '</div></details>'
        for call in row.get('calls', []):
            tool = call['tool']
            failed = not call.get('ok', False)
            held = call.get('status') == 'report'
            badge = '提交门' if held else CATEGORIES.get(tool, '其他')
            body += f'<div class="call{" error" if failed else ""}"><span class="badge">{badge}</span><span class="toolname">{esc(tool)}</span>'
            if held:
                body += '<span class="muted"> · 被提交门拦下，返回报告（不是拒绝）</span>'
            if failed:
                body += f'<span class="errorlabel"> · 调用失败 {esc(call.get("error_code") or "")}</span>'
            if call.get('status') == 'skipped_due_to_submit':
                body += '<span class="muted"> · 提交后跳过</span>'
            body += text_block(call.get('args', call.get('raw_arguments', {})), '调用参数')
            report = report_text(call) if held else None
            if report:
                body += f'<div class="reportbox">{esc(report)}</div>'
            else:
                if tool == 'ask':
                    body += answers_list(call)
                body += '<div class="muted">返回摘要 / 原文</div>' + text_block(call.get('result_text', '未记录'))
            if tool == 'view' and call.get('result_images'):
                body += '<div class="pair">'
                for j, path in enumerate(call['result_images']):
                    body += figure(jpeg(Image.open(resolve_asset(directory, path)), 640), f'Agent 实际看到的 T{j + 1}（归档缩小版）')
                body += '</div>'
            before, after = call.get('current_before'), call.get('current_after')
            if before != after:
                if after:
                    body += f'<p>当前掩码：{esc(before or "无")} → {esc(after)} · F1 {pct(metric_f1(call.get("eval")))}<br>{delta_label(call.get("eval"))}</p>'
                    body += figure(mask_assets(after), '本次更新后的掩码', 'thumb')
                else:
                    body += '<p class="muted">当前候选已清空：尚未产生掩码。</p>'
            body += '</div>'
        if handle:
            mask_assets(handle)
            body += f'<p class="muted">步末当前掩码 {esc(handle)} · F1 {pct(value)}</p>'
        else:
            body += '<p class="muted">尚未产生掩码（未设置当前候选；工具的中间产物仍见上方返回）。</p>'
        timeline.append(body + '</article>')
    final_handle = final.get('final_handle')
    if final_handle:
        mask_assets(final_handle)
    if not states or states[-1]['mask'] != final_handle:
        states.append(dict(mask=final_handle, label='最终提交（独立于步末当前候选）', metric=f'最终 F1 {pct(final_f1)}'))
    svg = chart(states, rows)
    timeline.append('</section>')

    right = '<aside><div class="sticky panel"><h2>掩码与结果</h2><div class="controls js-only"><button id="prev" type="button" aria-label="上一步">←</button><button id="next" type="button" aria-label="下一步">→</button>'
    right += f'<button type="button" data-select="{len(states)-1}">最终结果</button></div>'
    right += f'<p id="current-label">{esc(states[-1]["label"])}</p><p id="current-metric">{esc(states[-1]["metric"])}</p>'
    right += '<div data-mask-panel="none"' + (' hidden' if final_handle else '') + '><div class="empty">尚未产生掩码</div></div>'
    for handle, panel in panels.items():
        right += f'<div data-mask-panel="{esc(handle)}"' + ('' if handle == final_handle else ' hidden') + '>' + panel + '</div>'
    right += '<div class="legend"><span><i class="tp"></i>TP 命中</span><span><i class="fp"></i>FP 误检</span><span><i class="fn"></i>FN 漏检</span></div>'
    right += svg + '<p class="muted">仅绘制已记录的当前候选 F1；空缺不按 0 分处理。图像缩小不参与指标计算。</p></div></aside>'
    counts = Counter(c['tool'] for c in calls)
    errors = Counter(c['tool'] for c in calls if not c.get('ok', False))
    footer = '<footer><h2>最终提交</h2>'
    footer += f'<p>提交来源：{esc(final.get("submitted_by", "未记录"))} · 结束原因：{esc(final.get("termination_reason", "未记录"))} · summary 来源：{esc(final.get("summary_source", "未记录"))}</p>'
    footer += text_block(final.get('summary', '未记录'), threshold=100000)
    footer += ledger_section(directory)
    footer += f'<p>变化目标代理个数：{esc(final.get("change_component_count", "未记录"))} · 变化面积：{esc(final.get("change_area_px", "未记录"))} 像素。<strong>8 连通域是代理，不等于栋数。</strong></p>'
    footer += '<p>工具调用统计（含失败 / 跳过）：' + esc('；'.join(f'{k} × {v}（失败 / 跳过 {errors[k]}）' for k, v in sorted(counts.items()))) + '</p>'
    footer += '<p>F1 由记录器算，Agent 全程看不到；真值也不参与规划。空真值且空预测的逐图 F1 按现有评测约定为 1。展示图为 JPEG 缩小版，面积及指标取原分辨率记录。</p></footer>'
    state_json = json.dumps(states, ensure_ascii=False).replace('<', '\\u003c')
    answers = sum(len(entry.get('records', [])) for entry in read_json(directory / 'verification.json')) if (directory / 'verification.json').exists() else 0
    system = system_figure(task, counts, after_stop_count(directory), sum(c.get('status') == 'report' for c in calls), answers,
                           final.get('termination_reason', '未记录'))
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} · {esc(name)}</title><style>{CSS}</style></head><body>' + head + system + '<main class="layout">' + left + ''.join(timeline) + right + '</main>' + footer +
            '<dialog id="zoom-dialog" aria-label="放大图像"><button id="close-zoom" class="close" type="button">关闭（Esc）</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>' +
            f'<script id="states" type="application/json">{state_json}</script><script>{JS}</script></body></html>')


def render_hub(spec):
    """Entry page: the system figure and a table of the replay pages. Spec: title, lead, rows, notes."""
    rows = ''.join(f'<tr><td><a href="{esc(r["file"])}">{esc(r["label"])}</a></td><td>{esc(r["task"])}</td><td>{esc(r["run"])}</td>'
                   f'<td>{esc(r["score"])}</td><td>{esc(r["why"])}</td></tr>' for r in spec['rows'])
    notes = ''.join(f'<li>{esc(n)}</li>' for n in spec.get('notes', []))
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(spec["title"])}</title><style>{CSS}</style></head><body><header><h1>{esc(spec["title"])}</h1><p>{esc(spec["lead"])}</p></header>'
            + system_figure() + '<main class="hub"><section class="panel"><h2>四个回放</h2><div class="tablewrap"><table class="hubtable"><thead><tr><th>页面</th><th>任务</th><th>运行</th><th>F1</th><th>为什么选它</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div><ul>{notes}</ul></section></main></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', type=Path, help='JSON spec of the entry page; then --out is that page.')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--split', choices=['val', 'test'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config-summary', type=Path)
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    if args.hub:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_hub(read_json(args.hub)), encoding='utf-8')
        print(f'{args.out.name}: entry page')
        return
    if not args.run_dir or not args.split:
        parser.error('--run-dir and --split are required without --hub')
    document = render(args.run_dir, args.split,
                      read_json(args.config_summary) if args.config_summary else None,
                      read_json(args.fixed) if args.fixed else None, args.note)
    size = len(document.encode('utf-8'))
    if size >= 8_000_000:
        parser.error(f'页面 {size} 字节，超过 8 MB；请降低 jpeg() 默认质量后重新生成。')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding='utf-8')
    print(f'{args.out.name}: {size:,} bytes; {len(read_steps(args.run_dir))} trace rows (including initialization)')


if __name__ == '__main__':
    main()
