#!/usr/bin/env python3
"""Build a self-contained HTML review from rounds JSON and exported examples."""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from review_data import REPO, STATES, read_json


def text(value):
    if value is None:
        return '—'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def inline(value):
    """Small Markdown subset, with literal code spans and escaped HTML."""
    parts = re.split(r'(`[^`]+`)', text(value))
    return ''.join('<code>' + escape(p[1:-1]) + '</code>' if p.startswith('`')
                   else re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escape(p)) for p in parts)


def table(columns, rows, rendered=False, css=''):
    render = str if rendered else inline
    header = ''.join(f'<th scope="col">{inline(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{render(c)}</td>' for c in row) + '</tr>' for row in rows)
    return f'<div class="am-table-wrap"><table class="{css}"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def markdown(value):
    """Paragraphs, unordered/ordered lists, bold, code and simple pipe tables."""
    lines = text(value if value is not None else '未记录。').splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if i + 1 < len(lines) and '|' in line and re.fullmatch(r'[\s|:\-]+', lines[i + 1]) and '-' in lines[i + 1]:
            cells = lambda s: [c.strip() for c in s.strip().strip('|').split('|')]
            columns, rows = cells(line), []
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append(cells(lines[i]))
                i += 1
            out.append(table(columns, rows))
            continue
        match = re.match(r'^(?:([-*+])|\d+[.)])\s+(.+)', line)
        if match:
            tag = 'ul' if match[1] else 'ol'
            items = []
            while i < len(lines):
                match = re.match(r'^(?:([-*+])|\d+[.)])\s+(.+)', lines[i].strip())
                if not match or ('ul' if match[1] else 'ol') != tag:
                    break
                items.append(f'<li>{inline(match[2])}</li>')
                i += 1
            out.append(f'<{tag}>' + ''.join(items) + f'</{tag}>')
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            if re.match(r'^(?:[-*+]|\d+[.)])\s+', lines[i].strip()):
                break
            if i + 1 < len(lines) and '|' in lines[i] and re.fullmatch(r'[\s|:\-]+', lines[i + 1]):
                break
            paragraph.append(lines[i].strip())
            i += 1
        out.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    return ''.join(out)


def relative_url(path, output_dir):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing review image: {path}')
    return quote(Path(os.path.relpath(path, output_dir)).as_posix(), safe='/')


def figure(path, caption, output_dir, replay=False):
    src = relative_url(path, output_dir)
    return (f'<figure class="am-figure"><a href="{src}"><img class="{"replay" if replay else "chart"}" '
            f'src="{src}" alt="{escape(text(caption), quote=True)}" loading="lazy"></a>'
            f'<figcaption>{inline(caption)}</figcaption></figure>')


def figures(items, input_dir, output_dir):
    out = []
    for item in items:
        if Path(item['src']).is_absolute() or '://' in item['src']:
            raise ValueError('Figure src must be a local path relative to the rounds JSON')
        out.append(figure(input_dir / item['src'], item.get('caption', ''), output_dir))
    return ''.join(out)


def panel(letter, title, body):
    return (f'<section class="am-panel"><header class="am-panel-head"><span class="am-panel-id">{letter}</span>'
            f'<h2>{escape(title)}</h2></header><div class="am-panel-body am-md">{body}</div></section>')


def score(value):
    return '未记录' if value is None else f'{100 * float(value):.2f}%'


def example_html(example, round_id, figures_dir, output_dir):
    base = figures_dir / f'round_{round_id}'
    manifest = read_json(base / f'replay_{Path(example["name"]).stem}.json')
    if any(manifest[key] != example[key] for key in ('cfg', 'split', 'name')):
        raise ValueError(f'Example manifest does not match config/split/name: {example["name"]}')
    for state, value in example.get('f1', {}).items():
        if state in manifest['f1'] and abs(float(value) - manifest['f1'][state]) > .0005:
            raise ValueError(f'{example["name"]}: rounds JSON F1 differs from masks at {state}')
    body = f'<h3>{escape(example["name"])} · {escape(example["cfg"])} / {escape(example["split"])}</h3>'
    body += markdown(example.get('why', '未记录选择理由。'))
    body += figure(base / manifest['replay'], 'M0 → A → B → C；F1 为本图分数。点击查看原图。', output_dir, True)
    body += table(['状态', '本图 F1ᶜ'], [[state, score(manifest['f1'][state])] for state in STATES])
    body += f'<h3>MLLM 调用：{len(manifest["calls"])} 次</h3>'
    body += '<p>近景依次为实际送入模型的 T1、T2。T5 不产生 P(yes)；同一步含两次调用时，F1 前后值属于整步。</p>'
    rows = []
    for call in manifest['calls']:
        crops = '<div class="call-pair">' + ''.join(
            figure(base / path, f'T{t}', output_dir) for t, path in enumerate(call['thumbs'], 1)) + '</div>'
        rows.append([crops, f'{inline(call["step"])} / {inline(call["stage"])} / {call["kind"]}',
                     '<div class="raw">' + escape(text(call['raw'])) + '</div>',
                     '—' if call['p_yes'] is None else f'{float(call["p_yes"]):.4f}',
                     inline(call['action']) + '<br>' + inline(call['op']),
                     score(call['f1_before']) + ' → ' + score(call['f1_after']) + '<br>' + inline(call['f1_scope'])])
    body += table(['T1 | T2 近景', '步骤 / 阶段 / 调用', '回答原文', 'P(yes)', '执行的动作 / 掩码操作', '本步前后 F1ᶜ'], rows, True, 'calls-table') if rows else '<p>本图没有 MLLM 调用。</p>'
    if manifest['operations']:
        body += '<h3>C 阶段操作（SAM）</h3>'
        body += table(['步骤', '动作', '操作', '本步前后 F1ᶜ'], [[s['step'], s['action'], s['op'],
                      score(s.get('f1_before')) + ' → ' + score(s.get('f1_after'))] for s in manifest['operations']])
    return '<article class="example">' + body + '</article>'


SCRIPT = """
(() => {
  const picker = document.getElementById('round-select');
  const rounds = [...document.querySelectorAll('.round-panel')];
  function selectRound() {
    for (const round of rounds) round.hidden = round.id !== picker.value;
    document.getElementById('current-stage').textContent = document.getElementById(picker.value).dataset.stage;
  }
  picker.addEventListener('change', selectRound);
  selectRound();
  document.getElementById('mode-select').addEventListener('change', (event) => {
    document.documentElement.dataset.mode = event.target.value;
  });
})();
"""


def build_review(input_path, output_path, figures_dir):
    input_path, output_path, figures_dir = Path(input_path), Path(output_path), Path(figures_dir)
    data = read_json(input_path)
    rounds = data['rounds']
    if not rounds:
        raise ValueError('rounds must contain at least one round')
    if len({str(r['id']) for r in rounds}) != len(rounds):
        raise ValueError('Round IDs must be unique')
    content, options = [], []
    for index, row in enumerate(rounds):
        dom_id = f'round-{index}'
        options.append(f'<option value="{dom_id}">{escape(row["title"])}</option>')
        body = '<h2>' + escape(row['title']) + '</h2>'
        body += panel('配置', '本轮配置与改动', table(['配置项', '选择'], row['config']) + '<h3>改了什么</h3>' + markdown(row['changed']))
        for i, module in enumerate(('M0', 'M1', 'M2', 'M3', 'M4')):
            section = row['sections'][module]
            cards = []
            for key, title in [('did', '做了什么'), ('result', '得到了什么结果'),
                               ('issues', '遇到的关键问题'), ('next', '下一步怎么做，为什么')]:
                value = markdown(section.get(key, '未记录。'))
                if key == 'result':
                    value += figures(section.get('figs', []), input_path.parent, output_path.parent)
                cards.append(f'<div class="review-block"><h3>{title}</h3>{value}</div>')
            body += panel(chr(65 + i), module, '<div class="am-grid">' + ''.join(cards) + '</div>')
        summary = row['summary_table']
        body += panel('F', '本轮汇总', markdown(row.get('summary_note', '')) + table(summary['columns'], summary['rows'])
                      + figures(row.get('summary_figs', []), input_path.parent, output_path.parent))
        if not row['examples']:
            raise ValueError(f'Round {row["id"]}: at least one example is required')
        body += panel('G', '例子回放', ''.join(example_html(e, row['id'], figures_dir, output_path.parent) for e in row['examples']))
        body += panel('H', '与 Codex 的讨论', markdown(row.get('discussion', '未记录。')))
        content.append(f'<div class="round-panel" id="{dom_id}" data-stage="{escape(row["stage"], quote=True)}"'
                       + (' hidden' if index else '') + '>' + body + '</div>')
    if len(rounds) > 1:
        cross = data.get('cross_round', {})
        body = markdown(cross.get('text', '尚未记录轮次比较。'))
        body += figures(cross.get('figs', []), input_path.parent, output_path.parent)
        if cross.get('table'):
            body += table(cross['table']['columns'], cross['table']['rows'])
        content.append('<div id="cross-round">' + panel('I', '轮次之间的比较', body) + '</div>')
    css = Path(__file__).with_name('review.css').read_text(encoding='utf-8')
    document = f'''<!doctype html>
<html lang="zh-CN" data-theme="blueprint" data-mode="auto">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OVCD-v0 实验审核</title><style>{css}</style></head>
<body><main class="am-sheet"><div class="am-frame">
<header class="am-head"><h1>OVCD-v0 实验审核</h1>
<p class="am-sub">初始化 → 去误检 → 补漏 → 修边界 · 每轮完整记录 M0–M4</p>
<div class="am-head-meta"><span><b>source</b>{escape(input_path.name)} · Agent 轨迹与掩码</span>
<span><b>style</b>doc/ovcd_agent_v0_pipeline.html</span></div></header>
<div class="am-toolbar"><label class="am-pick" for="round-select">轮次<select id="round-select">{''.join(options)}</select></label>
<label class="am-pick" for="mode-select">明暗<select id="mode-select"><option value="auto">跟随系统</option><option value="light">亮</option><option value="dark">暗</option></select></label>
<span class="stage" id="current-stage" aria-live="polite">{escape(rounds[0]['stage'])}</span></div>
<noscript>请启用 JavaScript 来切换轮次与主题；当前显示第一轮。</noscript>
{''.join(content)}</div></main><script>{SCRIPT}</script></body></html>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, default=REPO / 'results/ovcd_v0_rounds.json')
    parser.add_argument('--out', type=Path, default=REPO / 'results/ovcd_v0_review.html')
    parser.add_argument('--figures', type=Path, default=REPO / 'figures/ovcd_v0')
    args = parser.parse_args()
    print(build_review(args.json, args.out, args.figures))


if __name__ == '__main__':
    main()
