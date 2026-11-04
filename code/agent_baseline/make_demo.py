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
    'match_instances': '比较', 'ask': '语义', 'select': '运算', 'to_mask': '运算',
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


CSS = r'''
:root{color-scheme:light;--bg:#f4f5f7;--panel:#fff;--text:#172530;--muted:#596975;--line:#cbd3da;--accent:#185b91;--tint:#edf4fa;--up:#187040;--down:#ad3440;--warn:#855418;--warnbg:#fff4df;--tp:#2cce78;--fp:#ef5050;--fn:#458dff;--backdrop:#0009}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#111920;--panel:#19232c;--text:#e5edf4;--muted:#a8b7c3;--line:#3b4b58;--accent:#8ac4f4;--tint:#243544;--up:#79d79a;--down:#ff9b9f;--warn:#f1c77d;--warnbg:#382f21}}
:root[data-theme=dark]{color-scheme:dark;--bg:#111920;--panel:#19232c;--text:#e5edf4;--muted:#a8b7c3;--line:#3b4b58;--accent:#8ac4f4;--tint:#243544;--up:#79d79a;--down:#ff9b9f;--warn:#f1c77d;--warnbg:#382f21}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;overflow-wrap:anywhere}
header,footer{padding:18px 24px;background:var(--panel)}header{border-bottom:1px solid var(--line)}h1{font-size:23px;line-height:1.3;margin:5px 0 10px}h2{font-size:17px;margin:0 0 10px}h3{font-size:15px;margin:0}p{margin:7px 0}.eyebrow,.muted,figcaption{color:var(--muted)}.warning{font-weight:700;color:var(--warn);background:var(--warnbg);padding:8px 12px;border-left:3px solid var(--warn)}
.headtop{display:flex;justify-content:space-between;gap:12px;align-items:start}.metrics{display:flex;flex-wrap:wrap;gap:8px 25px;margin:12px 0}.metric strong{display:block;font-size:21px;line-height:1.25;font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(180px,.8fr) minmax(320px,1.55fr) minmax(240px,1fr);gap:16px;padding:18px 24px;max-width:1800px;margin:auto}.layout>*{min-width:0}.panel,.step{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px}.step{margin-bottom:12px;scroll-margin-top:16px}.step.active{border-color:var(--accent);border-left:4px solid var(--accent);padding-left:9px}.stephead{display:flex;align-items:center;justify-content:space-between;gap:10px}.thought{border-left:2px solid var(--line);padding-left:10px;white-space:pre-wrap;margin:12px 0}.call{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}.call.error{border-left:3px solid var(--down);padding-left:10px}.errorlabel,.down{color:var(--down)}.up{color:var(--up)}.badge{background:var(--tint);color:var(--accent);padding:2px 6px;border-radius:3px;margin-right:6px}.toolname{font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;margin:8px 0}details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}
figure{margin:10px 0;min-width:0}figcaption{font-size:12px;margin-top:4px}img{display:block;width:100%;height:auto;max-width:100%;border-radius:3px}button,select{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:5px 9px;cursor:pointer;max-width:100%}button:hover{border-color:var(--accent)}button:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.zoom{display:block;padding:0;border:0;width:100%;cursor:zoom-in}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.thumb{max-width:180px}.sticky{position:sticky;top:12px}.sticky figure img{max-height:27vh;object-fit:contain;background:var(--bg)}.legend{display:flex;gap:12px;font-size:12px;flex-wrap:wrap}.legend i{display:inline-block;width:9px;height:9px;margin-right:4px}.legend .tp{background:var(--tp)}.legend .fp{background:var(--fp)}.legend .fn{background:var(--fn)}.chart{width:100%;display:block;color:var(--muted)}.chart text{fill:currentColor;font-size:10px}.chart .axis{stroke:var(--line)}.chart .trace{fill:none;stroke:var(--accent);stroke-width:2}.chart .point{fill:var(--panel);stroke:var(--accent);stroke-width:2;cursor:pointer}.chart .point.active{fill:var(--accent);stroke-width:3}.chart .cursor{stroke:var(--muted);stroke-dasharray:3 3}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.empty{padding:25px 10px;text-align:center;color:var(--muted);border:1px dashed var(--line);margin:12px 0}.js-only{display:none}.has-js .js-only{display:initial}[hidden]{display:none!important}footer{border-top:1px solid var(--line)}footer pre{max-width:1200px}dialog{border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);max-width:95vw;max-height:95vh;padding:14px}dialog::backdrop{background:var(--backdrop)}dialog img{max-width:85vw;max-height:80vh;object-fit:contain;width:auto}dialog .close{float:right;margin-bottom:10px}
@media(max-width:900px){.layout{grid-template-columns:1fr;padding:12px}.sticky{position:static}.sticky figure img{max-height:none}.inputs .pair{grid-template-columns:1fr 1fr}header,footer{padding:14px}.headtop{flex-wrap:wrap}}
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
    head = f'<header><div class="headtop"><div><div class="eyebrow">变化检测 · Agent 轨迹展示 / {esc(split)} / {esc(name)}</div><h1>{title}</h1></div><label class="js-only">外观 <select id="theme"><option value="auto">跟随系统</option><option value="light">浅色</option><option value="dark">深色</option></select></label></div>'
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
            body += f'<div class="call{" error" if failed else ""}"><span class="badge">{CATEGORIES.get(tool, "其他")}</span><span class="toolname">{esc(tool)}</span>'
            if failed:
                body += f'<span class="errorlabel"> · 调用失败 {esc(call.get("error_code") or "")}</span>'
            if call.get('status') == 'skipped_due_to_submit':
                body += '<span class="muted"> · 提交后跳过</span>'
            body += text_block(call.get('args', call.get('raw_arguments', {})), '调用参数')
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
    footer += f'<p>变化目标代理个数：{esc(final.get("change_component_count", "未记录"))} · 变化面积：{esc(final.get("change_area_px", "未记录"))} 像素。<strong>8 连通域是代理，不等于栋数。</strong></p>'
    footer += '<p>工具调用统计（含失败 / 跳过）：' + esc('；'.join(f'{k} × {v}（失败 / 跳过 {errors[k]}）' for k, v in sorted(counts.items()))) + '</p>'
    footer += '<p>F1 由记录器算，Agent 全程看不到；真值也不参与规划。空真值且空预测的逐图 F1 按现有评测约定为 1。展示图为 JPEG 缩小版，面积及指标取原分辨率记录。</p></footer>'
    state_json = json.dumps(states, ensure_ascii=False).replace('<', '\\u003c')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} · {esc(name)}</title><style>{CSS}</style></head><body>' + head + '<main class="layout">' + left + ''.join(timeline) + right + '</main>' + footer +
            '<dialog id="zoom-dialog" aria-label="放大图像"><button id="close-zoom" class="close" type="button">关闭（Esc）</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>' +
            f'<script id="states" type="application/json">{state_json}</script><script>{JS}</script></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config-summary', type=Path)
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--note', default='')
    args = parser.parse_args()
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
