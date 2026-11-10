"""Reproducible local screenshots; self-contained HTML fallback without a browser.

Use TASK_DATA_DB=file:task-preview?mode=memory&cache=shared to keep SQLite off
the shared disk and all generated files under the requested output directory.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import socket
import threading
import time
from contextlib import closing
from pathlib import Path

import task_data_api as api
import task_data_import as prepare
import task_data_store as store

STATIC = Path(__file__).with_name('static')


def esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def picture(db, kind, ident, key, caption):
    data = base64.b64encode(api.rendered(db, kind, ident, key)).decode('ascii')
    return f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{data}"><figcaption>{esc(caption)}</figcaption></figure>'


def mapping_panel(db):
    rows = []
    for m in store.mappings(db):
        cells = []
        for side in ('t1', 't2'):
            cells.append('<td>' + ''.join(f'<label><input type="checkbox" disabled {"checked" if i in m["value"][side] else ""}>{i}</label>' for i in range(1, 7)) + '</td>')
        rows.append(f'<tr><td><b>{esc(m["name"])}</b></td>{"".join(cells)}<td>快照只读</td></tr>')
    return '<details class="card td-mapping" open><summary>R1 映射表 · 6 个目的</summary><table><thead><tr><th>目的</th><th>T1 类别集合</th><th>→ T2 类别集合</th><th>修改</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table><p class="meta">1 低矮植被 · 2 非植被地表 · 3 树 · 4 水 · 5 建筑 · 6 运动场；仅取两期非零且不同的类别转换。</p><div class="td-notice">映射改动需要重新生成真值后再审。保存映射记录修改人和时间，不会自动重算。</div></details>'


def snapshot(db, view):
    meta = store.metadata(db)
    source = '合成演示数据' if meta['source'] == 'synthetic' else '真实任务数据'
    if meta.get('preview'):
        source += ' · 部分图像预览'
    header = f'<div class="td-toolbar"><div class="td-tabs"><button class="{"on" if view=="review" else ""}">真值审查</button><button class="{"on" if view!="review" else ""}">组件标注</button></div><span class="td-source">{source}</span><label>标注人 <input class="td-actor" placeholder="填写姓名" disabled></label></div>'
    notes = ''.join(f'<div class="td-notice">{esc(n)}</div>' for n in meta.get('notes', []))
    if view == 'review':
        e = next((e for e in store.examples(db) if e['task'] == 'R1'), None)
        content = '<div class="td-eyebrow">TASK DATA / GROUND TRUTH</div><h2>真值审查</h2>'
        content += f'<p>完整库共 {len(store.examples(db))} 个例子；快照展示首张 R1 卡。</p>' + mapping_panel(db)
        if e:
            content += f'<article class="card"><h3>R1 · {esc(store.PURPOSE_NAMES.get(e["group"],e["group"]))} · {esc(e["task_type"])}</h3><div class="meta">{esc(e["image"])} · {esc(e["selection"])} · 真值面积 {e["area"]:,} px</div><p class="td-instruction">{esc(e["instruction"])}</p><div class="td-grid">'
            content += ''.join(picture(db,'example',e['id'],k,c) for k,c in [('t1','T1 原图'),('t2','T2 原图'),('s1','T1 语义标签'),('s2','T2 语义标签'),('overlay','真值叠加在 T2')])
            content += '</div><div class="td-legend">' + ''.join(f'<span><i class="td-swatch" style="background:rgb({",".join(map(str,rgb))})"></i>{i} {esc(store.CLASS_NAMES[i])}</span>' for i,rgb in enumerate(store.PALETTE)) + '</div><p class="meta">黄色半透明填充、红色轮廓；面积按原始真值计算。</p><div class="td-row td-card-actions"><button class="td-choice" disabled>合理</button><button class="td-choice" disabled>不合理</button><input class="td-note" placeholder="可留一句话备注" disabled><button class="td-action" disabled>撤销</button></div></article>'
    else:
        number = 2 if view == 'second_pass' else 1
        records = store.components(db, number)
        content = f'<div class="td-eyebrow">TASK DATA / COMPONENT LABELS</div><h2>{"第二遍 · 盲标" if number==2 else "建筑变化组件标注"}</h2>'
        content += f'<div class="td-toolbar"><div class="td-tabs"><button class="{"on" if number==1 else ""}">第一遍</button><button class="{"on" if number==2 else ""}">第二遍</button></div><span>已标 {sum(bool(r["labeled_at"]) for r in records)} / {len(records)} <progress class="td-progress" max="{max(1,len(records))}" value="0"></progress></span><label>图 <select disabled><option>全部图</option></select></label><label>方向 <select disabled><option>全部方向</option></select></label></div>'
        if number == 2:
            content += '<div class="td-notice">第二遍按种子 1016 抽取 20%，请独立选择两个属性。</div>'
        if records:
            c = records[0]
            direction = '新建' if c['direction'] == 'new' else '拆除'
            content += f'<article class="card"><div class="td-row td-component-title"><h3>{esc(c["image"])} · {direction}</h3><span>面积 {c["area"]:,} px · bbox {c["bbox"]}</span></div><div class="td-component-layout"><div><div class="td-closeups">'
            content += picture(db,'component',c['id'],'t1','T1 近景'+(' · 主要判读期' if c['direction']=='demolished' else ''))
            content += picture(db,'component',c['id'],'t2','T2 近景'+(' · 主要判读期' if c['direction']=='new' else ''))
            content += '</div><div class="td-position">' + picture(db,'component',c['id'],'thumb','整图位置 · 红框与轮廓') + '</div></div><aside><b>屋顶颜色</b><div class="td-attributes">'
            content += ''.join(f'<button class="td-choice {"on" if number==1 and c["prefill_color"]==color else ""}" disabled><kbd>{i+1}</kbd>{color}</button>' for i,color in enumerate(store.COLORS))
            content += '</div><b>子类</b><div class="td-attributes">' + ''.join(f'<button class="td-choice" disabled><kbd>{key}</kbd>{label}</button>' for key,label in zip('QWER',store.SUBTYPES)) + '</div>'
            if number == 1:
                content += f'<p class="meta">HSV 预填：{esc(c["prefill_color"])}（可改）</p>'
            content += '</aside></div><div class="td-row td-card-actions"><button class="td-action" disabled>上一张</button><button class="td-choice" disabled>保存并下一张 <kbd>空格</kbd></button><button class="td-action" disabled>撤销 <kbd>Z</kbd></button></div></article>'
    base = store.read_text(STATIC / 'index.html')
    base_css = base.split('<style>', 1)[1].split('</style>', 1)[0]
    css = base_css + '\n' + store.read_text(STATIC / 'task_data.css')
    return '<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>任务数据 · 离线 HTML 快照</title><style>' + css + '</style></head><body><header><h1>任务数据</h1><div class="mode-switch"><a>MLLM 判断器审核</a><a>漏检候选审核</a><a class="on">任务数据</a></div></header><main><section class="td">' + header + '<p class="meta">离线 HTML 快照，控件只读。使用实际 API 渲染的图片；未经过浏览器截图。</p>' + notes + content + '</section></main></body></html>\n'


def screenshots(output):
    """Use a random loopback port; stop server/browser even on failure."""
    from playwright.sync_api import Error as PlaywrightError, sync_playwright
    import app
    import uvicorn

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app.app, host='127.0.0.1', port=port, workers=1, log_level='warning'))
        thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started:
                if time.monotonic() >= deadline:
                    raise RuntimeError('本地页面服务未能启动')
                time.sleep(.1)
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    try:
                        page = browser.new_page(viewport=dict(width=1440, height=1400), device_scale_factor=1)
                        for view, query in [('review','taskTab=review'),('components','taskTab=components'),('second_pass','taskTab=components&pass=2')]:
                            page.goto(f'http://127.0.0.1:{port}/?mode=task-data&{query}')
                            page.wait_for_selector('.td-example' if view=='review' else '#td-current-card article')
                            page.locator('.td img').evaluate_all("imgs => Promise.all(imgs.filter(i=>i.getBoundingClientRect().top<1600).map(i => i.complete ? Promise.resolve() : new Promise(r => {i.onload=r;i.onerror=r;})))")
                            page.screenshot(path=str(output / (view + '.png')), full_page=False)
                    finally:
                        browser.close()
            except PlaywrightError as exc:
                raise RuntimeError('浏览器未安装或页面截图失败') from exc
        finally:
            server.should_exit = True
            thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', type=Path)
    mode.add_argument('--run-root', type=Path)
    parser.add_argument('--second-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--html-only', action='store_true')
    parser.add_argument('--preview-images', type=int)
    args = parser.parse_args()
    if args.run_root and not args.second_root:
        parser.error('--run-root 需要 --second-root')
    # Keep database alive for both the preparation code and the API server.
    os.environ.setdefault('TASK_DATA_DB','file:task-preview?mode=memory&cache=shared')
    output = args.output.resolve()
    store.retry_io(lambda: output.mkdir(parents=True,exist_ok=True))
    with closing(store.connect()) as db:
        summary = (prepare.prepare_synthetic(db,args.synthetic) if args.synthetic else
                   prepare.prepare_real(db,args.run_root,args.second_root,limit_images=args.preview_images))
        for view in ('review','components','second_pass'):
            store.write_text(output / (view + '.html'), snapshot(db,view))
        status = '未请求浏览器截图' if args.html_only else ''
        if not args.html_only:
            try:
                screenshots(output)
                status = '浏览器截图成功'
            except (ImportError, OSError, RuntimeError) as exc:
                status = f'浏览器截图不可用：{type(exc).__name__}；已生成内嵌图片的 HTML 快照'
        prepare.export_files(db,output / 'exports')
        summary['capture'] = status
        store.write_text(output / 'capture.json', json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
