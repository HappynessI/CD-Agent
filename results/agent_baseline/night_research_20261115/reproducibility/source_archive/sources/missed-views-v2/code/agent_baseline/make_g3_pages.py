#!/usr/bin/env python3
"""Rebuild nine median-selected replay pages from existing traces only."""
from __future__ import annotations
import argparse
import base64
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_baseline.make_demo import (render, read_json, CSS, CURRENT_CSS, esc,
                                      pct, system_figure_current)
from agent_baseline.pick_demo import select_all
from agent_baseline.make_baselines_page import load_baselines, render_baselines, select_example
from PIL import Image

RULE = '三种子合并最终逐会话 F1 中位数；排除非 model 候选（全兜底除外）；最近者并列依成功工具类别数降序、图名、运行名'


class PageAudit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []
        self.links = []
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in {'meta','img','input','br','hr','link','source','wbr','area','base','embed','param','track','col'}:
            self.stack.append(tag)
        for name in ('src','href','action','poster'):
            value = attrs.get(name, '')
            if value.startswith(('http:', 'https:', '//', '/')):
                self.errors.append('外部或绝对资源引用')
            if name == 'src' and value and not value.startswith('data:'):
                self.errors.append('非内嵌资源')
        if tag == 'img' and attrs.get('src'):
            self.images.append(attrs['src'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append('HTML 标签未配对：' + tag)
        else:
            self.stack.pop()


def audit_html(document, image_limit=512):
    parser = PageAudit(); parser.feed(document); parser.close()
    if parser.stack:
        parser.errors.append('HTML 标签未闭合')
    # Only file paths, not ordinary leading slashes in prose or inline code.
    text_only = re.sub(r'data:image/jpeg;base64,[A-Za-z0-9+/=]+', '', document)
    if re.search(r'/(?:guisongxia01|persist_data|home|mnt|srv|tmp|data)/', text_only):
        parser.errors.append('服务器绝对路径')
    if re.search(r'url\(\s*[\"\']?(?:https?:|//)', document):
        parser.errors.append('外部 CSS 资源')
    for uri in parser.images:
        if not uri.startswith('data:image/jpeg;base64,'):
            parser.errors.append('图片不是内嵌 JPEG'); continue
        with Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))) as image:
            if max(image.size) > image_limit:
                parser.errors.append('图片长边超限')
    if parser.errors:
        raise ValueError('; '.join(sorted(set(parser.errors))))
    return dict(bytes=len(document.encode('utf-8')), images=len(parser.images), links=parser.links)


def render_index(records):
    rows = []
    for r in records:
        rows.append(f'<tr><td><a href="{r["file"]}">{esc(r["image"])}</a></td><td>{esc(r["task"])}</td><td>{esc(r["dataset"])}</td><td>{esc(r["run"])}<br>seed={esc(r["seed"])}</td><td>{pct(r["f1"])} / {pct(r["median"])}</td><td>{esc(RULE)}<br>会话 {r["population"]}，可选 {r["eligible"]}，兜底 {r["fallback_count"]}</td></tr>')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G3 / G4 多任务回放</title><style>'+CSS+CURRENT_CSS+'</style></head><body><header><h1>G3 / G4 多任务回放</h1><nav aria-label="主要导航"><p><a href="baselines.html">三种做法是什么</a> → 固定路线、规则与 Agent，同图分步比较。</p><p><a href="#typical-replays">9 个典型回放</a> → 逐任务查看输入、调用和最终掩码。</p><p><a href="../demo_g1/index.html">G1 旧回放</a> → 保留历史页面。</p></nav><p>按中位数选出的典型会话。每项合并三个种子的验证会话，逐会话 F1 不做整集混淆矩阵汇总。</p></header>'
            + system_figure_current('G3') + '<main class="hub"><section class="panel" id="typical-replays"><h2>九个任务</h2><div class="tablewrap"><table class="hubtable"><thead><tr><th>页面 / 图名</th><th>任务</th><th>数据</th><th>运行与种子</th><th>最终 F1 / 任务中位数</th><th>选择规则</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><p>这些是按中位数选的典型会话，不代表最好或最差。中位数包含已记录的兜底会话，候选排除非 model 提交，除非全部兜底。缺失 F1 不补为零。</p><p>T1.3 页使用 G4：最后 3 次回复及预算报告的当前答案提醒；有区域才在提交时求交；用途句由独立 --tool-hints 控制，轨迹未记录的开关不推断。</p><p>旧回放：<a href="../demo_g1/index.html">G1 回放</a> · <a href="../tools/index.html">核心工具</a></p></section></main></body></html>')


def build(root, out, selections=None):
    records = read_json(selections) if selections else select_all(root)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for record in records:
        directory = Path(root) / record['run'] / record['task'] / 'val' / record['image']
        info = read_json(directory / 'input.json')
        record['seed'] = info.get('seed', '未记录')
        record['file'] = record['page'] + '.html'
        baselines = load_baselines(root, record, directory)
        document = render(directory, 'val', selection=record, baselines=baselines)
        report = audit_html(document)
        if report['bytes'] > 3_000_000:
            # Preserve session and every step; reduce only the presentation JPEG quality.
            document = render(directory, 'val', selection=record, image_quality=40, baselines=baselines)
            report = audit_html(document)
        if report['bytes'] > 3_000_000:
            raise ValueError(record['file'] + ' 超过 3 MB')
        record['bytes'] = report['bytes']; record['embedded_images'] = report['images']
        (out / record['file']).write_text(document,encoding='utf-8')
        print(f'{record["file"]}: {record["run"]}/{record["image"]}, F1={record["f1"]:.6f}, median={record["median"]:.6f}, {record["bytes"]} bytes',flush=True)
    index = render_index(records); audit_html(index)
    (out / 'index.html').write_text(index,encoding='utf-8')
    example = select_example(root, records)
    if example['file'] not in {r['file'] for r in records}:
        directory = Path(root) / example['run'] / example['task'] / 'val' / example['image']
        document = render(directory, 'val', selection=example, baselines=load_baselines(root, example, directory))
        if audit_html(document)['bytes'] > 3_000_000:
            document = render(directory, 'val', selection=example, image_quality=40, baselines=load_baselines(root, example, directory))
        if audit_html(document)['bytes'] > 3_000_000:
            raise ValueError('补充回放超过 3 MB')
        (out / example['file']).write_text(document,encoding='utf-8')
    comparison = render_baselines(root, records, example)
    if audit_html(comparison)['bytes'] > 3_000_000:
        raise ValueError('三种做法页面超过 3 MB')
    (out / 'baselines.html').write_text(comparison,encoding='utf-8')
    (out / 'baseline_example.json').write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out / 'selection.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# G3 / G4 多任务回放', '', '只读取已有轨迹。所有图片内嵌，长边不超过 512 像素。打开 index.html。', '',
             '选择规则：'+RULE+'。中位数按三种子全部有最终 F1 的会话计算，兜底仍计入分布。类别按成功且未跳过的调用统计，分类来自 make_demo.CATEGORIES。',
             '浮点距离差不超过 1e-12 视为并列。缺最终 F1 的会话不补零；缺一个种子直接报错。selection.json 保存计数、种子和选择依据。', '',
             '| 页面 | 任务 / 数据 | 运行 / 图名 | 最终 F1 | 中位数 | 字节 |', '|---|---|---|---|---|---|']
    for r in records:
        lines.append(f'| [{r["page"]}]({r["file"]}) | {r["task"]} / {r["dataset"]} | {r["run"]} / {r["image"]} | {pct(r["f1"])} | {pct(r["median"])} | {r["bytes"]} |')
    lines += ['', '这些是典型会话，不代表最好或最差。旧 G1 在 ../demo_g1/index.html。', '',
              '复现：指定外部运行根目录，不把服务器路径写入代码。当前目录执行，先设 PYTHONPATH=code、TMPDIR=.runtime/tmp 和单线程变量。', '',
              '```bash', '$PY -m agent_baseline.make_g3_pages --run-root "$RUN_ROOT" --out-dir results/demo_g3', '$PY -m agent_baseline.make_tools_page --out-dir results/tools', '```', '',
              '输入第 4 格按归档展示给定掩码、反馈、区域或示例。缺记录写未记录。点/方位叠图只使用公开几何，不展示隐藏误差区域。',
              '桌面两侧 sticky，手机使用单栏并取消固定。父级 layout/aside/body 不设置 overflow:hidden。浅深主题沿用 G1 变量。',
              '所有回放保留失败调用、自动报告、返回掩码/候选；G4 提醒与求交只依轨迹字段标记。T1.3 本会话如无 region，不会伪造求交步骤。']
    lines += ['', 'baselines.html 解释固定路线、规则和 Agent。入口顶部提供三段导航。每个回放的对照面板在流程图和时间线之间。',
              '示例图沿用 LEVIR 中位数候选，并要求规则至少删过一个组件。选择记录在 baseline_example.json。',
              'LEVIR 规则掩码按已有 f37_corr 答案删除 no 的组件恢复，仅为离线展示，不调用模型。未缓存的固定匹配中间步骤明确标未缓存。',
              'C2/C4 的答案编号来自裁剪前组件。T5.2 是固定响应 C；T5.3 展示两个固定响应，验证器挑选不是删除规则。T3.1 对照按报告口径剔除示例，页头保留记录器原始 F1。']
    (out / 'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total > 30_000_000:
        raise ValueError('demo_g3 总量超过 30 MB')
    return records, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--selections',type=Path,help='Optional saved pick_demo --typical output.')
    args = parser.parse_args()
    _, total = build(args.run_root,args.out_dir,args.selections)
    print(f'总量：{total} bytes')


if __name__ == '__main__':
    main()
