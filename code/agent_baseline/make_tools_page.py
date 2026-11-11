#!/usr/bin/env python3
"""Build an offline Chinese catalog from the planner's registered schemas.

No model, worker, network or experiment is started. Usage: python -m
agent_baseline.make_tools_page --out-dir results/tools (with code on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_baseline import tools
from agent_baseline.state import Session

# These editorial summaries describe the implementations in tools.py,
# tools_t13.py, t13_worker.py, examples_t3x.py, map_input.py and loop_json.py.
# Parameter names, types, constraints and switches are extracted below.
META = {
    'view': ('看', '查看两期影像，可叠加轮廓、组件编号并裁切区域。', 'view#k、两张图、坐标与缩放信息', '纯绘图', '不改变'),
    'inspect': ('看', '按面积列出掩码组件或实例的编号、面积和边界框。', '组件表、总数和分页信息', '纯运算', '不改变'),
    'segment_text': ('分割', '在一个时相按类别文字分割目标实例。', 'inst#k、数量、面积和分数统计', 'SAM3（缓存或 worker）', '不改变'),
    'segment_prompt': ('分割', '用一个框或带标签的点分割单个时相的目标。', 'mask#k、面积、组件数、sam_quality', 'SAM ViT-H', '不改变'),
    'segment_everything': ('分割', '在一个时相生成不限类别的区域候选。', 'props#k、数量、总面积和统计', 'SAM ViT-H（DynamicEarth 区域提案，可读缓存）', '不改变'),
    'segment_semantic': ('分割', '在一个时相生成指定前景类别的语义掩码。', 'mask#k、面积和组件数', 'SegEarth-OV', '不改变'),
    'match_instances': ('比较', '用跨时相的单实例重叠判断出现、消失和仍存在。', 'appeared / disappeared / present_t2 的 inst#k 及数量', '纯重叠运算', '不改变'),
    'classify': ('比较', '计算各候选区域的类别平均概率，背景自动加入。', '新 props#k、逐 ID 类别概率', 'SegEarth-OV', '不改变'),
    'match_exemplar': ('比较', '按外观或时相特征差与示例的余弦相似度排列候选。', '新 props#k、逐 ID 相似度；可选正例减负例分数', 'DINOv2 + 纯余弦运算', '不改变'),
    'extract_features': ('特征', '提取并缓存一个时相的 patch 特征网格。', 'feat#k、时相、网格形状', 'DINOv2', '不改变'),
    'compare_features': ('特征', '按区域两期平均特征的 1−cosine 给候选附上变化分数。', '新 props#k、分数范围', '纯运算（输入 DINOv2 特征）', '不改变'),
    'select': ('运算', '通过编号、面积或分数条件筛选实例并保留 ID。', '新 inst#k 或 props#k、数量和面积', '纯运算', '不改变'),
    'to_mask': ('运算', '将一个或多个实例集合并成二值掩码。', 'mask#k、面积、组件数', '纯运算', '默认改变；set_current=false 时不改变'),
    'combine': ('运算', '对两个已有掩码求并、交或相减。', 'mask#k、面积、组件数', '纯运算', '默认改变；set_current=false 时不改变'),
    'morph': ('运算', '对已有掩码进行开闭运算或填洞。', 'mask#k、面积、组件数', '纯运算', '默认改变；set_current=false 时不改变'),
    'python': ('运算', '在隔离子进程中对只读数组做运算并保存新掩码。', '零个或多个 mask#k、标准输出文字及统计', '纯 NumPy / SciPy 运算', '默认将最后保存掩码设为当前；set_current=false 时不改变'),
    'remove_region': ('修改', '删除指定的完整组件，剩余组件保留 ID。', 'mask#k、面积、组件数和增删像素数', '纯运算', '默认改变；set_current=false 时不改变'),
    'add_region': ('修改', '从另一个掩码或实例集中加入所选区域。', 'mask#k、面积、组件数和增删像素数', '纯运算', '默认改变；set_current=false 时不改变'),
    'refine_region': ('修改', '用提示分割替换目标组件，并裁到编辑区域内。', 'mask#k、面积、组件数和增删像素数', 'SAM ViT-H', '默认改变；set_current=false 时不改变'),
    'ask': ('核验', '向独立验证器询问掩码组件是否符合任务变化要求。', '逐组件 answer、type、reason、p_yes、status；剩余未核验数', '可配置验证器；CLI 默认 qwen3.7-plus，G3/G4 本批轨迹用 qwen3.7-flash；也支持本地验证器', '不改变；更新核验账本'),
    'submit': ('结束', '提交已有掩码或空答案，并写下决定说明。', '最终 mask#k、面积、组件数、submitted；可返回报告', '纯运算；提交门可触发独立验证器', '通过提交门后设为当前及最终答案'),
    'plan': ('规划', '写入或替换待办清单，不执行清单中的动作。', '清单文字和状态', '纯记录', '不改变'),
    'load_skill': ('规划', '读取已启用技能的说明文本。', '技能名称与说明文字', '纯文本读取', '不改变'),
}


def registered_schemas(gate5=False):
    """Union of schemas actually offered by run_session, including opt-in variants."""
    session = Session((1, 1))
    # extend_schema only checks whether external_examples is nonempty.
    session.external_examples = (('example#0', 'exmask#1'),)
    return [*tools.session_schemas(session, python=True, tools_t13=True, gate5=gate5),
            tools.PLAN_SCHEMA, tools.SKILL_SCHEMA]


def cli_options():
    """Read add_argument declarations without importing/running the online loop."""
    from agent_baseline.make_demo import read_text
    tree = ast.parse(read_text(Path(__file__).with_name('loop_json.py')))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            names = [v.value for v in node.args if isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.startswith('--')]
            for name in names:
                values = {}
                for kw in node.keywords:
                    try:
                        values[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass
                result[name] = values
    return result


def switch_for(name, options):
    if name in tools.T13_NAMES:
        flag = '--tools-t13'
    elif name in ('python', 'plan'):
        flag = '--' + name
    elif name == 'load_skill':
        flag = '--skills'
    else:
        return dict(label='默认开', default=True, flag=None)
    if flag not in options:
        raise ValueError('Missing registered CLI switch: ' + flag)
    return dict(label=flag + ('（all 或按工具名开启）' if flag == '--tools-t13' else ''),
                default=options[flag].get('default', False), flag=flag, declaration=options[flag])


def rules_for(name):
    rules = ['G4：若此工具新建 mask#k，会计入最新掩码提醒；旧输入掩码及 region#0 除外。' if name in ('segment_prompt','segment_semantic','to_mask','combine','morph','remove_region','add_region','refine_region','python') else 'G4：此工具不直接产生最新 mask 提醒。']
    if name in ('segment_text', 'segment_everything'):
        rules += ['G5：仅 --gate5 时提供可选 region 参数。整图分割后求交；保留至少 1 像素的实例。像素统计按并集去重。不传 region 保留原行为。']
    if name == 'segment_prompt':
        rules += ['G3：只有 box 且无 points 时，框占比超过 big_box_share（默认 0.10）报 box_too_large；不作用于 refine_region。']
    if name == 'ask':
        rules += ['G3：自动选择仅包含反馈点半径 32 像素/方位格内或被编辑的相关组件；显式 ids 不经过此筛选。每次最多 64 个组件，无法解析不等于 no。']
    if name == 'submit':
        rules += ['开启 --verify-stage：首次提交及尚欠核验组件会被拦下，返回报告；答案是证据，规划器决定是否修改。G2/G3 预算报告给出复核机会；异常或预算耗尽仍会兜底。',
                  'G3：报告区域外像素及已核验的反向类型；方向只报告。',
                  'G4：有 region#0 时在实际提交处求交，保存前后句柄及删除像素数；显式提交与兜底都适用。']
    if name in ('compare_features', 'segment_semantic', 'match_exemplar'):
        rules += ['--tool-hints off/a/b：仅追加用途句，参数不变；独立于 --gate4，默认 off。']
    if name in ('segment_text','segment_prompt','refine_region','segment_everything','extract_features','classify','segment_semantic'):
        rules += ['--map-input：T1 为旧地图，光学工具只允许 time=t2。']
    if name in ('compare_features','match_exemplar'):
        rules += ['--map-input：不支持此操作。']
    return rules


def build_catalog(gate5=False):
    options = cli_options()
    entries = []
    for item in registered_schemas(gate5=gate5):
        fun = item['function']; name = fun['name']; spec = fun['parameters']
        group, description, output, backend, changes = META[name]
        parameters = []
        for key, value in spec['properties'].items():
            parameters.append(dict(name=key, type=value.get('type', '未定义'),
                required=key in spec.get('required', []), meaning=value.get('description', '未定义'),
                schema=value, switch='--external-examples' if key == 'negative' else '--gate5' if key == 'region' and name in ('segment_text', 'segment_everything') else None))
        default_schema = next((s['function']['parameters'] for s in tools.SCHEMAS if s['function']['name'] == name), None)
        tasks = ('T1.2 / T1.3 / T3.1 / T3.2 / T3.3 等；按开关启用，不按任务名硬限制'
                 if name in tools.T13_NAMES else 'T1 / T3 / T4 / T5；实际取决于会话输入与开关，代码未定义逐任务白名单')
        entries.append(dict(name=name, group=group, description=description,
            parameters=parameters, schema=spec, base_schema=default_schema,
            source_description=fun['description'], output=output, backend=backend,
            switch=switch_for(name, options), tasks=tasks, changes_current_mask=changes,
            rules=rules_for(name)))
    return entries


CLASSES = [
    ('读取工具', '读取影像或已有结果，不执行操作。', ['view', 'inspect']),
    ('执行工具', '执行操作，产生新的实例、掩码或特征。默认把新掩码设为当前掩码。',
     ['segment_text', 'segment_prompt', 'segment_everything', 'segment_semantic',
      'match_instances', 'match_exemplar', 'classify', 'extract_features', 'compare_features',
      'select', 'to_mask', 'combine', 'morph', 'python', 'remove_region', 'add_region', 'refine_region']),
    ('事件触发工具', '调用后启动验证器或提交门。', ['ask', 'submit']),
    ('规划工具', '只写清单或读说明，不处理影像。G1–G4 都没有打开。', ['plan', 'load_skill']),
]


def render_catalog(catalog):
    """G1 style (results/agent_baseline/demo_g1): the toolbox figure, then one card per tool in the G1 classes."""
    from html import escape
    from agent_baseline.site_svg import agent_figure, SWITCH_TOOLS
    esc = lambda v: escape(str(v), quote=True)
    css = ''.join((Path(__file__).resolve().parent / 'site_assets' / name).read_text(encoding='utf-8') for name in ('hub.css', 'site.css'))
    by_name = {item['name']: item for item in catalog}
    missing = set(by_name) - {n for _, _, names in CLASSES for n in names}
    if missing:
        raise ValueError(f'Tools without a class: {sorted(missing)}')
    default = sum(1 for item in catalog if item['switch']['default'] is True)
    sections = []
    for title, note, names in CLASSES:
        cards = []
        for name in names:
            item = by_name.get(name)
            if item is None:
                continue
            on = item['switch']['default'] is True
            badge = '<span class="badge">默认可用</span>' if on else f'<span class="badge off">需 {esc(item["switch"]["flag"])}</span>'
            params = ''.join(f'<li><code>{esc(p["name"])}</code>{"（必填）" if p["required"] else ""}</li>' for p in item['parameters']) or '<li>没有参数</li>'
            prow = ''.join(f'<tr><td><code>{esc(p["name"])}</code></td><td>{esc(p["type"])}</td><td>{"是" if p["required"] else "否"}</td><td>{esc(p["meaning"])}</td></tr>' for p in item['parameters'])
            cards.append(f'<article class="toolcard{"" if on else " sw"}"><h3><code>{esc(name)}</code>{badge}</h3><p>{esc(item["description"])}</p>'
                         f'<dl><dt>输入</dt><dd><ul>{params}</ul></dd><dt>输出</dt><dd>{esc(item["output"])}</dd><dt>后端</dt><dd>{esc(item["backend"])}</dd>'
                         f'<dt>当前掩码</dt><dd>{esc(item["changes_current_mask"])}</dd></dl>'
                         f'<details><summary>参数说明和规则</summary>'
                         + (f'<div class="tablewrap"><table class="hubtable"><thead><tr><th>参数</th><th>类型</th><th>必填</th><th>说明（代码原文）</th></tr></thead><tbody>{prow}</tbody></table></div>' if prow else '')
                         + '<ul>' + ''.join(f'<li>{esc(r)}</li>' for r in item['rules']) + '</ul></details></article>')
        sections.append(f'<section class="panel example"><h2>{esc(title)}（{len(cards)} 个）</h2><p class="muted">{esc(note)}</p><div class="toolcards">{"".join(cards)}</div></section>')
    head = ('<header><div class="eyebrow"><a href="../demo_g3/index.html">回放列表</a></div><h1>核心工具</h1>'
            '<p>Agent 只能通过下面的原子工具工作。没有一次输出变化掩码的工具。工具名和参数从代码中注册的定义读取。</p>'
            f'<div class="metrics"><div class="metric"><strong>{len(catalog)}</strong><span>工具总数</span></div>'
            f'<div class="metric"><strong>{default}</strong><span>默认可用</span></div>'
            f'<div class="metric"><strong>{len(catalog) - default}</strong><span>要用命令行开关打开</span></div></div></header>')
    svg = agent_figure('G3', ['T1 较早影像', 'T2 较晚影像', '指令（一句话任务）', '给定掩码、区域或示例', '系统提示和工具定义'], '其他输入',
                       switch_tools=SWITCH_TOOLS, harness_notes=['python、plan、load_skill 默认关闭，图中没有画出。'],
                       toolbox_note='14 个默认工具和 6 个要开关的工具。工具只生成带新编号的结果。')
    figure = (f'<section class="panel system"><h2>工具在系统中的位置</h2><figure><div class="sysscroll">{svg}</div>'
              '<figcaption class="syscap"><strong>只有规划器可以调用工具箱。虚线框的工具默认关闭，用 --tools-t13 打开。</strong></figcaption></figure></section>')
    rules = ('<section class="panel example"><h2>结果编号、验证器和提交门</h2><div class="capgrid">'
             '<div><h3>结果编号</h3><ul><li>工具只生成带新编号的结果，不修改已有结果。</li><li>编号种类：mask#k、inst#k、props#k、feat#k、view#k。</li>'
             '<li>输入编号：给定掩码 mask#0、区域 region#0、旧地图 map#0、外部示例 example#k、exmask#k。</li>'
             '<li>当前掩码是一个单独的指针。set_current=false 时，工具不修改当前掩码。</li></ul></div>'
             '<div><h3>验证器（ask）</h3><ul><li>验证器只接收组件的 T1、T2 裁剪图和验证问题。</li><li>验证器不接收规划器的理由和对话。</li>'
             '<li>输出：yes 或 no、变化类型、理由、p_yes。p_yes 没有校准。</li><li>每次最多验证 64 个组件。答案无法解析时，不等于 no。</li>'
             '<li>ask 不修改掩码。规划器决定是否删除组件。</li></ul></div>'
             '<div><h3>提交门（submit）</h3><ul><li>第一次 submit 时，提交门暂停提交并返回报告。</li><li>再次 submit 时，掩码含未验证组件才暂停。</li>'
             '<li>G3：报告列出区域外像素和方向错误。</li><li>G4：有 region#0 时，提交的掩码与区域求交。</li>'
             '<li>预算用完或会话异常结束时，harness 提交当前掩码。</li></ul></div></div></section>')
    extra = ('.toolcards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}'
             '.toolcard{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:2px;padding:10px 12px;background:var(--panel);min-width:0}'
             '.toolcard.sw{border-left-style:dashed}.toolcard h3{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:15px;margin:0 0 6px}'
             '.toolcard h3 code{font-family:var(--font-mono);font-size:14px}.toolcard .badge{font-size:12px;font-weight:400}.toolcard .badge.off{background:var(--warnbg);color:var(--warn)}'
             '.toolcard dl{display:grid;grid-template-columns:64px 1fr;gap:4px 8px;margin:8px 0;font-size:13px}.toolcard dt{color:var(--muted)}.toolcard dd{margin:0}'
             '.toolcard dd ul{margin:0;padding-left:16px}.toolcard code{font-family:var(--font-mono);font-size:12.5px}')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>核心工具</title><style>'
            + css + extra + '</style></head><body>' + head + figure + ''.join(sections) + rules + '</body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate5', action='store_true', help='Include G5 segmentation region parameters.')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog(gate5=args.gate5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'tools_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.out_dir / 'index.html').write_text(render_catalog(catalog),encoding='utf-8')
    print(f'工具目录：{len(catalog)} 个工具')


if __name__ == '__main__':
    main()
