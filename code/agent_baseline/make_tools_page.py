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


def registered_schemas():
    """Union of schemas actually offered by run_session, including opt-in variants."""
    session = Session((1, 1))
    # extend_schema only checks whether external_examples is nonempty.
    session.external_examples = (('example#0', 'exmask#1'),)
    return [*tools.session_schemas(session, python=True, tools_t13=True),
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


def build_catalog():
    options = cli_options()
    entries = []
    for item in registered_schemas():
        fun = item['function']; name = fun['name']; spec = fun['parameters']
        group, description, output, backend, changes = META[name]
        parameters = []
        for key, value in spec['properties'].items():
            parameters.append(dict(name=key, type=value.get('type', '未定义'),
                required=key in spec.get('required', []), meaning=value.get('description', '未定义'),
                schema=value, switch='--external-examples' if key == 'negative' else None))
        default_schema = next((s['function']['parameters'] for s in tools.SCHEMAS if s['function']['name'] == name), None)
        tasks = ('T1.2 / T1.3 / T3.1 / T3.2 / T3.3 等；按开关启用，不按任务名硬限制'
                 if name in tools.T13_NAMES else 'T1 / T3 / T4 / T5；实际取决于会话输入与开关，代码未定义逐任务白名单')
        entries.append(dict(name=name, group=group, description=description,
            parameters=parameters, schema=spec, base_schema=default_schema,
            source_description=fun['description'], output=output, backend=backend,
            switch=switch_for(name, options), tasks=tasks, changes_current_mask=changes,
            rules=rules_for(name)))
    return entries


def render_catalog(catalog):
    from agent_baseline.make_demo import CSS, esc, text_block
    groups = {}
    for item in catalog:
        groups.setdefault(item['group'], []).append(item)
    cards = []
    for group, entries in groups.items():
        cards.append(f'<section><h2>{esc(group)}</h2><div class="catalog-grid">')
        for item in entries:
            rows = []
            for p in item['parameters']:
                detail = dict(p['schema'])
                detail.pop('description', None)
                rows.append(f'<tr><td><code>{esc(p["name"])}</code></td><td>{esc(p["type"])}</td><td>{"是" if p["required"] else "否"}</td><td>{esc(p["meaning"])}{text_block(detail,"schema 约束")}{"仅 --external-examples" if p["switch"] else ""}</td></tr>')
            cards.append(f'<article class="panel"><h3><code>{item["name"]}</code></h3><p>{esc(item["description"])}</p><div class="tablewrap"><table><thead><tr><th>参数</th><th>类型</th><th>必填</th><th>含义 / 约束</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                         f'<p><strong>输出：</strong>{esc(item["output"])}</p><p><strong>后端：</strong>{esc(item["backend"])}</p><p><strong>开关：</strong>{esc(item["switch"]["label"])}</p><p><strong>任务：</strong>{esc(item["tasks"])}</p><p><strong>当前答案：</strong>{esc(item["changes_current_mask"])}</p><ul>'
                         + ''.join(f'<li>{esc(rule)}</li>' for rule in item['rules']) + '</ul>'
                         + text_block(item['source_description'],'代码中的用途原文') + '</article>')
        cards.append('</div></section>')
    css = '''.catalog{max-width:1400px;margin:auto;padding:20px}.catalog-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:14px;margin-bottom:24px}.catalog-grid>article{min-width:0}.catalog h2{margin:18px 0 10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}td{min-width:50px}td:last-child{min-width:160px}.dataflow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.dataflow>div{flex:1;min-width:150px;border:1px solid var(--line);padding:12px;background:var(--tint)}.catalog-grid li{margin:6px 0}.catalog-grid ul{padding-left:20px}@media(max-width:600px){.catalog{padding:12px}}'''
    flow = '<section class="panel"><h2>数据流</h2><div class="dataflow"><div>输入<br>T1、T2、指令<br>可选给定掩码 / 区域 / 示例 / 反馈 / 旧地图</div><span>→</span><div>规划器调用原子工具<br>查看、分割、比较、特征、运算、修改、核验</div><span>→</span><div>句柄与证据<br>mask#k、inst#k、props#k、feat#k<br>输入 region#0、map#0<br>外部 example#k、exmask#k；视图 view#k</div><span>→</span><div>submit → 提交门<br>报告 → 规划器复核<br>通过后输出二值掩码</div></div></section>'
    foot = '''<section class="panel"><h2>句柄、核验与提交</h2><p>工具创建新句柄，保留旧 mask / inst / props / feat 数据。select 和 remove_region 保留未删区域的 ID，其余编辑可重编号。当前答案是单独的指针；set_current=false 保留中间候选，不覆盖当前答案。map#0 与 region#0 是输入，不自动成为当前答案。特征网格只读；外部示例会缓存特征，核验账本与待办清单也会更新，因此“只出新句柄”不等于没有任何状态变化。</p>
<p>map_input.py 与 feedback.py 不注册额外工具。旧地图通过 --map-input 进入 map#0；外部示例通过 --external-examples 扩展 match_exemplar 的 exemplar，并增加可选 negative。反馈点和方位进入指令及 G3 相关组件筛选。完整参数来自 session_schemas，schema 没有单独定义参数含义时写“未定义”。</p>
<p>ask 不接收规划器的对话或理由。后端接收组件的 T1/T2 裁图与任务核验问题；--ask-correspondence 开启时是近景两张与环境两张，关闭时为两张近景。输出 yes/no、type、reason、可空 p_yes 及解析状态；p_yes 未校准。ask 不自动修改掩码。G3 自动核验受反馈相关性约束，显式 ids 仍可指定组件。</p>
<p>提交门需 --verify-stage。首次提交或仍欠核验时返回报告，规划器复核后编辑或再次 submit。G3 大框限制只用于 segment_prompt；区域外像素与方向违反进入报告。G4 在有区域时才对实际提交掩码与 region#0 求交；当前答案提醒仅比较句柄，方向违反仍只报告。异常/预算耗尽仍可兜底，页面选例会排除非 model 提交，除非全部兜底。</p></section>'''
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>核心工具 · 变化检测 Agent</title><style>' + CSS + css + '</style></head><body><header><h1>核心工具</h1><p>Agent 只能通过这些原子工具工作，没有一键输出变化掩码的工具。</p><p>工具名与参数直接读取注册 schema；后端及行为说明依据实现。共 ' + str(len(catalog)) + ' 个工具（含可选开关）。</p></header><main class="catalog">' + flow + ''.join(cards) + foot + '</main></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'tools_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.out_dir / 'index.html').write_text(render_catalog(catalog),encoding='utf-8')
    print(f'工具目录：{len(catalog)} 个工具')


if __name__ == '__main__':
    main()
