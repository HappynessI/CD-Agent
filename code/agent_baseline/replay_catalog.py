"""Inventory saved Agent sessions and render the G3 replay catalog."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

# Each entry points to a recorded configuration, never to a fresh model run.
GROUPS = (
    ('levir_c0_range', 'json_qwen37flash_g3_s1', 'T1.1xC0', 'LEVIR', '建筑变化', 'imc', 2),
    ('second_c0_g2', 'second_g2_any_s1', 'T1.1xC0', 'SECOND', '建筑变化', 'imc', 1),
    ('second_t12', 'json_qwen37flash_g3_t12_any_s1', 'T1.2xC0', 'SECOND', '建筑或低矮植被参与的变化', 't12', 2),
    ('second_c1t', 'json_qwen37flash_g3_c1t_appeared_s1', 'T1.1xC1t', 'SECOND', '低矮植被或非植被地表转为建筑', 'c1t', 1),
    ('second_removed', 'second_g5_disappeared_s1', 'T1.1xC1', 'SECOND', '拆除建筑', 'direction', 1),
    ('second_c2_g5', 'second_g5box_any_s1', 'T1.1xC2', 'SECOND', '框内建筑变化', 'region', 2),
    ('second_c4_g5', 'second_g5box_appeared_s1', 'T1.1xC4', 'SECOND', '框内新建建筑', 'region', 1),
    ('second_t31_g4', 'json_qwen37flash_g4_t31_any_s1', 'T3.1', 'SECOND', '图内变化示例', 't31', 2),
    ('levir_t41', 'json_qwen37flash_g3_t41_any_s1', 'T4.1', 'LEVIR', '指定面积的建筑变化', 't41', 2),
    ('second_t51c1', 't51c1_g3hint_appeared_s1', 'T5.1xC1', 'SECOND', '修正给定的新建建筑掩码', 't51', 2),
    ('second_t51_removed', 't51hint_hint_g3_disappeared_s1', 'T5.1xC1', 'SECOND', '修正掩码并保留拆除建筑', 't51', 1),
    ('whu_c0', 'whu_g3_any_s1', 'T1.1xC0', 'WHU', '建筑变化', 'imc', 2),
    ('whu_new', 'whu_g3_appeared_s1', 'T1.1xC1', 'WHU', '新建建筑', 'direction', 1),
    ('whu_removed', 'whu_g3_disappeared_s1', 'T1.1xC1', 'WHU', '拆除建筑', 'direction', 1),
    ('whu_t54', 'json_qwen37flash_g3_t54_any_s1', 'T5.4', 'WHU', '用新影像更新旧地图', 't54', 2),
)

TASK_LABELS = {
    'T1.1': '单类别', 'T1.2': '多类别',
    'T2.1': '地图可查', 'T2.2': '地图不可查',
    'T3.1': '图内点选', 'T3.2': '外部目标示例', 'T3.3': '变化对示例', 'T3.4': '正负示例',
    'T4.1': '属性描述', 'T4.2': '功能描述', 'T4.3': '事件影响分析', 'T4.4': '空间指代变化检测',
    'T5.1': '自主修正', 'T5.2': '点框反馈', 'T5.3': '语言反馈', 'T5.4': '地图更新',
}
TASK_FAMILIES = {'T1': '常见类别', 'T2': '稀缺类别', 'T3': '视觉示例', 'T4': '描述定义', 'T5': '已有结果更新'}
CONSTRAINTS = {'C0': '无附加约束', 'C1': '变化', 'C2': '空间', 'C3': '排除', 'C4': '组合'}
PENDING = {('T4.3', 'C0'): 'SpaceNet8，128 个洪水影响候选',
           ('T4.4', 'C0'): 'SECOND，128 个空间指代候选'}
FEEDBACK_REVIEW = {('T5.2', 'C0'), ('T5.3', 'C0')}
OFFLINE = {('T3.2', 'C0'): '有固定路线结果', ('T3.3', 'C0'): '有固定检索对照',
           ('T3.4', 'C0'): '有固定路线结果'}
OFFICIAL = {'whu_c0': ('any', 'c0'), 'whu_new': ('appeared', 'new'), 'whu_removed': ('disappeared', 'demolished')}
WHU_REVIEW = 'whu_gt_final_20261114'


def task_constraint(task):
    subtask, _, constraint = task.partition('x')
    return subtask, constraint[:2] or 'C0'


def official_scores(repo):
    folder = repo / 'results/agent_baseline' / WHU_REVIEW
    rows = {}
    with (folder / 'rescored_predictions.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            rows[row['config'], row['name'], row['direction']] = row
    return folder, rows


def inventory(root, repo):
    from agent_baseline.make_demo import mask_path, read_json, read_mask, resolve_asset
    from agent_baseline.make_site import exemplar_path, harness_of, measures

    official_folder, rescored = official_scores(repo)
    selected, population = [], []
    groups = [(spec, sorted((root / spec[1] / spec[2] / 'val').glob('*/final.json'))) for spec in GROUPS]
    for arm in ('p0v2', 'p1v2', 'p2v2'):
        for task, family in (('T1.1xC0', 'imc'), ('T5.1', 't51')):
            archive = repo / 'results/agent_baseline/harness1020_20261113/traces' / arm / task
            key = f'lev_{arm}_{task[1]}'
            description = '建筑变化' if family == 'imc' else '修正建筑变化掩码'
            groups.append(((key, arm, task, 'LEVIR', description, family, 1), sorted(archive.glob('*/final.json'))))

    for (key, run, task, dataset, description, family, limit), paths in groups:
        candidates = []
        unavailable = 0
        for path in paths:
            directory = path.parent
            if not (directory / 'input.json').is_file() or not (directory / 'steps.jsonl').is_file():
                unavailable += 1
                continue
            final, info = read_json(path), read_json(directory / 'input.json')
            metrics = final.get('eval') or {}
            if metrics.get('f1') is None or not final.get('mask_path') or not final.get('final_handle'):
                unavailable += 1
                continue
            image = directory.name
            inputs = info['source_images']
            gt = info.get('gt_mask_path') or str(Path(inputs[0]).parent.parent / 'label' / f'{image}.png')
            official = None
            if key in OFFICIAL:
                direction, subdir = OFFICIAL[key]
                existing = rescored.get((run, f'{image}.png', direction))
                if existing:
                    gt = str(official_folder / 'gt_official_v3' / subdir / f'{image}.png')
                    official = {k: existing[k] for k in ('gt_version', 'corrected', 'ignored_pixels')}
                    if direction != 'any':
                        official['ignore_path'] = str(official_folder / 'gt_official_v3/ignore' / f'{image}.png')
                else:
                    unavailable += 1
                    continue
            final_mask = resolve_asset(directory, final['mask_path'])
            if not all(resolve_asset(directory, item).is_file() for item in (*inputs, gt, final_mask)):
                unavailable += 1
                continue
            if not mask_path(directory, final['final_handle']).is_file():
                unavailable += 1
                continue
            comparison = official['corrected'] if official else metrics
            nonempty = comparison.get('tp', 0) + comparison.get('fn', 0) > 0
            if family == 't31':
                exemplar = exemplar_path(directory, info)
                if exemplar is None or not exemplar.is_file():
                    unavailable += 1
                    continue
                include = ~read_mask(exemplar)
                truth = read_mask(resolve_asset(directory, gt)) & include
                comparison = measures(read_mask(final_mask) & include, truth)
                nonempty = bool(truth.any())
            version = harness_of(info)
            if directory.is_relative_to(repo):
                source = directory.relative_to(repo).as_posix()
                version = f'{version} / {run.upper()}'
            else:
                source = directory.relative_to(root).as_posix()
            candidates.append(dict(task=task, dataset=dataset, run=run, image=image, version=version,
                                   description=description, family=family, session=str(directory), source=source,
                                   f1=metrics['f1'], selection_score=comparison['f1'], official=official, gt_path=gt,
                                   termination=final.get('termination_reason'), nonempty=nonempty,
                                   submitted_by=final.get('submitted_by')))

        nonempty = [case for case in candidates if case['nonempty']]
        normal = [case for case in nonempty if case['submitted_by'] == 'model'] or nonempty
        if not normal:
            normal = candidates
        picked = []
        if normal:
            best = max(normal, key=lambda case: (case['selection_score'], case['image']))
            picked.append((best, '较高 F1' if best['nonempty'] else '空真值示例', 'high'))
        if limit == 2 and len(nonempty) > 1:
            worst = min((case for case in nonempty if case is not best),
                        key=lambda case: (case['selection_score'], case['image']))
            picked.append((worst, '较低 F1', 'low'))
        for case, label, suffix in picked:
            selected.append(dict(case, page=f'{key}_{suffix}', selection=label))
        population.append(dict(group=key, run=run, task=task, discovered=len(paths), eligible=len(candidates),
                               unavailable=unavailable, nonempty=len(nonempty),
                               selected=[case['image'] for case, _, _ in picked]))
    return dict(records=selected, groups=population)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = inventory(args.run_root.resolve(), args.source_repo.resolve())
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(output["records"])} archived Agent sessions from {len(output["groups"])} configurations')


def normalize(records, historical):
    from agent_baseline.make_site import TASKS

    for record in historical:
        key = record['page']
        yield dict(record, file=f'{key}.html', description=TASKS[key][1], source=record['run'],
                   selection='原中位数案例', historical=True)
    for record in records:
        yield dict(record, file=f'{record["page"]}.html')


def metric_label(record):
    from agent_baseline.make_demo import pct

    if record.get('official'):
        return (f'{pct(record["official"]["corrected"]["f1"])} 官方 GT v3'
                f' / {pct(record["f1"])} 历史')
    if task_constraint(record['task'])[0] == 'T3.1':
        adjusted = record.get('scores', {}).get('agent', record.get('exemplar_excluded_f1'))
        label = f'{pct(record["f1"])} 含示例'
        return f'{label} / {pct(adjusted)} 扣示例' if adjusted is not None else label
    return pct(record['f1'])


def replay_switcher(record, siblings):
    from agent_baseline.make_demo import esc

    links = []
    for case in siblings:
        text = esc(f'{case["dataset"]} · {case["image"]} / {case["version"]}')
        if case['page'] == record['page']:
            links.append(f'<strong aria-current="page">{text}</strong>')
        else:
            links.append(f'<a href="{esc(case["file"])}">{text}</a>')
    return '<nav class="replay-switcher" aria-label="同任务与约束的案例"><span>切换本组案例</span>' + ''.join(links) + '</nav>'


def render_catalog(records):
    from agent_baseline.make_site import ASSETS, page, terms_block
    from agent_baseline.make_demo import esc

    records = sorted(records, key=lambda row: (task_constraint(row['task']), row['dataset'], row['page']))
    core = [row for row in records if not row['task'].startswith('T1.3')]
    archive = [row for row in records if row['task'].startswith('T1.3')]
    groups = Counter(task_constraint(row['task']) for row in core)
    fields = {
        'task': ('任务', {key: f'{key} {label}' for key, label in TASK_LABELS.items()}),
        'constraint': ('附加约束', {key: f'{key} {label}' for key, label in CONSTRAINTS.items()}),
        'dataset': ('数据集', {value: value for value in sorted({row['dataset'] for row in core})}),
        'harness': ('Harness', {value: value for value in sorted({row['version'] for row in core})}),
    }
    controls = []
    for name, (label, options) in fields.items():
        values = ''.join(f'<option value="{esc(key)}">{esc(title)}</option>' for key, title in options.items())
        controls.append(f'<label>{label}<select name="{name}"><option value="">全部</option>{values}</select></label>')

    def coverage_cell(task, constraint):
        key = task, constraint
        if key in groups:
            feedback = '<br><a class="case-name" href="data_review.html#T5">新数据待审核</a>' if key in FEEDBACK_REVIEW else ''
            return (f'<a href="#replays" data-coverage-task="{task}" data-coverage-constraint="{constraint}">'
                    f'{groups[key]} 个回放</a>' + feedback)
        if key in PENDING:
            return f'<a href="data_review.html#{task}">待审核</a><span class="case-name">{esc(PENDING[key])}</span>'
        if key in OFFLINE:
            return f'<span class="muted">Agent 未运行</span><span class="case-name">{esc(OFFLINE[key])}</span>'
        return '<span class="muted">未运行</span>'

    coverage_rows = []
    family = None
    for task, label in TASK_LABELS.items():
        first_in_family = family != task[:2]
        family = task[:2]
        family_label = f'<span class="case-name">{family} {TASK_FAMILIES[family]}</span>' if first_in_family else ''
        cells = ''.join(f'<td>{coverage_cell(task, constraint)}</td>' for constraint in CONSTRAINTS)
        coverage_rows.append(f'<tr class="{"family-start" if first_in_family else ""}"><th scope="row">{family_label}{task} {label}</th>{cells}</tr>')
    coverage_heads = ''.join(f'<th scope="col">{key}<span class="case-name">{label}</span></th>' for key, label in CONSTRAINTS.items())

    def case_row(record):
        task, constraint = task_constraint(record['task'])
        recipe = record['run'] if not record.get('historical') else '原入口'
        return (f'<tr data-replay-row data-task="{task}" data-constraint="{constraint}" '
                f'data-dataset="{esc(record["dataset"])}" data-harness="{esc(record["version"])}">'
                f'<td><a href="{esc(record["file"])}">{esc(record["image"])}</a>'
                f'<span class="case-name">{esc(record["description"])}</span></td>'
                f'<td class="case-tag">{task} × {constraint}</td>'
                f'<td class="case-tag">{esc(record["dataset"])}</td>'
                f'<td>{esc(record["version"])}<span class="case-name">{esc(recipe)}</span></td>'
                f'<td class="case-score">{esc(metric_label(record))}</td><td>{esc(record["selection"])}</td></tr>')

    archive_links = ''.join(
        f'<li><a href="{esc(row["file"])}">{esc(row["image"])}</a> · '
        f'{esc(row["dataset"])} · {esc(row["version"])} · 原单图 F1 {esc(metric_label(row))}</li>'
        for row in archive
    )
    body = (
        '<header class="catalog-head"><h1>变化检测 Agent 回放</h1>'
        '<p>查看真实会话的输入、工具调用、预测二值掩码、真值与单图评分。</p>'
        '<nav class="catalog-links" aria-label="主要导航"><a href="#replays">筛选回放</a>'
        '<a href="#coverage">任务覆盖</a><a href="data_review.html">数据审核</a>'
        '<a href="baselines.html">三种做法</a><a href="errors.html">典型错误</a>'
        '<a href="#archive">历史归档</a></nav></header><main class="catalog">'
        '<section class="panel" id="replays"><h2>会话回放</h2>'
        '<p>原案例沿用中位数选样。新增案例从已保存的指定配置选取较高或较低的单图 F1，优先使用非空真值。'
        'T3.1 按扣除示例后的 F1 选样，WHU C0/C1 按官方 GT v3 选样。</p>'
        '<form class="catalog-filters js-only" id="catalog-filters">' + ''.join(controls) + '<button type="reset">清除筛选</button></form>'
        f'<p id="filter-status" class="catalog-filter-status" role="status">共 {len(core)} 个会话</p>'
        '<p id="filter-empty" hidden>没有符合当前筛选的回放。可以清除筛选或查看任务覆盖。</p>'
        '<div class="tablewrap" tabindex="0" role="region" aria-label="会话回放表"><table><thead><tr><th scope="col">图像与指令</th><th scope="col">任务 × 约束</th>'
        '<th scope="col">数据</th><th scope="col">Harness 与配置</th><th scope="col">单图 F1</th>'
        '<th scope="col">选样依据</th></tr></thead><tbody>' + ''.join(map(case_row, core)) + '</tbody></table></div></section>'
        '<section class="panel coverage" id="coverage"><h2>T1–T5 × C0–C4</h2>'
        '<p>任务定义以 11 月 13 日修订为准。未运行表示当前研究归档没有可展示的 Agent 会话。约束列表示任务以外额外施加的要求。</p>'
        '<div class="tablewrap" tabindex="0" role="region" aria-label="任务覆盖表"><table><thead><tr><th scope="col">核心任务</th>' + coverage_heads
        + '</tr></thead><tbody>' + ''.join(coverage_rows) + '</tbody></table></div></section>'
        '<section class="panel" id="archive"><h2>历史归档</h2><h3>旧 T1.3 全部变化</h3>'
        '<p>2026-11-13 起退出核心任务体系。此处保留原指令、评分及逐步回放。</p><ul>' + archive_links
        + '</ul><nav class="catalog-links" aria-label="历史入口">'
        '<a href="original_index.html">原 G3 / G4 首页</a><a href="original_selection.json">原选样记录</a>'
        '<a href="../demo_g1/index.html">G1 回放</a>'
        '<a href="../trace_behavior_review_20261113/index.html">20 条历史轨迹行为复核</a></nav></section>'
        '<section class="panel"><h2>口径与工具</h2><nav class="catalog-links" aria-label="文档与工具">'
        '<a href="../../../doc/agent_baseline/task_system.html">任务定义</a>'
        '<a href="../tools/index.html">核心工具</a><a href="coverage.json">覆盖与选样清单</a></nav>'
        + terms_block(False) + '</section></main>'
    )
    script = (ASSETS / 'replay_catalog.js').read_text(encoding='utf-8')
    return page('变化检测 Agent 回放', body, f'<script>{script}</script>')


def review_data(repo):
    from agent_baseline.make_demo import jpeg, open_image

    tasks = repo / 'results/agent_baseline/task_review_20261113'
    sources = {
        'T4.3': tasks / 't43/full-v1/manifest.jsonl',
        'T4.4': tasks / 't44/v1/tasks.jsonl',
    }
    identifiers = {'T4.3': ('0_18_63-road', '0_12_14-building'), 'T4.4': ('-train_1051',)}
    result = {}
    for task, path in sources.items():
        with path.open() as source:
            rows = [json.loads(line) for line in source]
        chosen = [next(row for row in rows if row['id'].endswith(suffix)) for suffix in identifiers[task]
                  if any(row['id'].endswith(suffix) for row in rows)]
        if task == 'T4.3' and not any(row['target'] == 'building' for row in chosen):
            chosen.append(next(row for row in rows if row['target'] == 'building'))
        cards = []
        for row in chosen:
            fields = [(row['image_t1'], 'T1 较早影像'), (row['image_t2'], 'T2 较晚影像'),
                      (row['gt_mask'], '候选真值')]
            if row.get('meta', {}).get('reference_mask'):
                fields.append((row['meta']['reference_mask'], '空间参照掩码'))
            cards.append(dict(id=row['id'], instruction=row['instruction'],
                              images=[(jpeg(open_image(file), 512, 70), label) for file, label in fields]))
        result[task] = dict(total=len(rows), cards=cards, source=path.relative_to(tasks.parent).as_posix())
    return result


def render_reviews(data, records):
    from agent_baseline.make_site import page, figure
    from agent_baseline.make_demo import esc

    sections = []
    for task, title in (('T4.3', '洪水事件影响'), ('T4.4', '空间指代变化')):
        value = data[task]
        examples = []
        for card in value['cards']:
            images = ''.join(figure(uri, label) for uri, label in card['images'])
            examples.append(f'<article class="review-case"><h3>{esc(card["id"])}</h3><p>{esc(card["instruction"])}</p>'
                            f'<div class="pair">{images}</div></article>')
        sections.append(f'<section class="panel" id="{task}"><h2>{task} {title}</h2>'
                        f'<p>待人工审核 · {value["total"]} 条候选 · 候选真值用于审核。</p>'
                        f'<p><a href="../{esc(value["source"])}">查看完整候选清单</a></p>'
                        '<div class="review-gallery">' + ''.join(examples) + '</div></section>')
    whu = ''.join(f'<li><a href="{esc(record["file"])}">{esc(record["description"])} · {esc(record["image"])}</a>'
                  f' · F1 {esc(metric_label(record))}</li>' for record in records if record.get('official'))
    body = (
        '<header class="catalog-head"><nav class="catalog-links" aria-label="回放入口">'
        '<a href="index.html">回放首页</a></nav><h1>任务数据审核</h1>'
        '<p>查看候选样本、标签口径和待人工确认的问题。</p></header><main class="catalog">'
        + ''.join(sections) +
        '<section class="panel" id="T5"><h2>T5.2 / T5.3 反馈样本</h2>'
        '<p>LEVIR 共 104 条候选，19 条可视化样本在原数据审核档案中均标为待审核。'
        '现有 G3 回放使用旧任务输入与历史评分。</p>'
        '<a href="../data_revision_20261113/v2/review_v2_19/index.html">查看 19 条原始审核样本</a></section>'
        '<section class="panel" id="C2"><h2>C2 框内变化</h2>'
        '<p>待人工审核：框可能截断完整变化对象。把框、GT 碎块和原影像放在一起检查样本设计。</p>'
        '<a href="../trace_behavior_review_20261113/case_07.html">查看原 case-07</a></section>'
        '<section class="panel" id="WHU"><h2>WHU 官方 GT 重评分</h2>'
        '<p>C0 使用官方变化标签。C1 在官方变化像素上判定方向，方向不明的像素在 C1 评分中忽略。'
        '回放页同时显示历史分数和对应的离线重评分。</p>'
        '<ul>' + whu + '</ul><a href="../whu_gt_final_20261114/repair_report.md">查看修复报告及划分口径</a></section>'
        '</main><dialog id="review-zoom" aria-label="放大图像"><button type="button">关闭（Esc）</button><img alt=""></dialog>'
    )
    script = '''<script>
    const reviewZoom = document.getElementById('review-zoom');
    reviewZoom.querySelector('button').addEventListener('click', () => reviewZoom.close());
    document.querySelectorAll('.zoom').forEach(button => button.addEventListener('click', () => {
      const original = button.querySelector('img');
      const image = reviewZoom.querySelector('img');
      image.src = original.src;
      image.alt = original.alt;
      reviewZoom.showModal();
    }));
    </script>'''
    return page('任务数据审核 · 变化检测 Agent', body, script)


def audit_catalog(out, source_repo, pages):
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit

    from agent_baseline.make_g3_pages import audit_html

    class Anchors(HTMLParser):
        def __init__(self, target):
            super().__init__()
            self.ids = set()
            self.feed(target.read_text(encoding='utf-8'))

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if 'id' in attributes:
                self.ids.add(attributes['id'])

    root = out.parents[2]
    audit, anchors = {}, {}
    for filename in pages:
        file = out / filename
        data = audit_html(file.read_text(encoding='utf-8'))
        links = data.pop('links')
        for href in links:
            parsed = urlsplit(href)
            target = (file.parent / unquote(parsed.path)).resolve() if parsed.path else file
            if not target.exists() and target.is_relative_to(root):
                target = source_repo / target.relative_to(root)
            if not target.is_file():
                raise ValueError(f'{filename} has a missing target {href}')
            if parsed.fragment and target.suffix == '.html':
                if target not in anchors:
                    anchors[target] = Anchors(target).ids
                if unquote(parsed.fragment) not in anchors[target]:
                    raise ValueError(f'{filename} has a missing anchor {href}')
        audit[filename] = dict(data, relative_links=len(links))
    return {'pages': audit, 'total_pages': len(audit), 'local_links_checked': sum(row['relative_links'] for row in audit.values()),
            'embedded_images': sum(row['images'] for row in audit.values())}


def build_catalog(root, out, selection, cache, stage, source_repo):
    from filecmp import cmp

    from agent_baseline.make_demo import read_json
    from agent_baseline.make_site import Store, render_replay

    out, cache, source_repo = out.resolve(), cache.resolve(), source_repo.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    original = read_json(out / 'original_selection.json')
    records_file = cache / 'catalog.json'
    if stage != 'page':
        if selection is None:
            raise ValueError('--catalog data stage requires an inventory from replay_catalog')
        scanned = read_json(selection)
        records = list(normalize(scanned['records'], original))
        for record in records:
            if record.get('historical'):
                continue
            store = Store(cache / f'{record["page"]}.json', 'build')
            _, info = render_replay(root, record, store)
            record.update(info)
            store.save()
        data_store = Store(cache / 'review.json', 'build')
        data_store.get('reviews', lambda: review_data(source_repo))
        data_store.save()
        records_file.write_text(json.dumps(dict(records=records, groups=scanned['groups']), ensure_ascii=False), encoding='utf-8')
    if stage == 'data':
        return

    metadata = read_json(records_file)
    records = metadata['records']
    siblings = {}
    for record in records:
        siblings.setdefault(task_constraint(record['task']), []).append(record)
    documents = {}
    for record in records:
        if record.get('historical'):
            continue
        store = Store(cache / f'{record["page"]}.json', 'read')
        document, _ = render_replay(root, dict(record, siblings=siblings[task_constraint(record['task'])]), store)
        documents[record['file']] = document
    documents['index.html'] = render_catalog(records)
    data_store = Store(cache / 'review.json', 'read')
    documents['data_review.html'] = render_reviews(data_store.data['reviews'], records)

    for name, text in documents.items():
        (out / name).write_text(text, encoding='utf-8')
    public = [{k: v for k, v in row.items() if k not in {'session', 'gt_path'}} for row in records]
    for row in public:
        if row.get('official'):
            row['official'] = {k: v for k, v in row['official'].items() if k != 'ignore_path'}
    coverage = Counter(task_constraint(row['task']) for row in records if not row['task'].startswith('T1.3'))
    coverage_rows = []
    for task in TASK_LABELS:
        for constraint in CONSTRAINTS:
            key = task, constraint
            status = 'replay' if key in coverage else 'pending_review' if key in PENDING else 'not_run'
            cell = dict(task=task, constraint=constraint, status=status, count=coverage[key],
                        evidence=PENDING.get(key) or OFFLINE.get(key))
            if key in FEEDBACK_REVIEW:
                cell['new_feedback_data'] = 'pending_review'
            coverage_rows.append(cell)
    (out / 'coverage.json').write_text(json.dumps(dict(records=public, coverage=coverage_rows, configurations=metadata['groups']),
                                                   ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    existing = [row['file'] for row in records if row.get('historical')]
    audit = audit_catalog(out, source_repo, [*documents, *existing, 'original_index.html', 'baselines.html', 'errors.html'])
    if not cmp(out / 'selection.json', out / 'original_selection.json', shallow=False):
        raise ValueError('Legacy selection differs from its preserved original')
    audit['legacy_selection_preserved'] = True
    source_site = source_repo / out.relative_to(out.parents[2])
    if out != source_site:
        for name in existing:
            if not cmp(out / name, source_site / name, shallow=False):
                raise ValueError(f'Historical replay changed: {name}')
        audit['historical_replays_verified'] = len(existing)
        preserved = {'original_index.html': 'index.html', 'original_selection.json': 'selection.json'}
        for backup, legacy in preserved.items():
            reference = source_site / backup
            if not reference.is_file():
                reference = source_site / legacy
            if not cmp(out / backup, reference, shallow=False):
                raise ValueError(f'Historical snapshot changed: {backup}')
        audit['historical_snapshots_verified'] = list(preserved)
    (cache / 'page_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{audit["total_pages"]} HTML pages, {audit["embedded_images"]} embedded images, {audit["local_links_checked"]} links')


if __name__ == '__main__':
    main()
