"""Read-only p_yes audit. No model client, API, GPU or job submission.

Run from batch97_pyes with the interpreter and environment in NOTES.md.
Only development data enters select_band; report data is evaluated afterwards.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from agent_baseline.analysis.io_retry import read_text, read_image, retry_io, write_json, write_text

DEFAULT_RUN = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
GRID = tuple(i / 10 for i in range(11))


def probability(value):
    """Missing values remain missing: never substitute binary answers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def bin_index(value):
    p = probability(value)
    return 10 if p is None else min(9, int(p * 10))


def segments(rows):
    out = [dict(band=f'[{i/10:.1f},{(i+1)/10:.1f}' + (']' if i == 9 else ')'),
                components=0, tp=0, fp=0) for i in range(10)]
    out.append(dict(band='p_yes 缺失', components=0, tp=0, fp=0))
    for row in rows:
        if row.get('answer') == 'no':
            dest = out[bin_index(row.get('p_yes'))]
            dest['components'] += 1
            dest['tp'] += row['tp']
            dest['fp'] += row['fp']
    return out


def f1(counts):
    c = np.asarray(counts, dtype=float)
    total = c.sum(axis=0) if c.ndim == 2 else c
    tp, fp, fn = total
    return float(200 * tp / max(1, 2 * tp + fp + fn))


def signed(value):
    return f'{value:+.3f}' if 0 < abs(value) < .005 else f'{value:+.2f}'


def in_band(row, band):
    p = probability(row.get('p_yes'))
    return row.get('answer') == 'no' and band is not None and p is not None and band[0] <= p <= band[1]


def vectors(data, band=None, perfect=False, all_no=False):
    """Image rows (TP, FP, FN). Perfect = retain only strictly TP-majority no's."""
    out = np.zeros((len(data['images']), 3), dtype=np.int64)
    index = {name: i for i, name in enumerate(data['images'])}
    out[:, 2] = [data['truth_pixels'][name] for name in data['images']]
    for row in data['rows']:
        rescue = row.get('answer') == 'no' and (all_no or in_band(row, band))
        if perfect:
            rescue = rescue and row['tp'] > row['fp']
        if row.get('answer') != 'no' or rescue:
            i = index[row['image']]
            out[i, 0] += row['tp']
            out[i, 1] += row['fp']
            out[i, 2] -= row['tp']
    return out


def select_band(development, grid=GRID):
    """Fixed grid, closed intervals a<b; baseline/no change is also a candidate.

    Maximize pooled development F1; ties choose narrower then lower interval.
    This function deliberately has no report-set argument.
    """
    best_band, best_score = None, f1(vectors(development))
    best_width = 0.
    for i, a in enumerate(grid):
        for b in grid[i+1:]:
            score = f1(vectors(development, (a, b)))
            width = round(b - a, 10)
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and width < best_width):
                best_band, best_score, best_width = [a, b], score, width
    return best_band


def paired_ci(changed, base, samples=5000, seed=1016):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, len(base), len(base))
        values.append(f1(changed[idx]) - f1(base[idx]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def evaluate(data, band, selection_available=True):
    base = vectors(data)
    result = dict(rule_f1=f1(base), rule_counts=base.sum(0).tolist())
    for name, perfect, all_no in [('band', False, False), ('perfect_band', True, False),
                                  ('perfect_all_no', True, True)]:
        if not selection_available and not all_no:
            continue
        changed = vectors(data, band, perfect, all_no)
        result[name] = dict(f1=f1(changed), delta_f1=f1(changed) - f1(base),
                            ci95=paired_ci(changed, base), counts=changed.sum(0).tolist())
    if not selection_available:
        result['band'] = None
        result['perfect_band'] = None
    return result


def specifications(run):
    pools = {}
    levir_truth = Path('/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD')
    def levir(role, split, names, tag):
        return dict(name='levir_'+role, names=names, mask_dir=f'init/sam3_imc_p85/{split}',
                    answers=f'ask_eval/{tag}/f37_corr.jsonl', truth_dir=str(levir_truth/f'{split}_original/label'),
                    truth_threshold=127, truth_source=f'LEVIR-CD/{split}_original/label')
    pools['levir'] = dict(label='LEVIR 0.85', dev=levir('dev','train','dev2/names.txt','dev2'),
                          report=levir('report','val','dev2/val_names.txt','val_sam3_imc_p85'))
    for dataset, directions in [('second',('appeared','disappeared')), ('whu',('any','appeared','disappeared'))]:
        for direction in directions:
            gt = {'any':'c0','appeared':'new','disappeared':'demolished'}[direction]
            report = dict(name=f'{dataset}_{direction}_report',
                names='second/second_val128.txt' if dataset == 'second' else 'whu/names_val128.txt',
                mask_dir=f'{dataset}/fixed/{direction}', answers=f'{dataset}/rule/{direction}/answers.jsonl',
                settings=f'{dataset}/rule/{direction}/settings.json', saved_rule=f'{dataset}/rule/{direction}/masks',
                truth_dir=str(run/f'{dataset}/{"gt_official_v3" if dataset == "whu" else "gt"}/{gt}'), truth_source=f'{dataset}/{"gt_official_v3" if dataset == "whu" else "gt"}/{gt}')
            dev = dict(name=f'{dataset}_{direction}_dev',
                names='second_dev/names.txt' if dataset == 'second' else 'whu/names_dev60.txt',
                mask_dir=f'second_dev/second/fixed/{direction}' if dataset == 'second' else f'whu/fixed/{direction}',
                answers=f'second_dev/rule_card_defined/{direction}/{direction}/answers.jsonl' if dataset == 'second' else f'whu/rule/{direction}/answers.jsonl',
                truth_dir=str(run/f'second_dev/second/gt/{gt}') if dataset == 'second' else str(run/f'whu/gt_official_v3/{gt}'),
                truth_source=f'second_dev/second/gt/{gt}' if dataset == 'second' else f'whu/gt_official_v3/{gt}')
            if dataset == 'second':
                dev.update(settings=f'second_dev/rule_card_defined/{direction}/{direction}/settings.json',
                           saved_rule=f'second_dev/rule_card_defined/{direction}/{direction}/masks')
            pools[f'{dataset}_{direction}'] = dict(label=dataset.upper()+' '+{'any':'C0','appeared':'新建','disappeared':'拆除'}[direction], dev=dev, report=report)
    return pools


def settings_audit(run, spec):
    path = spec.get('settings')
    if not path or not exists(run/path):
        return dict(settings_file=path, available=False,
                    variant='f37_corr', model='qwen3.7-flash', evidence='ask 记录 variant 与仓库 VARIANTS；无逐请求 settings 快照')
    setting = json_file(run/path)
    return dict(available=True, settings_file=path, direction=setting['direction'], scope=setting.get('scope'),
                variant=setting['variant'], model='qwen3.7-flash', ask_variant=setting.get('ask_variant'),
                targets=sorted(set(str(t) for t in setting.get('targets',[]))),
                mask_dir=str(Path(setting['mask_dir']).relative_to(run)),
                ids_match_names={Path(n).stem for n in setting['ids']} == {Path(n).stem for n in read_text(run/spec['names']).split()},
                evidence='settings + 仓库 f37_corr 定义；原始请求未归档')


def request_estimates(data, report_data, name):
    inv, ref = data['inventory'], report_data['inventory']
    unit = inv['unit_cost_cny'] or ref['unit_cost_cny']
    issues = []
    if inv['missing_masks']:
        # There is no candidate pool to count. Report-set density is only a budget proxy.
        projected = math.ceil(ref['components'] * len(inv['missing_masks']) / ref['images_expected'])
        issues.append(dict(pool=name, reason='开发固定掩码缺失，开发 ask 缺失；先生成固定路线再点数组件',
                           components=None, requests=None, estimated_requests=projected,
                           estimate_basis='同方向报告集组件数 × 缺失开发图数 / 报告图数，向上取整；不是实测',
                           unit_cost_cny=unit, estimated_cost_cny=projected*unit if unit else None))
        return issues
    missing_p = inv['answers'] - inv['p_yes_readable']
    for reason, count in [('缺 ask 答案',inv['missing_answers']), ('已有 ask 的 p_yes 缺失',missing_p)]:
        if count:
            issues.append(dict(pool=name, reason=reason, components=count, requests=count,
                               unit_cost_cny=unit, estimated_cost_cny=count*unit if unit else None,
                               estimate_basis='每组件一请求；同池记录 cost_cny / priced_requests；不保证重问能读到 p_yes'))
    return issues


def plot_pool(output, label, parts):
    os.environ['MPLCONFIGDIR'] = str(output.parent.parent/'tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # A CJK font for the Chinese labels; set CJK_FONT to a .ttf/.otf path, otherwise matplotlib's default is used.
    font = FontProperties(fname=os.environ['CJK_FONT']) if os.environ.get('CJK_FONT') else FontProperties()
    fig, axes = plt.subplots(1,len(parts),figsize=(8*len(parts),4.8),squeeze=False)
    for ax,(role,data) in zip(axes[0],parts):
        if not data['complete_masks']:
            ax.set_axis_off()
            ax.set_title(label+' · '+role+'（缺候选池记录）',fontproperties=font)
            ax.text(.5,.5,'缺固定路线和 ask；无法分段',transform=ax.transAxes,
                    ha='center',fontproperties=font)
            continue
        table = segments(data['rows'])
        x = np.arange(11)
        ax.bar(x,[r['tp'] for r in table],label='真变化 TP',color='#348abd')
        ax.bar(x,[r['fp'] for r in table],bottom=[r['tp'] for r in table],label='误检 FP',color='#e89b45')
        ax.set_xticks(x,[f'{i/10:.1f}–{(i+1)/10:.1f}' for i in range(10)]+['缺失'],rotation=45,fontproperties=font)
        ax.set_title(label+' · '+role+f"（no：{sum(r['components'] for r in table)} 个）",fontproperties=font)
        ax.set_ylabel('组件内像素数',fontproperties=font)
        ax.set_xlabel('p_yes 分段；前九段右开，末段含 1',fontproperties=font)
        ax.legend(prop=font)
    fig.tight_layout()
    retry_io(lambda: fig.savefig(output,dpi=150))
    plt.close(fig)


def markdown(report):
    lines = ['# 验证器 p_yes 离线分析','',
        '只读已有记录。未调用 API，未用 GPU，未提交 Slurm。路径均相对 R；外部真值仅列数据集内位置。', '',
        'F1 为整集累计像素分数（0–100）；变化单位为分。按图配对 bootstrap 5000 次，种子 1016，95% 百分位区间。',
        '规则只删除 answer=no。未解析和缺答案保留。p_yes 缺失单列，绝不以二值答案填充。',
        '固定网格 0, 0.1, …, 1，候选为全部 a<b 的闭区间 [a,b]，另含“不保留任何 no”。只在开发集最大化 F1；并列取更窄，再取更低区间。',
        '完美复核仅保留 TP>FP 的整组件（恰好一半不留）；真值只用于天花板，不代表可实现的复核器，也不是所有组件子集的数学最优值。', '',
        '## 盘点','', '| 池 | 划分 | 图（有掩码/应有） | 组件 | 答案 | 可读 p_yes | no 可读/总数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,pool in report['pools'].items():
        for role in ('dev','report'):
            inv = pool[role]['inventory']
            rate = '无法统计' if inv['p_yes_fraction'] is None else f"{100*inv['p_yes_fraction']:.2f}%"
            comps = str(inv['components']) if inv['images_with_masks'] else '未知（缺掩码）'
            lines.append(f"| {pool['label']} | {'开发' if role=='dev' else '报告'} | {inv['images_with_masks']}/{inv['images_expected']} | {comps} | {inv['answers']} | {inv['p_yes_readable']}（{rate}） | {inv['no_p_yes_readable']}/{inv['no_components']} |")
    lines += ['', '## 固定区间的结果', '', '| 池 | 开发区间 | 开发规则→区间 F1 | 报告规则→区间 F1 | 报告 Δ [95% CI] | 完美复核区间 Δ | 全部 no 完美复核 Δ |', '|---|---|---|---|---|---|---|']
    for name,pool in report['pools'].items():
        dev,rep = pool['dev']['evaluation'],pool['report']['evaluation']
        if pool['selection_available']:
            band = pool['selected_band']
            bandstr = str(band) if band is not None else '不恢复 no'
            value = rep['band']
            lines.append(f"| {pool['label']} | {bandstr} | {dev['rule_f1']:.2f}→{dev['band']['f1']:.2f} | {rep['rule_f1']:.2f}→{value['f1']:.2f} | {signed(value['delta_f1'])} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}] | {rep['perfect_band']['delta_f1']:+.2f} | {rep['perfect_all_no']['delta_f1']:+.2f} |")
        else:
            lines.append(f"| {pool['label']} | 无开发记录，未选择 | — | {rep['rule_f1']:.2f}→未评估 | — | — | {rep['perfect_all_no']['delta_f1']:+.2f} |")
    lines += ['', '全部 no 天花板包括 p_yes 缺失组件。WHU 的天花板仅作诊断，不能代替开发集选型或据此启动 G6。区间固定，bootstrap 中不重新选区间。', '', '## 问法、模型和 settings', '']
    for pool in report['pools'].values():
        lines += [f"- {pool['label']}：{pool['settings_comparison']}"]
        for role in ('dev','report'):
            src = pool[role]['inventory']['sources']
            lines += [f"  - {'开发' if role=='dev' else '报告'}："+'；'.join(f'{k}=`{v}`' for k,v in src.items())+'。']
    lines += ['', '## no 组件分段', '']
    for name,pool in report['pools'].items():
        lines += [f"### {pool['label']}", '', f"![{pool['label']}](pyes_{name}.png)", '', '| 划分 | p_yes | 组件数 | TP 像素 | FP 像素 |', '|---|---|---:|---:|---:|']
        for role in ('dev','report'):
            for row in pool[role]['segments']:
                lines.append(f"| {'开发' if role=='dev' else '报告'} | {row['band']} | {row['components']} | {row['tp']} | {row['fp']} |")
    lines += ['', '## 需要补问（只列清单，未执行）', '', '| 池 | 原因 | 组件数 | 请求数 | 估计费用（元） |', '|---|---|---:|---:|---:|']
    for issue in report['needs_asking']:
        n = issue.get('requests')
        requests = str(n) if n is not None else f"未知；预算暂按 {issue['estimated_requests']}"
        cost = issue['estimated_cost_cny']
        lines.append(f"| {issue['pool']} | {issue['reason']} | {issue['components'] if issue['components'] is not None else '未知'} | {requests} | {cost:.4f} |")
    lines += ['',f"不重叠项目合计预算约 {report['estimated_followup_cost_cny']:.4f} 元。每组件一请求；缺失概率的补问包含已有 yes 和 no。只补 no 的备选请求数也保存在 JSON。重问不能保证得到 p_yes，不能无上限重试。", '', report['cost_note'], '', '## 结论', '', *report['conclusions'], '', '## 局限', '',
        'WHU 开发 60 张没有固定路线掩码和 ask；组件数与精确请求数不可从现有文件恢复，费用按报告池组件密度作预算估计。',
        '置信读数不是校准概率。缺失与问法、历史请求有关，不能当随机缺失。LEVIR 验证集过去已用于问法开发，本次只保证区间由 dev2 选择。',
        '未补问，未运行 Agent，未设计或上线 G6；由 Claude 审核后决定是否补问。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--reuse-inventory', action='store_true', help='Reuse this directory\'s explicit offline inventory cache')
    args = parser.parse_args()
    work = Path.cwd().resolve()
    # All outputs are tied to cwd; no output argument can redirect outside it.
    os.environ['TMPDIR'] = str(work/'tmp')
    retry_io(lambda: (work/'tmp').mkdir(exist_ok=True))
    retry_io(lambda: (work/'results').mkdir(exist_ok=True))
    started = time.monotonic()
    run = args.run_root.resolve()
    specs = specifications(run)
    cache = work/'tmp/pyes_inventory.json'
    if args.reuse_inventory:
        cached = json_file(cache)
        from agent_baseline.whu_gt import VERSION
        if cached.get('whu_gt_version') != VERSION:
            raise ValueError('Stale WHU inventory: rebuild without --reuse-inventory')
        datasets = cached['datasets']
        comparisons = cached['comparisons']
        inventory_seconds = cached['inventory_seconds']
    else:
        datasets, comparisons = {}, {}
        for name,spec in specs.items():
            datasets[name] = {}
            for role in ('dev','report'):
                print('inventory',name,role,flush=True)
                data = load_pool(run,spec[role])
                data['settings'] = settings_audit(run,spec[role])
                datasets[name][role] = data
                write_json(work/'tmp/pyes_inventory_progress.json', datasets)
            dev,rep = datasets[name]['dev'],datasets[name]['report']
            # Lists, not just nonempty masks: disjointness must hold for every planned image.
            names_dev = {Path(n).stem for n in read_text(run/spec['dev']['names']).split()}
            names_rep = {Path(n).stem for n in read_text(run/spec['report']['names']).split()}
            if names_dev & names_rep:
                raise ValueError(f'Development/report overlap: {name}')
            if name == 'levir':
                comparisons[name] = 'L1=f37_corr，模型 qwen3.7-flash，四图对应问法。dev2 与报告集使用同一变体；ask_eval 无独立 settings。历史 p_yes 可读率不同，不能据此假设请求完全同分布。'
            elif name.startswith('second'):
                old = json_file(run/'code_k/agent_baseline/scopes/dev_card_defined.json')
                current = json_file(work/'code/agent_baseline/scopes/building_rebuild_excluded.json')
                if old != current:
                    raise ValueError('Development scope card differs from final scope')
                for role in ('dev','report'):
                    cfg = datasets[name][role]['settings']
                    if not cfg['ids_match_names'] or cfg['variant'] != 'f37_corr' or cfg['direction'] != name.split('_',1)[1]:
                        raise ValueError('SECOND settings do not match chosen pool')
                comparisons[name] = 'settings：开发 scope=dev_card_defined，报告 scope=building_rebuild_excluded，名称不同；已逐字段核对 code_k 中开发范围卡与当前最终卡，内容完全相同（含“范围卡加一句定义”）。direction、f37_corr 相同；模型按变体定义为 qwen3.7-flash。未归档逐请求完整 prompt，不能验证服务端历史变更。'
            else:
                comparisons[name] = '报告 settings：building_rebuild_excluded + f37_corr，模型定义为 qwen3.7-flash，与 SECOND 最终范围卡同口径。开发图未找到固定路线和 ask，无法比较开发请求。报告 answers 中没有开发图 uid。'
        inventory_seconds = time.monotonic()-started
        from agent_baseline.whu_gt import VERSION
        write_json(cache,dict(whu_gt_version=VERSION,datasets=datasets,comparisons=comparisons,inventory_seconds=inventory_seconds))
    # Freeze every selection from development data before any report F1 evaluation.
    selections = {}
    for name,pair in datasets.items():
        inv = pair['dev']['inventory']
        # An unparsed reply is a known keep under the frozen rule, not a missing image.
        available = pair['dev']['complete_masks'] and inv['missing_answers'] == 0
        selections[name] = dict(available=available, band=select_band(pair['dev']) if available else None)
    write_json(work/'results/pyes_selection.json',dict(grid=GRID,source='development only',selections=selections))
    report = dict(grid=GRID, interval='closed [a,b], a<b', bootstrap=dict(samples=5000,seed=1016,unit='paired images'),
                  pools={}, needs_asking=[], conclusions=[], inventory_seconds=inventory_seconds,
                  reused_inventory=args.reuse_inventory)
    for name,pair in datasets.items():
        print('evaluate',name,flush=True)
        chosen = selections[name]
        pool = dict(label=specs[name]['label'],selected_band=chosen['band'],selection_available=chosen['available'],
                    settings_comparison=comparisons[name])
        for role,data in pair.items():
            # No baseline is claimed for a development set without its candidate masks.
            evaluation = evaluate(data,chosen['band'],chosen['available']) if data['complete_masks'] else None
            pool[role] = dict(inventory=data['inventory'],settings=data['settings'],
                              segments=segments(data['rows']),evaluation=evaluation)
            if role == 'dev' or data['inventory']['missing_answers'] or data['inventory']['answers'] != data['inventory']['p_yes_readable']:
                issues = request_estimates(data,pair['report'],name+'_'+role)
                for issue in issues:
                    issue['no_only_alternative_requests'] = data['inventory']['no_components']-data['inventory']['no_p_yes_readable'] if data['complete_masks'] else None
                report['needs_asking'].extend(issues)
        report['pools'][name] = pool
    report['estimated_followup_cost_cny'] = sum(i['estimated_cost_cny'] or 0 for i in report['needs_asking'])
    report['cost_note'] = ('费用按原记录 cost_cny 的均价（已有 ledger 口径），不是新调用实测或账单。'
        'LEVIR 报告记录保留了早期较高计价；不把它当作当前 flash 单价。WHU 开发预算沿用同方向报告均价，'
        '组件数按 60/128 的报告密度估计。改首词元格式来提高可读率会改变问法；若决定补问，应另列实验，不与原记录直接混作同一池。')
    for name,pool in report['pools'].items():
        rep = pool['report']['evaluation']
        if pool['selection_available']:
            delta = rep['band']['delta_f1']
            perfect = rep['perfect_band']['delta_f1']
            report['conclusions'].append(f"{pool['label']}：开发区间迁移到报告集的 ΔF1={signed(delta)} 分，{'达到' if delta>=1 else '未达到'} +1 分门槛；区间内完美复核为 {perfect:+.2f} 分。")
        else:
            report['conclusions'].append(f"{pool['label']}：开发数据缺失，不能判断开发区间是否达到 +1 分；全部 no 的 TP 主导组件天花板为 {rep['perfect_all_no']['delta_f1']:+.2f} 分，仅作诊断。")
    ceilings = [report['pools'][f'second_{direction}']['report']['evaluation']['perfect_all_no']['delta_f1']
                for direction in ('appeared','disappeared')]
    report['conclusions'].append(f'SECOND 全部 no 的 TP 主导组件天花板为新建 {ceilings[0]:+.2f}、拆除 {ceilings[1]:+.2f} 分；它与开发区间迁移收益是两项结果，不能把天花板写成已通过区间迁移门槛。')
    report['conclusions'].append('第 10.5 节 V1 在 dev2 全 200 张为 +0.34，但追加的独立 170 张确认区间跨零，因此 L2 没上线；不是证明 V1 无效。V4 针对置信读数可用性。本次利用已有 L1 的 p_yes，不修复 V4，也不证明 flash 规划器复核能实现真值天花板；需统计复核引入的 FP 和净收益。')
    report['analysis_seconds'] = time.monotonic()-started
    write_json(work/'results/pyes_band.json',report)
    write_text(work/'results/pyes_band.md',markdown(report))
    for name,pair in datasets.items():
        plot_pool(work/f'results/pyes_{name}.png',report['pools'][name]['label'],
                  [('开发集',pair['dev']),('报告集',pair['report'])])
    print(json.dumps(dict(status='complete',seconds=report['analysis_seconds'],conclusions=report['conclusions']),ensure_ascii=False),flush=True)


def json_file(path):
    return json.loads(read_text(path))


def exists(path):
    # Path.exists may hide I/O errors on some Python versions; stat must retry.
    try:
        retry_io(lambda: path.stat())
        return True
    except FileNotFoundError:
        return False


def answers(path):
    if not exists(path):
        return {}, dict(lines=0, duplicates=0)
    seq = [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    by_uid = {row['uid']: row for row in seq}
    return by_uid, dict(lines=len(seq), duplicates=len(seq) - len(by_uid))


def component_pixels(mask, truth, valid=None):
    """Same scipy 8-connectivity and scan order as common.components, no min area.

    bincount avoids repeatedly scanning a whole image for each component.
    """
    from scipy import ndimage
    labels, n = ndimage.label(mask, np.ones((3, 3), bool))
    valid = np.ones_like(mask, bool) if valid is None else valid
    areas = np.bincount(labels[valid], minlength=n+1)
    tp = np.bincount(labels[truth & valid], minlength=n+1)
    return [(i, int(tp[i]), int(areas[i] - tp[i])) for i in range(1, n+1)]


def load_pool(run, spec):
    """Inventory all expected components, including images with zero components."""
    names = [Path(n).stem for n in read_text(run / spec['names']).split()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image names')
    answer_rows, audit = answers(run / spec['answers'])
    rows, truths, expected = [], {}, set()
    missing_masks, saved_mismatch = [], []
    for k, stem in enumerate(names):
        mask_path = run / spec['mask_dir'] / (stem + '.png')
        if not exists(mask_path):
            missing_masks.append(stem)
            continue
        mask = read_image(mask_path) > 0
        truth_path = Path(spec['truth_dir']) / (stem + '.png')
        if 'gt_official_v3' in truth_path.parts:
            from agent_baseline.evaluation import load_reference
            truth, valid = load_reference(truth_path)
        else:
            truth = read_image(truth_path) > spec.get('truth_threshold', 0)
            valid = np.ones_like(truth, bool)
        if mask.ndim != 2 or mask.shape != truth.shape:
            raise ValueError(f'Unexpected mask/truth shape for {stem}')
        truths[stem] = int(truth.sum())
        pixels = component_pixels(mask, truth, valid)
        for ident, tp, fp in pixels:
            uid = f'{stem}:{ident}'
            expected.add(uid)
            reply = answer_rows.get(uid, {})
            rows.append(dict(image=stem, uid=uid, tp=tp, fp=fp,
                             answer=reply.get('answer'), p_yes=probability(reply.get('p_yes'))))
        if spec.get('saved_rule'):
            saved = read_image(run / spec['saved_rule'] / (stem + '.png')) > 0
            # Compare by component counts using the exact labels and no policy.
            base_tp = sum(tp for ident,tp,fp in pixels if answer_rows.get(f'{stem}:{ident}',{}).get('answer') != 'no')
            base_fp = sum(fp for ident,tp,fp in pixels if answer_rows.get(f'{stem}:{ident}',{}).get('answer') != 'no')
            if [int((saved & truth & valid).sum()), int((saved & ~truth & valid).sum())] != [base_tp, base_fp]:
                saved_mismatch.append(stem)
        if k % 25 == 0:
            print(spec['name'], k+1, '/', len(names), flush=True)
    matched = {uid: reply for uid, reply in answer_rows.items() if uid in expected}
    readable = sum(probability(a.get('p_yes')) is not None for a in matched.values())
    no = [row for row in rows if row['answer'] == 'no']
    costs = [a['cost_cny'] for a in matched.values() if isinstance(a.get('cost_cny'), (int,float))]
    audit.update(images_expected=len(names), images_with_masks=len(truths), components=len(rows),
                 answers=len(matched), valid_answers=sum(a.get('answer') in ('yes','no') for a in matched.values()),
                 p_yes_readable=readable, p_yes_fraction=readable/max(1,len(matched)) if matched else None,
                 no_components=len(no), no_p_yes_readable=sum(row['p_yes'] is not None for row in no),
                 missing_answers=len(expected-set(answer_rows)), extra_answers=len(set(answer_rows)-expected),
                 missing_masks=missing_masks, saved_rule_count_mismatches=saved_mismatch,
                 cost_cny=sum(costs), priced_requests=len(costs),
                 unit_cost_cny=sum(costs)/len(costs) if costs else None,
                 sources={k:v for k,v in spec.items() if k in ('names','mask_dir','answers','settings','saved_rule')})
    # External truth locations are recorded by dataset role, not absolute path.
    audit['truth_source'] = spec['truth_source']
    return dict(images=[n for n in names if n in truths], truth_pixels=truths, rows=rows,
                inventory=audit, complete_masks=not missing_masks)


if __name__ == '__main__':
    main()
