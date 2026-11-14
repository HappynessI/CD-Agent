"""Readable reports from saved development metrics; no inference or GT selection."""
import argparse
import json
from pathlib import Path
from .io_retry import write_text


def load(path):
    return json.loads(path.read_text())


def number(value):
    return '—' if value is None else f'{value:.3f}'


def t3_report(root):
    data = load(root/'dev60/summary.json')
    lines = ['# T3 semantic bridge: fixed development results', '',
        f"Completed tasks: {data['tasks']}/60. Model: qwen3.7-flash.", '',
        'The two single-date exemplar requests contain no task category. DINO and bridge use the same cached proposals; verification uses the same eight-view question and shared candidate responses. All scores exclude the supplied exemplar. Oracle class ranking is an offline diagnostic only.', '',
        'Class agreement is agreement with existing labels, not adjudicated visual understanding accuracy. Individual grounding and label ambiguities require the saved-input audits.', '',
        '| Group | Tasks | Before agreement | After agreement | Pair agreement |',
        '|---|---:|---:|---:|---:|']
    for group, row in data['groups'].items():
        rates = [number(100*row['semantic'][key]) if row['semantic'][key] is not None else '—'
                 for key in ('before_correct','after_correct','pair_correct')]
        lines.append(f"| {group} | {row['tasks']} | {' | '.join(rates)} |")
    lines += ['', '| Group | Method | F1 | Precision | Recall | Candidate pixel coverage | Component recall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group, row in data['groups'].items():
        for method, metrics in row['metrics'].items():
            fields = [metrics.get(k) for k in ('f1_c','precision','recall')]
            fields += [100*metrics[k] if metrics.get(k) is not None else None
                       for k in ('pixel_coverage','instance_recall')]
            lines.append(f"| {group} | {method} | {' | '.join(map(number,fields))} |")
    lines += ['', '## Individual tasks and failure evidence', '',
        'Any true/false pixel overlap is a diagnostic signal, not a semantic error label. Empty residual-GT tasks remain in the fixed set and are identified below. Complete raw responses, input images and usage are retained under each request archive.', '',
        '| Task | Group | Remaining GT pixels | Full proposal coverage | Failure signals | Metrics |',
        '|---|---|---:|---:|---|---|']
    for row in data['per_task']:
        upper = row['proposal_upper']
        path = 'dev60/tasks/'+row['id']+'/metrics.json'
        coverage = upper.get('pixel_coverage_upper')
        lines.append(f"| {row['id']} | {row['group']} | {upper['truth_pixels']} | "
            f"{number(100*coverage) if coverage is not None else '—'} | "
            f"{', '.join(row['failures']) or 'none recorded'} | [evidence]({path}) |")
    lines += ['', 'See [manual visual audit](manual_dev_review.md) and [smoke review](manual_smoke_review.md).', '',
        'These are fixed-pipeline offline results. They do not establish Agent adoption or three-seed online gains.', '']
    write_text(root/'offline_report.md','\n'.join(lines))


def missed_report(root):
    sources = load(root/'pilot/full_sources_summary.json')
    data = load(root/'pilot/score_summary.json')
    lines = ['# find_missed sources and controlled verifier results', '',
        'Model: qwen3.7-flash. Only SECOND/LEVIR development data. WHU historical old-GT figures are excluded.', '',
        'All five cached sources are retained. Ranking is source round-robin, descending area within a source, then original index; exact duplicates retain provenance. GT is read only for scoring. S3/S4 thresholds were historically chosen on these development labels, so source gains are descriptive, not held-out generalization.', '',
        '## Full source complementarity', '',
        '| Group | Images | Source | Added TP | Added FP | Unique TP vs all others | Unique FP | Contributing images | Missing-pixel recall |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for group, row in sources['groups'].items():
        for source, values in row['sources'].items():
            recall = values['recall']['pixel_recall']
            lines.append(f"| {group} | {row['images']} | {source} | {values['tp']} | {values['fp']} | "
                f"{values['unique_vs_all_tp']} | {values['unique_vs_all_fp']} | {values['contributing_images_vs_all']} | "
                f"{number(100*recall) if recall is not None else '—'} |")
    lines += ['', 'Unique contribution is relative to the union of all other sources; zero unique pixels does not imply a source is uninformative. Pair overlaps and concentration by image are in [full source evidence](pilot/full_sources_summary.json).', '',
        '## Same-candidate verifier comparison', '',
        'The 30-image subset is fixed lexicographic 10/group. Top-5/10 are prefixes of Top-20. Every arm sees the same candidate masks and crop geometry. SoM adds a support-anchored mark; structured adds the same explicit state questions for every candidate. This is addition to a fixed historical initial answer, not an Agent result.', '',
        '| Group | Images | k | Arm | F1 | Precision | Recall | Candidate missing-pixel recall |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for group, row in data['groups'].items():
        base = row['baseline']
        lines.append(f"| {group} | {row['images']} | — | initial | "
            + ' | '.join(number(base[k]) for k in ('f1','precision','recall'))+' | — |')
        for k, budget in row['budgets'].items():
            recall = budget['candidate_recall']['pixel_recall']
            for arm in ('four','som','structured'):
                metrics = budget[arm]
                values = [number(metrics[key]) for key in ('f1','precision','recall')] if metrics is not None else ['incomplete']*3
                lines.append(f"| {group} | {row['images']} | {k} | {arm} | {' | '.join(values)} | "
                    f"{number(100*recall) if recall is not None else '—'} |")
    lines += ['', '## Per-image verifier diagnostics', '',
        'False accept/reject below uses candidate TP>FP as a pixel-benefit proxy, not an adjudicated semantic label. Refer to original images/responses before assigning a model error. Counts use Top-20 once, avoiding repeated counting at Top-5/10.', '',
        '| Image/group | Arm | Answers/expected | Unknown | Truncated | Proxy false accept | Proxy false reject | Evidence |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in data['details']:
        for arm, values in row['budgets']['20']['variants'].items():
            errors = values['verifier_errors']
            link = f"pilot/scores/{row['group']}/{row['stem']}.json"
            lines.append(f"| {row['group']}/{row['stem']} | {arm} | {values['answered']}/{values['expected']} | "
                f"{values['unknown']} | {values['truncated']} | {errors['tp_fp_proxy_false_accept']} | "
                f"{errors['tp_fp_proxy_false_reject']} | [case]({link}) |")
    lines += ['', 'See [manual smoke audit](manual_smoke_review.md) for checked localization failures. No offline acceptance directly changes an Agent answer.', '']
    write_text(root/'offline_report.md','\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t3',type=Path)
    parser.add_argument('--missed',type=Path)
    args=parser.parse_args()
    if args.t3: t3_report(args.t3)
    if args.missed: missed_report(args.missed)


if __name__=='__main__': main()
