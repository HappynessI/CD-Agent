"""Import the zero-shot verifier results that exist before this store (run on the server).

  exp3:  OUT/exp3/<dataset>/candidates.jsonl + mllm_<tag>.jsonl   (p_yes and generated answer)
  exp6j: <run>/predictions/<stage>/<dataset>/<model>_<view>_<packing>/config.json + part-*.jsonl
         (p_yes / logit_diff only, no generation; accuracy then falls back to p_yes > 0.5)

Examples:
  python import_existing.py --db verifier.db exp3 --tag qwen35_4b --model qwen3.5-4b
  python import_existing.py --db verifier.db exp6j --run-dir /path/to/exp6j/run
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from store import RunRecorder, zero_shot_template

PROBE = Path(__file__).resolve().parents[2] / 'code/probe'
DEFAULT_OUT = Path('/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def exp3_prompts():
    """PROMPT / GENERAL_PROMPT from exp3_mllm.py without importing torch."""
    module = ast.parse((PROBE / 'exp3_mllm.py').read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in module.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('PROMPT', 'GENERAL_PROMPT')}


def as_sample(row, dataset, split='test'):
    return {**{k: v for k, v in row.items() if k not in ('dataset', 'weight')}, 'dataset': dataset, 'split': split}


def import_exp3(args):
    prompts = exp3_prompts()
    for dataset in args.datasets:
        base = args.out / args.subdir / dataset
        path = base / f'mllm_{args.tag}.jsonl'
        if not path.exists():
            print('skip', path)
            continue
        cands = {r['cid']: r for r in read_jsonl(base / 'candidates.jsonl')}
        key = 'PROMPT' if dataset in ('levir', 'whu') else 'GENERAL_PROMPT'
        prompt = prompts[key]
        with RunRecorder(args.db, name=f'exp3_{args.tag}_{dataset}', model=args.model, prompt_text=prompt,
                         prompt_name=f'exp3_{key.lower()}', template=zero_shot_template(prompt),
                         experiment='exp3', readout='token_prob',
                         config={'source': str(path)}, replace=True) as rec:
            for m in read_jsonl(path):
                rec.record(as_sample(cands[m['cid']], dataset), p_yes=m.get('p_yes'),
                           yes_no_mass=m.get('yes_no_mass'), answer=m.get('answer'),
                           confidence=m.get('confidence'), raw_text=m.get('raw'))
        print('imported', path)


def import_exp6j(args):
    for config_path in sorted((args.run_dir / 'predictions').glob('*/*/*/config.json')):
        folder = config_path.parent
        config = json.loads(config_path.read_text())
        name = f"exp6j_{config['stage']}_{config['dataset']}_{folder.name}"
        parts = [r for shard in range(config['shards']) if (folder / f'part-{shard:03d}.jsonl').exists()
                 for r in read_jsonl(folder / f'part-{shard:03d}.jsonl')]
        rows = {r['cid']: r for r in config['samples']}
        stage_prompt = 'exp3' if config['stage'] == 'step1' else f"exp6j_step2_{config['view']}"
        with RunRecorder(args.db, name=name, model=config['model'], prompt_text=config['prompt'],
                         prompt_name=f"{stage_prompt}_{config['dataset']}",
                         template=zero_shot_template(config['prompt'], config['view']),
                         experiment=f"exp6j_{config['stage']}", readout='token_prob',
                         view=config['view'], packing=config['packing'],
                         config={k: v for k, v in config.items() if k not in ('samples', 'prompt')},
                         replace=True) as rec:
            for p in parts:
                rec.record(as_sample(rows[p['cid']], config['dataset']), p_yes=p.get('p_yes'),
                           logit_diff=p.get('logit_diff'), yes_no_mass=p.get('yes_no_mass'),
                           seconds=p.get('seconds'))
        print('imported', name, f'{len(parts)}/{len(rows)}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', type=Path, required=True)
    sub = p.add_subparsers(dest='source', required=True)
    a = sub.add_parser('exp3')
    a.add_argument('--tag', required=True)
    a.add_argument('--model', required=True)
    a.add_argument('--out', type=Path, default=DEFAULT_OUT)
    a.add_argument('--subdir', default='exp3')
    a.add_argument('--datasets', nargs='+', default=['levir', 'whu', 'sysu', 'second'])
    b = sub.add_parser('exp6j')
    b.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    (import_exp3 if args.source == 'exp3' else import_exp6j)(args)


if __name__ == '__main__':
    main()
