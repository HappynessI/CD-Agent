"""Generate both validation task types without running models or reading labels."""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline

from common import LEVIR
from agent_baseline.loop_json import RUN
from agent_baseline.tasks import INSTRUCTIONS


def generate(output, dataset=LEVIR, init_root=None):
    dataset = Path(dataset)
    init_root = Path(init_root) if init_root else RUN / 'init' / 'sam3_imc_p85' / 'val'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as file:
        for path in sorted((dataset / 'val_original' / 'A').glob('*.png')):
            for task, instruction in INSTRUCTIONS.items():
                row = dict(id=path.name, image_t1=str(path.resolve()),
                    image_t2=str((dataset / 'val_original' / 'B' / path.name).resolve()),
                    instruction=instruction, task_type=task, constraint=None,
                    target='building', change_direction='any',
                    init_mask=str((init_root / path.name).resolve()) if task == 'T5.1' else None,
                    gt_mask=str((dataset / 'val_original' / 'label' / path.name).resolve()), meta={})
                file.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-root', type=Path, default=LEVIR)
    parser.add_argument('--init-root', type=Path, help='Directory containing the given validation masks.')
    args = parser.parse_args()
    generate(args.output, args.dataset_root, args.init_root)


if __name__ == '__main__':
    main()
