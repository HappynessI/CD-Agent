"""Read G5 segmentation clipping provenance for offline attribution, without models."""
import argparse
import json
from pathlib import Path


def read_clips(directory):
    """Yield call identifiers, arguments and union-pixel clipping statistics."""
    directory = Path(directory)
    for line in (directory / 'steps.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        step = json.loads(line)
        for call in step.get('calls', []):
            stats = (call.get('provenance') or {}).get('region_clip')
            if stats is not None:
                yield dict(session=str(directory), step=step['step'], tool=call['tool'],
                           tool_call_id=call['tool_call_id'], args=call['args'],
                           output_handles=call.get('handles_created', []), **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='Session or run directory; may be repeated.')
    parser.add_argument('--output', type=Path, required=True, help='Output JSONL file.')
    args = parser.parse_args()
    directories = sorted({path.parent for run in args.run for path in run.rglob('steps.jsonl')})
    with args.output.open('w') as stream:
        for directory in directories:
            for row in read_clips(directory):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
