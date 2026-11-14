"""Archive explicitly selected night code snapshots and execution records on Slurm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from importlib import metadata

from .io_retry import retry_io, write_json


SOURCE_SUFFIXES = {'.py', '.sh', '.sbatch', '.yaml', '.yml'}


def copy_record(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    retry_io(lambda: shutil.copyfile(source, destination))
    return dict(source=str(source), archive=str(destination), bytes=source.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Archive through Slurm')
    plan = json.loads(args.plan.read_text())
    packages = {}
    for name in ('numpy', 'scipy', 'Pillow', 'torch', 'transformers', 'openai', 'requests'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    write_json(args.output / 'archive_environment.json', dict(
        executable=sys.executable, python=sys.version, packages=packages,
        scope='Archive/report interpreter only; original experiment interpreters are recorded in job scripts.'))
    records = []
    for entry in plan['snapshots']:
        source = Path(entry['path'])
        if not source.is_dir():
            raise FileNotFoundError(source)
        copied = []
        for folder in ('code', 'tools'):
            for path in sorted((source / folder).rglob('*')):
                if not path.is_file() or path.is_symlink():
                    continue
                if path.suffix not in SOURCE_SUFFIXES and path.name != 'vision_t13.example.json':
                    continue
                if any(part in ('__pycache__', '.git', '.venv') for part in path.parts):
                    continue
                copied.append(copy_record(path, args.output / 'sources' / entry['name'] / path.relative_to(source)))
        if not copied:
            raise ValueError(f'No source code archived from {source}')
        records.append(dict(kind='source_snapshot', name=entry['name'], path=str(source), files=copied))
        write_json(args.output / 'inventory.json', dict(complete=False, records=records))
    for directory in plan['experiments']:
        source = Path(directory)
        if not (source / 'state.json').is_file():
            raise FileNotFoundError(source / 'state.json')
        paths = [source / name for name in ('exp.json', 'state.json', 'metric.py')]
        for folder in ('jobs', 'logs', 'events', 'hooks'):
            paths.extend(sorted((source / folder).glob('*')))
        copied = [copy_record(path, args.output / 'execution' / source.name / path.relative_to(source))
                  for path in paths if path.is_file()]
        records.append(dict(kind='execution', name=source.name, files=copied))
        write_json(args.output / 'inventory.json', dict(complete=False, records=records))
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'inventory.json', dict(complete=True, records=records,
        slurm_job_id=os.environ['SLURM_JOB_ID'],
        scope='Code, job definitions, logs and observations only. No credentials, model weights or dataset copies. '
              'Raw requests, responses and metrics remain in their original result directories. '
              'Execution states are observations at archival time; later Git/notification finalization can differ.'))


if __name__ == '__main__':
    main()
