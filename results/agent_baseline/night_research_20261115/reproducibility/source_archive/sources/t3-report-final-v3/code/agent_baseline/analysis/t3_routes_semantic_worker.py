"""Persistent SegEarth worker for route A; Slurm GPU only."""
import contextlib
import json
import sys
import traceback
from agent_baseline.t13_worker import DynamicBackend
from .io_retry import read_image
from .t3_routes_cache import write_npz


def main():
    wire, backend = sys.stdout, None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                if backend is None:
                    backend = DynamicBackend(request['config'], request['cache'])
                image = read_image(request['image'], rgb=True)
                with torch.inference_mode():
                    probabilities = backend.semantic(image, request['classes'])
                write_npz(request['output'], data=probabilities)
            result = dict(ok=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            result = dict(ok=False)
        wire.write(json.dumps(result) + '\n')
        wire.flush()


if __name__ == '__main__':
    main()
