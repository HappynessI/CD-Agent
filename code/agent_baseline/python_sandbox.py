"""Array-only subprocess boundary; the child receives neither Session nor recorder."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .state import ToolError

WALL_SECONDS = 60  # includes starting the interpreter and importing NumPy and SciPy
CPU_SECONDS = 5
MEMORY_BYTES = 2 * 1024**3


def calculate(code, masks, instances, rgb, shape):
    # Scratch is independent of the trace directory. No user paths or environment are serialized.
    # Memory-backed where available: a timeout on network storage must not read as a failed calculation.
    with tempfile.TemporaryDirectory(prefix='agent_arrays_', dir='/dev/shm' if os.access('/dev/shm', os.W_OK) else None) as scratch:
        scratch = Path(scratch)
        arrays = dict(masks, **instances, t1=rgb[0], t2=rgb[1])
        np.savez(scratch / 'input.npz', **arrays)
        (scratch / 'code.txt').write_text(code)
        specification = dict(masks=list(masks), instances=list(instances), shape=list(shape))
        (scratch / 'spec.json').write_text(json.dumps(specification))
        worker = Path(__file__).with_name('python_worker.py')
        # -I ignores user site and PYTHONPATH. Inherited credentials/proxies are absent.
        env = dict(PATH='/usr/bin:/bin', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (scratch / 'stdout.txt').open('w+b') as log:
            try:
                done = subprocess.run([sys.executable, '-I', str(worker), str(CPU_SECONDS), str(MEMORY_BYTES)],
                    cwd=scratch, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    close_fds=True, timeout=WALL_SECONDS)
            except subprocess.TimeoutExpired:
                raise ToolError('python_timeout', 'Python exceeded its wall time limit.') from None
            log.seek(0)
            stdout = log.read(4096).decode('utf-8', 'replace')
        output = scratch / 'output.json'
        if done.returncode or not output.exists():
            raise ToolError('python_failed', 'Python failed or exceeded a resource limit. ' + stdout[-2000:])
        result = json.loads(output.read_text())
        if result.get('error'):
            raise ToolError('python_failed', result['error'])
        masks_out = [np.unpackbits(np.frombuffer(base64.b64decode(data), np.uint8), count=int(np.prod(shape))).reshape(shape).astype(bool)
                     for data in result['masks']]
        return masks_out, stdout
