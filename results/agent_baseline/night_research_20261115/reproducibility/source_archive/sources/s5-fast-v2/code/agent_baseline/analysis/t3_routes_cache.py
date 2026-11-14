"""Read diagnostic vision cache without launching workers or modifying it."""
import json
from pathlib import Path
import numpy as np
from .io_retry import read_text, retry_io


class CachedVision:
    def __init__(self, root):
        self.root = Path(root) / 'dynamic'
        self.inventory = json.loads(read_text(self.root / 'inventory.json'))

    def for_row(self, row):
        paths = [str(Path(row[k]).resolve()) for k in ('image_t1', 'image_t2')]
        def call(operation, **kwargs):
            found = [(i, r) for i, r in enumerate(self.inventory)
                if [s['path'] for s in r['sources']] == paths and r['operation'] == operation
                and r.get('time') == kwargs.get('time') and r.get('classes') == kwargs.get('classes')]
            if len(found) != 1:
                raise ValueError(f'Expected one cached {operation}/{kwargs.get("time")}, got {len(found)}')
            index, record = found[0]
            for source in record['sources']:
                stat = retry_io(lambda p=source['path']: Path(p).stat())
                if stat.st_size != source['size'] or stat.st_mtime_ns != source['mtime']:
                    raise ValueError('Source changed since diagnostic cache creation')
            def load():
                with np.load(self.root / f'item_{index:06d}.npz', allow_pickle=False) as z:
                    if operation == 'segment_everything':
                        h, w = map(int, z['shape'])
                        return np.unpackbits(z['data'], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
                    return z['data'].copy()
            return retry_io(load)
        return call


def write_npz(path, **arrays):
    path = Path(path)
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp.npz')
        np.savez(temp, **arrays)
        temp.replace(path)
    retry_io(write)


def read_npz(path):
    def load():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    return retry_io(load)
