"""Persistent subprocess workers. Host imports no torch, mmseg or torchange."""
import atexit
import json
import os
from pathlib import Path
import subprocess
import threading
import numpy as np


class WorkerPool:
    """One serial GPU worker per backend/environment; shared across image sessions."""
    def __init__(self, config, cache):
        self.config, self.cache = config, Path(cache).resolve()
        self.processes, self.locks = {}, {}
        self.start_lock = threading.Lock()
        atexit.register(self.close)

    def for_pair(self, paths, mci=None):
        paths = [str(Path(p).resolve()) for p in paths]
        def call(operation, **kwargs):
            if operation == 'segment_everything' and mci is not None:
                with np.load(mci, allow_pickle=False) as z:
                    h,w = map(int,z['image_shape'])
                    masks = np.unpackbits(z['masks_packed'],axis=1,count=h*w,bitorder='little').reshape(-1,h,w).astype(bool)
                    return masks[z['source_time'] == (1 if kwargs['time'] == 't1' else 2)]
            kind = 'anychange' if operation == 'anychange' else 'dynamic'
            with self.start_lock:
                if kind not in self.processes:
                    cfg = self.config[kind]
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', NO_ALBUMENTATIONS_UPDATE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                    env.update(cfg.get('env',{}))
                    runtime = self.cache / 'runtime' / kind
                    runtime.mkdir(parents=True, exist_ok=True)
                    for name, folder in [('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib'),('TORCH_HOME','torch'),('HF_HOME','hf')]:
                        target = runtime / folder
                        target.mkdir(exist_ok=True)
                        env[name] = str(target)
                    # The parent path is explicit so either backend Python can import this checkout.
                    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH','')
                    self.cache.mkdir(parents=True,exist_ok=True)
                    log = (self.cache / (kind+'.stderr.log')).open('a')
                    proc = subprocess.Popen([cfg['python'],'-B','-m','agent_baseline.t13_worker',kind],
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
                    self.processes[kind] = (proc,log)
                    self.locks[kind] = threading.Lock()
            proc,_ = self.processes[kind]
            with self.locks[kind]:
                request = dict(operation=operation, paths=paths, cache=str(self.cache), config=self.config[kind], **kwargs)
                proc.stdin.write(json.dumps(request)+'\n')
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'{kind} worker exited; inspect {self.cache / (kind+".stderr.log")}')
                response = json.loads(line)
                if not response['ok']:
                    raise RuntimeError(response['error'])
                with np.load(response['path'],allow_pickle=False) as z:
                    if operation in ('segment_everything','anychange'):
                        h,w = map(int,z['shape'])
                        return np.unpackbits(z['data'],axis=1,count=h*w).reshape(-1,h,w).astype(bool)
                    return z['data'].copy()
        return call

    def close(self):
        for proc,log in self.processes.values():
            if proc.poll() is None:
                proc.stdin.close()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait()
            log.close()
        self.processes.clear()
