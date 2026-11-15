"""Retry observed shared-disk EIO during Python imports, then run a module.

Use the prescribed Python. This only retries file reads/stats, never API/model
execution. Necessary on this staging mount where import failures were observed.
"""
import _frozen_importlib_external as external
import sys
import time


def retry_eio(function):
    def call(*args, **kwargs):
        for attempt in range(3):
            try:
                return function(*args, **kwargs)
            except OSError as exc:
                if exc.errno != 5 or attempt == 2:
                    raise
                time.sleep(.1 * (attempt + 1))
    return call


external._path_stat = retry_eio(external._path_stat)
external.SourceFileLoader.get_data = retry_eio(external.SourceFileLoader.get_data)
external.FileFinder._fill_cache = retry_eio(external.FileFinder._fill_cache)
# NumPy's import-time installation metadata is outside SourceFileLoader.
from importlib.metadata import PathDistribution
PathDistribution.read_text = retry_eio(PathDistribution.read_text)
if __name__ == '__main__':
    import runpy
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name='__main__', alter_sys=True)
