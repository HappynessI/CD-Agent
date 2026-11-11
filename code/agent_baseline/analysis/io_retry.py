"""Small shared-filesystem helpers: retry local I/O, never model/API calls."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image


def retry_io(call):
    for attempt in range(3):
        try:
            return call()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.1 * (attempt + 1))


def read_text(path):
    return retry_io(lambda: Path(path).read_text())


def write_text(path, text):
    path = Path(path)
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    retry_io(write)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_image(path, rgb=False):
    def read():
        with Image.open(path) as im:
            return np.array(im.convert('RGB') if rgb else im)
    return retry_io(read)


def save_image(path, array):
    path = Path(path)
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.png')
        Image.fromarray(array).save(temporary)
        temporary.replace(path)
    retry_io(write)
