import tempfile
import threading
import unittest
from pathlib import Path

from agent_baseline.analysis import io_retry


class ConcurrentWriteTest(unittest.TestCase):
    def test_parallel_writers_to_same_file(self):
        # Regression: a fixed '<name>.tmp' made parallel shards race on os.replace and crash.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'settings.json'
            errors = []
            def run(i):
                try:
                    for _ in range(50):
                        io_retry.write_json(path, {'version': 1})
                except Exception as e:
                    errors.append(e)
            threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
            for t in threads: t.start()
            for t in threads: t.join()
            self.assertEqual(errors, [])
            self.assertEqual(io_retry.read_text(path), '{\n  "version": 1\n}\n')
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ['settings.json'])


if __name__ == '__main__':
    unittest.main()
