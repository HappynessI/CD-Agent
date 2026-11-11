"""The page step must work from the cache alone and fail clearly when a value is missing."""
import tempfile
import unittest
from pathlib import Path

from agent_baseline.make_site import Store


class StoreTest(unittest.TestCase):
    def test_build_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'page.json'
            calls = []
            build = Store(path, 'build')
            self.assertEqual(build.get('a', lambda: calls.append(1) or [1, 'x']), [1, 'x'])
            build.get('a', lambda: calls.append(1) or [2])   # computed once per key
            build.save()
            self.assertEqual(len(calls), 1)
            read = Store(path, 'read')
            self.assertEqual(read.get('a', lambda: self.fail('the page step must not compute')), [1, 'x'])
            with self.assertRaises(KeyError):
                read.get('missing', lambda: None)


if __name__ == '__main__':
    unittest.main()
