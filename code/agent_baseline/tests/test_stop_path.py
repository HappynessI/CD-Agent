import os
import unittest
from pathlib import Path
from unittest import mock
from agent_baseline.planner import stop_path


class StopPathTest(unittest.TestCase):
    def test_default_without_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('AGENT_STOP_FILE', None)
            self.assertEqual(stop_path('/run/STOP.json'), Path('/run/STOP.json'))

    def test_override_per_key(self):
        with mock.patch.dict(os.environ, {'AGENT_STOP_FILE': '/run/STOP_k2.json'}):
            self.assertEqual(stop_path('/run/STOP.json'), Path('/run/STOP_k2.json'))


if __name__ == '__main__':
    unittest.main()
