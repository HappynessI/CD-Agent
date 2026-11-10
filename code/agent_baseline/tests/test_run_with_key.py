import tempfile
import unittest
from pathlib import Path
from agent_baseline.run_with_key import read_key


class ReadKeyTest(unittest.TestCase):
    def write(self, text):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / 'k.csv'
        p.write_text(text, encoding='utf-8')
        return p

    def test_console_export(self):
        key, url = read_key(self.write('id,1\napiKey,sk-abc\nopenAiCompatible,https://example.test/v1\n'))
        self.assertEqual((key, url), ('sk-abc', 'https://example.test/v1'))

    def test_plain_key(self):
        self.assertEqual(read_key(self.write('sk-plain\n')), ('sk-plain', None))

    def test_invalid(self):
        with self.assertRaises(SystemExit):
            read_key(self.write('nothing here at all\n'))


if __name__ == '__main__':
    unittest.main()
