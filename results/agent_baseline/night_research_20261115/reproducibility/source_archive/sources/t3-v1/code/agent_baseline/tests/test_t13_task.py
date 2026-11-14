import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from agent_baseline.analysis.make_t13_tasks import task_row
from agent_baseline.scope_t13 import ASK_CANDIDATES, session_prompts
from agent_baseline.state import Session
from agent_baseline.tasks import read_tasks, instruction_for


class T13TaskTest(unittest.TestCase):
    def manifest(self, **change):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        row = task_row('a.png', d, d, 'describe')
        row.update(change)
        (d / 'm.jsonl').write_text(json.dumps(row) + '\n')
        return d / 'm.jsonl'

    def test_row_reads_and_keeps_variant(self):
        row = read_tasks(self.manifest(), 'T1.3')[0]
        self.assertEqual(row['target'], 'any change')
        self.assertEqual(row['ask_variant'], 'describe')
        self.assertIsNone(row['init_mask'])

    def test_variant_changes_only_question(self):
        base = session_prompts(Session((1, 1), target='any change'))
        chosen = session_prompts(Session((1, 1), target='any change', ask_variant='describe'))
        self.assertEqual(base['SYSTEM'], chosen['SYSTEM'])
        self.assertTrue(chosen['VERIFY'].startswith(ASK_CANDIDATES['describe']))
        self.assertEqual(base['VERIFY'].split('Reply with JSON only:')[1], chosen['VERIFY'].split('Reply with JSON only:')[1])

    def test_variant_rejected_outside_t13(self):
        with self.assertRaises(ValueError):
            read_tasks(self.manifest(task_type='T1.1xC0', target='building'))

    def test_t13_requires_any_change(self):
        with self.assertRaises(ValueError):
            instruction_for('T1.3', Session((1, 1), target='building'))


if __name__ == '__main__':
    unittest.main()
