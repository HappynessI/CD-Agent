import json
import tempfile
import unittest
from pathlib import Path
from agent_baseline.analysis.night_missed_rank_inventory import inventory


class InventoryTest(unittest.TestCase):
    def test_reuse_and_completed_slots_are_not_new_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(inventory(root)['status'],'pending_prepare')
            old=root/'old.json';old.write_text('{}')
            (root/'answers').mkdir();(root/'answers/done.json').write_text('{}')
            row=dict(group='levir_any',stem='a',split='confirm',s5_gate='dino_010',
                candidates=[dict(uid='done'),dict(uid='reuse',old_answer=str(old)),dict(uid='new')])
            (root/'manifest.json').write_text(json.dumps([row]))
            result=inventory(root)
            self.assertEqual(result['total']['new_request_slots'],1)
            self.assertEqual(result['total']['reusable_old_answer'],1)
            self.assertEqual(result['max_attempts_bound'],3)


if __name__=='__main__':unittest.main()
