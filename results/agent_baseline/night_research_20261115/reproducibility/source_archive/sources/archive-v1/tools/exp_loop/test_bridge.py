import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch

import bridge
import loop


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.exp=Path(self.temp.name)
        binding=self.exp/'binding.json';binding.write_text('{}')
        (self.exp/'exp.json').write_text(json.dumps(dict(agent_bridge=str(binding))))
        state=dict(status='running',history=[],queue=['next.sbatch'],cost_cny=.3)
        loop.request_review(self.exp,state,'submission_failed','invalid resource request')

    def test_receipt_is_persisted_and_duplicate_not_injected(self):
        with patch.object(bridge,'inject',AsyncMock(return_value='turn-example')) as inject:
            self.assertEqual(bridge.deliver(self.exp),'accepted')
            self.assertEqual(bridge.deliver(self.exp),'already accepted')
            self.assertEqual(inject.call_count,1)
            payload=json.loads(inject.call_args.args[1])
            self.assertEqual(payload['kind'],'submission_failed')
            self.assertEqual(payload['cost_cny'],.3)
        self.assertEqual(loop.load(self.exp)['review']['delivery'],'accepted')

    def test_delivery_error_stays_pending_for_fallback(self):
        with patch.object(bridge,'inject',AsyncMock(side_effect=OSError('socket unavailable'))):
            with self.assertRaises(OSError):bridge.deliver(self.exp)
        state=loop.load(self.exp)
        self.assertEqual(state['review']['delivery'],'pending')
        self.assertIn('socket unavailable',state['review']['delivery_error'])
        self.assertFalse(state['finalize_requested'])


if __name__=='__main__':unittest.main()
