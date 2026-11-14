import unittest
import numpy as np
from agent_baseline.analysis.night_unpack_diagnostic import compare_unpack


class UnpackDiagnosticTest(unittest.TestCase):
    def test_abba_equal_for_padding_duplicates_empty_and_strided(self):
        masks=np.zeros((3,7,7),bool);masks[:2,1:5,1:5]=True
        bits=np.packbits(masks.reshape(3,-1),axis=1)
        for source in (bits,bits[::2],bits[:0]):
            result=compare_unpack(source,(7,7))
            self.assertTrue(result['all_equal'])
            self.assertEqual([r['method'] for r in result['rounds']],list('ABBA'))
            self.assertEqual(len({r['pixels_set'] for r in result['rounds']}),1)
            for row in result['rounds']:
                self.assertIn('cpu_seconds',row['unpack']);self.assertIn('wall_seconds',row['index'])
        self.assertFalse(compare_unpack(bits[::2],(7,7))['input']['c_contiguous'])


if __name__=='__main__':unittest.main()
