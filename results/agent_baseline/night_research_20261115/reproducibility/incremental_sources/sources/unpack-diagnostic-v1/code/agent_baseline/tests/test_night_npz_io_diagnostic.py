from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from agent_baseline.analysis.night_npz_io_diagnostic import diagnostic


class NpzDiagnosticTest(unittest.TestCase):
    def test_timing_fields_and_exact_arrays_in_both_orders(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.npz';masks=np.zeros((3,7,7),bool);masks[:2,1:4,1:4]=True
            np.savez_compressed(path,masks=np.packbits(masks.reshape(3,-1),axis=1),shape=[7,7],scores=[.5,.6,.7])
            for order in ('AB','BA'):
                events=[];result=diagnostic(path,order,lambda name,value:events.append(name))
                self.assertTrue(result['equal']['keys_equal']);self.assertTrue(all(result['equal']['fields'].values()))
                self.assertEqual(result['decode']['unique_nonempty'],1)
                self.assertEqual(result['decode']['rows'],3)
                self.assertEqual(len(result['zip_members']),3)
                for arm in ('A','B'):
                    self.assertIn('wall_seconds',result['arms'][arm]['open_directory'])
                    for key in ('masks','shape','scores'):
                        self.assertIn('cpu_seconds',result['arms'][arm]['keys'][key]['materialize'])
                        self.assertIn('wall_seconds',result['arms'][arm]['keys'][key]['copy'])
                self.assertEqual(events[0],'begin_'+order[0])
                self.assertIn('read_bytes',events)


if __name__=='__main__':unittest.main()
