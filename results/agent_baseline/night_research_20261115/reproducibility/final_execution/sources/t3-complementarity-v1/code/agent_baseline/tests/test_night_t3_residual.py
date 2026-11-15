import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from agent_baseline.analysis.night_t3_residual import candidate_mode,reuse_answer,SYSTEM,T3_QUESTION,MODEL,API_SEED


class ResidualTests(unittest.TestCase):
    def test_support_subtraction_empty_and_exact_reuse(self):
        ex=np.array([[1,0],[0,0]],bool);mask=np.array([[1,1],[0,0]],bool)
        mode,residual=candidate_mode(mask,ex)
        self.assertEqual(mode,'request_residual');self.assertFalse((residual&ex).any())
        np.testing.assert_array_equal(residual,[[0,1],[0,0]])
        self.assertEqual(candidate_mode(ex,ex)[0],'skip_empty')
        self.assertEqual(candidate_mode(residual,ex)[0],'reuse')

    def test_reused_request_retains_source_with_zero_new_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'input.json').write_text(json.dumps(dict(system=SYSTEM,question=T3_QUESTION,model=MODEL,seed=API_SEED,temperature=0,max_tokens=768)))
            (folder/'answer.json').write_text(json.dumps(dict(parsed=dict(change='yes'),cost_cny=1.25)))
            answer=reuse_answer(folder)
            self.assertEqual(answer['cost_cny'],0);self.assertEqual(answer['original_cost_cny'],1.25)
            self.assertFalse(answer['new_api_request']);self.assertEqual(answer['answer'],'yes')


if __name__=='__main__':unittest.main()
