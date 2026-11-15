"""JSON-only regression; execute with Slurm after the original score completes."""
import argparse
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from agent_baseline.analysis.night_missed_rank import aggregate_records,recorded_scores

class RankAggregateTest(unittest.TestCase):
    def test_missing_record_and_identity_rejected(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);item=dict(group='g',stem='a',split='tune')
            (out/'manifest.json').write_text(json.dumps([item]))
            with self.assertRaisesRegex(ValueError,'Missing score'):recorded_scores(out)
            path=out/'scores/g/a.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(item,split='confirm')))
            with self.assertRaisesRegex(ValueError,'identity differs'):recorded_scores(out)
            path.write_text(json.dumps(item));self.assertEqual(recorded_scores(out),[item])

    def test_existing_scores_reproduce_original_metrics_and_intervals(self):
        reference=os.environ.get('NIGHT_RANK_REFERENCE')
        if not reference:self.skipTest('Set NIGHT_RANK_REFERENCE to completed rank_V0 for real-record regression')
        out=Path(reference);old=json.loads((out/'score_summary.json').read_text())
        records=recorded_scores(out)
        self.assertEqual(len(records),len(old['images']))
        new=aggregate_records(records,old['verifier'])
        for group,splits in old['groups'].items():
            for split,rankings in splits.items():
                for ranking,ks in rankings.items():
                    for k,expected in ks.items():
                        actual=new['groups'][group][split][ranking][k]
                        for field in ('complete','metrics','delta_vs_ungated_same_verifier','paired_image_bootstrap95',
                                      'delta_vs_gated_balanced','paired_vs_gated_balanced_bootstrap95'):
                            if field in expected:self.assertEqual(actual[field],expected[field],(group,split,ranking,k,field))
                        if actual['complete'] and ranking=='balanced':
                            self.assertEqual(actual['delta_vs_gated_balanced'],0)
                            self.assertEqual(actual['paired_vs_gated_balanced_bootstrap95'],[0.,0.])
        # An actual recorded ranking equal to its gated control must have zero
        # paired difference for every resampled image set, independent of GT.
        same=copy.deepcopy(records)
        for row in same:row['rankings']['evidence']=copy.deepcopy(row['rankings']['balanced'])
        result=aggregate_records(same,old['verifier'])
        for splits in result['groups'].values():
            for rankings in splits.values():
                for value in rankings['evidence'].values():
                    if value['complete']:
                        self.assertEqual(value['delta_vs_gated_balanced'],0)
                        self.assertEqual(value['paired_vs_gated_balanced_bootstrap95'],[0.,0.])

if __name__=='__main__':unittest.main()
