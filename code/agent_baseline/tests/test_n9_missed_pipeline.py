import unittest
from agent_baseline.analysis.n9_missed_pipeline import make_request
from agent_baseline.analysis.n9_missed_score import summarize_line
from agent_baseline.analysis.n9_missed_protocol import configurations, candidate_indices


class N9PipelineTests(unittest.TestCase):
    def test_verifiers_share_seed_and_fixed_view_pairs(self):
        row = dict(shape=[32,32], direction='appeared', scope='building_rebuild_excluded', mask_index=7)
        paths = [f'/example/{i}.jpg' for i in range(4)]
        requests = {v: make_request(row, paths, v) for v in ('V0','V2','V3','V4')}
        for v in requests:
            self.assertEqual(requests[v]['parameters'], requests['V0']['parameters'])
            self.assertEqual(requests[v]['messages'][0], requests['V0']['messages'][0])
            self.assertEqual(requests[v]['messages'][1]['content'][:-1], requests['V0']['messages'][1]['content'][:-1])
        self.assertIn('Blue pixels', requests['V2']['messages'][1]['content'][-1]['text'])
        self.assertIn('Blue pixels', requests['V4']['messages'][1]['content'][-1]['text'])
        self.assertIn('INSIDE', requests['V3']['messages'][1]['content'][-1]['text'])

    def test_any_source_comparison_and_profile_grid(self):
        self.assertEqual([len(configurations(p)) for p in ('any','build','demolish')], [16,24,12])
        c = dict(index=0, sources=['S3'], semantic_delta=.9, dino=.8, area=200)
        config = configurations('any')[0]
        self.assertEqual(candidate_indices([c], dict(config, sources='all')), [0])
        self.assertEqual(candidate_indices([c], dict(config, sources='S1_S2_S5')), [])
        c['sources'].append('S1_02')
        self.assertEqual(candidate_indices([c], dict(config, sources='S1_S2_S5')), [0])

    def test_cross_line_pooled_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_line([dict(group='levir_any'), dict(group='whu_any')])

if __name__ == '__main__': unittest.main()
