import unittest
from agent_baseline.analysis.night_t33_grounding import selection


class T33SelectionTests(unittest.TestCase):
    def test_query_and_example_scene_isolation(self):
        rows=[dict(id=f'query{i}_{j}',image_t1=f'/query/{i}.png',meta=dict(example_provenance=[dict(name=f'source{i//2}.png')])) for i in range(60) for j in range(2)]
        chosen,pilot,split=selection(rows)
        self.assertEqual(len(chosen),60);self.assertEqual(len(pilot),12)
        source_splits={}
        for row in chosen:
            name=row['meta']['example_provenance'][0]['name']
            source_splits.setdefault(name,set()).add(split[row['id']])
        self.assertTrue(all(len(parts)==1 for parts in source_splits.values()))
        other=selection(list(reversed(rows)))
        self.assertEqual([r['id'] for r in chosen],[r['id'] for r in other[0]])
        self.assertEqual(split,other[2])


if __name__=='__main__':unittest.main()
