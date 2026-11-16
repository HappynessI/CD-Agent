import json,tempfile,unittest
from pathlib import Path
from agent_baseline.analysis.n9_t3_full_confirm import inputs

class FullConfirm(unittest.TestCase):
    def fixture(self,root):
        def write(name,value):
            p=root/name;p.write_text(json.dumps(value));return p
        approval=write('approval.json',dict(status='approved',approval_reference='unit fixture, not actual approval'))
        write('partition.json',dict(query=dict(confirm=['q.png']),external_sources=dict(confirm=['e.png'])))
        same=dict(id='same',task_type='T3.1',meta=dict(split='confirm'),image_t1='q.png')
        external=dict(id='external',task_type='T3.3',meta=dict(split='confirm'),image_t1='q.png',examples=[dict(image_t1='e.png',image_t2='e.png')])
        write('t3_1_clean_confirm.json',[same]);write('t3_3_clean_confirm.json',[external])
        selections=[write('tune31.json',{'T3.1':dict(rank='delta',k=5)}),write('tune33.json',{'T3.3':dict(rank='joint',k=8)})]
        return approval,selections
    def test_separate_frozen_configs_no_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval,selections=self.fixture(root)
            rows,methods,_=inputs(root,approval,selections)
            self.assertEqual([r['task_type'] for r in rows],['T3.1','T3.3'])
            self.assertEqual(methods['T3.1']['rank'],'delta');self.assertEqual(methods['T3.3']['k'],8)
            # Fixture has no report, RGB, labels or GT files: selection reads only declared metadata.
            with self.assertRaisesRegex(ValueError,'both task types'):inputs(root,approval,selections[:1])
    def test_pending_and_source_leak_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval,selections=self.fixture(root)
            approval.write_text(json.dumps(dict(status='pending_review')))
            with self.assertRaisesRegex(ValueError,'human approval'):inputs(root,approval,selections)
            approval.write_text(json.dumps(dict(status='approved',approval_reference='fixture')))
            p=root/'t3_3_clean_confirm.json';rows=json.loads(p.read_text());rows[0]['examples'][0]['image_t1']='tune_source.png';p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'source partition'):inputs(root,approval,selections)

if __name__=='__main__':unittest.main()
