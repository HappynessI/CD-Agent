import unittest
import os,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.n9_t3_online_manifest import select_tasks,validate_frozen,freeze
from agent_baseline.analysis.n9_t3_online_loop import inference_rows
from agent_baseline.analysis.n9_t3_online_adapter import QUESTION,COUNTS,NOT_COUNTS,align_user_instruction

class Contract(unittest.TestCase):
    def test_freeze_t31_ignores_example_for_scoring(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);e=np.zeros((8,8),np.uint8);e[:2,:2]=255;old=np.zeros_like(e);old[7,7]=255
            Image.fromarray(e).save(root/'e.png');Image.fromarray(old).save(root/'ignore.png')
            rows=[dict(id=f'task{i}',task_type='T3.1',meta=dict(split='confirm'),image_t1=str(root/'e.png'),image_t2=str(root/'e.png'),gt_mask=str(root/'e.png'),exemplar_mask=str(root/'e.png'),gt_ignore_mask=str(root/'ignore.png')) for i in range(20)]
            (root/'t3_1_clean_confirm.json').write_text(json.dumps(rows))
            (root/'approval.json').write_text(json.dumps(dict(status='approved',approval_reference='unit test fixture')))
            (root/'config.json').write_text(json.dumps({'T3.1':dict(rank='delta',k=5)}))
            freeze(root,root/'approval.json',root/'selection.json',root/'config.json',root/'frozen')
            frozen=[json.loads(x) for x in (root/'frozen/tasks.jsonl').read_text().splitlines()]
            self.assertEqual(len(frozen),20);self.assertTrue(all(r['task_type']=='T3.1' for r in frozen))
            ignored=np.asarray(Image.open(frozen[0]['gt_ignore_mask']))>0
            self.assertEqual(int(ignored.sum()),5);self.assertTrue(ignored[7,7]);self.assertTrue(ignored[:2,:2].all())
            from agent_baseline.analysis.n9_t3_fixed_confirm import cohort
            tasks,method,_=cohort(root,root/'frozen',root/'config.json')
            self.assertEqual([r[0] for r in tasks],[r['id'] for r in frozen]);self.assertEqual(method,dict(rank='delta',k=5))
            (root/'config.json').write_text(json.dumps({'T3.1':dict(rank='joint',k=8)}))
            with self.assertRaisesRegex(ValueError,'saved tune selection'):cohort(root,root/'frozen',root/'config.json')
    def test_verifier_receives_example_and_candidate(self):
        from agent_baseline.analysis.n9_t3_online_adapter import install
        from agent_baseline.state import Session
        from agent_baseline import tools,loop_json
        class Ask:
            context=True
            session=None
            def __call__(self,pairs,question):
                self.pairs=pairs;self.question=question
                return [dict(answer='yes',type='real',reason='same transition',p_yes=.9) for _ in pairs],{}
        with patch.dict(os.environ,N9_ONLINE_OPT_IN='1'),tempfile.TemporaryDirectory() as folder:
            install();session=Session((32,32),target='any change');e=np.zeros((32,32),bool);e[:8,:8]=1
            session.exemplar_handle=session.add_mask(e);candidate=np.zeros_like(e);candidate[16:24,16:24]=1
            key=session.add_mask(candidate);ask=Ask()
            tool=tools.Toolbox([np.zeros((32,32,3),np.uint8)]*2,folder,session=session,ask=ask)
            result=tool.ask([dict(mask=key)])
            self.assertEqual(result['results'][0]['answer'],'yes');self.assertEqual(len(ask.pairs[0]),4)
            self.assertIn('upper panel',ask.question);self.assertEqual(len(list(Path(folder).glob('n9_verifier_inputs/pair_*/*.png'))),4)
            text=loop_json.system_prompt('v4',30,session)[2]
            self.assertIn('SAME directed',text);self.assertNotIn('no category restriction',text)
    def test_selection_image_diversity(self):
        rows=[dict(id=f'{i}_{j}',image_t1=f'image{i}',task_type='T3.1') for i in range(25) for j in range(3)]
        rows += [dict(id=f'external{i}',image_t1=f'external{i}',task_type='T3.3') for i in range(30)]
        chosen=select_tasks(rows);lookup={r['id']:r for r in rows}
        self.assertEqual(len(chosen),20);self.assertEqual(len({lookup[x]['image_t1'] for x in chosen}),20)
        self.assertEqual(chosen,select_tasks(list(reversed(rows))))
        self.assertTrue(all(lookup[x]['task_type']=='T3.1' for x in chosen))
    def test_reject_mixed_frozen_quota(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'freeze.json').write_text(json.dumps(dict(approval=dict(status='approved',approval_reference='test fixture'),seeds=[1016,2027,3038])))
            rows=[dict(id=str(i),task_type='T3.1') for i in range(20)]
            (root/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            validate_frozen(root)
            rows[0]['task_type']='T3.3'
            (root/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'exclusively T3.1'):validate_frozen(root)
    def test_whitelist(self):
        row=dict(id='n9_online_001',task_type='T3.3',instruction='same directed transition',gt_mask='recorder-only',meta={'transition':'SECRET'},target='SECRET',examples=[dict(role='positive',image_t1='a',image_t2='b',mask='e',transition='SECRET')])
        out=inference_rows(lambda:[row])()[0]
        self.assertNotIn('SECRET',str(out));self.assertEqual(out['gt_mask'],'recorder-only');self.assertEqual(out['meta'],{})
    def test_semantic_contract(self):
        self.assertIn('SAME source AND destination',QUESTION);self.assertIn('upper panel',QUESTION)
        self.assertIn('different transition',NOT_COUNTS[0]);self.assertIn('T1 source',COUNTS[0])
        messages=[dict(role='user',content=[dict(type='text',text='Find other changes including the example itself.')])]
        self.assertNotIn('including the example itself',str(align_user_instruction(messages)))
        self.assertIn('excluding the example itself',str(messages))

if __name__=='__main__':unittest.main()
