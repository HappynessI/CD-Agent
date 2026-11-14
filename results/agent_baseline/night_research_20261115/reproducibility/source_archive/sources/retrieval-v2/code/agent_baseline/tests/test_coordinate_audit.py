import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.coordinate_audit import coordinate_error,load_journal,select_dev,empty_session


class CoordinateAuditTests(unittest.TestCase):
    def call(self,box,message='box must be an ordered box inside the full image.',code='bad_args'):
        return dict(tool='segment_prompt',args=dict(box=box),error_code=code,result_text=json.dumps(dict(message=message)))
    def test_empty_initialization_is_distinct_from_missing_real_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            (directory/'steps.jsonl').touch();(directory/'messages.jsonl').touch()
            self.assertTrue(empty_session(directory))
            (directory/'steps.jsonl').write_text('{"step":1}\n')
            self.assertFalse(empty_session(directory))

    def test_bounds_are_separate_from_other_argument_errors(self):
        self.assertIsNotNone(coordinate_error(self.call([0,0,768,768]),[512,512]))
        for call in (self.call([80,80,40,40]),self.call([0,0,0,10]),
                     self.call([0,0,768,768],'time must be one of t1,t2.'),
                     self.call([0,0,768,768],code='box_too_large')):
            self.assertIsNone(coordinate_error(call,[512,512]))
        self.assertIsNone(coordinate_error(self.call([0,0,768,768]),[1024,1024]))
    def test_point_exclusive_bounds(self):
        call=dict(args=dict(points=[[512,10]]),error_code='bad_args',result_text=json.dumps(dict(message='points must lie inside the full image.')))
        self.assertIsNotNone(coordinate_error(call,[512,512]))
        call=dict(args=dict(exemplar=dict(point=[512,10])),error_code='bad_args',result_text=json.dumps(dict(message='Point must be inside the image.')))
        self.assertIsNotNone(coordinate_error(call,[512,512]))
    def test_resume_preserves_partial_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sessions.jsonl';p.write_bytes(b'{"session":"a"}\n{"sess')
            self.assertEqual(load_journal(p),[dict(session='a')])
            self.assertEqual(p.with_suffix('.partial').read_bytes(),b'{"sess')
            self.assertEqual(load_journal(p),[dict(session='a')])
    def test_selection_never_uses_report_or_test_images(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);manifest=out/'dev.jsonl'
            manifest.write_text(''.join(json.dumps(dict(image_t1=f'{n}.png',id=n))+'\n' for n in ('a','b')))
            rows=[dict(task='T1.3',name=n,coordinate_bad_args=v,split=split,events=[1],session=n) for n,v,split in [('a',2,'val'),('b',99,'test'),('report',100,'val')]]
            select_dev(rows,manifest,out)
            selection=json.loads((out/'selection.json').read_text())['tasks']
            self.assertEqual([r['name'] for r in selection],['a','b'])
            self.assertEqual(selection[1]['historical_coordinate_errors'],0)


if __name__=='__main__':unittest.main()
