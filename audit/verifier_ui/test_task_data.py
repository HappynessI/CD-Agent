"""Lightweight, single-process tests. Temporary files stay under TMPDIR/current UI."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app
import store as legacy
import task_data_api as api
import task_data_import as prepare
import task_data_store as store
import task_data_preview as preview


class TaskDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get('TMPDIR', Path(__file__).parent / '.runtime/tmp'))
        root.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=root, prefix='task-data-test-')
        cls.path = Path(cls.temp.name)
        cls.db = store.connect(':memory:')
        prepare.prepare_synthetic(cls.db, cls.path / 'synthetic')

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls.temp.cleanup()

    def test_selection_quartiles_random_shortage(self):
        rows = [dict(id=str(i), area=i) for i in range(13)]
        selected = prepare.select_examples(rows)
        self.assertEqual(selected, prepare.select_examples(list(reversed(rows))))
        self.assertEqual(len({r['id'] for r in selected}), 3)
        self.assertLessEqual(selected[0]['area'], 3)
        self.assertGreaterEqual(selected[1]['area'], 10)
        self.assertEqual(selected[2]['selection'], '随机')
        self.assertEqual(len(prepare.select_examples(rows[:3])), 2)
        self.assertEqual(prepare.select_examples([dict(id='empty', area=0)]), [])

    def test_hsv_bins(self):
        for rgb, expected in [((20, 80, 220), '蓝'), ((220, 45, 20), '红或橙'),
                              ((235, 235, 225), '白或浅灰'), ((25, 25, 30), '深灰或黑'),
                              ((20, 150, 45), '绿'), ((130, 90, 130), '其他或混色'), ((0, 0, 0), '看不清')]:
            image = np.tile(np.array(rgb, np.uint8), (8, 8, 1))
            self.assertEqual(prepare.prefill_color(image, np.ones((8, 8), bool)), expected)
        self.assertEqual(prepare.prefill_color(image, np.zeros((8, 8), bool)), '看不清')

    def test_component_area_direction_eight_connectivity_and_crop(self):
        a, b = np.zeros((150, 150), np.uint8), np.zeros((150, 150), np.uint8)
        b[0:7, 0:9] = 5  # 63 pixels: excluded
        b[30:38, 30:38] = 5  # exactly 64: included
        b[38:46, 38:46] = 5  # diagonal joins the former: one 128-pixel component
        a[85:93, 85:93] = 5  # demolished, exactly 64
        rgb1 = np.full((150, 150, 3), (220, 40, 20), np.uint8)
        rgb2 = np.full((150, 150, 3), (20, 70, 220), np.uint8)
        rows = prepare.component_records('x.png', a, b, rgb1, rgb2, {'t1':'a','t2':'b'}, 'test')
        self.assertEqual(sorted(r['area'] for r in rows), [64, 128])
        self.assertEqual({r['direction']:r['prefill_color'] for r in rows}, {'new':'蓝','demolished':'红或橙'})
        for r in rows:
            x0,y0,x1,y1 = r['assets']['crop']
            self.assertGreaterEqual(x1-x0, 96)
            self.assertGreaterEqual(y1-y0, 96)

    def test_blind_sample_reproducible(self):
        ids = [f'c{i}' for i in range(20)]
        first = prepare.blind_sample(ids)
        self.assertEqual(len(first), 4)
        self.assertEqual(first, prepare.blind_sample(list(reversed(ids))))

    def test_review_label_undo_mapping_export_and_reimport(self):
        with closing(store.connect(':memory:')) as db:
            summary = prepare.prepare_synthetic(db, self.path / 'write-fixture')
            self.assertEqual((summary['examples'],summary['components'],summary['second_pass']), (9,20,4))
            example = store.examples(db)[0]
            ident = example['id']
            with db:
                store.save_review(db, ident, 'reasonable', '可以', 'Ethan')
                store.save_review(db, ident, 'unreasonable', '修正', 'Ethan')
                store.undo(db, 'review', 'Ethan', target=ident)
            self.assertEqual(store.examples(db)[0]['verdict'], 'reasonable')
            component = store.components(db, 2)[0]
            cid = component['id']
            with db:
                store.save_label(db,cid,1,'蓝',store.SUBTYPES[0],'Ethan')
                store.save_label(db,cid,1,'绿',store.SUBTYPES[1],'Ethan')
                store.undo(db,'component','Ethan',1)
                store.save_label(db,cid,2,'蓝',store.SUBTYPES[0],'Second reviewer')
                store.save_mapping(db,'ecology',{'t1':[1], 't2':[2,5]},'Ethan')
            first = next(r for r in store.components(db) if r['id']==cid)
            self.assertEqual(first['color'], '蓝')
            blind = next(r for r in store.components(db,2) if r['id']==cid)
            self.assertNotIn('prefill_color',blind)
            self.assertNotIn('second_result',blind)
            self.assertEqual(blind['annotator'],'Second reviewer')
            prepare.export_files(db,self.path/'export')
            review=json.loads(store.read_text(self.path/'export/task_review.json'))
            labels=json.loads(store.read_text(self.path/'export/second_component_labels.json'))
            self.assertEqual(review['mapping_history'][0]['annotator'],'Ethan')
            self.assertTrue(review['mapping_history'][0]['modified_at'])
            self.assertEqual(review['mappings'][0]['value'],{'t1':[1], 't2':[2,5]})
            label = next(r for r in labels['components'] if r['id']==cid)
            for key in ('id','image','direction','bbox','area','color','subtype','prefill_color',
                        'prefill_changed','annotator','labeled_at','second_result'):
                self.assertIn(key,label)
            self.assertNotIn('assets',label)
            self.assertEqual(labels['agreement']['paired'],1)
            self.assertEqual(labels['agreement']['both'],1)
            self.assertEqual(label['prefill_changed'], label['color']!=label['prefill_color'])
            prepare.prepare_synthetic(db,self.path/'write-fixture')
            self.assertEqual(store.examples(db)[0]['verdict'],'reasonable')
            self.assertEqual(store.mappings(db)[0]['value'],{'t1':[1], 't2':[2,5]})
            with db:
                store.undo(db,'review','Ethan',target=ident)
                store.undo(db,'component','Second reviewer',2)
            self.assertIsNone(store.examples(db)[0]['verdict'])
            self.assertIsNone(next(r for r in store.components(db,2) if r['id']==cid)['color'])

    def test_io_retry(self):
        with patch('task_data_store.time.sleep'), patch('pathlib.Path.read_text',side_effect=[OSError('transient'),OSError('transient'),'ok']) as read:
            self.assertEqual(store.read_text('ignored'),'ok')
            self.assertEqual(read.call_count,3)

    def test_snapshot_uses_images_and_hides_blind_prefill(self):
        review = preview.snapshot(self.db,'review')
        first = preview.snapshot(self.db,'components')
        second = preview.snapshot(self.db,'second_pass')
        self.assertEqual(review.count('src="data:image/png;base64,'),5)
        for name in store.PURPOSE_NAMES.values():
            self.assertIn(name,review)
        self.assertEqual(second.count('src="data:image/png;base64,'),3)
        self.assertIn('HSV 预填',first)
        self.assertNotIn('HSV 预填',second)
        self.assertNotIn('td-choice on',second)

    def test_apis_blindness_images_and_legacy_interface(self):
        uri='file:task_data_api_test?mode=memory&cache=shared'
        with patch.dict(os.environ,{'TASK_DATA_DB':uri}), closing(store.connect()) as db:
            prepare.prepare_synthetic(db,self.path/'api-fixture')
            def old_database():
                conn=legacy.connect(':memory:')
                with conn:
                    conn.execute("INSERT INTO prompt_version(id,name,text,created_at) VALUES ('old','Old prompt','unchanged','2026-01-01')")
                return conn
            # Call the registered handlers in this thread. This also works on login
            # nodes where sockets and TestClient's cross-thread portal are unavailable.
            with patch.object(app,'db',side_effect=old_database):
                before=app.prompts()
                catalog=api.catalog()
                self.assertEqual(len(catalog['examples']),9)
                ident=catalog['examples'][0]['id']
                self.assertEqual(api.review(ident,api.ReviewInput(verdict='reasonable',note='ok',annotator='A')),dict(ok=True))
                self.assertEqual(api.mapping('ecology',api.MappingInput(t1=[1],t2=[5],annotator='A')),dict(ok=True))
                first=api.components()['items']
                blind=api.components(pass_number=2)['items']
                self.assertEqual(len(blind),4)
                self.assertNotIn('prefill_color',blind[0])
                self.assertNotIn('assets',blind[0])
                cid=blind[0]['id']
                for number, color in [(1,'蓝'),(2,'绿')]:
                    result=api.label(cid,api.LabelInput(pass_number=number,color=color,subtype=store.SUBTYPES[0],annotator='A'))
                    self.assertEqual(result,dict(ok=True))
                blind_row=next(r for r in api.components(pass_number=2)['items'] if r['id']==cid)
                self.assertEqual(blind_row['color'],'绿')
                self.assertNotIn('prefill_color',blind_row)
                self.assertEqual(app.prompts(),before)
                self.assertEqual(before[0]['id'],'old')
                self.assertEqual(api.undo(api.UndoInput(kind='component',pass_number=2,annotator='A'))['target'],cid)
                for kind, target, key in [('example',ident,'overlay'),('example',ident,'s1'),('component',first[0]['id'],'t1'),('component',cid,'thumb')]:
                    response=api.image(kind,target,key)
                    self.assertEqual(response.status_code,200)
                    image=Image.open(io.BytesIO(response.body))
                    self.assertLessEqual(max(image.size),512)
                self.assertEqual(api.export('task_review.json').status_code,200)
                self.assertTrue(any(getattr(r,'path',None)=='/api/prompts' for r in app.app.routes))

    def test_new_schema_does_not_change_legacy_data(self):
        with closing(legacy.connect(':memory:')) as db:
            before=[tuple(r) for r in db.execute("SELECT name,sql FROM sqlite_master WHERE name NOT LIKE 'task_%' ORDER BY name")]
            legacy.add_prompt(db,'old','Legacy body')
            rows=[tuple(r) for r in db.execute('SELECT * FROM prompt_version')]
            prepare.prepare_synthetic(db,self.path/'legacy-fixture')
            self.assertEqual(rows,[tuple(r) for r in db.execute('SELECT * FROM prompt_version')])
            after=[tuple(r) for r in db.execute("SELECT name,sql FROM sqlite_master WHERE name NOT LIKE 'task_%' ORDER BY name")]
            self.assertEqual(before,after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
