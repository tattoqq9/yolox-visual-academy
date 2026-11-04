"""Unit 06 structural, numerical and private-dataset contracts; stdlib only."""
from __future__ import annotations
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'site/lab/06-tiny'
MODULE = ROOT / 'tools/unit06/dataset_contract.py'
spec=importlib.util.spec_from_file_location('dataset_contract',MODULE)
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class TestUnit06Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads((ROOT/'tools/unit06/examples/coco_train.json').read_text('utf-8'))
        cls.images = ROOT/'tools/unit06/examples/images'

    def test_01_math_app_and_page(self):
        for name in ('math.js','app.js','index.html','styles.css','synthetic-tiny.png'):
            self.assertTrue((LAB/name).is_file(),name)
    def test_02_site_links_and_curriculum(self):
        home=(ROOT/'site/index.html').read_text('utf-8')
        self.assertIn('lab/06-tiny/index.html',home)
        items=json.loads((ROOT/'site/assets/curriculum.json').read_text('utf-8'))['units']
        self.assertEqual(next(x for x in items if x['id']=='06-signed-ltrb')['status'],'prototype-ready')
        self.assertIn('learn/06-tiny.html',home)
    def test_03_no_misleading_real_data_claim(self):
        html=(LAB/'index.html').read_text('utf-8')
        self.assertIn('人工値',html)
        self.assertIn('Signed LTRBの優位性は未検証',html)
        self.assertIn('比較は仮説提出後に公開',html)
    def test_04_page_has_all_controls(self):
        html=(LAB/'index.html').read_text('utf-8')
        for uid in ['gridCanvas','boxSize','centerX','centerY','stride','scoreLevel','threshold','candidateRows','idea','reveal','comparison','twDelta','rightDelta','diagnosis']:
            self.assertIn(f'id="{uid}"',html)
    def test_05_no_dataloading_or_uploads(self):
        js=(LAB/'app.js').read_text('utf-8')
        for term in ('fetch(', 'XMLHttpRequest', 'sendBeacon(', 'localStorage','FileReader','FormData'):
            self.assertNotIn(term,js)
    def test_06_student_intentionally_incomplete(self):
        txt=(ROOT/'site/code/unit06/student.py').read_text('utf-8')
        self.assertEqual(txt.count('NotImplementedError('),7)
    def test_07_zip_starter(self):
        with zipfile.ZipFile(ROOT/'site/downloads/unit06_python_starter.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(set(z.namelist()),{'student.py','test_student.py','START_HERE.md'})
    def test_08_synthetic_example_valid_and_dont_publish(self):
        m=contract.validate(self.example,self.images,'train')
        self.assertEqual(m['schema'],contract.SCHEMA)
        self.assertEqual(m['annotation_count'],1)
        self.assertFalse(m['publishable'])
        self.assertFalse(m['rights_reviewed'])
        self.assertEqual(m['images'][0]['annotations'][0]['bbox_xywh_px'],[46.0,62.0,4.0,4.0])
    def test_09_sample_png_actual_dimensions(self):
        self.assertEqual(contract.image_dimensions(self.images/'synthetic-tiny.png'),(128,128))
    def test_10_bbox_out_of_bounds_rejected(self):
        d=json.loads(json.dumps(self.example));d['annotations'][0]['bbox']=[125,20,10,2]
        with self.assertRaisesRegex(ValueError,'out-of-bounds'):
            contract.validate(d,self.images,'train')
    def test_11_nan_and_bad_category_rejected(self):
        d=json.loads(json.dumps(self.example));d['annotations'][0]['bbox'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'invalid bbox'):
            contract.validate(d,self.images,'train')
        d=json.loads(json.dumps(self.example));d['annotations'][0]['category_id']=999
        with self.assertRaisesRegex(ValueError,'unknown image or category'):
            contract.validate(d,self.images,'train')
    def test_12_missing_or_mismatched_dimensions_rejected(self):
        d=json.loads(json.dumps(self.example));d['images'][0]['width']=127
        with self.assertRaisesRegex(ValueError,'dimensions'):
            contract.validate(d,self.images,'train')
    def test_13_path_traversal_rejected(self):
        d=json.loads(json.dumps(self.example));d['images'][0]['file_name']='../../source.jpg'
        with self.assertRaises((ValueError,FileNotFoundError)):
            contract.validate(d,self.images,'train')
    def test_14_ignore_and_crowd_preserved(self):
        d=json.loads(json.dumps(self.example));d['annotations'][0]['ignore']=1;d['annotations'][0]['iscrowd']=True
        x=contract.validate(d,self.images,'val')
        self.assertEqual(x['images'][0]['annotations'][0]['ignore'],True)
        self.assertEqual(x['images'][0]['annotations'][0]['iscrowd'],True)
    def test_15_cross_split_leakage_rejected(self):
        train=contract.validate(self.example,self.images,'train')
        with self.assertRaisesRegex(ValueError,'leakage'):
            contract.validate(self.example,self.images,'val',[train])
    def test_16_same_split_compare_rejected(self):
        train=contract.validate(self.example,self.images,'train')
        with self.assertRaisesRegex(ValueError,'another split'):
            contract.validate(self.example,self.images,'train',[train])
    def test_17_rights_flag_still_nonpublishable(self):
        m=contract.validate(self.example,self.images,'test',rights_reviewed=True)
        self.assertTrue(m['rights_reviewed'])
        self.assertFalse(m['publishable'])
    def test_18_cli_private_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            x=subprocess.run([sys.executable,str(MODULE),'--images',str(self.images),'--annotations',str(ROOT/'tools/unit06/examples/coco_train.json'),'--split','train','--output',str(p)],capture_output=True,text=True,timeout=20)
            self.assertEqual(x.returncode,0,x.stderr)
            self.assertFalse(json.loads(p.read_text())['publishable'])
    def test_19_cli_public_path_rejected(self):
        x=subprocess.run([sys.executable,str(MODULE),'--images',str(self.images),'--annotations',str(ROOT/'tools/unit06/examples/coco_train.json'),'--split','train','--output',str(ROOT/'site/assets/forbidden.json')],capture_output=True,text=True,timeout=20)
        self.assertNotEqual(x.returncode,0)
        self.assertIn('must NOT be stored',x.stderr)
    def test_21_jpeg_header_dimensions(self):
        path=ROOT/'site/lab/03-nano/input416.jpg'
        self.assertEqual(contract.image_dimensions(path),(416,416))
    def test_22_multilabel_and_ignore_preserved(self):
        d=json.loads(json.dumps(self.example))
        d['categories'].append({'id':2,'name':'person'})
        d['annotations'].append({'id':2,'image_id':1,'category_id':2,'bbox':[15,20,30,50],'ignore':1,'iscrowd':1})
        result=contract.validate(d,self.images,'test')
        self.assertEqual(result['annotation_count'],2)
        self.assertEqual(result['images'][0]['annotations'][1]['category_id'],2)
        self.assertTrue(result['images'][0]['annotations'][1]['ignore'])
    def test_23_public_site_cannot_include_real_data_artifacts(self):
        banned={'.pt','.pth','.ckpt','.onnx'}
        self.assertFalse([p for p in (ROOT/'site').rglob('*') if p.suffix.lower() in banned])
        self.assertFalse((ROOT/'site/out').exists())

    def test_20_private_contract_and_provenance(self):
        doc=(ROOT/'tools/unit06/DATASET_CONTRACT_JA.md').read_text('utf-8')
        for phrase in ('train / val / test','publishable','COCO','SHA-256','利用条件'):
            self.assertIn(phrase,doc)

if __name__=='__main__':unittest.main()
