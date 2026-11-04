"""Unit 05 publication contracts (stdlib only, safe on GitHub Actions)."""
from __future__ import annotations
import json,math,re,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
LAB=SITE/'lab'/'05-training'

class TestUnit05Public(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((LAB/'trace05.json').read_text('utf-8'))
        cls.exp=cls.data['experiments']

    def test_01_schema_and_source(self):
        self.assertEqual(self.data['schema'],'yva.unit05.public.v1')
        self.assertEqual(self.data['provenance']['training'],'actual CPU PyTorch updates')
        self.assertTrue(self.data['provenance']['not_head_detector'])
    def test_02_no_weights_published(self):
        self.assertFalse(self.data['provenance']['trained_weights_public'])
        self.assertFalse(any(p.suffix in {'.pt','.pth','.ckpt'} for p in SITE.rglob('*') if p.is_file()))
    def test_03_two_experiments(self):
        self.assertEqual(set(self.exp),{'one','eight'})
        for name,count in [('one',1),('eight',8)]:
            self.assertEqual(self.exp[name]['provenance']['train_count'],count)
            self.assertEqual(self.exp[name]['provenance']['steps'],600)
    def test_04_model_topology(self):
        for item in self.exp.values():
            self.assertEqual(item['provenance']['parameter_count'],896754)
            self.assertEqual(item['provenance']['input'],128)
            self.assertEqual(item['provenance']['seed'],20261104)
    def test_05_snapshot_steps(self):
        expected=[0,1,5,25,50,100,200,400,600]
        for item in self.exp.values():self.assertEqual([x['step'] for x in item['snapshots']],expected)
    def test_06_snapshots_have_valid_stats(self):
        for item in self.exp.values():
            for s in item['snapshots']:
                for split in ('training','validation'):
                    self.assertTrue(0<=s[split]['ap50']<=1)
                    self.assertGreater(s[split]['num_gt'],0)
                    self.assertEqual(s[split]['tp']+s[split]['fn'],s[split]['num_gt'])
                    self.assertGreaterEqual(s[split]['fp'],0)
    def test_07_positive_final_weights_hash(self):
        for item in self.exp.values():
            hashes=[s['weight_sha256'] for s in item['snapshots']]
            self.assertEqual(len(set(hashes)),len(hashes))
            self.assertTrue(all(re.fullmatch('[0-9a-f]{64}',h) for h in hashes))
    def test_08_actual_loss_components(self):
        for item in self.exp.values():
            for s in item['snapshots'][1:]:
                v=s['loss'];self.assertTrue(all(math.isfinite(v[x]) for x in ('total','reg','obj','cls')))
                self.assertAlmostEqual(v['reg']+v['obj']+v['cls'],v['total'],delta=5.e-5)
                self.assertGreaterEqual(v['positives'],1)
    def test_09_examples_disjoint_and_have_gt(self):
        self.assertEqual(len(self.data['validation_examples']),6)
        for i,e in enumerate(self.data['validation_examples']):
            self.assertEqual(e['image'],f'val-{i}.png')
            self.assertEqual(len(e['gt']),4)
            self.assertTrue(all(0<=x<=128 for x in e['gt']))
    def test_10_images_are_real_png(self):
        for i in range(6):
            d=(LAB/f'val-{i}.png').read_bytes()
            self.assertTrue(d.startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertEqual(int.from_bytes(d[16:20],'big'),128)
            self.assertEqual(int.from_bytes(d[20:24],'big'),128)
    def test_11_predicted_boxes_and_scores(self):
        for item in self.exp.values():
            self.assertEqual(len(item['final_predictions']),6)
            for i,e in enumerate(item['final_predictions']):
                self.assertEqual(e['image_index'],i)
                for p in e['predictions']:
                    self.assertEqual(len(p['box']),4)
                    self.assertTrue(all(math.isfinite(float(x)) for x in p['box']))
                    self.assertTrue(0<=p['score']<=1)
    def test_12_reloaded_model_verified(self):
        self.assertTrue(self.data['provenance']['checkpoint_reloaded_and_verified'])
        for item in self.exp.values():
            self.assertEqual(item['snapshots'][-1]['validation']['num_gt'],6)
    def test_13_original_result_not_reused(self):
        self.assertIn('NOT the earlier recorded',self.data['provenance']['caution'])
        self.assertFalse(math.isclose(self.exp['eight']['snapshots'][-1]['validation']['ap50'],.7857,abs_tol=.01))
    def test_14_site_navigation(self):
        home=(SITE/'index.html').read_text('utf-8')
        self.assertIn('lab/05-training/index.html',home)
        curricula=json.loads((SITE/'assets/curriculum.json').read_text('utf-8'))
        self.assertEqual(next(x for x in curricula['units'] if x['id']=='05-implementation')['status'],'prototype-ready')
    def test_15_ui_controls(self):
        html=(LAB/'index.html').read_text('utf-8')
        for id in ('predictionCanvas','curveCanvas','stepSlider','scoreSlider','imageChoice','showGT','showPred'):
            self.assertIn(f'id="{id}"',html)
    def test_16_js_has_schema_and_fetch(self):
        js=(LAB/'app.js').read_text('utf-8')
        self.assertIn('yva.unit05.public.v1',js)
        self.assertIn("fetch('trace05.json'",js)
    def test_17_student_starter_is_intentionally_incomplete(self):
        src=(SITE/'code/unit05/student.py').read_text('utf-8')
        self.assertEqual(src.count('NotImplementedError('),7)
        self.assertTrue((SITE/'learn/05-training.html').exists())
    def test_18_zip_contents(self):
        p=SITE/'downloads/unit05_pytorch_starter.zip'
        self.assertTrue(p.exists())
        with zipfile.ZipFile(p) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(set(z.namelist()),{'student.py','test_student.py','START_HERE.md','nano_reference.py'})

if __name__=='__main__':unittest.main()
