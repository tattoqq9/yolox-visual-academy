"""Unit 04 is self-contained, numerical, and distinct from actual trained detection."""
import hashlib, importlib.util, json, math, unittest, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'site/lab/04-detection/trace04.json'
REF=ROOT/'tools/unit04/reference.py'
spec=importlib.util.spec_from_file_location('yva_unit04_reference',REF)
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

class TestUnit04(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(P.read_text())
        cls.previous=ROOT/'site/lab/03-fpn-head/trace.json'
        cls.d3=json.loads(cls.previous.read_text())

    def test_provenance_and_sha(self):
        p=self.d['provenance'];self.assertFalse(p['learned']);self.assertEqual(p['seed'],20261104)
        self.assertIn('UNTRAINED',p['weights']);self.assertIn('synthetic',p['gt'])
        self.assertEqual(p['source_sha256'],hashlib.sha256(self.previous.read_bytes()).hexdigest())

    def test_all_raw_channels_match_original_tensor(self):
        for level,size,stride in [('p3',52,8),('p4',26,16),('p5',13,32)]:
            raw=self.d['levels'][level];self.assertEqual((raw['height'],raw['width'],raw['stride']),(size,size,stride))
            samples={x['index']:x['values'] for x in self.d3['nodes'][level+'.raw']['samples']}
            for idx,name in enumerate(['tx','ty','tw','th','obj','cls0']):
                self.assertEqual(raw['raw'][name],samples[idx])
                self.assertEqual(len(raw['raw'][name]),size*size)
                self.assertTrue(all(math.isfinite(v) for v in raw['raw'][name]))

    def test_decode_integer_grid(self):
        pred=ref.decode_box((.25,-.5,math.log(2),math.log(3)),3,5,8)
        for x,y in zip(pred,(26,36,16,24)):self.assertAlmostEqual(x,y)

    def test_decode_real_trace_sample(self):
        raw=self.d['levels']['p3']['raw'];x,y=24,14;i=y*52+x
        pr=ref.decode_box(tuple(raw[k][i] for k in ['tx','ty','tw','th']),x,y,8)
        self.assertGreater(pr[2],0);self.assertGreater(pr[3],0);self.assertTrue(0<=pr[0]<416)

    def test_iou_small_large(self):
        got=ref.iou_xyxy((0,0,4,4),(1,0,5,4));self.assertAlmostEqual(got,.6)
        got=ref.iou_xyxy((0,0,80,80),(1,0,81,80));self.assertAlmostEqual(got,79/81)

    def test_geometry_half_center_and_radius(self):
        self.assertTrue(ref.geometry(0,0,8,(4,4,10,10)))
        self.assertFalse(ref.geometry(0,0,8,(16,4,10,10)))

    def test_dynamic_k_top_ten(self):
        self.assertEqual(ref.dynamic_k([.9]*8),7)
        self.assertEqual(ref.dynamic_k([.9]*20),9)
        self.assertEqual(ref.dynamic_k([.01]*20),1)

    def test_simota_conflicts(self):
        ks,owner=ref.simota([[.5,1,2],[.4,3,4]],[[.7,.7,.7],[.8,.8,.8]])
        self.assertEqual(ks,[2,2]);self.assertEqual(owner,[1,0,-1])

    def test_sigmoid_bce_numeric_derivative(self):
        for z in (-5,-.2,0,.2,5):
            for t in (0,.5,1):
                h=1e-4;grad=(ref.bce_with_logits(z+h,t)-ref.bce_with_logits(z-h,t))/(2*h)
                self.assertAlmostEqual(grad,ref.sigmoid(z)-t,places=6)

    def test_nms_classes(self):
        boxes=[('a',(0,0,10,10),.9,0),('b',(1,0,11,10),.8,1),('c',(1,1,11,11),.7,0)]
        self.assertEqual(ref.nms(boxes,.5,class_aware=True),['a','b'])
        self.assertEqual(ref.nms(boxes,.5,class_aware=False),['a'])

    def test_web_references_and_starter(self):
        idx=(ROOT/'site/lab/04-detection/index.html').read_text()
        lab=(ROOT/'site/lab/04-detection/app.js').read_text()
        lesson=(ROOT/'site/learn/04-detection.html').read_text()
        self.assertIn('実PyTorch',idx);self.assertIn('人工',idx)
        self.assertIn('trace04.json',lab);self.assertIn('get_geometry_constraint',lesson)
        self.assertIn('lab/04-detection/index.html',(ROOT/'site/index.html').read_text())
        with zipfile.ZipFile(ROOT/'site/downloads/unit04_python_starter.zip') as z:
            self.assertEqual(set(z.namelist()),{'student.py','test_student.py','START_HERE.md'})
            self.assertIn(b'NotImplementedError',z.read('student.py'))

if __name__=='__main__':unittest.main()
