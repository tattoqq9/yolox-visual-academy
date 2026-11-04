"""Unit 03-C real-tensor trace integrity checks. Python stdlib only; CI safe."""
import json
import math
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class Unit03CTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=ROOT/'site/lab/03-fpn-head/trace.json'
        cls.d=json.loads(cls.path.read_text(encoding='utf8'))
        cls.n=cls.d['nodes']

    def test_schema_provenance(self):
        self.assertEqual(self.d['schema'],'yolox-academy.neck-head-trace.v1')
        self.assertFalse(self.d['provenance']['learned'])
        self.assertEqual(self.d['provenance']['seed'],20261104)
        self.assertIsNone(self.d['provenance']['checkpoint'])
        self.assertIn('UNTRAINED',self.d['provenance']['warning'])
        self.assertEqual(self.d['config'],{'depth':.33,'width':.25,'depthwise':True,'numClasses':80,'headOutputsPerCell':85})

    def test_model_params(self):
        p=self.d['params']
        self.assertEqual(p,{'total':912159,'backbone':459968,'neck':347424,'head':104767})
        self.assertEqual(sum(v for k,v in p.items() if k!='total'),p['total'])

    def test_nodes_and_shape(self):
        self.assertEqual(len(self.n),38)
        expected={'dark3':[1,64,52,52],'dark4':[1,128,26,26],'dark5':[1,256,13,13],
                  'lateral':[1,128,13,13],'up0':[1,128,26,26],'cat0':[1,256,26,26],
                  'tdp4':[1,128,26,26],'reduce':[1,64,26,26],'up1':[1,64,52,52],
                  'cat1':[1,128,52,52],'p3':[1,64,52,52],'down0':[1,64,26,26],
                  'cat2':[1,128,26,26],'p4':[1,128,26,26],'down1':[1,128,13,13],
                  'cat3':[1,256,13,13],'p5':[1,256,13,13]}
        for k,shape in expected.items():self.assertEqual(self.n[k]['shape'],shape,k)
        for level,w in [('p3',52),('p4',26),('p5',13)]:
            for suffix,c in [('stem',64),('cls_feat',64),('reg_feat',64),('reg',4),('obj',1),('cls',80),('raw',85)]:
                self.assertEqual(self.n[level+'.'+suffix]['shape'],[1,c,w,w])

    def test_serialized_channels_match_shape(self):
        for k,n in self.n.items():
            H,W=n['shape'][2:]
            self.assertTrue(n['samples'],k)
            self.assertEqual(len(n['samples']),len(n['sampleIndices']))
            self.assertEqual([c['index'] for c in n['samples']],n['sampleIndices'])
            self.assertEqual(len(set(n['sampleIndices'])),len(n['sampleIndices']))
            for sample in n['samples']:
                self.assertTrue(0<=sample['index']<n['shape'][1])
                self.assertEqual(len(sample['values']),H*W,k)
                self.assertTrue(all(math.isfinite(v) for v in sample['values']),k)
                self.assertLessEqual(sample['min'],sample['max'])

    def test_all_dataflow_edges_exist(self):
        for k,n in self.n.items():
            for parent in n['inputs']:self.assertIn(parent,self.n,f'{k} input {parent} missing')

    def value(self,key,ch,x,y):
        n=self.n[key]
        sample=next(s for s in n['samples'] if s['index']==ch)
        return sample['values'][y*n['shape'][3]+x]

    def assertApprox(self,a,b):
        self.assertTrue(math.isclose(a,b,rel_tol=5e-6,abs_tol=3e-14),(a,b))

    def test_nearest_upsample_exact_source(self):
        for src,up in [('lateral','up0'),('reduce','up1')]:
            for ch in [0,self.n[src]['sampleIndices'][-1]]:
                self.assertIn(ch,self.n[up]['sampleIndices'])
                for x,y in [(0,0),(3,4),(7,10)]:
                    expected=self.value(src,ch,x,y)
                    for dy in (0,1):
                        for dx in (0,1):self.assertApprox(expected,self.value(up,ch,x*2+dx,y*2+dy))

    def test_concat_preserves_sources(self):
        cases=[('up0','dark4','cat0',128),('up1','dark3','cat1',64),('down0','reduce','cat2',64),('down1','lateral','cat3',128)]
        for a,b,out,cut in cases:
            self.assertEqual(self.n[out]['params'],0)
            self.assertEqual(self.n[out]['shape'][1],self.n[a]['shape'][1]+self.n[b]['shape'][1])
            for c in (0,cut-1,cut,self.n[out]['shape'][1]-1):
                src,src_ch=(a,c) if c<cut else (b,c-cut)
                self.assertIn(c,self.n[out]['sampleIndices'])
                self.assertIn(src_ch,self.n[src]['sampleIndices'])
                for x,y in [(0,0),(2,3)]:
                    self.assertApprox(self.value(src,src_ch,x,y),self.value(out,c,x,y))

    def test_head_raw_logits_order(self):
        mapping=[('reg',0,0),('reg',1,1),('reg',2,2),('reg',3,3),('obj',0,4),('cls',0,5),('cls',1,6),('cls',7,12),('cls',79,84)]
        for level in ('p3','p4','p5'):
            for suffix,ch,idx in mapping:
                for x,y in [(0,0),(1,2)]:
                    self.assertApprox(self.value(level+'.'+suffix,ch,x,y),self.value(level+'.raw',idx,x,y))

    def test_all_head_probs_are_bias_initialization(self):
        for level in ('p3','p4','p5'):
            x=self.value(level+'.obj',0,0,0)
            p=1/(1+math.exp(-x))
            self.assertTrue(.009 < p < .011,p)

    def test_total_predictions(self):
        c=self.d['counts'];self.assertEqual(c['total'],3549)
        self.assertEqual(c['p3'],2704);self.assertEqual(c['p4'],676);self.assertEqual(c['p5'],169)
        self.assertEqual(c['rawScalars'],301665)
        self.assertEqual(3549*(4+1+1),21294) # 1-class hypothetical, not this trace

    def test_ui_and_textbook_do_not_claim_trained(self):
        html=(ROOT/'site/lab/03-fpn-head/index.html').read_text(encoding='utf8')
        textbook=(ROOT/'site/learn/03-fpn-head.html').read_text(encoding='utf8')
        js=(ROOT/'site/lab/03-fpn-head/app.js').read_text(encoding='utf8')
        self.assertIn('未学習',html);self.assertIn('未学習',textbook)
        self.assertIn('provenance?.learned!==false',js)
        self.assertIn('lab/03-fpn-head/index.html',(ROOT/'site/index.html').read_text())
        self.assertTrue((ROOT/'site/downloads/unit03c_pytorch_starter.zip').is_file())
        self.assertTrue((ROOT/'site/assets/003Z-selfie.png').is_file())
        self.assertLess(self.path.stat().st_size,3*1024*1024)

    def test_student_remains_todo(self):
        text=(ROOT/'site/code/unit03c/student.py').read_text()
        self.assertGreaterEqual(text.count('NotImplementedError'),5)
        import zipfile
        with zipfile.ZipFile(ROOT/'site/downloads/unit03c_pytorch_starter.zip') as z:
            self.assertEqual(set(z.namelist()),{'student.py','test_student.py','START_HERE.md'})
            self.assertIn(b'NotImplementedError',z.read('student.py'))

if __name__=='__main__':unittest.main()
