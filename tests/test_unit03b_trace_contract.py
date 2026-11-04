"""Static trace contract checks, no torch dependency (safe for GitHub Pages CI)."""
import unittest
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class TraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=json.loads((ROOT/'site/lab/03-nano/trace.json').read_text(encoding='utf-8'))
    def test_metadata(self):
        self.assertEqual(self.t['schema'],'yolox-academy.nano-trace.v1')
        self.assertEqual(self.t['params']['total'],912159)
        self.assertEqual(sum(self.t['params'][s] for s in ['backbone','neck','head']),912159)
        self.assertEqual(self.t['experimental_config'],{'depth':0.33,'width':0.25,'depthwise':True,'classes':80,'input':[1,3,416,416]})
    def test_untrained_is_labeled(self):
        self.assertFalse(self.t['provenance']['trained'])
        self.assertEqual(self.t['provenance']['type'],'random-initialized')
        self.assertIn('NOT a trained',self.t['provenance']['warning'])
    def test_shape_and_native_res_values(self):
        target={'focus':(16,208,2),'dark2':(32,104,4),'dark3':(64,52,8),'dark4':(128,26,16),'dark5':(256,13,32)}
        for k,(c,size,stride) in target.items():
            stage=self.t['stages'][k]
            self.assertEqual(stage['shape'],[1,c,size,size],k)
            self.assertEqual(stage['stride'],stride)
            self.assertEqual(len(stage['channels']),3)
            for ch in stage['channels']:
                self.assertLess(ch['index'],c)
                self.assertEqual(len(ch['values']),size*size,k)
                self.assertTrue(all(isinstance(v,(int,float)) for v in ch['values']))
    def test_csp_real_branches(self):
        shapes=[self.t['branches'][k]['shape'][1] for k in ['branch1','branch1_out','branch2','branch_concat','branch_output']]
        self.assertEqual(shapes,[32,32,32,64,64])
        self.assertEqual(self.t['branches']['branch_concat']['paramCount'],0)
    def test_fpn_and_head(self):
        self.assertEqual(self.t['fpn']['shapes'],[[1,64,52,52],[1,128,26,26],[1,256,13,13]])
        self.assertEqual(self.t['head']['rawShapes'],[[1,85,52,52],[1,85,26,26],[1,85,13,13]])
    def test_no_pretrained_weights_committed(self):
        for name in (ROOT/'site').rglob('*'):
            self.assertNotIn(name.suffix.lower(),{'.pt','.pth','.ckpt'})
    def test_website_contract(self):
        self.assertTrue((ROOT/'site/lab/03-nano/index.html').is_file())
        self.assertTrue((ROOT/'site/learn/03-nano.html').is_file())
        text=(ROOT/'site/lab/03-nano/app.js').read_text()
        self.assertIn('random-initialized',text)
        self.assertNotIn('XMLHttpRequest(',text)
        self.assertIn('lab/03-nano/index.html',(ROOT/'site/index.html').read_text())
        self.assertTrue((ROOT/'site/downloads/unit03b_pytorch_starter.zip').is_file())

if __name__=='__main__':unittest.main()
