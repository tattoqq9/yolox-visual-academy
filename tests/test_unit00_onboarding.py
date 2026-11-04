"""Unit00 pedagogical gates: no invented model outputs and trace values are real."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'site/lab/00-overview'


class Unit00OnboardingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (P/'index.html').read_text('utf8')
        cls.js = (P/'app.js').read_text('utf8')
        cls.text = (ROOT/'site/learn/00-overview.html').read_text('utf8')
        cls.trace = json.loads((P/'actual-untrained-trace.json').read_text('utf8'))
        cls.bbpath = ROOT/'site/lab/03-nano/trace.json'
        cls.npath = ROOT/'site/lab/03-fpn-head/trace.json'
        cls.bb = json.loads(cls.bbpath.read_text('utf8'))
        cls.neck = json.loads(cls.npath.read_text('utf8'))

    def test_all_six_steps_and_accessible_navigation(self):
        self.assertEqual(self.html.count('data-stage="'), 6)
        self.assertIn('aria-label="処理の順番"', self.html)
        self.assertIn('id="playFlow"', self.html)
        self.assertIn('id="prevFlow"', self.html)
        self.assertIn('id="nextFlow"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('prefers-reduced-motion', (P/'styles.css').read_text('utf8'))

    def test_stage_terms_and_links(self):
        for word in ['Convolution','Kernel','Backbone','PAFPN','Head','BBox','Objectness','Classification','NMS']:
            self.assertIn(word, self.html+self.js+self.text)
        for suffix in ['01-cnn','03-nano','03-fpn-head','04-detection']:
            self.assertIn(suffix, self.js)
        self.assertIn('画像分類', self.html)
        self.assertIn('顔認証', self.html)

    def test_illustrative_bbox_is_not_passed_off_as_inference(self):
        for word in ['人手', 'AIの検出結果ではありません','未学習','説明用']:
            self.assertIn(word, self.html)
        self.assertIn('index !== 5', self.js)
        self.assertIn("$('#illustrativeBox').hidden = index !== 5", self.js)

    def test_untrained_source_provenance_exact(self):
        self.assertEqual(self.trace['schema'],'yva.unit00.untrained-preview.v1')
        self.assertFalse(self.trace['trained'])
        self.assertEqual(self.trace['type'], 'real-pytorch-forward-subsamples')
        self.assertFalse(self.bb['provenance']['trained'])
        self.assertFalse(self.neck['provenance']['learned'])
        self.assertEqual(self.trace['input']['seed'], self.bb['provenance']['seed'])
        self.assertEqual(self.trace['sha256']['backbone'], hashlib.sha256(self.bbpath.read_bytes()).hexdigest())
        self.assertEqual(self.trace['sha256']['neck'], hashlib.sha256(self.npath.read_bytes()).hexdigest())

    def test_every_sample_is_exact_value_from_source(self):
        self.assertEqual([x['id'] for x in self.trace['maps']], ['dark2','dark3','p3'])
        for m in self.trace['maps']:
            if m['id']=='p3':
                source=self.neck['nodes']['p3']
                values=source['samples'][0]['values']
            else:
                source=self.bb['stages'][m['id']]
                values=source['channels'][0]['values']
            shape=source['shape']
            _, channels, h, w = shape
            target_w,target_h=m['sample']['width'],m['sample']['height']
            self.assertEqual(m['shape'],shape)
            self.assertEqual(target_w*target_h,len(m['sample']['values']))
            self.assertEqual(len(values),w*h)
            self.assertIn('no interpolation',m['sample']['method'])
            for j,v in enumerate(m['sample']['values']):
                y,x=divmod(j,target_w)
                iy=min(h-1,int((y+.5)*h/target_h))
                ix=min(w-1,int((x+.5)*w/target_w))
                self.assertEqual(v, values[iy*w+ix])

    def test_unit01_explains_grayscale_and_downsampling(self):
        html=(ROOT/'site/lab/01-cnn/index.html').read_text('utf8')
        js=(ROOT/'site/lab/01-cnn/app.js').read_text('utf8')
        book=(ROOT/'site/learn/01-cnn.html').read_text('utf8')
        for word in ['グレー値','48列','Kernel']:
            self.assertIn(word, js+book)
        for node in ['originalSample','sampleCanvas','zoomSample']:
            self.assertIn(node, js)
        self.assertIn('unit01-orientation', html)
        self.assertIn('00-overview', html)
        self.assertIn('実際のYOLOX', js+book)

    def test_homepage_and_curriculum_begin_at_zero(self):
        home=(ROOT/'site/index.html').read_text('utf8')
        cur=json.loads((ROOT/'site/assets/curriculum.json').read_text('utf8'))
        self.assertEqual(cur['units'][0]['id'], '00-overview')
        self.assertIn('lab/00-overview/index.html', home)
        self.assertIn('learn/00-overview.html', home)
        self.assertEqual(cur['units'][0]['status'], 'prototype-ready')

    def test_live_http_verification_includes_new_assets(self):
        t=(ROOT/'scripts/verify_live.py').read_text('utf8')
        for asset in ['lab/00-overview/index.html','lab/00-overview/app.js',
                      'lab/00-overview/actual-untrained-trace.json','learn/00-overview.html']:
            self.assertIn(asset,t)


if __name__=='__main__':
    unittest.main()
