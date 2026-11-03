"""学生用。TODOを埋めてから実行: python -m unittest discover -s tests -p test_student.py -v"""
from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'practice'))
import cnn_from_scratch as m

class StudentTests(unittest.TestCase):
    def test_01_rgb(self):
        rgb=np.array([[[255,0,0],[255,255,255]]]);np.testing.assert_allclose(m.rgb_to_gray(rgb)[0],[.299,1],atol=1e-12)
    def test_02_size(self):
        self.assertEqual(m.output_size(48,3,1,2),24)
        self.assertEqual(m.output_size(32,3,1,2),16)
    def test_03_one_convolution(self):
        x=np.arange(1,10).reshape(3,3);k=np.array([[1,0,-1]]*3)
        np.testing.assert_allclose(m.conv2d_single(x,k,stride=1,padding=0),[[-6.]])
    def test_04_padding(self):
        x=np.ones((2,2));np.testing.assert_allclose(m.conv2d_single(x,np.ones((3,3))),[[4,4],[4,4]])
    def test_05_multi_channel(self):
        x=np.ones((3,4,4));w=np.ones((2,3,3,3))
        y=m.conv2d_multi(x,w,stride=2,padding=1)
        self.assertEqual(y.shape,(2,2,2));self.assertEqual(y[0,0,0],12)
    def test_06_negative_relu(self):
        x=np.ones((1,3,3));w=-np.ones((1,1,3,3))
        self.assertEqual(m.conv2d_multi(x,w,padding=0,relu=False)[0,0,0],-9)
        self.assertEqual(m.conv2d_multi(x,w,padding=0,relu=True)[0,0,0],0)
if __name__=='__main__':unittest.main()
