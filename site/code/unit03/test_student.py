"""Student tests: python3 -m unittest discover -s tests -v"""
import unittest
import numpy as np
from student import conv2d_same, batch_norm_spatial, silu, csp_concept

class TestUnit03(unittest.TestCase):
    def test_conv_center(self):
        a=np.arange(20,dtype=float).reshape(4,5)/10
        k=np.array([[0,0,0],[0,1,0],[0,0,0]])
        np.testing.assert_allclose(conv2d_same(a,k),a)
    def test_conv_blur(self):
        a=np.ones((3,3),dtype=float)
        r=conv2d_same(a,np.ones((3,3))/9)
        self.assertAlmostEqual(r[1,1],1.0)
        self.assertAlmostEqual(r[0,0],4/9)
    def test_conv_shape(self):
        self.assertEqual(conv2d_same(np.zeros((12,16)),np.ones((3,3))).shape,(12,16))
    def test_bn_constant(self):
        np.testing.assert_allclose(batch_norm_spatial(np.ones((2,3)),gamma=2,beta=-1),-np.ones((2,3)))
    def test_bn_mean(self):
        a=np.arange(12,dtype=float).reshape(3,4)
        self.assertAlmostEqual(float(batch_norm_spatial(a).mean()),0.,places=10)
    def test_silu(self):
        np.testing.assert_allclose(silu(np.array([-1.,0.,1.])),[-.26894142137,0,.73105857863],rtol=1e-8)
    def test_csp(self):
        x=np.zeros((4,5,4));x[:,:,0]=3;x[:,:,1]=4;x[:,:,2]=1;x[:,:,3]=2
        y=csp_concept(x)
        self.assertEqual(y.shape,x.shape)
        np.testing.assert_allclose(y[:,:,0],x[:,:,0]);np.testing.assert_allclose(y[:,:,1],x[:,:,1])
        self.assertAlmostEqual(float(y[1,2,2]),1.)
        np.testing.assert_allclose(y[:,:,3],0)

if __name__=='__main__':unittest.main()
