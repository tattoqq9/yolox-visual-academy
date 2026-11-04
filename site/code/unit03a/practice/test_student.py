import unittest
import numpy as np
from student import conv2d_hwc, spatial_batch_norm, silu, csp_split_concat

class Unit03AStudentTests(unittest.TestCase):
    def test_conv_shape_and_27_terms(self):
        x=np.ones((5,5,3),dtype=float)
        w=np.ones((4,3,3,3),dtype=float)
        actual=conv2d_hwc(x,w,2,1)
        self.assertEqual(actual.shape,(3,3,4))
        self.assertAlmostEqual(float(actual[1,1,0]),27)
        self.assertAlmostEqual(float(actual[0,0,0]),12)

    def test_batchnorm(self):
        x=np.array([[[1.],[3.]],[[5.],[7.]]])
        y=spatial_batch_norm(x,gamma=2,beta=1,eps=1e-5)
        self.assertEqual(y.shape,x.shape)
        self.assertAlmostEqual(float(np.mean(y)),1,places=6)

    def test_silu(self):
        x=np.array([-1.,0.,1.]);actual=silu(x)
        np.testing.assert_allclose(actual,x/(1+np.exp(-x)),rtol=1e-10,atol=1e-10)

    def test_csp_split(self):
        x=np.arange(16,dtype=float).reshape(2,2,4)
        m=np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        b=np.array([[0,0,1,0],[0,0,0,1]],dtype=float)
        result=csp_split_concat(x,m,b)
        np.testing.assert_allclose(result,x)

if __name__=='__main__': unittest.main()
