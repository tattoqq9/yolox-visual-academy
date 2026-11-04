"""After completing student.py: python -m unittest test_student -v"""
import unittest
import torch
from torch import nn
from student import focus_space_to_depth, conv_weight_count, DepthwiseSeparableConv, csp_concat

class TestUnit03B(unittest.TestCase):
    def test_focus_shape(self):
        x=torch.zeros(1,3,416,416)
        self.assertEqual(tuple(focus_space_to_depth(x).shape),(1,12,208,208))
    def test_focus_exact_order(self):
        x=torch.arange(16).reshape(1,1,4,4).float()
        y=focus_space_to_depth(x)
        expected=[[[0.,2.],[8.,10.]],[[4.,6.],[12.,14.]],[[1.,3.],[9.,11.]],[[5.,7.],[13.,15.]]]
        self.assertEqual(tuple(y.shape),(1,4,2,2))
        for i in range(4):torch.testing.assert_close(y[0,i],torch.tensor(expected[i]))
    def test_conv_weights(self):
        self.assertEqual(conv_weight_count(64,64,3,False),36864)
        self.assertEqual(conv_weight_count(64,64,3,True),4672)
    def test_depthwise_shapes_and_groups(self):
        m=DepthwiseSeparableConv(16,32,3,2)
        x=m(torch.randn(1,16,16,16))
        self.assertEqual(tuple(x.shape),(1,32,8,8))
        convs=[v for v in m.modules() if isinstance(v,nn.Conv2d)]
        self.assertEqual(len(convs),2)
        self.assertEqual((convs[0].groups,convs[0].kernel_size),(16,(3,3)))
        self.assertEqual((convs[1].groups,convs[1].kernel_size),(1,(1,1)))
        self.assertEqual(sum(p.numel() for p in m.parameters()),16*9+16*32+2*(16+32))
    def test_csp_cat(self):
        a=torch.ones(2,32,7,9)
        b=2*torch.ones(2,32,7,9)
        o=csp_concat(a,b)
        self.assertEqual(tuple(o.shape),(2,64,7,9))
        torch.testing.assert_close(o[:,:32],a)
        torch.testing.assert_close(o[:,32:],b)

if __name__=='__main__':unittest.main()
