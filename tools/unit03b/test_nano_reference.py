"""Torch CPU numerical tests. Run: python3 -m unittest tools/unit03b/test_nano_reference.py -v"""
import unittest
import torch
from torch import nn
from tools.unit03b.nano_reference import BaseConv, Bottleneck, CSPLayer, CSPDarknet, DWConv, Focus, YOLOX, count_params

class NanoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def test_nano_parameter_count(self):
        m=YOLOX(80)
        self.assertEqual(count_params(m),912159)
        self.assertEqual(count_params(m.backbone.backbone),459968)
        self.assertEqual(count_params(m.backbone)-459968,347424)
        self.assertEqual(count_params(m.head),104767)
    def test_focus_shape(self):
        m=Focus(3,16)
        self.assertEqual(tuple(m(torch.zeros(1,3,416,416)).shape),(1,16,208,208))
    def test_focus_channel_order(self):
        # Hook pre-Conv: verify exact rearrangement using an identity projection into conv input.
        x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        patches=torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],dim=1)
        torch.testing.assert_close(patches[:,0],torch.tensor([[[0,2],[8,10]]],dtype=torch.float32))
        torch.testing.assert_close(patches[:,1],torch.tensor([[[4,6],[12,14]]],dtype=torch.float32))
        torch.testing.assert_close(patches[:,2],torch.tensor([[[1,3],[9,11]]],dtype=torch.float32))
        torch.testing.assert_close(patches[:,3],torch.tensor([[[5,7],[13,15]]],dtype=torch.float32))
    def test_depthwise_groups_and_params(self):
        standard=BaseConv(64,64,3);sep=DWConv(64,64,3)
        self.assertEqual(standard.conv.groups,1)
        self.assertEqual(sep.dconv.conv.groups,64)
        self.assertEqual(sep.pconv.conv.groups,1)
        self.assertEqual(count_params(standard),36864+128)
        self.assertEqual(count_params(sep),64*9+64*64+2*64+2*64)
    def test_csp_recomposition(self):
        m=CSPLayer(64,64,3,depthwise=True).eval()
        x=torch.randn(1,64,13,13)
        with torch.inference_mode():
            a=m.m(m.conv1(x));b=m.conv2(x)
            ref=m.conv3(torch.cat((a,b),dim=1))
            actual=m(x)
            torch.testing.assert_close(actual,ref)
            self.assertEqual(tuple(actual.shape),(1,64,13,13))
    def test_backbone_stage_shapes(self):
        m=CSPDarknet().eval()
        traced=[]
        hooks=[m.get_submodule(n).register_forward_hook(lambda _,__,x,n=n:traced.append((n,tuple(x.shape)))) for n in ('stem','dark2','dark3','dark4','dark5')]
        with torch.inference_mode(): m(torch.zeros(1,3,416,416))
        for h in hooks:h.remove()
        expected=[('stem',(1,16,208,208)),('dark2',(1,32,104,104)),('dark3',(1,64,52,52)),('dark4',(1,128,26,26)),('dark5',(1,256,13,13))]
        self.assertEqual(traced,expected)
    def test_full_80_class_output_shapes(self):
        m=YOLOX(80).eval()
        with torch.inference_mode():out=m(torch.randn(1,3,416,416))
        self.assertEqual([tuple(z.shape) for z in out],[(1,85,52,52),(1,85,26,26),(1,85,13,13)])
    def test_one_class_output_shapes(self):
        m=YOLOX(1).eval()
        with torch.inference_mode():out=m(torch.randn(1,3,320,320))
        self.assertEqual([tuple(z.shape) for z in out],[(1,6,40,40),(1,6,20,20),(1,6,10,10)])
    def test_bottleneck_shortcut(self):
        m=Bottleneck(16,16,shortcut=True,depthwise=True).eval()
        self.assertTrue(m.use_add)
        with torch.inference_mode():
            inp=torch.randn(1,16,8,8)
            torch.testing.assert_close(m(inp),m.conv2(m.conv1(inp))+inp)

if __name__ == '__main__':unittest.main()
