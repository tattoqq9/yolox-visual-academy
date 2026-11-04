"""Unit 03-C student exercises. Run: python -m unittest test_student -v"""
import unittest
import torch
from torch import nn
from student import upsample_nearest2x, concat_channels, pafpn_top_down, pafpn_bottom_up, DecoupledRawHead


class Capture(nn.Module):
    def __init__(self, cin, cout):
        super().__init__();self.conv=nn.Conv2d(cin,cout,1,bias=False);self.seen=None
    def forward(self,x):
        self.seen=x.detach().clone();return self.conv(x)


class TestUnit03C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23);torch.set_num_threads(2)

    def test_nearest_shape(self):
        x=torch.randn(1,3,13,13);self.assertEqual(tuple(upsample_nearest2x(x).shape),(1,3,26,26))

    def test_nearest_exact(self):
        x=torch.tensor([[[[1.,2.],[3.,4.]]]])
        self.assertTrue(torch.equal(upsample_nearest2x(x),torch.tensor([[[[1.,1.,2.,2.],[1.,1.,2.,2.],[3.,3.,4.,4.],[3.,3.,4.,4.]]]])))

    def test_concat_order(self):
        a=torch.ones(2,3,4,5);b=2*torch.ones(2,2,4,5)
        z=concat_channels(a,b)
        self.assertEqual(tuple(z.shape),(2,5,4,5));torch.testing.assert_close(z[:,:3],a);torch.testing.assert_close(z[:,3:],b)

    def test_top_down_shapes_and_sources(self):
        d3=torch.randn(1,64,52,52);d4=torch.randn(1,128,26,26);d5=torch.randn(1,256,13,13)
        lateral=Capture(256,128);p4=Capture(256,128);reduce=Capture(128,64);p3=Capture(128,64)
        out,fpn1,fpn0=pafpn_top_down(d3,d4,d5,lateral,p4,reduce,p3)
        self.assertEqual(tuple(out.shape),(1,64,52,52));self.assertEqual(tuple(fpn1.shape),(1,64,26,26));self.assertEqual(tuple(fpn0.shape),(1,128,13,13))
        torch.testing.assert_close(p4.seen[:,:128],upsample_nearest2x(fpn0));torch.testing.assert_close(p4.seen[:,128:],d4)
        torch.testing.assert_close(p3.seen[:,:64],upsample_nearest2x(fpn1));torch.testing.assert_close(p3.seen[:,64:],d3)

    def test_bottom_up_shapes_and_sources(self):
        p3=torch.randn(1,64,52,52);fpn1=torch.randn(1,64,26,26);fpn0=torch.randn(1,128,13,13)
        bu2=nn.Conv2d(64,64,3,2,1);c3n3=Capture(128,128);bu1=nn.Conv2d(128,128,3,2,1);c3n4=Capture(256,256)
        outputs=pafpn_bottom_up(p3,fpn1,fpn0,bu2,c3n3,bu1,c3n4)
        self.assertEqual([tuple(v.shape) for v in outputs],[(1,64,52,52),(1,128,26,26),(1,256,13,13)])
        torch.testing.assert_close(c3n3.seen[:,:64],bu2(p3));torch.testing.assert_close(c3n3.seen[:,64:],fpn1)
        torch.testing.assert_close(c3n4.seen[:,:128],bu1(outputs[1]));torch.testing.assert_close(c3n4.seen[:,128:],fpn0)

    def make_head(self,classes):
        def factory(a,b):return nn.Sequential(nn.Conv2d(a,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())
        return DecoupledRawHead(classes,(64,128,256),64,factory).eval()

    def test_head_eighty_class_shapes(self):
        m=self.make_head(80)
        with torch.inference_mode(): out=m((torch.randn(1,64,8,8),torch.randn(1,128,4,4),torch.randn(1,256,2,2)))
        self.assertEqual([tuple(x.shape) for x in out],[(1,85,8,8),(1,85,4,4),(1,85,2,2)])

    def test_head_one_class_shapes(self):
        m=self.make_head(1)
        with torch.inference_mode(): out=m((torch.randn(1,64,8,8),torch.randn(1,128,4,4),torch.randn(1,256,2,2)))
        self.assertEqual([tuple(x.shape) for x in out],[(1,6,8,8),(1,6,4,4),(1,6,2,2)])

    def test_head_raw_order_and_branch_sharing(self):
        m=self.make_head(80)
        x=torch.randn(1,64,8,8)
        with torch.inference_mode():
            stem=m.stems[0](x)
            cls=m.cls_preds[0](m.cls_convs[0](stem))
            reg_feat=m.reg_convs[0](stem)
            reg=m.reg_preds[0](reg_feat);obj=m.obj_preds[0](reg_feat)
            expected=torch.cat([reg,obj,cls],dim=1)
            actual=m((x,torch.randn(1,128,4,4),torch.randn(1,256,2,2)))[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertNotEqual(id(m.cls_convs[0]),id(m.reg_convs[0]))

if __name__=='__main__':unittest.main()
