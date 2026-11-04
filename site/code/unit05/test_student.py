"""Unit 05 self-check. TODO stubs deliberately fail until completed by the learner."""
import unittest
import torch
from student import (build_model,flatten_grid,decode_boxes,iou_tensor,
                     simota_one_gt,update_once,infer_and_ap50)

class TestUnit05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.manual_seed(11);torch.set_num_threads(2)
    def test_01_model_params(self):
        model=build_model(1)
        self.assertEqual(sum(p.numel() for p in model.parameters()),896754)
        with torch.no_grad():
            model.eval();pred=model(torch.rand(1,3,64,64))
        self.assertEqual([list(t.shape) for t in pred],[[1,6,8,8],[1,6,4,4],[1,6,2,2]])
    def test_02_flatten_grid(self):
        raw=[torch.zeros(1,6,n,n) for n in (4,2,1)]
        p,g,s=flatten_grid(raw)
        self.assertEqual(tuple(p.shape),(21,6));self.assertEqual(tuple(g.shape),(21,2))
        self.assertEqual(tuple(s.shape),(21,1))
        self.assertTrue(torch.equal(g[:5],torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[0.,1.]])))
    def test_03_decode_boxes(self):
        box=decode_boxes(torch.tensor([[0.,0.,0.,0.]]),torch.tensor([[3.,5.]]),torch.tensor([[8.]]))
        torch.testing.assert_close(box,torch.tensor([[24.,40.,8.,8.]]))
    def test_04_iou_gradient(self):
        p=torch.tensor([[2.,2.,4.,4.]],requires_grad=True)
        gt=torch.tensor([3.,2.,4.,4.]);iou=iou_tensor(p,gt)
        self.assertAlmostEqual(float(iou.item()),.6,places=6)
        iou.sum().backward();self.assertIsNotNone(p.grad)
    def test_05_assignment(self):
        raw=torch.zeros(3,6);grid=torch.tensor([[0.,0.],[1.,0.],[12.,0.]])
        stride=torch.ones(3,1)*8;gt=torch.tensor([4.,4.,8.,8.]);positive=simota_one_gt(raw,grid,stride,gt)
        self.assertGreater(len(positive),0);self.assertTrue(all(0<=int(i)<3 for i in positive))
    def test_06_backprop_updates_weights(self):
        m=build_model(1);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        before=next(m.parameters()).detach().clone()
        value=update_once(m,torch.rand(1,3,64,64),torch.tensor([20.,20.,12.,14.]),opt)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertFalse(torch.equal(before,next(m.parameters()).detach()))
    def test_07_ap_metrics(self):
        m=build_model(1)
        result=infer_and_ap50(m,[(torch.rand(1,3,64,64),torch.tensor([32.,32.,18.,18.]))])
        self.assertIn('ap50',result)
        self.assertTrue(0<=result['ap50']<=1)
        self.assertEqual(result['tp']+result['fn'],1)

if __name__=='__main__':unittest.main()
