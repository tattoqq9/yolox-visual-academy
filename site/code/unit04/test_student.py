"""Run after completing all TODOs: python3 -m unittest -v test_student.py"""
import math
import unittest
from student import sigmoid,bce_with_logits,decode_box,iou_xyxy,geometry,dynamic_k,simota,nms

class TestStudent(unittest.TestCase):
    def test_sigmoid_origin(self): self.assertAlmostEqual(sigmoid(0),0.5)
    def test_sigmoid_stability(self):
        self.assertAlmostEqual(sigmoid(-1000),0)
        self.assertAlmostEqual(sigmoid(1000),1)
    def test_bce_origin(self): self.assertAlmostEqual(bce_with_logits(0,1),math.log(2))
    def test_bce_stability(self): self.assertAlmostEqual(bce_with_logits(-1000,1),1000)
    def test_decode_grid_not_center(self): self.assertEqual(decode_box((0,0,0,0),3,5,8),(24,40,8,8))
    def test_decode_log_width(self):
        a=decode_box((.25,-.5,math.log(2),math.log(3)),3,5,8)
        for got,want in zip(a,(26,36,16,24)):self.assertAlmostEqual(got,want)
    def test_iou_same(self):self.assertEqual(iou_xyxy((0,0,4,4),(0,0,4,4)),1)
    def test_iou_one_pixel_tiny(self):self.assertAlmostEqual(iou_xyxy((0,0,4,4),(1,0,5,4)),.6)
    def test_iou_no_overlap(self):self.assertEqual(iou_xyxy((0,0,4,4),(5,5,7,7)),0)
    def test_geometry_uses_half_center(self):self.assertTrue(geometry(0,0,8,(4,4,1,1)))
    def test_geometry_strict_edge(self):self.assertFalse(geometry(0,0,8,(16,4,1,1)))
    def test_dynamic_k(self):self.assertEqual(dynamic_k([.9]*8),7)
    def test_dynamic_k_min(self):self.assertEqual(dynamic_k([0]*3),1)
    def test_simota_conflict(self):
        cost=[[.5,1,2],[.4,3,4]];iou=[[.7,.7,.7],[.8,.8,.8]]
        ks,owner=simota(cost,iou)
        self.assertEqual(ks,[2,2]);self.assertEqual(owner,[1,0,-1])
    def test_nms_same_class(self):
        boxes=[('A',(0,0,10,10),.9,0),('B',(1,0,11,10),.8,0),('C',(50,50,60,60),.7,0)]
        self.assertEqual(nms(boxes,.5),['A','C'])
    def test_nms_class_aware(self):
        boxes=[('A',(0,0,10,10),.9,0),('B',(1,0,11,10),.8,1)]
        self.assertEqual(nms(boxes,.5),['A','B'])
if __name__=='__main__':unittest.main()
