"""Run from this directory: python3 -m unittest -v test_student.py"""
import math
import unittest
from student import (iou_xyxy, grid_centers, geometry_candidates, dynamic_k,
                     decode_yolox, decode_signed_ltrb, valid_signed_ltrb)

class StudentTest(unittest.TestCase):
    def test_iou_one_pixel_shift_of_4px_square(self):
        self.assertAlmostEqual(iou_xyxy((0,0,4,4),(1,0,5,4)),0.6)
    def test_grid_centers_and_order(self):
        c=grid_centers(128,8)
        self.assertEqual(len(c),256)
        self.assertEqual(c[0],(0,0,4,4))
        self.assertEqual(c[-1],(15,15,124,124))
    def test_candidate_geometry(self):
        grid=grid_centers(128,8)
        out=geometry_candidates(grid,48,64,8)
        self.assertGreater(len(out),0)
        self.assertTrue(all(abs(x-48)<12 and abs(y-64)<12 for _,_,x,y in out))
    def test_dynamic_k(self):
        self.assertEqual(dynamic_k([.9]*11),9)
        self.assertEqual(dynamic_k([0.01,0.02]),1)
        self.assertEqual(dynamic_k([]),0)
    def test_yolox_raw_box(self):
        actual=decode_yolox((.5,.5,math.log(.25),math.log(.25)),5,7,8)
        for a,b in zip(actual,(43,59,45,61)):
            self.assertAlmostEqual(a,b)
    def test_signed_distance_can_be_negative(self):
        self.assertEqual(decode_signed_ltrb((4,4,-2,-2),50,50),(46,46,48,48))
    def test_signed_invalid_box(self):
        self.assertTrue(valid_signed_ltrb((4,4,-2,-2)))
        self.assertFalse(valid_signed_ltrb((-2,2,1,1)))
        self.assertFalse(valid_signed_ltrb((1,-5,1,2)))

if __name__=='__main__': unittest.main()
