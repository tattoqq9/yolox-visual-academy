"""学生用の採点可能な基礎テスト。実行: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'practice'))
from student import conv2d_same,receptive_field_stats,grid_centers,decode_yolox_box,geometry_candidates

class TestStrideAndRF(unittest.TestCase):
    def setUp(self):
        self.img=np.arange(1,17,dtype=np.float64).reshape(4,4)
        self.identity=np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float64)
        self.ones=np.ones((3,3),dtype=np.float64)

    def test_01_identity_stride1(self):
        np.testing.assert_allclose(conv2d_same(self.img,self.identity,1),self.img)

    def test_02_stride2_shape(self):
        got=conv2d_same(np.ones((16,16)),self.ones,2)
        self.assertEqual(got.shape,(8,8))

    def test_03_stride4_shape(self):
        got=conv2d_same(np.ones((16,16)),self.ones,4)
        self.assertEqual(got.shape,(4,4))

    def test_04_padding_corner_sum(self):
        # 左上は入力の4画素 1+2+5+6 のみ参照できる
        self.assertAlmostEqual(conv2d_same(self.img,self.ones)[0,0],14.)

    def test_05_kernel_is_not_flipped(self):
        k=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],float)
        self.assertAlmostEqual(conv2d_same(self.img,k)[1,1],-8.)

    def test_06_constant_blur_inside(self):
        k=self.ones/9
        self.assertAlmostEqual(conv2d_same(np.ones((16,16))*90,k)[5,5],90)

    def test_07_convolution_float(self):
        got=conv2d_same(np.ones((2,2))*2,self.ones/9)
        self.assertAlmostEqual(got[0,0],8/9)

    def test_08_rf_empty(self):
        self.assertEqual(receptive_field_stats([]),{'size':1,'jump':1,'start':.5})

    def test_09_rf_all_ones(self):
        s=receptive_field_stats([(3,1,1)]*3)
        self.assertEqual((s['size'],s['jump'],s['start']),(7,1,.5))

    def test_10_rf_all_twos(self):
        s=receptive_field_stats([(3,2,1)]*3)
        self.assertEqual((s['size'],s['jump'],s['start']),(15,8,.5))

    def test_11_rf_mixed(self):
        s=receptive_field_stats([(3,2,1),(3,2,1),(3,1,1)])
        self.assertEqual((s['size'],s['jump']),(15,4))

    def test_12_rf_start_with_no_padding(self):
        self.assertEqual(receptive_field_stats([(3,1,0)])['start'],1.5)

    def test_13_grid_shapes(self):
        self.assertEqual(grid_centers(416,8).shape,(52,52,2))
        self.assertEqual(grid_centers(416,16).shape,(26,26,2))
        self.assertEqual(grid_centers(416,32).shape,(13,13,2))

    def test_14_grid_corner(self):
        np.testing.assert_allclose(grid_centers(416,8)[0,0],[4,4])

    def test_15_grid_22_12(self):
        np.testing.assert_allclose(grid_centers(416,8)[12,22],[180,100])

    def test_16_yolox_decode(self):
        np.testing.assert_allclose(decode_yolox_box((.75,.25,0,0),22,12,8),[182,98,8,8])

    def test_17_small_width(self):
        box=decode_yolox_box((.5,.5,np.log(.5),np.log(.5)),22,12,8)
        np.testing.assert_allclose(box,[180,100,4,4])

    def test_18_bbox_center_outside_cell(self):
        self.assertGreater(decode_yolox_box((3,0,0,0),22,12,8)[0],(22+1)*8)

    def test_19_geometry_candidates_small_gt(self):
        sizes=[int(geometry_candidates((184,101),416,s).sum()) for s in [8,16,32]]
        self.assertEqual(sizes,[6,9,9])

    def test_20_geometry_candidate_boundaries(self):
        # 1.5sで境界上の点は候補に含まれない (< を使用)
        mask=geometry_candidates((16,16),32,8)
        self.assertFalse(mask[0,0]) # anchor (4,4), each difference is 12
        self.assertTrue(mask[1,1])  # anchor (12,12)

if __name__=='__main__':unittest.main()
