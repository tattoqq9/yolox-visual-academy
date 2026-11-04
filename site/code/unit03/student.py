"""Unit03-A: 学生が穴埋めする実装。NumPyのみ使用可。"""
import numpy as np


def conv2d_same(x, kernel):
    """x[H,W] と kernel[3,3] の相互相関。zero padding=1, stride=1。"""
    raise NotImplementedError("3×3の積和を自分で実装")


def batch_norm_spatial(x, gamma=1.0, beta=0.0, eps=1e-5):
    """入力2D配列の空間平均・母分散で正規化。"""
    raise NotImplementedError("mean, variance, gamma, beta を実装")


def silu(x):
    """SiLU(x) = x * sigmoid(x)。"""
    raise NotImplementedError("SiLUを実装")


def csp_concept(x):
    """教育用4チャネル分岐: [C0,C1,edge(blur(C2)),blur(edge(C3))]。"""
    raise NotImplementedError("2経路を計算し、channel方向にConcat")
