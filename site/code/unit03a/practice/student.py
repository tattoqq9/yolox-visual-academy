"""YOLOX Visual Academy Unit 03-A — STUDENT WORK.
入力テンソルは NumPy [H,W,C]。物体検出用ライブラリのConv/Bnを使わず実装してください。
各関数を別のNumPy演算で書くことは可能ですが、まずはループで1つの値を追ってください。
"""
import numpy as np


def conv2d_hwc(x: np.ndarray, weights: np.ndarray, stride: int = 2, pad: int = 1) -> np.ndarray:
    """入力 [H,W,Cin]、重み [Cout,Cin,3,3] → [Hout,Wout,Cout]。"""
    # TODO: 出力サイズとゼロpaddingを計算する
    # TODO: (out_y,out_x,out_ch)ごとに全チャネル×3×3の積和を求める
    raise NotImplementedError("Lesson 02: 27項の畳み込み")


def spatial_batch_norm(x: np.ndarray, gamma: float = 1.0, beta: float = 0.0, eps: float = 1e-5):
    """1枚入力のH,W平均・分散を使用（PyTorchのtrain()との違いに注意）。"""
    # TODO: チャネルごとに平均・分散を求める（ddof=0）
    # TODO: y = gamma*(x-mu)/sqrt(var+eps)+beta
    raise NotImplementedError("Lesson 03: BatchNorm")


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x)"""
    # TODO: 数値が大きい場合のオーバーフローにも注意
    raise NotImplementedError("Lesson 04: SiLU")


def csp_split_concat(x: np.ndarray, w_main: np.ndarray, w_bypass: np.ndarray):
    """まずCSPの分岐・チャネル結合だけを実装する。
    w_main/w_bypass: [2,4]。結果 [H,W,4]。
    公式YOLOXの完全なCSPLayerではなく基本原理の演習。
    """
    # TODO: 1×1射影による2経路を実装し、チャネル方向へconcat
    raise NotImplementedError("Lesson 05: CSP split / concat")
