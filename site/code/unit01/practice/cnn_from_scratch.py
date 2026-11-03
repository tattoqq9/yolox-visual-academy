"""Unit 01 | 手を動かす演習。

TODO を順に実装してください。
ルール: numpy は使用可 / scipy.signal と torch.nn.Conv2d は使用不可。
各関数は float64 を保ち、演算途中でクリップや丸めをしないこと。

必要な配列形状:
  conv2d_single: image [H,W], kernel [K,K] => output [H_out,W_out]
  conv2d_multi:  image [C_in,H,W], weights [C_out,C_in,K,K]
                => output [C_out,H_out,W_out]

PyTorch Conv2d と同じく、Kernel を反転しない cross-correlation を実装します。
"""
from __future__ import annotations
import numpy as np


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB image [H,W,3], pixel range [0,255] -> gray [H,W] in [0,1].

    G = .299*(R/255) + .587*(G/255) + .114*(B/255)
    Note: this is luma approximation, not a color-science linear luminance.
    """
    # TODO 1: 画像をfloat64へ変換し、3チャネルの重み付き和を返す
    raise NotImplementedError("TODO 1: rgb_to_gray")


def output_size(input_size: int, kernel_size: int, padding: int, stride: int) -> int:
    """floor((input_size + 2*padding - kernel_size)/stride) + 1"""
    # TODO 2: 出力サイズを求める。strideは必ず正、kernel_sizeも正
    raise NotImplementedError("TODO 2: output_size")


def conv2d_single(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray:
    """単一チャネルの2D畳み込み（cross-correlation）をループで実装。

    ヒント:
       (1) np.pad(..., ((padding,padding), (padding,padding)))
       (2) 各出力 (oy,ox) から入力開始位置を (oy*stride,ox*stride) とする
       (3) 参照範囲の patch と kernel を要素ごとに掛けて合計する
    """
    # TODO 3: 端はゼロpadding。負の出力をReLUで消さない
    raise NotImplementedError("TODO 3: conv2d_single")


def conv2d_multi(image: np.ndarray, weights: np.ndarray, bias: np.ndarray | None = None,
                 stride: int = 1, padding: int = 1, relu: bool = False) -> np.ndarray:
    """複数入力・複数出力チャネルの畳み込み。

    入力の全C_inチャネルを合計して初めて1つの出力チャネルとなる。
    weights[out_channel,in_channel,:,:] は独立したKernel。
    bias[out_channel] は該当出力チャネルに一度だけ加える。
    """
    # TODO 4: 4重ループ (out_channel, in_channel, oy, ox) などで実装
    # TODO 5: relu=True のときだけ出力をmax(0,z)に変換
    raise NotImplementedError("TODO 4/5: conv2d_multi")
