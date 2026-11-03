"""Unit 02 学生用 — まだ完成していません（TODOを実装する課題です）。

依存: numpyのみ。学習済みYOLOXの重みやネットワーク接続は不要です。
ブラウザのUnit02と同じ座標規約・計算を独立実装してください。
"""
import numpy as np


def conv2d_same(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """2D・3x3・zero padding=1 の相互相関（kernelを反転させない）。

    入力: image (H,W), kernel (3,3), stride >=1
    出力: (floor((H+2-3)/stride)+1, floor((W+2-3)/stride)+1)
    ヒント: まず全要素を0で初期化して、各出力位置で9項を足す。
    """
    # TODO 1: shape、入力範囲チェック、9項の積和
    raise NotImplementedError('TODO 1: conv2d_same')


def receptive_field_stats(layers: list[tuple[int,int,int]]) -> dict[str, float]:
    """理論受容野の縦横サイズ R、累積Stride J、最初の受容野中心 start を求める。

    layers = [(kernel_size, stride, padding), ...]
    Dilation=1、左右対称のpadding、奇数kernelのみ考える。
    pixel(0)の中心座標を0.5とする。startの単位は入力の画素。
    戻り値: {'size': R, 'jump': J, 'start': start}
    """
    # TODO 2: R=1,J=1,start=0.5から層をたどる
    raise NotImplementedError('TODO 2: receptive_field_stats')


def grid_centers(image_size: int, stride: int) -> np.ndarray:
    """YOLOX割当用の格子中心 (H,W,2)を生成する。末尾軸の順は (x,y)。

    例: image_size=416,stride=8 -> (52,52,2)
    grid_centers(416,8)[12,22] は (180,100)。
    """
    # TODO 3: 0始まりのg_x,g_yから (g+0.5)*s を作る
    raise NotImplementedError('TODO 3: grid_centers')


def decode_yolox_box(raw: tuple[float,float,float,float], gx: int, gy: int, stride: int) -> np.ndarray:
    """標準YOLOXの (tx,ty,tw,th) から (cx,cy,w,h) を返す。

    中心値にsigmoidによるセル内制限は加えない。
    """
    # TODO 4: 中心は線形、幅高さは exp
    raise NotImplementedError('TODO 4: decode_yolox_box')


def geometry_candidates(gt_xy: tuple[float,float], image_size: int, stride: int) -> np.ndarray:
    """GT中心に対するYOLOX幾何候補の真偽マップ (H,W)。

    条件: abs(anchor_x-gt_x) < 1.5*stride, Y方向も同様。
    これはSimOTAの最終割当ではなく、最初の幾何候補のみ。
    """
    # TODO 5: 格子中心の座標とGT中心を比較
    raise NotImplementedError('TODO 5: geometry_candidates')
