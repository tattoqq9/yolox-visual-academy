# Unit 03-A: CNN Block を NumPy でスクラッチ実装

CNN基礎の学生が、可視化された計算をNumPyで再現する演習です。

## 使い方

```bash
python3 -m unittest discover -s tests -v
```

`student.py` の4関数を実装してください。テストの正解を見ずに、ブラウザの数式と数値で検証すること。

- `conv2d_same(x,kernel)`: 入力とKernelはいずれも2次元、stride1、zero padding1
- `batch_norm_spatial(x,gamma,beta,eps=1e-5)`: 空間全体の母平均/母分散
- `silu(x)`: x sigmoid(x)
- `csp_concept(x)`: [H,W,4]の簡略モデル、channel0/1はそのまま、channel2はblur後edge、channel3はedge後blurの2段変換を行って結合

本演習のCSPは教育用の単純化であり、YOLOXの公式CSPLayerとは異なります。BNも学習時ミニバッチ統計・推論時running statisticsを省いた教育用です。
