# YOLOX Visual Academy · Unit 04 — Pythonスクラッチ課題

**学習対象：** YOLOXのBBox Decode、IoU、NMS、現行`get_geometry_constraint`の候補抽出、Dynamic K、重複割当の解消、BCEWithLogits。

## 手順

1. 最初に[教科書](../../learn/04-detection.html)と[シミュレータ](../../lab/04-detection/index.html)を交互に操作する。
2. `student.py` の8箇所の `NotImplementedError` を自分で置き換える。
3. Python標準ライブラリのみでテストできる。

```bash
cd site/code/unit04
python3 -m unittest -v test_student.py
```

未完成の状態で失敗するのは**正常**です。`simota()`は最初から全ネットワークを学習する処理ではなく、与えられたcost行列から正例を選ぶ操作です。幾何条件とクラスコストの全計算は、ブラウザのStep 03と追加問題で習得します。

## 確認課題

- YOLOX Decodeの`gx`と、SimOTA候補中心の`gx+0.5`が異なることを説明できるか。
- 4px四方のBBoxが1px横にずれるとIoU=0.6になる理由を面積で説明できるか。
- NMSが正例割当ではなく**推論後の重複削除**であることを説明できるか。
- Dynamic Kの実数和を整数化すること、競合解消後にKが減りうることを理解したか。
- `∂BCEWithLogits(z,t)/∂z = sigmoid(z)-t` を自分で導けるか。

**注意：** シミュレータのrawは未学習モデルから、GT・学習候補は架空の教材用データから生成されています。学習済みYOLOXの精度を示すものではありません。

## ソース

参照公式：<https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py>

公式コードは本配布物に複製せず、数式と挙動を独立実装しています。歴史的なYOLOX実装には幾何候補の取り方が異なる版が存在します。学習時に使用するYOLOXのリビジョンを必ず確認してください。
