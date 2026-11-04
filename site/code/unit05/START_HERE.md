# Unit 05 — YOLOX-Nanoを自分で学習させる

## このUnitのゴール

Unit 01〜04の数式・モジュールを統合し、**重みを更新できる1クラスのYOLOX-Nano**を組み立て、実際のBBoxとAP50を評価することです。

1. `build_model` — Focus、Backbone、PAFPN、Decoupled Headからモデルを構築
2. `flatten_grid` — P3/P4/P5の出力を1つの候補配列に連結
3. `decode_boxes` — raw4値から画像座標のBBoxを復元
4. `iou_tensor` — 微分可能なIoU計算
5. `simota_one_gt` — 1つのGTに対する幾何候補・Dynamic K・正例割当
6. `update_once` — Loss・逆伝播・Optimizerによる実際の1更新
7. `infer_and_ap50` — Sigmoid、NMS、AP50評価を統合

## 実行方法（WSL・Python）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pillow
python -m unittest -v test_student.py
```

演習開始時の `NotImplementedError` は正常な状態です。全7課題を埋めてテストを通過させてください。テストは段階ごとに実行可能です。

**参照用のネットワーク構造**として `nano_reference.py` を同梱しています。すぐにimportするのではなく、演算とテンソル形状を理解して自作してみてください。完全版の訓練器は公開リポジトリの `tools/unit05/` にありますが、まず演習を自力で解くことを推奨します。

合成データのみで動きます。セルフィーや人物写真、公式学習済みチェックポイントは訓練に使用していません。実験で得られるAP50は実世界の精度指標ではありません。
