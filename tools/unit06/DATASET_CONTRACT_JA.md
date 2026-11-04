# YOLOX Visual Academy — 将来の正式データ受入仕様（v0.6）

## 方針

- 現在のUnit 01〜06は**合成データ・未学習参照値・教育用実測値**を明確にラベル付けして公開する。
- 教材が一通り完成したあと、利用者から提供される画像とBBoxラベルを正式版へ差し替える。
- **素材を受け取るまで、対象データのカテゴリ、数、利用条件、教師・評価用の境界を推定しない。**
- 公開済みのSynthetic教材を後から消すのではなく、正式版と切替可能な状態を目指す。

## 初期対応フォーマット

COCO JSON (`images`, `categories`, `annotations`) ＋ PNG/JPEG画像。BBox座標は**原画像ピクセルの左上XYWH**。
`category_id`, `ignore`, `iscrowd` を保持し、分類を勝手に減らさない。カテゴリ名の例として `head`, `person` は使えるが、実データの定義は受入時に確定する。

```
{
  "images": [{"id": 1, "file_name": "example.jpg", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "head"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [50, 60, 20, 24], "ignore": 0, "iscrowd": 0}]
}
```

`tools/unit06/examples/` は人工四角形の実行サンプルで、頭部データではない。

## 非公開ゲート

```
python3 tools/unit06/dataset_contract.py \
  --images /secure/train/images \
  --annotations /secure/train/annotations.json \
  --split train

python3 tools/unit06/dataset_contract.py \
  --images /secure/val/images \
  --annotations /secure/val/annotations.json \
  --split val \
  --compare-manifest out/unit06/dataset_manifest_train.json
```

- 重複ID、存在しない画像、画像寸法不一致、不正BBox（NaN、面積0、範囲外）、不正カテゴリ、invalid flagを拒否。
- 画像はSHA-256で重複を検出し、分割比較を指定した場合は同一画像のリークを拒否。
- **近重複・被写体の一致・撮影系列からのリークは自動では保証しない。** 受入後に別監査する。
- デフォルトのmanifestは `out/unit06/`（Git ignore済み）。公開サイトの `site/` と追跡対象コードへ出力禁止。
- `--rights-reviewed` は申告フラグを残すだけで、出版権限は与えない。`publishable` は常にfalse。

## 正式版移行フェーズ（今は未実装）

1. 受入前監査：画像とGTの権利、person/head/ignore定義、取得同意、匿名化、再配布可否を確認。
2. Schema validation：COCO/追加ラベル形式を共通正規形に変換し、train / val / testのハッシュを固定。
3. 研究データ処理：プライベート環境で本学習・評価・トレース取得。
4. パブリッシュ審査：画像を一切公開せず可視化値だけ出す、または利用許可済みサブセットのみ公開。
5. Unit 01〜06の公式版データトレースを一括更新し、旧Synthetic教材は基礎学習の選択肢として維持。
6. 学生が完成できるか初学者レビュー、数値・ブラウザ・ライセンス・CI再検証後に正式版リリース。

**注意：モデル重み `.pt` や訓練・評価用の実データは公開GitHub Pagesの対象に含めない。**
