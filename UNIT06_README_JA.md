# YOLOX Visual Academy — Unit 06（研究演習版）

## 公開サイト

- シミュレータ：https://tattoqq9.github.io/yolox-visual-academy/lab/06-tiny/index.html
- 教科書：https://tattoqq9.github.io/yolox-visual-academy/learn/06-tiny.html
- 学生用ZIP：https://tattoqq9.github.io/yolox-visual-academy/downloads/unit06_python_starter.zip

## Unit 06でできること

- 128×128の人工四角形のGTを、2〜24pxで変化させる
- Stride4/8/16/32のGrid位置・中心半径1.5を操作する
- 幾何候補・最大IoU・SimOTAのDynamic K・Objectness×Class Scoreを分離する
- Scoreの低い架空候補と高い架空候補を切り替え、BBox品質とは独立な失敗を観察する
- 自分のBBox回帰仮説を15文字以上記述してからSigned LTRBと比較する
- 左右逆転による無効BBoxと順序保証の条件を学ぶ
- COCO形式のラベル受入と、学習/検証/テスト分離の機構を準備する

**注意**：算出値・予測BBox・Objectness・Classificationは人工の数式教材値です。正式な頭部検出実験・学習済み重みではありません。今回は「改善した」と結論しません。

## WSLで最新版を確認

```bash
cd ~/projects/yolox-visual-academy
git status --short
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --test tests/unit04_math.test.cjs
node --test tests/unit06_math.test.cjs
node --check site/lab/06-tiny/math.js
node --check site/lab/06-tiny/app.js
bash scripts/preview.sh
```

Chrome で `http://localhost:8765/lab/06-tiny/index.html` を開く。

## 操作チェック

1. Strideを8→4にして幾何候補数の変化を見る。
2. 架空の信頼度を低→高にし、正例のScore通過数の変化を見る。
3. GTサイズを2pxにして、IoUや参照中心との関係を調べる。
4. 仮説を記入して「比較式を表示」を押す。Signed LTRBの右辺を−8pxに変更し、無効BBoxの可能性を確認する。
5. リロードしたときに仮説ノートが保存されていないことを確認する（プライバシー上の仕様）。

## 将来の正式画像・ラベル導入

一通りの教材と品質検証が完了した時点で、ユーザーからデータとラベルを用意いただきます。現在は**COCO JSON受入のローカル検査のみ**に対応します。

```bash
python3 tools/unit06/dataset_contract.py \
  --images tools/unit06/examples/images \
  --annotations tools/unit06/examples/coco_train.json \
  --split train
```

出力先は `out/unit06/dataset_manifest_train.json`（gitignore済み）。
訓練/評価画像の素材権利、肖像権、GT品質の検査、分割間の近重複などは別の品質ゲートです。**このコマンドは画像もラベルも外部送信しません。公開対象へ自動反映もしません。**

詳細：`tools/unit06/DATASET_CONTRACT_JA.md`。

## 安全な変更範囲

`tattoqq9/yolox-visual-academy` のみ変更。頭部検出の実研究用リポジトリ、Tatto Engine、ローカル共同環境のプロセス・変数・ビルド構成には影響しません。
