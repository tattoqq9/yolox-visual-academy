# YOLOX Visual Academy — Unit 00（v0.7）

## 目的と位置づけ

CNNの式や実装へ入る前に、入力RGB画像からHeadのraw値・BBox・NMSまでの全体像を理解する。頭部検出（物体検出）と、画像全体のClassificationや個人識別を区別する。

1. [シミュレータ](site/lab/00-overview/index.html)：説明用の6段階、用語辞典、未学習PyTorch実テンソルの縮小サンプル
2. [教科書](site/learn/00-overview.html)：各処理の学習上の目的と対応Unitへの接続
3. [Unit 01](site/lab/01-cnn/index.html)：グレー値、48×32リサイズ、3×3 Kernelの説明と実画像の数値を追跡

## 教育上の区分

- Unit00で表示する頭部BBoxは **手作業で付けた概念的な例**。学習済みYOLOXの推論結果ではない。
- Unit00の図（Kernel、特徴の板、Head分岐）は **模式図**。
- Unit00の「本当に計算したテンソル」は、既存の `site/lab/03-nano/trace.json` と `site/lab/03-fpn-head/trace.json` から取り出した、**固定seed・未学習の実PyTorch forwardの数値**。保存された値は元ソースにビット数値一致し、縮小表示は画素の座標間引きで、平均・補間をしていない。色の表示のみ正負と最大絶対値でスケールする。
- 学習済み頭部検出モデルの特徴量と正式ラベルの統合は **未実施**。後日の明示的な権利確認とモデル設定/チェックポイント/前処理の固定後に実施する。

## コマンド

```bash
cd ~/projects/yolox-visual-academy
git status --short
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --test tests/unit04_math.test.cjs tests/unit06_math.test.cjs
bash scripts/preview.sh
```

ローカルページ：`http://localhost:8765/lab/00-overview/index.html`。公開版：`https://tattoqq9.github.io/yolox-visual-academy/lab/00-overview/index.html`

実数値の抽出を再実行する場合：

```bash
python3 tools/unit00/export_existing_trace.py
python3 -m unittest tests.test_unit00_onboarding -v
```

このコマンドは入力となる公開済みJSONに読み取りアクセスし、Unit00用JSONのみを書き換える。研究用プロジェクトには触れない。

ブラウザのスモーク試験（オプション；Playwright PythonとChromiumが必要）：

```bash
python3 tools/unit00/smoke_gui.py
```

検証環境のChromiumはlocalhostにアクセスできないため、このスモーク試験は**同一HTML/CSS/JS/画像/JSONを一時的にインライン化**してブラウザで操作する。静的HTTP配信は別途確認する。これは公開URLを直接操作した検証の代用ではない。

## 修了条件

初学者がUnit00を操作した後に、(1)分類と検出の違い、(2)RGB→畳み込み→Backbone→PAFPN→Head→後処理、(3)Kernel・特徴マップ・BBoxの役割、(4)模式図と実計算値の違い、を自分の言葉で説明できること。
