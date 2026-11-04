# YOLOX Visual Academy v0.6 — WSLで開発・確認する

本リポジトリは既に `tattoqq9/yolox-visual-academy` としてGitHubに公開済みです。
**過去の新規リポジトリ作成スクリプトを再実行する必要はありません。**
頭部検出研究・Tatto Engineには触れません。

公開版：https://tattoqq9.github.io/yolox-visual-academy/

## 既存WSL環境の更新

```bash
cd ~/projects/yolox-visual-academy
git status --short
# 変更中のファイルがある場合は、先に手元の作業を保全すること
git pull --ff-only

python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --test tests/unit04_math.test.cjs
node --test tests/unit06_math.test.cjs
bash scripts/preview.sh
```

公開後のHTTP確認（GUIの代用ではありません）：

```bash
python3 scripts/verify_live.py \
  https://tattoqq9.github.io/yolox-visual-academy/ \
  --attempts 6 --interval 10
```

## 各章の学習先

- Unit01：`site/lab/01-cnn/index.html`
- Unit02：`site/lab/02-grid/index.html`
- Unit03-A：`site/lab/03-block/index.html`
- Unit03-B：`site/lab/03-nano/index.html`
- Unit03-C：`site/lab/03-fpn-head/index.html`
- Unit04：`site/lab/04-detection/index.html`
- Unit05：`site/lab/05-training/index.html`（本物のPyTorch更新記録／合成図形）
- Unit06：`site/lab/06-tiny/index.html`（合成候補・研究仮説）

各章に日本語教科書、ブラウザ演習、Python課題が含まれています。

## 正式データは後日に導入する

当面は学習構造と操作を完成させることが目的です。一通りレビューした後、ユーザーが提供する画像・ラベルを安全に取り込む工程を設けます。
COCO形式の非公開検証は `tools/unit06/dataset_contract.py` を参照。データの公開許諾・GT品質・モデル権利は別途確認します。実画像とラベルを公開GitHubリポジトリへ直接commitしないでください。

## 関連資料

- `ROADMAP.md`：現状と今後
- `UNIT06_README_JA.md`：Unit06検証
- `tools/unit06/DATASET_CONTRACT_JA.md`：正式画像・GTの受入仕様
- `RELEASE_NOTES_v0.6.md`：今回の差分と実証範囲
