# YOLOX Visual Academy Unit 05 — 復旧・実装・検証手順

公開後の実験室：<https://tattoqq9.github.io/yolox-visual-academy/lab/05-training/index.html>

## 重要な注記

2026-11-04の実験ログは残っていますが、コードと重みはその時点で失われました。本版は合成データ生成器から作り直した**2026-11-05の再構築版**です。元のAP50とは一致しません。新記録：1枚学習 val AP50=0.0185、8枚学習 val AP50=0.1019。評価対象は人工長方形6枚のみです。

## WSLで公開版を取得

```bash
cd ~/projects/yolox-visual-academy
git status --short
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --test tests/unit04_math.test.cjs
node --check site/lab/05-training/app.js
bash scripts/preview.sh
```

Chromeで <http://localhost:8765/lab/05-training/index.html> を開き、1枚/8枚、更新回数、検証画像、スコア閾値を動かします。

## 実学習の再現（ローカル・重みは公開しない）

```bash
# NumPy、PyTorch、Pillowが必要。既存環境にあればそのまま利用
python3 tools/unit05/train.py --steps 600 --train-count 1 \
  --trace out/unit05/overfit.json --checkpoint out/unit05/overfit.pt
python3 tools/unit05/train.py --steps 600 --train-count 8 \
  --trace out/unit05/smallset.json --checkpoint out/unit05/smallset.pt
python3 tools/unit05/evaluate.py --checkpoint out/unit05/smallset.pt --split val
python3 tools/unit05/export_public.py
```

`export_public.py`は学習済み重みのSHA-256と再評価値を照合し、公開に適した数値・BBox・人工画像のみを書き出します。`.pt`ファイルはGitや静的サイトに置きません。

### 演習

`site/downloads/unit05_pytorch_starter.zip`には学生用の7課題とネットワーク参照構造が入っています。未完成時の`NotImplementedError`は期待される挙動です。完成版のモデル訓練器とは分離してあります。

### ローカル・公開HTTP検査

```bash
python3 scripts/verify_live.py \
  https://tattoqq9.github.io/yolox-visual-academy/ \
  --attempts 6 --interval 10
```

このHTTP検査がPASSしても公開ブラウザ上の実操作は別の確認項目です。

## 安全な変更範囲

変更は独立した`tattoqq9/yolox-visual-academy`のみに行います。頭部検出研究とTatto Engineの作業ディレクトリ、Gitブランチ、ビルド、共有設定には一切アクセスしません。
