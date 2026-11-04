# YOLOX Visual Academy — Unit 03-C 公開・確認手順

## 更新元

独立リポジトリ：`tattoqq9/yolox-visual-academy`
公開先：`https://tattoqq9.github.io/yolox-visual-academy/lab/03-fpn-head/index.html`

Unit 01〜03-Bの公開ファイルは変更せず、Unit 03-Cを追加します。Tatto Engineと頭部検出研究リポジトリには一切書き込みません。

## WSLから最新版を取得・検証

```bash
cd ~/projects/yolox-visual-academy
git status --short
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --check site/lab/03-fpn-head/app.js
bash scripts/preview.sh
```

ブラウザで `http://localhost:8765/lab/03-fpn-head/index.html` を開いてください。

- Top-down：`up0` と `cat0` のチャネルごとの値の出所を確認
- Bottom-up：`down0 → cat2 → p4 → down1 → cat3 → p5` のShapeを確認
- Head：P3/P4/P5を切り替え、分類枝と回帰枝、reg4/obj1/cls80の値を確認
- ヒートマップのクリック／x,y座標入力・3D回転／チャネル切替を確認
- 80クラス→1クラスの仮想切替はShape計算のみであることを確認
- DevTools Consoleに未処理のJavaScriptエラーがないことを確認

公開環境のHTTP検証（全28ファイル・アセット）：

```bash
python3 scripts/verify_live.py https://tattoqq9.github.io/yolox-visual-academy/ --attempts 6 --interval 10
```

PyTorchでトレースの数値を再生成する場合：

```bash
# 必要なライブラリ: torch, numpy, pillow（既存環境があれば追加導入は不要）
python3 tools/unit03c/export_neck_head.py --output /tmp/unit03c-trace.json
```

公開版はPyTorchの固定seedの**未学習モデル**で取得した実テンソルです。`site/lab/03-fpn-head/trace.json` は選択チャネルのみが保存された教育用データです。頭部検出性能や学習済み特徴の意味を示していません。

## 学生演習

公開サイトの `downloads/unit03c_pytorch_starter.zip` は未完成コードです。
演習フォルダ内で `python3 -m unittest test_student -v` を実行して、自分で5個のTODOを完成させます。PyTorchが必要です。参考解答は公開配布に含みません。

公開後のブラウザ操作結果は、実機で確認してください。GitHub Actions成功だけでは、インタラクションの完全動作まで証明されません。
