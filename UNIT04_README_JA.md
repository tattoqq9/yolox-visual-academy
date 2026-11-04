# YOLOX Visual Academy · Unit 04 操作・検証手順

公開URL（デプロイ後）: <https://tattoqq9.github.io/yolox-visual-academy/lab/04-detection/index.html>

## WSLに最新版を取得

```bash
cd ~/projects/yolox-visual-academy
git status --short
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
node --test tests/unit04_math.test.cjs
node --check site/lab/04-detection/math.js
node --check site/lab/04-detection/app.js
bash scripts/preview.sh
```

ブラウザで `http://localhost:8765/lab/04-detection/index.html` を開く。

## 確認する操作

- [ ] 01：P3/P4/P5切替で格子形状が52²/26²/13²に変化。スライダーを変更すると **仮想編集** になり、復元ボタンで実PyTorch値へ戻る。
- [ ] 02：4px四方・1pxずれのIoU=0.6。NMS「クラス別」のON/OFFで残る候補が変わる。
- [ ] 03：GT中心を移動すると候補位置・Cost・Dynamic Kが変わる。表の候補を選ぶとキャンバスで強調される。
- [ ] 04：Obj/Cls LogitとBBox横ズレを変更すると、Lossと偏微分が変化する。
- [ ] すべての画像・操作がローカルで動く。DevToolsに未処理のエラーなし。
- [ ] PCとスマホ表示で横スクロールと操作不能がない。

## 公開側のHTTP確認

```bash
python3 scripts/verify_live.py \
  https://tattoqq9.github.io/yolox-visual-academy/ \
  --attempts 6 --interval 10
```

HTTPの成功だけでは実ブラウザでのGUI操作は証明されません。

## この教材で「実測」と「例示」を分ける

- Step01のraw値のみ、Unit03-Cで生成した<strong>未学習固定seed</strong>のPyTorch実forward。
- Step02〜04のBBox、GT、クラス、候補、Lossは計算理解のための**人工例**。写真に対する正答GTではない。
- full YOLOXの訓練・GPU update・SimOTA配列全量・AP測定は実施しない。
- ブラウザは静的。利用者のローカル写真はサーバーへアップロードしない。
- 学生用ZIPの`NotImplementedError`は学習課題であり、未完成状態のテストFAILは正常。

## 安全な開発範囲

変更対象は独立した `tattoqq9/yolox-visual-academy` のみ。頭部検出研究・Tatto Engineのコード、環境、ビルド、Git状態は変更しない。

## ソースについて

数学的参照：<https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py> と <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/losses.py>。

YOLOX公式のmainが将来変わる可能性あり。今回の候補幾何は2026-11-04に確認した`get_geometry_constraint`（中心半径1.5×stride）です。学術的再現には使用commitを記録すること。
