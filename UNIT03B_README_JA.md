# YOLOX Visual Academy — Unit 03-B 公開手順

公開ページ（反映後）：https://tattoqq9.github.io/yolox-visual-academy/lab/03-nano/index.html

教材の学習順： Unit03-A の数値計算 → Unit03-B の実構造 → `site/code/unit03b/` のPyTorch演習。

## WSLで最新を取得

```bash
cd ~/projects/yolox-visual-academy
git pull --ff-only
python3 scripts/check_site.py site
python3 -m unittest discover -s tests -v
bash scripts/preview.sh
```

Windows Chrome またはEdgeで http://localhost:8765/lab/03-nano/index.html を開き、以下を試してください。

- [ ] セルフィーの前処理画像が表示される
- [ ] Focus, dark2, dark3, dark4, dark5 の出力shapeを順番に表示できる
- [ ] 3D特徴マップをドラッグで回転できる
- [ ] チャネル選択とマップ上の座標によって出力値が変わる
- [ ] Depthwise比較の数値：64→64, K=3でStandard 36,864、DW+PW 4,672
- [ ] CSP分岐のConcat前が32+32ch、後が64ch
- [ ] ページの警告が**未学習重み**と明示している
- [ ] DevTools Consoleに赤い例外が出ない

`Ctrl+C`でサーバ停止。`pytest`は必須ではありません。PyTorchテストは**学習用に独立した環境**で行ってください。

```bash
python3 -m unittest tools/unit03b/test_nano_reference.py -v
```

Web教材を開くためにPyTorchをインストールする必要はありません。student演習はPyTorchが必要です。

## 注意

同梱の`trace.json`は公式の設定に対応するYOLOX-Nanoを固定seedの**未学習重み**で実行した値です。形状・演算の可視化が目的で、識別性能を示していません。YOLOX公式の学習済みチェックポイントやユーザーの頭部検出研究用のデータ・重みは一切含みません。

ソースはこの独立した `tattoqq9/yolox-visual-academy` に限定して変更します。その他の研究・ゲームエンジンのリポジトリは対象外です。
