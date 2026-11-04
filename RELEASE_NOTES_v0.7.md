# YOLOX Visual Academy v0.7 — Unit 00 / Unit 01 入門改訂 (2026-11-05)

## 変更点

- **Unit 00**「AIはどうやって頭を見つけるのか？」を新設。画像分類と物体検出を区別し、RGB入力→Convolution→Backbone→PAFPN→Decoupled Head→NMS/BBoxの6段階を順に操作できる。
- 各段階の専門用語を数式なしで解説し、既存Unitへリンク。全体地図を前後に行き来可能。
- 既存Unit 03-B / 03-Cの固定seed未学習PyTorch forwardから、計算済み特徴マップの実値を23KB程度の静的JSONとして抽出。出典SHA・元テンソル形状・チャネル番号・サンプリング方法を保存。学習済み検出モデルの結果ではないことを明示。
- **Unit 01** に全体地図を追加。Kernelの平易な初出説明、RGB→グレー値の教育用変換の位置付け、写真→48×32画像→選択近傍の画素拡大を新設。
- ホームページの最初の行動を「Unit 00を始める」に変更。Unit 00の教科書とリンクを追加。
- `scripts/check_site.py`・`scripts/verify_live.py`・GitHub Actionsに新ページと計算根拠の検査を追加。

## 検証ゲート

- Python unittest：98件中97 PASS、1 skip（Unit 01〜06の既存テストを含む）。
- Node：Unit04 11件＋Unit06 12件＝23件 PASS。
- HTML：19ページ・内部参照245件 PASS（更新時点）。
- `tools/unit00/export_existing_trace.py` で既存実テンソルから再生成できること、配列・座標ごとの値が一致することをテスト。
- Unit00/Unit01のPC 1400×920、モバイル390×844：操作スモークPASS。JavaScript例外0、横スクロール0。
- ローカルHTTPからUnit00/01のHTML・JS・JSONがGETできることを確認。公開URLでのGUI操作は別途検証する。

## 未対応と注意点

- Unit00の頭部BBoxは**人が示した模式図**。モデルの推論結果ではない。Head出力値の図も概念的表示。
- 特徴マップの実値は**未学習モデル**。正式な学習済みYOLOXの特徴量・画像/ラベル・性能評価はこのリリースには含めない。
- ラベルの正式受入はv1.1以降の独立工程。自動でモデル重みや私有データを公開しない。
- シミュレータの基礎数値と一部の小モデル演習は教育用簡略化。公式YOLOX trainerへの完全一致は保証しない。
- v1.0へ向けた初学者レビューとYouTube制作は未完了。
