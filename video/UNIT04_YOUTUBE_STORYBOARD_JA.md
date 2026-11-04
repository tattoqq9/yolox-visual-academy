# YouTube 第4・5回案：BBox Decode・SimOTA・Loss

**目的**：授業前に4pxのBBoxを1pxずらしたIoUを自分で答えられる。授業後にはYOLOXのLoss計算をコードにできる。

| シーン | 表示・操作 | 説明する数値 |
| --- | --- | --- |
| 01 導入 | 003Z-selfie.pngとUnit03-CのHead raw85値 | 416入力・P3=52² |
| 02 Decode | 実forwardのtx/ty/tw/thスライダー | grid+raw と exp(raw) |
| 03 反例 | 4px/80px BBoxを横へ1px動かす | IoU=0.6 vs 0.9753 |
| 04 NMS | 重複した候補からScore順に減らす | NMSとSimOTAは別役割 |
| 05 正例割当 | 2つのGT中心を動かして候補の色を切替 | 1.5×stride geometry |
| 06 Dynamic K | IoU top10合計とcost、競合 | KはGTごとに変化 |
| 07 Loss | z_objとz_clsを動かす | d BCE/dz = sigmoid(z)-t |
| 08 研究質問 | 4pxや0-IoU時の微分を比較 | 学習が困難になる条件を仮説化 |

**放送倫理・事実関係**：Step01のみ未学習の実PyTorch raw。Step02〜04のGT・候補は人工作成。学習済み頭部認識をしたと主張しない。公式YOLOXソースから実装差を必ず確認。モデルやGTの捏造を実験結果として公表しない。

公開用サイトは静的で、ユーザー画像をネットワークに送信しない。フル学習、公式チェックポイント、研究データは教材から隔離する。
