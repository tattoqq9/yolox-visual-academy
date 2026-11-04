# YOLOX Visual Academy | 公開版ロードマップ

最終目標は「CNN基礎を知る学生がYOLOX-Nanoをスクラッチ実装し、極小頭部の問題からSigned LTRBのような改善案を設計・評価できること」です。

| Unit | 内容 | 現状（2026-11-04） |
|---|---|---|
| 01 | 画像・RGB・畳み込み・特徴マップ | 公開済み。セルフィー初期画像とGUIをユーザー確認済み |
| 02 | Stride・受容野・格子・幾何候補 | 公開済み。セルフィー初期画像とGUIをユーザー確認済み |
| 03-A | Conv→BatchNorm→SiLU→CSP型の分岐・結合 | レビュー用実装。教育用固定重み（学習済みYOLOXではない） |
| 03-B | CSPDarknetの実モデル構造、PAFPN、Decoupled Head | 未実装 |
| 04 | BBox decode、IoU、NMS、SimOTA、Loss・勾配 | 未実装 |
| 05 | YOLOX-Nanoをゼロから組み、学習と評価 | 未実装 |
| 06 | 極小頭部の原因分解、Signed LTRB仮説・比較実験 | 未実装 |

各段階の公開条件：数学／Python／JS／ブラウザ／公開リンク／第三者レビュー（推奨）の各ゲートを区別する。Unit 03-AのCSPは概念を理解するための簡略系であり、YOLOX公式CSPLayerと直接互換ではない。

YouTubeはサイト・教材の同じ計算を記録し、教材更新時に矛盾が生じないようにする。現在は動画制作準備段階であり、公開は未実施。
