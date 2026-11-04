# リリースゲート（2026-11-04）

## 3種類の検証を混同しない

1. **数学・静的ゲート**：`python3 scripts/check_site.py site`、`node --test tests/test_unit03_math.cjs`、`python3 -m unittest discover -s tests -p 'test_site_*.py' -v`。
2. **ブラウザ操作ゲート**：`python3 scripts/browser_smoke.py`（Playwright Chromium＋ローカルHTTPサーバー）。ChatGPT側のブラウザはローカルURLが管理者設定で拒否されることがあり、その場合のインメモリ検査は代替的な表示・操作試験でありHTTP検査ではない。
3. **公開サイトゲート**：GitHub Actions成功、GitHub Pages公開、主要URLのHTTP成功、公開先のブラウザ操作を確認。

## 既存研究への影響防止

- `tatto-engine-phase0-m01`、頭部検出研究用ディレクトリへの読み書き・停止・プロセス変更禁止。
- 本教材repoのmainは数値・GUIの検証完了後に更新する。
- 計算と描画は同じ `site/lab/03-blocks/math.js` を利用。
- 学習済み重みと固定の教育用重みを混同しない。

## 公開上の未決定事項

- 教材全体のオープンソースライセンス：未選定。配布・翻案・第三者利用の条件を決めるまで、ユーザー以外が自由に再配布できるとは案内しない。
- 人物写真 `003Z-selfie.png` の権利・本人の許諾範囲：現ユーザーによるサイト使用の指示は得ているが、第三者の再配布・二次利用まで保証されているわけではない。
- YouTube動画の公開：未実施。
- 初学者による理解度試験：未実施。
