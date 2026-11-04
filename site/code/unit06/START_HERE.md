# Unit 06 — Tiny BBox Research Exercises

まず https://tattoqq9.github.io/yolox-visual-academy/lab/06-tiny/index.html を操作し、
比較式を開く前に自分の仮説を記入してください。

## 演習目的

7つのTODOを自力で実装します。IoUの幾何、Gridの中心点、SimOTA候補の中心制約、
Dynamic K、YOLOX Decode、符号付きLTRB、幅高さの有効性を区別します。

```bash
python3 -m unittest -v test_student.py
```

課題コードは意図的に未完成です。最初のテストが失敗するのは正しい動作です。
このZIPにはYOLOX学習済み重み、実人物画像、実頭部GT、秘密の研究ログを含みません。

## 注意

- Signed LTRBの4値は `l,t,r,b` の**符号付き距離**です。幅 = l+r、高さ = t+b。
- Signed LTRBの生出力には順序保証がありません。モデルが常に有効BBoxを返すとは限りません。
- YOLOXとSigned LTRBはパラメータの単位・勾配が異なります。スライダー値をそのまま公平な性能比較に使えません。
- 実性能の検証には同一データ、同一seed群、等しい学習手順、同一のAP/Recall、BBox品質とScoreの分離が必要です。
