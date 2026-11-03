# Unit 02：NumPyでStride・受容野・YOLOX格子を自作する

**対象:** CNNの基礎知識があり、Python/NumPyを使える初学者

## 目的

シミュレータの数値を「見た」段階から、自力で数式をコードへ変換する段階へ進みます。ネットワーク学習は不要です。

## 手順

1. Python 3.10以上とNumPyを用意する: `python -m pip install numpy`
2. `practice/student.py` の TODO 1–5 を順に埋める
3. ZIPを展開した `unit02/` ディレクトリで `python -m unittest discover -s tests -v` を実行する
4. 最初は未実装なのでテストが失敗します。全20件PASSを目指します
5. WebのUnit02「01 Strideと畳み込み」で「入力16×16のCSVを保存」
6. `python practice/compare_browser.py unit02_gray16.csv --stride 2 --kernel edge --ox 3 --oy 1` で再計算する

## 各課題のヒント

- **TODO 1**: `H_out=(H+2-3)//stride+1`。`image[y,x]` は範囲内でのみ参照し、それ以外は0。
- **TODO 2**: 最初は `R=1`,`J=1`,`start=0.5`。各層でRを増やし、最後にJを更新。paddingはstartに影響。
- **TODO 3**: `np.meshgrid(...,indexing='ij')` で列X/行Yを作り、`(g+0.5)*stride`。
- **TODO 4**: Headの中心予測は `(g+t)*stride`、幅・高さは `exp(t)*stride`。
- **TODO 5**: GT中心からのX距離とY距離をそれぞれ `1.5*stride` と比較。境界は `<` であり `<=` ではありません。

## 注意事項

- Unit02は、16×16にダウンサンプルした画像での固定Kernel実験です。これは学習済みYOLOXの中間特徴ではありません。
- GUIの416画像へ描いた赤枠は人工的なGTです。人手でアノテーションした頭部を意味しません。
- 受容野Rは理論上の入力依存範囲。CNNの実効受容野・YOLOX実モデルの受容野とは違います。
- 幾何候補はSimOTAの最終正例ではありません。

## 参考

- YOLOX公式: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
- Distill receptive fields: https://distill.pub/2019/computing-receptive-fields/
