# Unit 01 — NumPyで畳み込みをゼロから実装する

対象: CNNの基礎を聞いたことはあるが、内部計算をまだ自力で書いたことがない学習者。

## 実行環境

Python 3.10以上とNumPyが必要です。`python -m pip install numpy` で導入できます。

## 学び方

1. まずブラウザ教材の「01 画像とRGB」で自分の画像を選び、数字をクリックして確認。
2. 「02 畳み込み」で`明るさ1チャネル`の9項を順に計算し、Kernelも変更。
3. 「03 特徴マップ」の再生で出力が埋まることを確認。
4. 「04 チャネル」「05 次の層」で出力数とstrideを理解。
5. `practice/cnn_from_scratch.py` の5つのTODOを実装。**スターターには解答実装を含めていません**。
6. リポジトリのルートで次のテストを実行。

```powershell
python -m unittest discover -s tests -p test_student.py -v
```

最初はTODOがあるため、テストが失敗するのが正常です。

## 実装前の手計算

### 問題A — 1チャネル、paddingなし

```
入力X = [[1,2,3],
         [4,5,6],
         [7,8,9]]
Kernel = [[ 1,0,-1],
          [ 1,0,-1],
          [ 1,0,-1]]
```

- 出力は何×何になる？
- 数値はいくつ？
- Kernelを180度回転させると符号はどう変わる？PyTorch `Conv2d`ではどちらを使う？

### 問題B — 複数チャネル

入力が3チャネル、出力Kernelが2個のとき、重みテンソルの形状は `[2,3,3,3]`。なぜ出力が6チャネルではなく2チャネルなのか説明する。

### 問題C — stride

入力が48×32、kernel3、padding1、stride2のとき、出力が24×16になる理由を図と数式の両方で示す。

## ブラウザとPythonで同じ値を比較

ブラウザの最終章で**48×32のRGBサンプルCSV**を保存してください。必ずそのCSVからPythonで明るさを計算し、選択位置にKernelを適用すると、ブラウザで確認した出力値を再現できます（表示時の丸めは除く）。

実行例：

```powershell
python practice/compare_browser.py samples_rgb_48x32.csv 22 12 blur
```

## このUnitでまだ学ばないもの

この教材の固定Kernelは学習済みYOLOX重みではありません。続くUnitでBackbone、Neck、格子、BBox、SimOTA、Lossへ進みます。
