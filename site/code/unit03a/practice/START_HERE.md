# Unit 03-A — スクラッチ実装を始める

**推奨順序：** Web実験室 → 教科書 → `student.py` → テスト → 実験室に戻る。

1. Python 3.11以降と NumPy を用意します。`python3 -m pip install numpy`
2. `student.py` の4関数を実装します。`NotImplementedError` を消してください。
3. `python3 -m unittest -v test_student.py` を実行します。
4. Web実験室で任意のピクセルをクリックし、27項の和と自分のConv出力を比較します。
5. 最後に、`torch.nn.Conv2d` の入力 `[N,C,H,W]` と教材の `[H,W,C]` の変換を説明してみましょう。

**注意：** 教材ではBatchNormを1枚の画像内の H,W で計算しています。PyTorchの学習時BatchNormはバッチ N を含む N,H,W の平均・分散を使います。評価時は通常 running stats を用いるので、この関数と常に同じ値ではありません。

教育用CSPは分岐・Concatの課題から始めます。公式YOLOXのCSPLayer完全実装はUnit 03-B以降で扱います。
