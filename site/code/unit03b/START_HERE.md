# Unit 03-B — 公式YOLOX-Nano構造のPyTorchスクラッチ演習

**学習目標：** Focusのチャネル再配置、Depthwise Separable Convの定義、CSPのConcatを自分で実装する。

## WSLでの準備

Python 3.10以上とPyTorchが必要です。環境は必ずこの教材用の独立した仮想環境を使用してください。ほかの研究・開発環境へインストールしないこと。

```bash
cd ~/projects/yolox-visual-academy
python3 -m venv .venv-unit03b
source .venv-unit03b/bin/activate
python3 -m pip install --upgrade pip
# 既にPyTorchが使える場合はこの行は不要。
python3 -m pip install torch
cd site/code/unit03b
python3 -m unittest test_student -v
```

初回のテストは`NotImplementedError`で失敗します。`student.py`の4箇所を実装してPASSさせてください。`nn.Conv2d`による重みの学習は次のユニットへ発展させます。

## 次元と計算の説明

- `Focus`は`[N,C,H,W]`を`[N,4C,H/2,W/2]`にし、例えば`[N,3,416,416]`から`[N,12,208,208]`へ変換する。チャネル順TL,BL,TR,BR。
- `Depthwise`: `groups=cin`とすると各入力チャネルを別々にK×Kで計算する。
- `Pointwise`: 1×1 Convで全入力チャネルを混合してcoutチャネルへ変える。
- `CSP`は2枝の出力を`dim=1`（チャネル次元）で結合する。
- BNの学習パラメータγ/βは重み数の式には含めない。実Convブロックの総パラメータ数には加える。

## 教師用の参照実装

`tools/unit03b/nano_reference.py`は公式構造に対応する動作する参考モデルです。先に学生版を完成させてから確認してください。参照モデルの学習パラメータ数は80クラス構成で912,159個です。

## 実測テンソルを更新したい場合

```bash
# プロジェクトルートへ戻る
cd ~/projects/yolox-visual-academy
python3 -m pip install pillow numpy  # トレース生成時のみ必要
python3 tools/unit03b/export_trace.py --output /tmp/nano_trace_untrained.json --preview /tmp/nano_input.jpg
```

既定のtraceは**未学習重み**で生成した数値です。公式の`yolox_nano.pth`(80クラス)を入手し、信頼できるチェックポイントであることを確認した場合だけ、次のように比較用ファイルへ出力できます。配布・公開は権利確認後に別途判断します。

```bash
python3 tools/unit03b/export_trace.py \
  --weights /path/to/yolox_nano.pth \
  --output /tmp/nano_trace_trained.json \
  --preview /tmp/nano_preprocessed.jpg
```

学習済み実験の入力は、16bitや正規化済みRGBではなく、入力画像を416×416へリサイズ・padding後、BGRの0〜255 float32とするものです。前処理にPillowを使うため、OpenCV公式デモと数ピクセルの値が異なる可能性があります。

`trace.json`には学習済みモデルの「頭部検出」結果は含まれません。前処理や前提を確認してから各層を解釈すること。

チェックポイントを指定しただけでは学習済みと保証できません。独立に検証している場合だけ `--checkpoint-status verified-pretrained` を明示してください。
