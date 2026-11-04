# Unit 03-C — YOLOPAFPNとDecoupled HeadのPyTorch演習

## 学習目標

Unit 03-Bで実装したConv/Depthwise Convolutionを使って、複数のテンソルの経路を**自分で接続**してください。完成済みモデルをただ呼ぶのではなく、チャンネル方向のConcatの順序、Upsampleの値の複製、Headのraw値の並びを正確に再現することが目標です。

## 始め方（WSL）

```bash
cd ~/projects/yolox-visual-academy
python3 -m venv .venv-unit03c
source .venv-unit03c/bin/activate
python3 -m pip install torch  # 既存のPyTorch環境があれば不要
cd site/code/unit03c
python3 -m unittest test_student -v
```

最初の状態では `student.py` の `NotImplementedError` が発生します。完成していないことが正しい初期状態です。**テストが全PASSすることを自力実装の完了条件**とします。

## TODO

1. `upsample_nearest2x(x)`: `[N,C,H,W]` → `[N,C,2H,2W]`、最近傍、学習パラメータ0。
2. `concat_channels(a,b)`: 同じ `N,H,W` の2配列を `dim=1` で連結。順序 `[a,b]` を保持。
3. `pafpn_top_down`: dark5→lateral→upsample + dark4→C3_p4→reduce→upsample + dark3→C3_p3。返すのは `(p3, fpn1, fpn0)`。
4. `pafpn_bottom_up`: P3→stride2→concat(fpn1)→C3_n3=P4→stride2→concat(fpn0)→C3_n4=P5。返すのは `(p3,p4,p5)`。
5. `DecoupledRawHead`: Stem後、分類2Convと回帰2Convの別枝。回帰枝からBBox4とObj1を予測。`torch.cat([reg,obj,cls],dim=1)` をそのまま返す。**sigmoidやBBox Decodeを実施しない**。

## 数値・形状の根拠

- 入力416×416。Backboneの `dark3=(1,64,52,52)`, `dark4=(1,128,26,26)`, `dark5=(1,256,13,13)`。
- Neckの出力 `P3=(1,64,52,52)`, `P4=(1,128,26,26)`, `P5=(1,256,13,13)`。
- 80クラスでは各Headの出力チャネル数が85。1クラスでは6。ただし演習上の出力形状であり、公開トレースは80クラスです。
- 経路に使うConv/C3はテストからモジュールとして渡します。学生側はその接続を実装します。
- Stemは1×1のConv-BN-SiLU。3×3枝は `conv_factory(cin,cout)` で構築し、NumPyの固定Kernelと混同しないでください。

教科書：`../../learn/03-fpn-head.html`、シミュレータ：`../../lab/03-fpn-head/index.html`。

**注意**：公開用教材・演習に学習済みの重みや教師用完成コードは含めません。今回の演習は**配線と出力順序の理解**が対象で、損失関数とSimOTAによる学習はUnit 04です。
