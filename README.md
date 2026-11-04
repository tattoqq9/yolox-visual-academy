# YOLOX Visual Academy

一般公開している対話型教材（Unit 01・02）と、Unit 03-Aのレビュー版。最終的にYouTube・教科書・Web実験・NumPy/PyTorchを接続するための学習プロジェクトです。

**公開サイト:** https://tattoqq9.github.io/yolox-visual-academy/

## 既存WSL環境での更新

```bash
cd ~/projects/yolox-visual-academy
git pull --ff-only
python3 scripts/check_site.py site
bash scripts/preview.sh
```

`http://localhost:8765/` をWindowsブラウザで開く。Ctrl+Cで停止。

## 自動検証

```bash
python3 scripts/check_site.py site
node --test tests/test_unit03_math.cjs
python3 -m unittest discover -s tests -p 'test_site_*.py' -v
```

Webブラウザ検証は `pip install playwright` および `python -m playwright install chromium` を行い、previewサーバーを起動した上で `python scripts/browser_smoke.py` を使用。GitHub Actionsはmainの公開前に静的・数値・ブラウザを検証する。

学生向けコードは `site/code/unit03a/practice/`。`site/downloads/unit03a_numpy_starter.zip` にも同じ未完成コードが入る。**Unit 03-AのCSPは教育用簡略モデルで、公式YOLOXの学習済みネットワークではない。**

- [ROADMAP.md](ROADMAP.md)
- [docs/RELEASE_GATES.md](docs/RELEASE_GATES.md)
- [docs/YOUTUBE_EP03A_SCRIPT_JA.md](docs/YOUTUBE_EP03A_SCRIPT_JA.md)
- [CHANGELOG.md](CHANGELOG.md)

教材全体のライセンス条件は未確定。人物写真を第三者が再利用できると保証しません。
