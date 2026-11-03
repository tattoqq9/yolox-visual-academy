# YOLOX Visual Academy v0.2 — WSL / GitHub Pages 公開手順

このZIPは**新規の独立リポジトリ**を作るためのものです。既存のVoxelエンジン／頭部検出研究リポジトリを変更しません。研究用ファイルや提供された元の人物写真も含まれません。

## 0. ダウンロードと配置

1. ChatGPTから `yolox-academy-v0.2-wsl-pages.zip` をWindowsの `ダウンロード` フォルダへ保存してください。
2. WSL (Ubuntu) ターミナルを開きます。パスは**保存場所に応じて**変更してください。

```bash
mkdir -p ~/projects/yolox-visual-academy
cd ~/projects/yolox-visual-academy
unzip /mnt/c/Users/tatto/Downloads/yolox-academy-v0.2-wsl-pages.zip
```

`unzip` がない場合は `sudo apt update && sudo apt install -y unzip` で導入します。Windowsのユーザー名が異なる場合は `/mnt/c/Users/<Windowsユーザー名>/Downloads/` へ読み替えてください。ZIPには `site/`・`scripts/`・`.github/workflows/` が含まれています。

## 1. WSLでオフライン／ローカル確認

```bash
python3 scripts/check_site.py site
bash scripts/preview.sh
```

WindowsのChromeまたはEdgeで `http://localhost:8765/` を開いてください。
- Unit 01「畳み込み」とUnit 02「Stride・格子」の両方が開く
- 教科書↔実験へのリンクが動く
- 「画像を使う」で選んだファイルが表示される（アップロード先サーバーはありません）
- コンソールの赤いJavaScript例外がない

停止するにはターミナルで `Ctrl+C` を押します。

## 2. GitHubの認証 (初回のみ)

```bash
git --version
gh --version
gh auth login
gh auth status
git config --global user.name "あなたのGitHub表示名"
git config --global user.email "GitHubに登録済みのメールアドレス"
```

`gh` が無ければ Ubuntu の環境に応じたGitHub CLIの導入手順を https://cli.github.com/ で確認してください。`git config` を既に設定済みの場合は変更不要です。公開コミットに本名のメールアドレスを表示させたくない場合、GitHubの設定で取得できる `noreply` アドレスの利用を推奨します。

## 3. 独立した公開リポジトリを新規作成しpush

```bash
bash scripts/publish_new_repo.sh yolox-visual-academy
```

このスクリプトは、ローカル検証後に確認プロンプトを表示し、了承した場合だけ新規PUBLICリポジトリの作成とpushを行います。既存の `.git` があれば安全のため停止します。**既存リポジトリへの上書きやforce pushは実行しません。**

`gh repo create` で失敗した場合は、原因を確認してから手動で作成してください。既存リポジトリへのpushが必要な場合は独自の手順が必要です。

## 4. WSLからGitHub Pagesの公開・動作確認まで実施

```bash
bash scripts/enable_pages_and_verify.sh
```

このスクリプトは確認後、GitHub APIで**PagesをGitHub Actions方式に設定**し、デプロイを実行して、Actionsの完了を待ちます。成功後に主要10ページ・アセットを実際の公開URLから取得して `PASS/FAIL` を報告します。途中のAPI呼び出しで失敗したら、そこで停止します。

**GitHub APIの権限不足などで失敗した場合**は、GitHubで `Settings` → `Pages` → `Build and deployment` → `Source` を **GitHub Actions** に設定してください。その後 `Actions` → `Validate and deploy GitHub Pages` → `Run workflow` (main) を実行できます。アクションが成功したら、以下をWSLで実行します。

```bash
python3 scripts/verify_live.py https://<GitHubユーザー名>.github.io/yolox-visual-academy/ --attempts 12 --interval 10
```

公開URLは `Settings → Pages` で確認できます。主要URLのHTTP検査がPASSしても、GUI操作の確認とは別です。WindowsのChrome・Edgeで公開URLを開き、Unit 01・02のアニメーション、画像読み込み、教科書の章ジャンプを試してください。

## 5. 継続的な修正・再公開

初回公開後に `site/` 内を更新した場合は、以下で再公開できます。

```bash
python3 scripts/check_site.py site
git add site
git commit -m "Update YOLOX learning site"
git push
```

mainにpushすると、GitHub Actionsが自動的に公開内容を更新します。

## トラブルシューティング

- `unzip: cannot find` : ZIPのWindows保存場所とファイル名を再確認。`ls /mnt/c/Users/tatto/Downloads/*yolox*` で探す
- `git identity unknown` : Gitのuser.name / user.emailを設定
- `gh: command not found` : GitHub CLIをインストール
- GitHub Pagesが404 : `Source: GitHub Actions` を選んでいるか、ActionがPASSしたか、URLの最後の `/` を確認
- フロントが真っ白 : ブラウザの開発者ツール（F12 → Console）を確認してエラーを控える
- ZIPのダウンロード自体ができない : ChatGPTの通常のPCブラウザでリンクを開く。ファイル名は英数字のみ。別ブラウザでも改善しない場合はエラー文面を共有

## このパッケージに含まれるもの

- `site/` 送信される公開サイト (Unit 01/02, 教科書, JS, 学生用コード)
- `.github/workflows/deploy-pages.yml` 2つのUnitを自動検証・デプロイ
- `scripts/check_site.py` ローカルの静的ファイル/リンク検査
- `scripts/verify_live.py` 公開後のHTTP実測検査
- `scripts/preview.sh` ローカルHTTPサーバー
- `scripts/publish_new_repo.sh` 確認付き新規リポジトリ作成・push
- `scripts/enable_pages_and_verify.sh` APIでGitHub Pages設定、Actions実行、公開URL実測

このZIP内にはPythonの教員用解答はありません。YouTubeの投稿・ライセンス条件の決定・第三者レビューも未実施です。
