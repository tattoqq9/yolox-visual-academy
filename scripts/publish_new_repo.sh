#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="${1:-yolox-visual-academy}"
if [[ ! "$repo" =~ ^([A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: repo name invalid: $repo" >&2; exit 2
fi
for app in git gh python3; do
  command -v "$app" >/dev/null || { echo "ERROR: $app がありません。README_WSL_JA.mdを確認してください" >&2; exit 2; }
done
if [[ -e .git ]]; then
  echo "ERROR: 既存Gitリポジトリを変更しないため停止しました（.gitが存在します）。" >&2
  exit 2
fi
gh auth status || { echo "先に 'gh auth login' を実行してください" >&2; exit 2; }
git var GIT_AUTHOR_IDENT >/dev/null 2>&1 || {
  echo 'git user.name / user.email を設定してください（README参照）' >&2; exit 2;
}
python3 scripts/check_site.py site
echo
echo "新しい PUBLIC GitHubリポジトリ '$repo' に教材をpushします。"
echo '既存の研究用リポジトリには書き込みません。'
read -r -p '公開してよいですか？ [y/N] ' ok
[[ "$ok" == 'y' || "$ok" == 'Y' ]] || { echo '中止しました'; exit 0; }
git init -b main
git add --all
git commit -m 'Add YOLOX Visual Academy Units 01-02 with Pages deployment'
gh repo create "$repo" --public --source=. --remote=origin --push
echo
echo 'PUSH COMPLETE: GitHubにソースをpushしました。'
echo '続けて bash scripts/enable_pages_and_verify.sh を実行すると、Pages公開・検証へ進めます。'
echo 'APIの権限不足の場合はREADMEの手動設定手順を参照してください。'
