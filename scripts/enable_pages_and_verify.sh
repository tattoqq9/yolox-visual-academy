#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for app in gh python3 git; do
  command -v "$app" >/dev/null || { echo "ERROR: $app が必要です" >&2; exit 2; }
done
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: GitHubリモートoriginがありません。先にpublish_new_repo.shを実行してください。" >&2
  exit 2
fi
gh auth status >/dev/null
repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
echo "Target repository: $repo"
read -r -p 'このリポジトリのGitHub Pagesを有効化し、公開検証しますか？ [y/N] ' ok
[[ "$ok" == 'y' || "$ok" == 'Y' ]] || { echo '中止しました'; exit 0; }
if current="$(gh api "repos/$repo/pages" --jq '.build_type' 2>/dev/null)"; then
  if [[ "$current" != 'workflow' ]]; then
    echo "既存Pagesのbuild_typeが '${current}' です。変更せず停止します。"
    echo "GitHub Settings > Pages > Source で GitHub Actions に切り替えてから再実行してください。"
    exit 2
  fi
  echo 'Pages: existing workflow mode'
else
  echo 'Enabling Pages with GitHub Actions...'
  if ! gh api --method POST "repos/$repo/pages" -f build_type=workflow >/dev/null; then
    echo 'GitHub APIからPagesを設定できませんでした。Settings > Pages > Source をGitHub Actionsに設定し、再実行してください。' >&2
    exit 1
  fi
fi
echo 'Starting deployment workflow...'
gh workflow run deploy-pages.yml --ref main
sleep 4
run_id="$(gh run list --workflow deploy-pages.yml --limit 15 --json databaseId,event --jq '[.[] | select(.event=="workflow_dispatch")][0].databaseId // empty')"
if [[ -z "$run_id" ]]; then
  echo 'ワークフローIDが取得できません。GitHub Actionsページで実行結果を確認してください。' >&2
  exit 1
fi
echo "Watching GitHub Actions run $run_id ..."
gh run watch "$run_id" --exit-status
url="$(gh api "repos/$repo/pages" --jq '.html_url')"
if [[ -z "$url" || "$url" == "null" ]]; then
  echo '公開URLが取得できません。GitHub Settings > Pagesを確認してください。' >&2
  exit 1
fi
printf '\nPublished URL: %s\n' "$url"
python3 scripts/verify_live.py "$url" --attempts 12 --interval 10
echo 'PASS: GitHub Actionsが成功し、主要URLへのHTTPアクセスを確認しました。'
