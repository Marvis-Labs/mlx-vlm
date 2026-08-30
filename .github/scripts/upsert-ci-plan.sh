#!/usr/bin/env bash
set -euo pipefail

marker='<!-- mlx-vlm:ci:plan -->'
comment_id="$(
  gh api "repos/$CI_REPOSITORY/issues/$CI_PR/comments?per_page=100" \
    --jq "[.[] | select(.user.type == \"Bot\" and (.body | contains(\"$marker\")))][0].id // empty"
)"

if [[ -n "$comment_id" ]]; then
  jq -n --rawfile body "$CI_SUMMARY" '{body: $body}' |
    gh api --method PATCH "repos/$CI_REPOSITORY/issues/comments/$comment_id" --input - >/dev/null
else
  jq -n --rawfile body "$CI_SUMMARY" '{body: $body}' |
    gh api --method POST "repos/$CI_REPOSITORY/issues/$CI_PR/comments" --input - >/dev/null
fi
