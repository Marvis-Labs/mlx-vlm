#!/usr/bin/env bash
set -euo pipefail

marker="<!-- mlx-vlm-ci:attempt:${CI_ATTEMPT_ID} -->"
grep -Fq "$marker" "$CI_SUMMARY"
comment_id="$(
  gh api "repos/$CI_REPOSITORY/issues/$CI_PR/comments?per_page=100" \
    --jq "[.[] | select(.user.type == \"Bot\" and (.body | contains(\"$marker\")))][0].id // empty"
)"
[[ -z "$comment_id" ]] || exit 0

jq -n --rawfile body "$CI_SUMMARY" '{body: $body}' |
  gh api --method POST "repos/$CI_REPOSITORY/issues/$CI_PR/comments" --input - >/dev/null
