#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-PHMFactory-P01}"
DEST="${2:-../PHMFactory-P01}"
BRANCH=research/p01-operator-bias-20260907
command -v gh >/dev/null
command -v git >/dev/null
OWNER="$(gh api user --jq .login)"
[[ ! -e "$DEST" ]] || { echo "Refusing to overwrite $DEST" >&2; exit 2; }
if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "$OWNER/$NAME already exists; choose a new name or inspect it manually." >&2
  exit 2
fi
gh repo fork PHMbench/PHM-Vibench --fork-name "$NAME" --clone=false --remote=false
git clone "https://github.com/$OWNER/$NAME.git" "$DEST"
git -C "$DEST" remote add upstream https://github.com/PHMbench/PHM-Vibench.git
git -C "$DEST" fetch upstream "$BRANCH"
git -C "$DEST" switch -c "$BRANCH" "upstream/$BRANCH"
echo "Local checkout ready: $DEST ($BRANCH). No branch was pushed or merged."
